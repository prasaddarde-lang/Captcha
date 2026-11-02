"""Optional REST API (FastAPI).

Authentication: header ``X-API-Key`` must match env var ``CAPTCHA_FW_API_KEY``.
If that variable is unset, only loopback clients (127.0.0.1 / ::1) are accepted.
HTTPS: set ``CAPTCHA_FW_SSL_CERTFILE`` and ``CAPTCHA_FW_SSL_KEYFILE``.
"""
from __future__ import annotations

import hmac
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import Depends, FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from core.response_handler import TaskRequest, TaskResponse  # noqa: E402
from core.security import sanitize_text  # noqa: E402

LOOPBACK = {"127.0.0.1", "::1", "localhost", "testclient"}


class OperatorAction(BaseModel):
    operator: str = Field(min_length=1, max_length=80)
    notes: str | None = Field(default=None, max_length=500)


class FeedbackIn(BaseModel):
    execution_id: int
    actual_type: str = Field(min_length=1, max_length=40)
    reviewer: str | None = Field(default=None, max_length=80)


class ApproveIn(BaseModel):
    approver: str = Field(min_length=1, max_length=80)
    allow_regression: bool = False


class ProviderVerifyIn(BaseModel):
    site: str = Field(min_length=1, max_length=200)
    provider: str
    token: str = Field(min_length=1, max_length=4096)
    remote_ip: str | None = None


def create_app(framework=None) -> FastAPI:
    from core.framework import CaptchaFramework

    fw = framework or CaptchaFramework()
    app = FastAPI(title="CAPTCHA Management Framework API", version="1.0.0",
                  description="Detect CAPTCHA challenges and route them to authorized resolution workflows.")

    def auth(request: Request) -> None:
        expected = os.environ.get(fw.settings.api_key_env)
        if expected:
            supplied = request.headers.get("X-API-Key", "")
            if not hmac.compare_digest(supplied.encode(), expected.encode()):
                raise HTTPException(401, "invalid or missing X-API-Key")
        elif (request.client.host if request.client else "") not in LOOPBACK:
            raise HTTPException(403, f"set {fw.settings.api_key_env} to allow non-local clients")

    @app.exception_handler(ValueError)
    async def _value_error(_: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"detail": sanitize_text(str(exc), 500)})

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_version": fw.detector.classifier.version,
                "image_model_version": fw.image_classifier.version}

    @app.post("/v1/tasks", response_model=TaskResponse, dependencies=[Depends(auth)])
    def run_task(task: TaskRequest) -> TaskResponse:
        return fw.run(task)

    @app.get("/v1/verifications", dependencies=[Depends(auth)])
    def list_verifications(status: str | None = "PENDING") -> list[dict]:
        return fw.db.list_verifications(status.upper() if status else None)

    @app.get("/v1/verifications/{request_id}", dependencies=[Depends(auth)])
    def get_verification(request_id: str) -> dict:
        vr = fw.hv.get(request_id)
        if not vr:
            raise HTTPException(404, "unknown request_id")
        return vr

    @app.post("/v1/verifications/{request_id}/complete", dependencies=[Depends(auth)])
    def complete(request_id: str, body: OperatorAction) -> dict:
        if not fw.hv.complete(request_id, body.operator, body.notes):
            raise HTTPException(409, "request is not pending")
        return fw.hv.get(request_id)

    @app.post("/v1/verifications/{request_id}/reject", dependencies=[Depends(auth)])
    def reject(request_id: str, body: OperatorAction) -> dict:
        if not fw.hv.reject(request_id, body.operator, body.notes):
            raise HTTPException(409, "request is not pending")
        return fw.hv.get(request_id)

    @app.get("/v1/verifications/{request_id}/screenshot", dependencies=[Depends(auth)])
    def screenshot(request_id: str):
        vr = fw.hv.get(request_id)
        if not vr or not vr.get("screenshot_path"):
            raise HTTPException(404, "no screenshot")
        p = Path(vr["screenshot_path"]).resolve()
        try:
            p.relative_to(Path(fw.settings.screenshot_dir).resolve())
        except ValueError:
            raise HTTPException(403, "screenshot outside allowed directory")
        if not p.exists():
            raise HTTPException(404, "screenshot purged")
        return FileResponse(p, media_type="image/png")

    @app.post("/v1/feedback", dependencies=[Depends(auth)])
    def feedback(body: FeedbackIn) -> dict:
        if not fw.db.get_execution(body.execution_id):
            raise HTTPException(404, "unknown execution_id")
        return {"feedback_id": fw.db.record_feedback(body.actual_type, body.execution_id, reviewer=body.reviewer)}

    @app.get("/v1/executions", dependencies=[Depends(auth)])
    def executions(limit: int = 50) -> list[dict]:
        return fw.db.list_executions(min(max(limit, 1), 500))

    @app.get("/v1/reports/performance", dependencies=[Depends(auth)])
    def report() -> dict:
        from learning.performance_report import build_report
        return build_report(fw.db)

    @app.get("/v1/models", dependencies=[Depends(auth)])
    def models() -> list[dict]:
        from learning.model_evaluation import ModelRegistry
        ModelRegistry(fw.db, fw.settings).ensure_baseline()
        return fw.db.list_models()

    @app.post("/v1/models/{version}/approve", dependencies=[Depends(auth)])
    def approve(version: str, body: ApproveIn) -> dict:
        from learning.model_evaluation import ApprovalError, ModelRegistry
        try:
            return ModelRegistry(fw.db, fw.settings).approve(version, body.approver, body.allow_regression)
        except ApprovalError as exc:
            raise HTTPException(409, str(exc))

    @app.post("/v1/models/{version}/deploy", dependencies=[Depends(auth)])
    def deploy(version: str) -> dict:
        from learning.model_evaluation import ApprovalError, ModelRegistry
        try:
            m = ModelRegistry(fw.db, fw.settings).deploy(version)
        except ApprovalError as exc:
            raise HTTPException(409, str(exc))
        fw.reload_model()
        return m

    @app.get("/v1/models/image", dependencies=[Depends(auth)])
    def image_models() -> dict:
        from training.model_registry import ImageModelRegistry
        reg = ImageModelRegistry()
        return {"production": reg.production_version(), "loaded": fw.image_classifier.version,
                "models": reg.summary()}

    @app.post("/v1/models/image/{version}/approve", dependencies=[Depends(auth)])
    def approve_image_model(version: str, body: ApproveIn) -> dict:
        """A named reviewer approves a candidate; it is then promoted (with automatic rollback on failure)."""
        from training.model_registry import ApprovalError, ImageModelRegistry
        reg = ImageModelRegistry()
        try:
            reg.approve(version, body.approver, body.allow_regression)
            m = reg.promote(version)
        except ApprovalError as exc:
            raise HTTPException(409, str(exc))
        return {"version": version, "status": m["status"], "approved_by": m["approved_by"]}

    @app.post("/v1/provider/verify", dependencies=[Depends(auth)])
    def provider_verify(body: ProviderVerifyIn) -> dict:
        """Server-side siteverify for sites you own (token produced by a human)."""
        from core.resolution_engine import verify_provider_token
        entry = fw.settings.load_sites().get(body.site)
        if not entry:
            raise HTTPException(404, "site not registered in sites.json")
        try:
            return verify_provider_token(body.provider, body.token, entry, fw.settings, body.remote_ip,
                                         fw.http_client)
        except PermissionError as exc:
            raise HTTPException(403, str(exc))
        except RuntimeError as exc:
            raise HTTPException(503, str(exc))

    @app.get("/operator", response_class=HTMLResponse)
    def operator_console() -> str:
        return OPERATOR_HTML

    return app


OPERATOR_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>CAPTCHA Operator Console</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{font-family:Segoe UI,Arial,sans-serif;margin:20px;color:#222;max-width:1000px}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:8px;text-align:left}
button{padding:6px 12px;margin-right:6px;cursor:pointer}.ok{background:#1b7f3b;color:#fff;border:0}
.no{background:#b3261e;color:#fff;border:0}img{max-width:420px;border:1px solid #ccc;margin-top:6px}
.muted{color:#777} input{padding:6px}
</style></head><body>
<h1>CAPTCHA Operator Console</h1>
<p class="muted">Complete each challenge in the robot's browser window, then mark it complete here.
The workflow resumes automatically.</p>
<p>Operator name <input id="op" placeholder="Your name"> API key <input id="key" type="password" placeholder="X-API-Key (if set)"></p>
<table><thead><tr><th>Request</th><th>Task</th><th>Type</th><th>Host</th><th>Expires (UTC)</th><th></th></tr></thead>
<tbody id="rows"><tr><td colspan="6" class="muted">Loading...</td></tr></tbody></table>
<script>
const $ = s => document.querySelector(s);
try { $('#op').value = sessionStorage.getItem('op') || ''; $('#key').value = sessionStorage.getItem('key') || ''; } catch(e) {}
function hdrs(){ const h={'Content-Type':'application/json'}; const k=$('#key').value; if(k) h['X-API-Key']=k; return h; }
async function load(){
  try { sessionStorage.setItem('op',$('#op').value); sessionStorage.setItem('key',$('#key').value);} catch(e) {}
  const r = await fetch('/v1/verifications?status=PENDING',{headers:hdrs()});
  const tb = $('#rows');
  if(!r.ok){ tb.innerHTML = `<tr><td colspan=6>Error ${r.status}</td></tr>`; return; }
  const rows = await r.json();
  if(!rows.length){ tb.innerHTML = '<tr><td colspan=6 class="muted">No pending requests</td></tr>'; return; }
  tb.innerHTML = '';
  for(const v of rows){
    const tr = document.createElement('tr');
    const cells = [v.request_id, v.task_id, v.captcha_type||'', v.source_host||'', v.expires_at||''];
    cells.forEach(c => { const td=document.createElement('td'); td.textContent=c; tr.appendChild(td); });
    const td = document.createElement('td');
    const ok = document.createElement('button'); ok.className='ok'; ok.textContent='Completed'; ok.onclick=()=>act(v.request_id,'complete');
    const no = document.createElement('button'); no.className='no'; no.textContent='Reject'; no.onclick=()=>act(v.request_id,'reject');
    td.append(ok,no);
    if(v.screenshot_path){ const b=document.createElement('button'); b.textContent='Screenshot'; b.onclick=()=>shot(v.request_id,td); td.append(b); }
    tr.appendChild(td); tb.appendChild(tr);
  }
}
async function shot(id,td){ const r=await fetch(`/v1/verifications/${id}/screenshot`,{headers:hdrs()}); if(!r.ok) return;
  const img=document.createElement('img'); img.src=URL.createObjectURL(await r.blob()); td.appendChild(img); }
async function act(id,what){
  const op=$('#op').value.trim(); if(!op){ alert('Enter your operator name first'); return; }
  const r=await fetch(`/v1/verifications/${id}/${what}`,{method:'POST',headers:hdrs(),body:JSON.stringify({operator:op})});
  if(!r.ok) alert('Failed: '+r.status); load();
}
load(); setInterval(load, 4000);
</script></body></html>"""


def serve(host: str | None = None, port: int | None = None) -> None:
    import uvicorn
    from config import get_settings

    s = get_settings()
    kwargs = {}
    if s.ssl_certfile and s.ssl_keyfile:
        kwargs = {"ssl_certfile": s.ssl_certfile, "ssl_keyfile": s.ssl_keyfile}
    elif (host or s.api_host) not in ("127.0.0.1", "localhost", "::1"):
        print("WARNING: serving on a non-loopback interface without TLS. "
              "Set CAPTCHA_FW_SSL_CERTFILE / CAPTCHA_FW_SSL_KEYFILE.", file=sys.stderr)
    uvicorn.run(create_app(), host=host or s.api_host, port=port or s.api_port, log_level="info", **kwargs)


if __name__ == "__main__":
    serve()
