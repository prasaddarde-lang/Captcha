"""Performance reports (JSON + standalone HTML)."""
from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

from database.db_manager import DatabaseManager
from learning.analytics import CaptchaAnalytics


def build_report(db: DatabaseManager, evaluation: dict | None = None) -> dict:
    a = CaptchaAnalytics(db)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": a.summary(),
        "detection_accuracy": a.detection_accuracy(),
        "verification": a.verification_stats(),
        "recurring_failures": a.recurring_failures(),
        "daily_trend": a.daily_trend(),
        "models": [{k: m[k] for k in ("version", "status", "accuracy", "macro_f1", "approved_by", "deployed_at")}
                   for m in db.list_models()],
        "evaluation": evaluation,
    }


def _table(title: str, mapping: dict) -> str:
    if not mapping:
        return f"<h3>{html.escape(title)}</h3><p class='muted'>No data</p>"
    rows = "".join(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>" for k, v in mapping.items())
    return f"<h3>{html.escape(title)}</h3><table>{rows}</table>"


def render_html(report: dict) -> str:
    s = report["summary"]
    acc = report["detection_accuracy"]
    ver = report["verification"]
    rec = report["recurring_failures"]["recommendations"]
    models = "".join(
        f"<tr><td>{html.escape(str(m['version']))}</td><td>{m['status']}</td><td>{m['accuracy']}</td>"
        f"<td>{m['macro_f1']}</td><td>{html.escape(str(m['approved_by']))}</td></tr>" for m in report["models"])
    ev = report.get("evaluation") or {}
    per_class = "".join(
        f"<tr><td>{c}</td><td>{v['precision']}</td><td>{v['recall']}</td><td>{v['f1']}</td><td>{v['support']}</td></tr>"
        for c, v in (ev.get("per_class") or {}).items())
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>CAPTCHA Framework Report</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#222;max-width:1100px}}
h1{{margin-bottom:4px}} .muted{{color:#777}} .kpis{{display:flex;gap:16px;flex-wrap:wrap;margin:16px 0}}
.kpi{{border:1px solid #ddd;border-radius:8px;padding:12px 16px;min-width:150px}} .kpi b{{font-size:22px;display:block}}
table{{border-collapse:collapse;margin-bottom:16px}} td,th{{border:1px solid #ddd;padding:4px 10px;text-align:left}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}}
</style></head><body>
<h1>CAPTCHA Management Framework - Performance Report</h1>
<p class="muted">Generated {report['generated_at']}</p>
<div class="kpis">
 <div class="kpi"><b>{s['total_executions']}</b>executions</div>
 <div class="kpi"><b>{s['detection_rate']:.0%}</b>detection rate</div>
 <div class="kpi"><b>{acc['accuracy'] if acc['accuracy'] is not None else 'n/a'}</b>labelled accuracy ({acc['labelled_samples']})</div>
 <div class="kpi"><b>{ver['human_completion_rate'] if ver['human_completion_rate'] is not None else 'n/a'}</b>operator completion rate</div>
 <div class="kpi"><b>{s['avg_duration_ms']} ms</b>avg duration</div>
</div>
<div class="grid">
 <div>{_table('By CAPTCHA type', s['by_type'])}</div>
 <div>{_table('By status', s['by_status'])}</div>
 <div>{_table('By resolution method', s['by_method'])}</div>
 <div>{_table('By platform', s['by_platform'])}</div>
 <div>{_table('Human verification', ver['human_by_status'])}</div>
 <div>{_table('Accuracy per type (feedback)', acc['per_type'])}</div>
</div>
<h3>Model evaluation {html.escape(str(ev.get('model_version', '')))}</h3>
{f"<p>Accuracy {ev.get('accuracy')} | macro-F1 {ev.get('macro_f1')} | samples {ev.get('samples')}</p><table><tr><th>class</th><th>precision</th><th>recall</th><th>F1</th><th>support</th></tr>{per_class}</table>" if ev else "<p class='muted'>Not run</p>"}
<h3>Model registry</h3>
<table><tr><th>version</th><th>status</th><th>accuracy</th><th>macro F1</th><th>approved by</th></tr>{models}</table>
<h3>Recommendations</h3>
<ul>{''.join(f'<li>{html.escape(r)}</li>' for r in rec) or '<li class="muted">No recurring failures</li>'}</ul>
</body></html>"""


def write_report(db: DatabaseManager, out_dir: Path, evaluation: dict | None = None) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(db, evaluation)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jp = out_dir / f"performance_report_{stamp}.json"
    hp = out_dir / f"performance_report_{stamp}.html"
    jp.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    hp.write_text(render_html(report), encoding="utf-8")
    return {"json": str(jp), "html": str(hp)}
