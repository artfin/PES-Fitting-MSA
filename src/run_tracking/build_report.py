#!/usr/bin/env python3
"""Generate a self-contained static HTML run report (Phase 2 of run-tracking).

Scans a model folder, resolves each run's best validation RMSE (from the Phase-1
metrics.json when present, else by parsing the log), and renders a single
offline HTML file: a sortable leaderboard, reproducibility badges, and an
expandable per-run panel with intent, full config, and a diff vs a baseline.

Usage:
    python src/run_tracking/build_report.py --folder models/h2o-h2o
    python src/run_tracking/build_report.py --folder models/h2o-h2o \\
        --baseline water-extended-27-grad --out models/h2o-h2o/report.html
"""

import argparse
import datetime as _dt
import html
import json
import os
import sys

# Allow running as a script (python src/run_tracking/build_report.py ...).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_tracking.registry import (  # noqa: E402
    discover_runs,
    resolve_best_val_rmse,
    repro_status,
    config_summary,
    flatten_config,
)


def _esc(v):
    return html.escape("" if v is None else str(v))


def _fmt_rmse(v):
    return "{:.3f}".format(v) if isinstance(v, (int, float)) else "—"


def _fmt_date(ts):
    if not ts:
        return "—"
    return _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def build_rows(runs, baseline_cfg=None):
    """Turn Run objects into render-ready dicts, sorted by best val RMSE asc."""
    rows = []
    for run in runs:
        rmse, source = resolve_best_val_rmse(run)
        level, badge = repro_status(run)
        summ = config_summary(run.config)
        metrics = run.metrics or {}

        diff = {}
        if baseline_cfg is not None:
            flat = flatten_config(run.config)
            base = flatten_config(baseline_cfg)
            keys = set(flat) | set(base)
            for k in sorted(keys):
                a, b = base.get(k), flat.get(k)
                if a != b:
                    diff[k] = (a, b)

        rows.append({
            "stem": run.stem,
            "rmse": rmse,
            "rmse_source": source,
            "best_epoch": metrics.get("best_epoch"),
            "epochs_run": metrics.get("epochs_run"),
            "early_stopped": metrics.get("early_stopped"),
            "wall_time_s": metrics.get("wall_time_s"),
            "repro_level": level,
            "repro_badge": badge,
            "intent": run.intent,
            "summary": summ,
            "config_flat": flatten_config(run.config),
            "diff": diff,
            "date": _fmt_date(run.mtime),
            "sort_ts": run.mtime,
        })

    # Sort by RMSE ascending; runs without a value sink to the bottom.
    rows.sort(key=lambda r: (r["rmse"] is None, r["rmse"] if r["rmse"] is not None else 0.0))
    return rows


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

_CSS = """
:root {
  --bg: #ffffff; --fg: #1a1a1a; --muted: #6b7280; --line: #e5e7eb;
  --row: #f9fafb; --accent: #2563eb; --card: #f3f4f6;
  --ok: #16a34a; --warn: #d97706; --none: #9ca3af;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1115; --fg: #e6e8eb; --muted: #9aa2ad; --line: #262b33;
    --row: #161a20; --accent: #6ea8fe; --card: #161a20;
    --ok: #4ade80; --warn: #fbbf24; --none: #6b7280;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
.wrap { max-width: 1200px; margin: 0 auto; padding: 28px 20px 80px; }
h1 { font-size: 22px; margin: 0 0 2px; }
.sub { color: var(--muted); margin: 0 0 20px; font-size: 13px; }
.kpis { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 22px; }
.kpi { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
  padding: 12px 16px; min-width: 130px; }
.kpi .v { font-size: 22px; font-weight: 650; }
.kpi .l { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
.tablewrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 10px; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 9px 12px; border-bottom: 1px solid var(--line); white-space: nowrap; }
th { position: sticky; top: 0; background: var(--bg); cursor: pointer; user-select: none;
  font-size: 12px; text-transform: uppercase; letter-spacing: .03em; color: var(--muted); }
th.num, td.num { text-align: right; }
tbody tr:nth-child(4n+1), tbody tr:nth-child(4n+2) { background: var(--row); }
tr.run { cursor: pointer; }
tr.detail > td { background: var(--card); white-space: normal; padding: 0; }
tr.detail.hidden { display: none; }
.rmse { font-weight: 650; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-left: 6px; vertical-align: middle; }
.dot.manifest { background: var(--accent); }
.dot.log { background: transparent; border: 1px solid var(--muted); }
.badge { font-size: 11px; padding: 2px 8px; border-radius: 999px; font-weight: 600; }
.badge.reproducible { background: color-mix(in srgb, var(--ok) 18%, transparent); color: var(--ok); }
.badge.dirty { background: color-mix(in srgb, var(--warn) 20%, transparent); color: var(--warn); }
.badge.pre-tracking { background: color-mix(in srgb, var(--none) 22%, transparent); color: var(--none); }
.rank { color: var(--muted); }
.panel { padding: 16px 18px; }
.panel h3 { margin: 0 0 6px; font-size: 13px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }
.intent { font-style: italic; color: var(--fg); margin: 0 0 14px; max-width: 800px; }
.kv { display: grid; grid-template-columns: max-content 1fr; gap: 2px 16px; font-family: ui-monospace, monospace; font-size: 12.5px; }
.kv .k { color: var(--muted); }
.diff { font-family: ui-monospace, monospace; font-size: 12.5px; }
.diff .row { display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px; padding: 1px 0; }
.diff .from { color: var(--warn); } .diff .to { color: var(--ok); }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
@media (max-width: 720px) { .cols { grid-template-columns: 1fr; } }
.foot { color: var(--muted); font-size: 12px; margin-top: 22px; }
code { background: var(--card); padding: 1px 5px; border-radius: 5px; }
"""

_JS = """
function toggle(id){var d=document.getElementById(id);if(d)d.classList.toggle('hidden');}
function sortBy(idx, numeric){
  var tb=document.querySelector('tbody');
  var runs=[].slice.call(tb.querySelectorAll('tr.run'));
  var st=tb.getAttribute('data-sort-'+idx)==='asc'?'desc':'asc';
  tb.setAttribute('data-sort-'+idx, st);
  runs.sort(function(a,b){
    var x=a.children[idx].getAttribute('data-v'), y=b.children[idx].getAttribute('data-v');
    if(numeric){x=parseFloat(x); y=parseFloat(y);
      if(isNaN(x))x=Infinity; if(isNaN(y))y=Infinity; return st==='asc'?x-y:y-x;}
    x=x||''; y=y||''; return st==='asc'?x.localeCompare(y):y.localeCompare(x);
  });
  runs.forEach(function(r){
    var d=document.getElementById('d-'+r.getAttribute('data-stem'));
    tb.appendChild(r); if(d) tb.appendChild(d);
  });
}
"""


def _kpis(rows):
    n = len(rows)
    with_metric = sum(1 for r in rows if r["rmse"] is not None)
    manifests = sum(1 for r in rows if r["rmse_source"] == "manifest")
    repro = sum(1 for r in rows if r["repro_level"] == "reproducible")
    best = next((r for r in rows if r["rmse"] is not None), None)
    cards = [
        ("total runs", n),
        ("best val RMSE", _fmt_rmse(best["rmse"]) + " cm⁻¹" if best else "—"),
        ("best run", best["stem"] if best else "—"),
        ("with metrics", "{}/{}".format(with_metric, n)),
        ("tracked manifests", manifests),
        ("reproducible", repro),
    ]
    out = ['<div class="kpis">']
    for label, val in cards:
        out.append('<div class="kpi"><div class="v">{}</div><div class="l">{}</div></div>'
                    .format(_esc(val), _esc(label)))
    out.append("</div>")
    return "".join(out)


def _detail_panel(r):
    parts = ['<div class="panel">']
    if r["intent"]:
        parts.append('<p class="intent">{}</p>'.format(_esc(r["intent"])))

    parts.append('<div class="cols">')

    # Config column
    parts.append('<div><h3>config</h3><div class="kv">')
    for k, v in r["config_flat"].items():
        parts.append('<div class="k">{}</div><div>{}</div>'.format(_esc(k), _esc(v)))
    parts.append("</div></div>")

    # Diff or metrics column
    if r["diff"]:
        parts.append('<div><h3>diff vs baseline</h3><div class="diff">')
        for k, (a, b) in r["diff"].items():
            parts.append(
                '<div class="row"><div class="k">{}</div>'
                '<div><span class="from">{}</span> → <span class="to">{}</span></div></div>'
                .format(_esc(k), _esc(a), _esc(b)))
        parts.append("</div></div>")
    else:
        m = []
        if r["best_epoch"] is not None:
            m.append(("best epoch", r["best_epoch"]))
        if r["epochs_run"] is not None:
            m.append(("epochs run", r["epochs_run"]))
        if r["early_stopped"] is not None:
            m.append(("early stopped", r["early_stopped"]))
        if r["wall_time_s"] is not None:
            m.append(("wall time", "{:.0f} s".format(r["wall_time_s"])))
        if m:
            parts.append('<div><h3>metrics</h3><div class="kv">')
            for k, v in m:
                parts.append('<div class="k">{}</div><div>{}</div>'.format(_esc(k), _esc(v)))
            parts.append("</div></div>")

    parts.append("</div></div>")  # cols, panel
    return "".join(parts)


_COLS = [
    # (label, numeric)
    ("#", True), ("run", False), ("val RMSE", True), ("best ep", True),
    ("type", False), ("model", False), ("order", True), ("loss", False),
    ("pretrained", False), ("repro", False), ("date", False),
]


def render_html(rows, folder, baseline):
    head = "".join(
        '<th class="{cls}" onclick="sortBy({i},{num})">{label}</th>'.format(
            i=i, num="true" if num else "false",
            cls="num" if num else "", label=_esc(label))
        for i, (label, num) in enumerate(_COLS))

    body = []
    for rank, r in enumerate(rows, 1):
        s = r["summary"]
        src = r["rmse_source"]
        dot = '<span class="dot {}"></span>'.format(src) if src else ""
        rmse_v = "" if r["rmse"] is None else "{:.6f}".format(r["rmse"])
        cells = [
            ('num', str(rank), '<span class="rank">{}</span>'.format(rank)),
            ('', r["stem"], _esc(r["stem"])),
            ('num rmse', rmse_v, _fmt_rmse(r["rmse"]) + dot),
            ('num', r["best_epoch"], _esc(r["best_epoch"]) if r["best_epoch"] is not None else "—"),
            ('', s["type"], _esc(s["type"])),
            ('', s["hidden_dims"], _esc(s["hidden_dims"])),
            ('num', s["order"], _esc(s["order"])),
            ('', s["loss"], _esc(s["loss"])),
            ('', s["pretrained"], _esc(s["pretrained"]) or "—"),
            ('', r["repro_level"],
             '<span class="badge {}">{}</span>'.format(r["repro_level"], _esc(r["repro_badge"]))),
            ('', r["sort_ts"], _esc(r["date"])),
        ]
        tds = "".join('<td class="{cls}" data-v="{v}">{html}</td>'.format(
            cls=cls, v=_esc("" if v is None else v), html=inner)
            for cls, v, inner in cells)
        body.append('<tr class="run" data-stem="{stem}" onclick="toggle(\'d-{stem}\')">{tds}</tr>'
                    .format(stem=_esc(r["stem"]), tds=tds))
        body.append('<tr class="detail hidden" id="d-{stem}"><td colspan="{n}">{panel}</td></tr>'
                    .format(stem=_esc(r["stem"]), n=len(_COLS), panel=_detail_panel(r)))

    gen = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    base_note = " · baseline: <code>{}</code>".format(_esc(baseline)) if baseline else ""
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Run report — {folder}</title>
<style>{css}</style></head>
<body><div class="wrap">
<h1>Training run report</h1>
<p class="sub"><code>{folder}</code> · {n} runs · generated {gen}{base_note}</p>
{kpis}
<div class="tablewrap"><table>
<thead><tr>{head}</tr></thead>
<tbody>{body}</tbody>
</table></div>
<p class="foot">Sorted by validation RMSE. Click a row for config &amp; metrics; click a column header to re-sort.
Metric source: <span class="dot manifest"></span> metrics.json &nbsp; <span class="dot log"></span> parsed from log.
Repro badge shows the git commit each run was produced at.</p>
</div><script>{js}</script></body></html>""".format(
        folder=_esc(folder), css=_CSS, n=len(rows), gen=gen, base_note=base_note,
        kpis=_kpis(rows), head=head, body="".join(body), js=_JS)


def main():
    ap = argparse.ArgumentParser(description="Generate a static HTML run report.")
    ap.add_argument("--folder", required=True, help="model folder to scan, e.g. models/h2o-h2o")
    ap.add_argument("--out", default=None, help="output HTML path (default: <folder>/report.html)")
    ap.add_argument("--baseline", default=None,
                    help="run stem to diff every run's config against")
    args = ap.parse_args()

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        ap.error("folder not found: {}".format(folder))

    runs = discover_runs(folder)
    baseline_cfg = None
    if args.baseline:
        match = next((r for r in runs if r.stem == args.baseline), None)
        if match is None:
            ap.error("baseline run not found: {}".format(args.baseline))
        baseline_cfg = match.config

    rows = build_rows(runs, baseline_cfg=baseline_cfg)
    out = args.out or os.path.join(folder, "report.html")
    with open(out, "w") as f:
        f.write(render_html(rows, os.path.relpath(folder), args.baseline))

    with_metric = sum(1 for r in rows if r["rmse"] is not None)
    print("Wrote {} ({} runs, {} with a val-RMSE metric)".format(out, len(rows), with_metric))


if __name__ == "__main__":
    main()
