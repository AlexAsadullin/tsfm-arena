"""Render the leaderboard as Markdown and as a self-contained static HTML page."""

from __future__ import annotations

import html
from pathlib import Path

from tsfm_arena.leaderboard.board import CI, Leaderboard, Row, ScoreRow


def _num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _ci(ci: CI, digits: int = 3) -> str:
    if ci.estimate is None:
        return "n/a"
    if ci.low is None or ci.high is None:
        return _num(ci.estimate, digits)
    return f"{ci.estimate:.{digits}f} [{ci.low:.{digits}f}, {ci.high:.{digits}f}]"


def _dsr(row: Row) -> str:
    if row.dsr is None:
        return "n/a"
    label = "S6" if row.dsr_source == "S6" else "local, S6 unavailable"
    gate = "" if row.gate_passed is None else (", gate passed" if row.gate_passed else ", gate failed")
    return f"{row.dsr:.3f} ({label}{gate})"


def _window(s: ScoreRow) -> str:
    first = (s.first_origin or "")[:10]
    last = (s.last_origin or "")[:10]
    return f"{first} to {last} ({s.n_origins})"


MAIN_HEADER = (
    "| Model | Cutoff | Eligible window (origins) | Rank IC [CI] | ICIR | Net Sharpe [CI] | DSR "
    "| Turnover | MAE [CI] | CRPS | Dispersion | Collapse |"
)


def _main_line(row: Row, s: ScoreRow) -> str:
    cutoff = (row.pretraining_cutoff or "none")[:10]
    return (
        f"| {row.model} | {cutoff} | {_window(s)} | {_ci(s.rank_ic)} | {_num(s.icir)} | "
        f"{_ci(s.sharpe_net, 2)} | {_dsr(row)} | {_num(s.turnover, 2)} | {_ci(s.mae, 4)} | "
        f"{_num(s.crps, 4)} | {_num(s.dispersion_ratio, 2)} | {_num(s.collapse_rate, 2)} |"
    )


def render_markdown(board: Leaderboard) -> str:
    pct = round(board.confidence * 100)
    lines = [
        "# TSFM Arena leaderboard",
        "",
        f"Dataset `{board.dataset}` (`{board.dataset_fingerprint}`), horizon {board.horizon} bar(s), "
        f"embargo {board.embargo_days:g} days, costs {board.cost_bps:g} bps per unit turnover, "
        f"{board.n_trials} trials in family `{board.family}`. Generated {board.generated_at.isoformat()}.",
        "",
        f"Brackets are {pct}% stationary-bootstrap intervals over origins. Each model is scored only on "
        "origins after its pretraining cutoff plus the embargo. Dispersion is the cross-sectional standard "
        "deviation of forecasts divided by that of realised returns; collapse is the share of origins "
        "where it is below the configured threshold.",
        "",
        "## Each model on its eligible window",
        "",
        MAIN_HEADER,
        "|" + "---|" * 12,
    ]
    lines += [_main_line(r, r.eligible) for r in board.rows if r.eligible]
    if board.common_window:
        lines += [
            "",
            f"## Common window ({board.common_window[0][:10]} to {board.common_window[1][:10]})",
            "",
            "All models on the origins that are eligible for every model, so the rows are comparable.",
            "",
            "| Model | Origins | Rank IC [CI] | ICIR | Net Sharpe [CI] | MAE [CI] | CRPS |",
            "|---|---|---|---|---|---|---|",
        ]
        lines += [
            f"| {r.model} | {r.common.n_origins} | {_ci(r.common.rank_ic)} | {_num(r.common.icir)} | "
            f"{_ci(r.common.sharpe_net, 2)} | {_ci(r.common.mae, 4)} | {_num(r.common.crps, 4)} |"
            for r in board.rows
            if r.common
        ]
    contaminated = [r for r in board.rows if r.contaminated]
    if contaminated:
        lines += [
            "",
            "## Contaminated origins (not ranked)",
            "",
            "Scores on origins inside a model's pretraining window. For in-house models these origins are "
            "in the training set, so the gap to the eligible window measures look-ahead inflation.",
            "",
            "| Model | Window (origins) | Rank IC [CI] | Net Sharpe [CI] |",
            "|---|---|---|---|",
        ]
        lines += [
            f"| {r.model} | {_window(r.contaminated)} | {_ci(r.contaminated.rank_ic)} | "
            f"{_ci(r.contaminated.sharpe_net, 2)} |"
            for r in contaminated
            if r.contaminated
        ]
    skipped = [r for r in board.rows if r.status == "skipped"]
    if skipped:
        lines += ["", "## Not evaluated", ""]
        lines += [f"- {r.model}: {r.reason}" for r in skipped]
    return "\n".join(lines) + "\n"


_CSS = """
:root { --bg:#ffffff; --fg:#1d1d1f; --muted:#5f6368; --line:#d9dce1; --bar:#3b6fd8; --zero:#9aa0a6; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --bg:#121417; --fg:#e8eaed; --muted:#9aa0a6; --line:#33373d;
    --bar:#7aa2f7; --zero:#5f6368; }
}
:root[data-theme="dark"] { --bg:#121417; --fg:#e8eaed; --muted:#9aa0a6; --line:#33373d; --bar:#7aa2f7;
  --zero:#5f6368; }
body { background:var(--bg); color:var(--fg); font:14px/1.45 system-ui,sans-serif; margin:0;
  padding:24px 16px; }
main { max-width:1200px; margin:0 auto; }
p { color:var(--muted); max-width:80ch; }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
th, td { border-bottom:1px solid var(--line); padding:6px 8px; text-align:left; white-space:nowrap; }
th { font-weight:600; }
svg line.ci { stroke:var(--bar); stroke-width:2; } svg circle { fill:var(--bar); }
svg line.zero { stroke:var(--zero); stroke-dasharray:2 2; }
"""


def _ci_svg(ci: CI, lo: float, hi: float, width: int = 140) -> str:
    """Inline interval plot on a shared axis ``[lo, hi]`` with a dashed zero line."""
    if ci.estimate is None or ci.low is None or ci.high is None or hi <= lo:
        return ""

    def x(v: float) -> float:
        return 4 + (width - 8) * (min(max(v, lo), hi) - lo) / (hi - lo)

    zero = f'<line class="zero" x1="{x(0):.1f}" x2="{x(0):.1f}" y1="2" y2="14"/>' if lo < 0 < hi else ""
    return (
        f'<svg width="{width}" height="16" role="img" aria-label="interval">{zero}'
        f'<line class="ci" x1="{x(ci.low):.1f}" x2="{x(ci.high):.1f}" y1="8" y2="8"/>'
        f'<circle cx="{x(ci.estimate):.1f}" cy="8" r="3"/></svg>'
    )


def _axis(cis: list[CI]) -> tuple[float, float]:
    values = [v for c in cis for v in (c.low, c.high) if v is not None] + [0.0]
    pad = (max(values) - min(values)) * 0.05 or 1.0
    return min(values) - pad, max(values) + pad


def render_html(board: Leaderboard) -> str:
    esc = html.escape
    rows = [r for r in board.rows if r.eligible]
    ic_axis = _axis([r.eligible.rank_ic for r in rows if r.eligible])
    sr_axis = _axis([r.eligible.sharpe_net for r in rows if r.eligible])
    body = []
    for r in rows:
        s = r.eligible
        if s is None:
            continue
        body.append(
            "<tr>"
            f"<td>{esc(r.model)}</td><td>{esc((r.pretraining_cutoff or 'none')[:10])}</td>"
            f"<td>{esc(_window(s))}</td>"
            f"<td>{esc(_ci(s.rank_ic))}</td><td>{_ci_svg(s.rank_ic, *ic_axis)}</td>"
            f"<td>{esc(_num(s.icir))}</td>"
            f"<td>{esc(_ci(s.sharpe_net, 2))}</td><td>{_ci_svg(s.sharpe_net, *sr_axis)}</td>"
            f"<td>{esc(_dsr(r))}</td><td>{esc(_ci(s.mae, 4))}</td><td>{esc(_num(s.crps, 4))}</td>"
            f"<td>{esc(_num(s.dispersion_ratio, 2))}</td>"
            "</tr>"
        )
    skipped = "".join(
        f"<li>{esc(r.model)}: {esc(r.reason or '')}</li>" for r in board.rows if r.status == "skipped"
    )
    pct = round(board.confidence * 100)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TSFM Arena leaderboard</title><style>{_CSS}</style></head>
<body><main>
<h1>TSFM Arena leaderboard</h1>
<p>Dataset {esc(board.dataset)} ({esc(board.dataset_fingerprint or "no fingerprint")}), horizon
{board.horizon} bar(s), embargo {board.embargo_days:g} days, costs {board.cost_bps:g} bps per unit turnover,
{board.n_trials} trials. Each model is scored only after its pretraining cutoff plus the embargo. Intervals
are {pct}% stationary-bootstrap intervals over origins; the dashed line marks zero.</p>
<div class="scroll"><table>
<thead><tr><th>Model</th><th>Cutoff</th><th>Eligible window (origins)</th><th>Rank IC</th><th></th>
<th>ICIR</th><th>Net Sharpe</th><th></th><th>DSR</th><th>MAE</th><th>CRPS</th><th>Dispersion</th></tr></thead>
<tbody>{"".join(body)}</tbody></table></div>
{f"<h2>Not evaluated</h2><ul>{skipped}</ul>" if skipped else ""}
<p>Generated {esc(board.generated_at.isoformat())}.</p>
</main></body></html>
"""


def write_leaderboard(board: Leaderboard, out_dir: Path) -> dict[str, Path]:
    """Write ``leaderboard.md``, ``index.html`` and ``leaderboard.json`` into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "markdown": out_dir / "leaderboard.md",
        "html": out_dir / "index.html",
        "json": out_dir / "leaderboard.json",
    }
    paths["markdown"].write_text(render_markdown(board), encoding="utf-8")
    paths["html"].write_text(render_html(board), encoding="utf-8")
    paths["json"].write_text(board.model_dump_json(indent=2), encoding="utf-8")
    return paths
