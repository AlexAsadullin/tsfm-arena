# HANDOVER: tsfm-arena (Tradeq S1)

Last updated 2026-10-22 by the orchestrating session. The stopped agent only ran the test suite here.

## Goal

The project is a cutoff-aware benchmark of time-series foundation models on financial returns. It scores models by
decision metrics (rank correlation and long-short returns) instead of MSE, and it includes a rank-tuned in-house
finance model. The spec is in `../prompts/04-tsfm-arena.md`. The default port is 8604. The project produces contract
6.5 (forecast service) and consumes 6.1 (ledger) and 6.3 (PIT). For the report, the choice that needs evidence is the
model and its training, including the rank loss.

## Stack

Python 3.13, uv `.venv`, and pins from `../constraints.txt`. The project uses torch 2.14, transformers 5.20,
chronos-forecasting 2.3.2 (Chronos-2 and Chronos-Bolt), peft (LoRA), accelerate, pandas 3, numpy, scipy, FastAPI and
uvicorn, httpx, typer (the `tsfm-arena` CLI), pydantic-settings (`TSFM_*` env vars, `config/settings.yaml`, and the
model registry in `config/models.yaml`) and structlog.

## Architecture and data flow

A data source (`data/`: the I2 PIT client, local files or a synthetic panel with a planted, decaying signal) returns
a `ReturnPanel` as known at each origin. `evaluation/arena.py` runs a cutoff-aware rolling-origin evaluation of every
registry model (`models/factory.py` builds from `config/models.yaml`). The models are baselines (random walk,
historical mean, pooled ridge AR), Chronos with LoRA, an optional Kronos, and the in-house PatchTST-style model.
`evaluation/metrics.py` and `bootstrap.py` produce the metrics and stationary-bootstrap CIs. `leaderboard/` renders
Markdown and HTML. `pipeline/run.py` trains, evaluates, logs trials to S6 (`ledger/client.py`) and writes the
leaderboard. `training/` holds the losses (pointwise, quantile and rank losses such as spearman, listnet and
pairwise) and the trainers. `service/app.py` serves `/v1/forecast` and `/v1/rank`.

## Files

| Path | Purpose |
|---|---|
| src/tsfm_arena/models/{baselines,chronos,kronos,patchtst,factory,spec,base}.py | Forecaster adapters and the registry |
| src/tsfm_arena/training/{losses,trainer,lora_tune}.py | Rank and quantile losses, PatchTST training, LoRA rank-tuning |
| src/tsfm_arena/evaluation/{arena,metrics,bootstrap}.py | Rolling-origin evaluation and CIs |
| src/tsfm_arena/leaderboard/{board,render}.py | Leaderboard on the eligible and common windows |
| src/tsfm_arena/pipeline/{run,evaluate,training,report}.py | Pipelines; report.py backs `make report` |
| src/tsfm_arena/data/*.py | PIT client, sources, synthetic panel |
| config/models.yaml | Model registry, including the training config of the rank-tuned model |

## State at handover

Updated 2026-10-22 by the finishing agent.

1. Commits of the finishing session: 80afbbf (rank_loss listnet to spearman in config/models.yaml), adb1ddc
   (report.json bug fix), 6418ab7 (README claim and key results filled). `make lint typecheck test`: clean,
   71 passed, 98% coverage.
2. The rank_loss question is settled: `variant_specs` in pipeline/report.py overrides `rank_loss` for every
   variant, so report_tables.md does not depend on that field. A rerun at 20:59 under the committed spearman
   config reproduced every table cell of the 09:52 run (only the generation timestamp differs).
3. Bug fixed: `run_report` merged `result.paths` after its own keys, so the arena key `json` replaced
   `report.json`, the report payload overwrote `reports/arena/leaderboard.json`, and report.json was never
   written. A third `make report` after the fix (21:17) wrote report.json and an intact leaderboard.json, and
   report_tables.md again matched every cell.
4. docs/report.md is still untracked. The corrected text (placeholders T3_*, CHRONOS_* filled, "four of the
   five" corrected to "three of the four", LoRA gain stated as not significant, rule 8 note on rank_loss) was
   returned to the orchestrator, which writes and commits it.

## Mistakes and lessons

1. The config was changed to follow the report's Decision but was not committed. That risks a mismatch between the evaluated model and the shipped model. Check whether the evaluated variants read the changed field at all before rerunning (here they did not).
2. These lessons apply platform-wide. Compound shell commands were denied for subagents, so use `make -C` and `git -C`. Subagents may not write report files, so return the text to the orchestrator. Commit early, because of the rate limit.
3. The README and report shipped with template placeholders (RESULT_*, T3_*, CHRONOS_*). Grep for upper-case placeholders before calling a document done.
4. The report claimed that LoRA rank-tuning "raised the rank IC further" and kept it in the Decision, but its paired t-test (1.46, p = 0.15) is not significant and it lowered net Sharpe from 2.34 to 1.87. Every "raised" or "better" needs its test.
5. A missing output file (report.json) after an exit code of 0 signalled a dict-merge bug; check every file that a report cites.
6. `git commit -a` sweeps unrelated edits into a fix commit; commit by path.
