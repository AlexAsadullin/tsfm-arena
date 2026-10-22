# tsfm-arena

tsfm-arena benchmarks time-series forecasting models on cross-sections of asset returns. It scores each
model only on forecast dates after the model's declared training-data cutoff plus an embargo, and it ranks
models by decision metrics (rank IC, Sharpe after costs, deflated Sharpe) instead of MSE. On the
synthetic 30-asset panel, the PatchTST-style model that we trained on finance data reaches a rank IC of
0.021 on dates it never saw, but 0.170 on its own training dates, so an arena without the
cutoff rule would overstate it by a factor of 8. Spec: [S1 in the platform plan](../quant_platform_plan_full.md#s1-financial-time-series-foundation-models).

Terms used below: rank IC is the Spearman correlation between forecasts and realised returns across assets
on one date, averaged over dates. ICIR is mean IC divided by its standard deviation. DSR (deflated Sharpe
ratio) is the probability that the true Sharpe ratio exceeds the best Sharpe expected from the number of
trials run. CRPS (continuous ranked probability score) measures quantile-forecast quality. An origin is a
forecast date. MSE is mean squared error. PatchTST is a small Transformer forecaster that reads the return
history in fixed-length patches. LoRA (low-rank adaptation) fine-tunes a large model by training small
added matrices while the original weights stay frozen.

## Quickstart

```bash
make setup        # .venv with Python 3.13 via uv, locked deps, .env from .env.example
make lint typecheck test
make reproduce    # offline demo: synthetic panel, training, arena, trials, leaderboard
make report       # experiments behind docs/report.md (about 15 min on a laptop CPU)
make serve        # forecast service on 127.0.0.1:8604
```

Outputs go to `$TRADEQ_DATA_DIR/tsfm-arena/`: `leaderboard/` (leaderboard.md, index.html,
leaderboard.json), `reports/`, `checkpoints/`, `datasets/`. Trials go to the Research Ledger (S6) or, when
it is down, to `$TRADEQ_DATA_DIR/research-ledger/inbox/s1-<yyyymmdd>.jsonl`.

Chronos runs only when its weights are available: set `TSFM_ALLOW_DOWNLOADS=true` once to download them into
`$TRADEQ_DATA_DIR/cache/huggingface`; later runs use the cache offline. Tests never download.

## Configuration

Defaults live in `config/settings.yaml`; the model registry is `config/models.yaml`. Any setting can be
overridden by an environment variable with prefix `TSFM_` and `__` for nesting.

| Variable | Meaning |
|---|---|
| `TRADEQ_DATA_DIR` | data root shared by all Tradeq projects (required) |
| `TSFM_ALLOW_DOWNLOADS` | let adapters download model weights (default `false`) |
| `TSFM_DATA__SOURCE` | `synthetic`, `local` (Parquet/CSV) or `pit` (I2 service) |
| `TSFM_EVALUATION__EMBARGO` | gap after a model's cutoff before it is scored (ISO 8601, default `P30D`) |
| `TSFM_EVALUATION__LONG_SHORT__COST_BPS` | trading cost per unit of turnover (default 10) |
| `TSFM_LEDGER__BASE_URL`, `TSFM_LEDGER__MODE` | S6 address; `auto` falls back to the inbox file |
| `TSFM_DATA__PIT__BASE_URL` | I2 address |
| `TSFM_KRONOS__REPO_PATH` | checkout of the Kronos source, needed for the Kronos adapter |

Every registry entry declares `pretraining_cutoff`. Public models declare their release date, because
training-data cutoffs are not published. In-house models train only on data up to their cutoff; the
checkpoint stores the real training end, and loading fails if it is later than the declaration.

## Architecture

```
data/ (synthetic | local file | I2 client) --> ReturnPanel --> evaluation/arena.py (rolling origins, cutoff rule)
models/ (Forecaster protocol: baselines, PatchTST, Chronos, Kronos) --^          |
training/ (pinball, MSE, pairwise / ListNet / soft-Spearman rank losses, LoRA)   v
                         evaluation/metrics.py + bootstrap.py --> pipeline/ --> ledger/ (S6) + leaderboard/
service/app.py: FastAPI forecast service over the same registry and data sources
```

At each origin a model sees only data known at that time: the context window and, for refitted baselines,
the history up to the origin. With the I2 source, the arena re-queries with `known_at` equal to the origin.
Origins on or before `cutoff + embargo` are scored separately as contaminated and never ranked.

## Contracts

Produced: [6.5 forecast service](../prompts/00-platform-conventions.md#65-research-services):
`POST /v1/forecast` and `POST /v1/rank`, plus `GET /v1/models` and `GET /healthz`. A `known_at` inside a
model's pretraining window returns 409 (switch off with `TSFM_SERVICE__ENFORCE_CUTOFF=false`).

Consumed: [6.1 Research Ledger](../prompts/00-platform-conventions.md#61-research-ledger-s6-trial-capture-and-verdicts)
(one `TrialRecord` per evaluated model and config, verdicts read back) and
[6.3 Point-in-time Store](../prompts/00-platform-conventions.md#63-point-in-time-store-i2-as-known-at-t)
(`POST /v1/query`, Arrow or JSON; a 409 leakage refusal is raised, never bypassed).

Contract extensions (optional fields only): 6.5 responses also carry `schema_version`, `model` and
`known_at`, and the rank response carries `model_cutoff`. The I2 client sends symbols as
`filters = {"symbol": [...]}` and accepts a JSON body that is a row list or `{"rows": [...]}`.

## Key results

Synthetic data (30 assets, 1300 daily bars, planted momentum signal), horizon 1 day, costs 10 bps,
eligible window 2024-07-31 to 2026-07-22 (722 origins), 90% stationary-bootstrap intervals. Full tables and
tests: [docs/report.md](docs/report.md).

| Result (rank IC unless stated) | Value [90% CI] | Origins |
|---|---|---|
| PatchTST, MSE objective, after its cutoff (3 seeds) | 0.021 [0.009, 0.033] | 722 |
| Same model on its own training dates (mean of 3 seeds) | 0.170 (seeds 0.163 to 0.175) | 450 |
| PatchTST, soft-Spearman rank-tuning (registry model `patchtst-rank`) | 0.028 [0.018, 0.038] | 722 |
| Ridge regression on 10 lags (`ar-ridge`) | 0.057 [0.047, 0.067] | 1172 |
| Chronos-Bolt small, zero-shot | 0.037 [0.024, 0.051] | 572 |
| Momentum rule that matches the planted signal (ceiling) | 0.097 [0.086, 0.106] | 1172 |

No rank loss beat the MSE model significantly: the paired t-test on per-origin rank IC gives p = 0.14
(pairwise), 0.20 (soft-Spearman) and 0.87 (ListNet). `patchtst-rank` uses soft-Spearman because it keeps
forecasts on the return scale (dispersion ratio 0.57, MAE 0.030), while pairwise and ListNet forecasts
spread 12 to 16 times wider than realised returns.

## Limitations

The results come from synthetic data, so they show that the pipeline separates models; they are not market
evidence. Public cutoffs are release dates, which can be later than the true data cutoff but not earlier.
Kronos needs a source checkout and is disabled by default. LoRA rank-tuning supports Chronos-Bolt only.
Rank-loss models predict scores, not returns, so their MAE and CRPS are not comparable with MSE-trained
models.

## Roadmap

1. Monthly leaderboard refresh on I2 crypto data (needs the I1/I2 services running with history).
2. Intraday horizons: the code supports any bar length; the demo uses daily bars.
3. Kronos LoRA and Chronos-2 LoRA (different module layouts from Chronos-Bolt).
4. Cached GPU serving (`TSFM_DEVICE=cuda`) is configurable but untested here, because the laptop has no GPU.
