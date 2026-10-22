# Rank-tuned forecasters in a cutoff-aware arena: report for tsfm-arena v0.1

All numbers in this report come from the command make report, which writes the tables to
$TRADEQ_DATA_DIR/tsfm-arena/reports/report_tables.md and report.json. The run fixes every random seed, so a
second run reproduces the numbers that the report cites.

## Question

The present project asks whether a forecaster that we tune with a cross-sectional ranking loss ranks assets
better than the same forecaster that we train with mean squared error (MSE), and whether either beats simple
baselines and public time-series foundation models (TSFMs), which are large forecasters that their authors
pretrained on many unrelated series. The second question is how much a benchmark overstates a model when it
scores the model on dates that the model saw in training, because the arena exists to prevent that error.

The platform specification (section S1) summarises three findings that frame these questions. Rahimikia et
al. (2025) found that off-the-shelf TSFMs are weak on returns both zero-shot and after fine-tuning, while
models that the authors pretrained from scratch on financial data gave real gains. Noguer i Alonso and
Franklin (2026) concluded that TSFMs are useful priors rather than sources of excess return, and the
forecast-collapse study (2026) showed that per-series error metrics hide a collapse of cross-sectional
structure. We can conclude from these findings that the arena must score ranking quality across assets
rather than error per series, which is the design that the present project follows.

## Data

The present study uses synthetic data, because the point-in-time store (I2) holds no market history yet, so
its results show how the arena separates models and are not market evidence. The generator draws 30 assets
over 1300 daily bars from 2023-01-01, and each return is the sum of a market factor, which a beta per asset
scales, and an idiosyncratic part with Student-t shocks of 5 degrees of freedom. The idiosyncratic part
carries a planted signal: at each bar the generator takes the mean return of each asset over the last 5 bars,
it standardises that mean across assets, and the next return correlates with the result at a strength rho
that starts at 0.12 and halves every 1500 bars.

We changed two generator parameters during the study, and the reason was a measurement of the signal ceiling.
The planted strength started at 0.08 with a half-life of 900 bars, but the momentum rule that knows the
generator reached a rank IC of only 0.037 on the dates after mid-2024, which left too little signal to
separate the models over 722 dates. We therefore raised the strength from 0.08 to 0.12 and the half-life
from 900 to 1500 bars, which gives the momentum rule a rank IC of 0.079 on those dates, as Table 3 of report_tables.md shows.

## Methods

The arena scores a model only on forecast dates, which we call origins, that fall after the model's declared
pretraining cutoff plus an embargo of 30 days. At each origin the model receives the last 64 or more daily
returns of every asset, which end at the origin, and it forecasts the log return of the next day as a mean
and nine quantiles. Origins inside the pretraining window are scored in a separate table and are never
ranked. Public models declare their release date as the cutoff, because their authors do not publish the
end of their training data, and in-house models declare the last date that their training targets touch.

We score each model with five metrics, and the first three decide the ranking. The rank information
coefficient (rank IC) is the Spearman correlation between the forecasts and the realised returns across the
30 assets on one origin, which we average over origins. The net Sharpe ratio comes from a portfolio that
buys the top 20% of assets by forecast and sells the bottom 20% each day, which pays 10 basis points per
unit of turnover, and we annualise it with 365 days. The deflated Sharpe ratio (DSR), which corrects the
Sharpe ratio for the number of strategies that the researcher tried, is the probability that the true
Sharpe ratio exceeds the maximum that 22 trials would reach by chance (Bailey and Lopez de Prado, 2014). The
mean absolute error (MAE) and the continuous ranked probability score (CRPS), which averages the quantile
loss over the nine quantile levels, measure point and distribution accuracy. The dispersion ratio divides the
cross-sectional standard deviation of the forecasts by that of the realised returns, so a value near zero
means that the model forecasts nearly the same return for every asset.

Confidence intervals are 90% intervals from a stationary bootstrap (Politis and Romano, 1994), which resamples blocks of origins with a
mean length of 10 days so that it keeps their serial dependence, and each interval rests on 1000 resamples.

The in-house model is a small PatchTST network (Nie et al., 2023), which divides each 64-day return window by
its own standard deviation, cuts it into 8 patches of 8 days, and passes them through a two-layer Transformer
encoder with 32 dimensions. The network outputs a point forecast and nine quantiles, and the quantiles cannot
cross because the network outputs the lowest quantile plus positive increments. We train it only on data up
to 2024-06-30, which gives 483 training dates with 30 assets each, and every training batch contains whole
cross-sections of 16 dates, so that a ranking loss can compare assets on the same date.

We compare five training objectives for this network, and every objective runs with three seeds. The MSE
model (patchtst-mse) minimises the squared error of the point forecast plus the quantile loss of the
quantiles for 30 epochs, which makes it the reference for every other objective. The three rank-tuned models start from the MSE model of the same seed and train for
15 more epochs, in which the quantile loss keeps a weight of 0.2 and a ranking loss on the point forecasts
takes a weight of 1. The pairwise model (patchtst-pairwise) penalises every pair of assets whose forecasts
order them against their realised returns, with the logistic loss of RankNet (Burges et al., 2005). The
ListNet model (patchtst-listnet) turns the realised returns and the forecasts of one date into two softmax
distributions, and it minimises their cross-entropy (Cao et al., 2007). The soft-Spearman model
(patchtst-spearman) replaces the ranks of the forecasts with sigmoid sums of pairwise differences, which are
differentiable, and it maximises their correlation with the true ranks of the returns. The scratch ListNet
model (patchtst-listnet-scratch) trains the ListNet objective for 30 epochs from random weights, so it tests
whether the MSE stage helps.

The baselines are a random walk, which forecasts a zero return; the historical mean of the last 252 days;
a ridge regression (ar-ridge), which we refit every 30 origins on the last 365 days of all assets and which
regresses the next return on the last 10 returns of the same asset; and the momentum rule (xs-momentum),
which forecasts the 5-day mean return minus its cross-sectional average. The momentum rule computes the same
statistic as the planted signal, so it estimates the best rank IC that any model can reach on this data.

The public TSFMs are Chronos-Bolt small and Chronos-2 (Ansari et al., 2024; Amazon, 2025), which forecast
zero-shot from the last 512 returns. The LoRA Chronos-Bolt model (chronos-bolt-small-rank), where LoRA is
short for low-rank adaptation, adds low-rank adapter matrices of rank 4 (Hu et al., 2021) to the attention projections of Chronos-Bolt, it freezes all
other weights, and it trains the 147 456 adapter weights for 3 epochs with the ListNet loss on 219 dates up
to 2024-12-31. Kronos (Shi et al., 2025) has an adapter in the present project, but it needs a source
checkout that the test machine lacks, so the present study did not run it.

## Experiments and results

Rank-tuning did not raise the rank IC of the PatchTST network by a margin that the tests can separate from
noise, although all three ranking losses moved the estimate upward. As Table 1 shows, the MSE model reached
a rank IC of 0.021 with a 90% interval from 0.009 to 0.033 over 722 origins and three seeds, while the
pairwise model reached 0.031 (0.021 to 0.040), the soft-Spearman model 0.028 (0.018 to 0.038) and the ListNet
model 0.022 (0.013 to 0.032). Student's paired t-test on the per-origin differences between the pairwise
model and the MSE model equals 1.47 with a two-sided p-value of 0.14, while the critical value at 95%
confidence is 1.96, so we cannot reject the hypothesis that the mean difference of 0.0094 is zero (Table 2).
The same test gives 1.29 (p = 0.20) for the soft-Spearman model and 0.17 (p = 0.87) for the ListNet model,
and both values lie below the same critical value of 1.96.
The scratch ListNet model reached 0.027 against 0.022 for the ListNet model that started from the MSE model,
and the paired t-test of -0.93 (p = 0.35), whose absolute value lies below 1.96, supports the conclusion
that the MSE stage brought no measurable gain.

The ridge baseline ranked assets better than every PatchTST variant, and for three of the four variants that
we tested against it the difference is significant at 95% confidence. As Table 1 shows, the ridge baseline
reached a rank IC of 0.057 (0.047 to 0.067) over 1172 origins, against 0.021 for the MSE model, and the
paired t-test on their common 722 origins equals 2.64 in favour of the ridge baseline with p = 0.008, while
the critical value is 1.96 (Table 2). The tests for the ListNet and soft-Spearman models equal 2.53
(p = 0.012) and 2.08 (p = 0.038), and the pairwise model is the exception, because its test equals 1.58
with p = 0.11. The momentum rule
reached 0.097 (0.086 to 0.106), which shows that every learned model recovers at most 60% of the signal
that the data contains.

No PatchTST variant earned a positive Sharpe ratio after costs, because each of them rebalanced most of its
book every day. As the leaderboard in Table 4 shows, the PatchTST variants turned over 2.7 to 3.0 units of
gross exposure per day, which costs 27 to 30 basis points per day, against 1.6 units for Chronos-Bolt. The
MSE model earned a net Sharpe ratio of -1.55 (-2.99 to -0.02), and the best variant, the pairwise model,
earned -0.15 (-1.53 to 1.17), while the ridge baseline earned 2.30 (1.34 to 3.23) and Chronos-Bolt earned
2.34 (0.92 to 3.82). The DSR of every model except the momentum rule stays below 0.15, because 22 trials
raise the Sharpe ratio that chance alone would produce, so no learned model passes a DSR threshold of 0.95.

The choice of ranking loss changed how the forecasts spread across assets, which matters for a user who needs
return forecasts rather than ranks. As Table 1 shows, the dispersion ratio of the pairwise and ListNet models
lies between 12 and 16, because these losses depend only on differences between scores and leave their
scale free, so their MAE of 0.28 to 0.62 is 10 to 22 times the MAE of 0.028 of the MSE model. The
soft-Spearman model kept a dispersion ratio of 0.57 and an MAE of 0.030, because it standardises the soft
ranks before it correlates them, so a wider spread of the scores does not lower this loss while their
differences stay small next to the temperature of 1. The quantile forecasts kept their quality under every loss, since the CRPS of all
PatchTST variants lies between 0.0223 and 0.0235, against 0.0212 for the random walk.

The cutoff rule removed an inflation of the rank IC that is several times larger than the effect of any
loss. As Table 3 shows, the MSE model reached a rank IC of 0.170 on the 450 origins before its cutoff,
which are its own training dates, and 0.021 on the 722 origins after it, so the ratio is 8.00. The
momentum rule, which cannot overfit, fell only from 0.125 to 0.079 between the same two windows, which is a
ratio of 1.58, so we can conclude that signal decay explains a small part of the gap and that overfitting
explains the rest. The soft-Spearman model showed the largest inflation, because it reached 0.399 before its
cutoff and 0.028 after it, which is a ratio of 14.46. An arena that scored it on all 1172 origins would have
given it a mean rank IC of about 0.170, which lies above the 0.097 of the momentum rule.

Chronos-Bolt ranked assets better zero-shot than the PatchTST network that we trained on finance data, while
the LoRA rank-tuning raised its rank IC by an amount that the tests cannot separate from noise. As Table 1
shows, Chronos-Bolt reached 0.037 (0.024 to 0.051) on
572 origins after its cutoff, against 0.021 for the MSE model, and on the common window of the 245 origins
that are eligible for every model, Chronos-Bolt reached 0.038 while the LoRA Chronos-Bolt model reached 0.050
(Table 4). Student's paired t-test of Chronos-Bolt against the MSE model on their 572 common origins equals
2.13 with p = 0.034, which exceeds the critical value of 1.96, so the gap of 0.017 is not random (Table 2).
The same test of the LoRA model against zero-shot Chronos-Bolt on 538 origins equals 1.46 with p = 0.15, so
we cannot reject the hypothesis that the mean difference of 0.011 is zero. The LoRA model also widened its
forecasts, because its dispersion ratio rose from 0.13 to 0.93 and its MAE rose from 0.0269 to 0.0375, while
its net Sharpe ratio of 1.87 lies below the 2.34 of the zero-shot model (Table 4). Chronos-2 reached 0.023 (-0.002 to 0.045) on its 245 eligible origins, so its
interval includes zero. On synthetic data the Chronos models cannot have memorised outcomes, so their drop
before and after the cutoff in Table 3 should match the drop of the momentum rule, but it does not.
Chronos-Bolt fell by a ratio of 1.94 where the momentum rule fell by 1.45 on the same origins, and
Chronos-2 fell by 2.03 where the momentum rule fell by 1.32. Neither model can have memorised the synthetic
outcomes, so the excess drop has a cause that the present study did not identify, and one candidate is that a
model which captures the signal only in part loses a larger share of its rank IC when the signal weakens. The
LoRA model fell by 1.91 where the momentum rule fell by 1.43, which is close to the ratio of zero-shot
Chronos-Bolt, so its LoRA training on dates up to 2024-12-31 added no measurable memorisation.

## Comparison with analogs

The present study reverses the direction of the Rahimikia et al. (2025) finding on this data, and the reason
lies in the data rather than in the methods. The specification (section S1) states that Rahimikia et al.
found off-the-shelf TSFMs weak and models pretrained from scratch on financial data stronger, whereas in the
present study Chronos-Bolt reached a rank IC of 0.037 against 0.021 for the PatchTST network that we
pretrained from scratch. Our from-scratch network saw 483 dates of 30 assets, about 14 500 windows, which is
far less data than a finance pretraining corpus, so we can conclude only that a small in-house model does not
beat a public TSFM here, not that the published finding is wrong.

The results agree with Noguer i Alonso and Franklin (2026), whom the specification (section S1) cites for the
conclusion that TSFMs are useful priors rather than alpha engines. In the present study the best TSFM, the
LoRA Chronos-Bolt model, reached a rank IC of 0.048, which is half the 0.097 of the momentum rule that knows
the signal and below the 0.057 of a ridge regression on 10 lags.

The forecast-collapse study (2026), which the specification cites, reports that per-series metrics hide a
collapse of cross-sectional structure, and the present study confirms that pattern. As Table 1 and Table 4
show, the random walk had a CRPS of 0.0212 and an MAE of 0.0270, which tie with the ridge baseline and lie
below the 0.0224 and 0.0280 of the MSE model, while the random walk has no rank IC at all because it forecasts
the same value for every asset. Chronos-Bolt had a dispersion ratio of 0.13 and a collapse rate of 0.20 (Table
4), so on a fifth of the origins its forecasts spread across assets by less than a tenth of the realised
spread, and an MAE ranking would not reveal this.

Kronos (Shi et al., 2025) could not run, because the present project has no Kronos source checkout, and the
specification (section S1) notes that the headline gains of Kronos are zero-shot gains over general models
rather than gains over a tuned asset-specific model. The present study therefore makes no claim about Kronos,
and its adapter is tested only against a stand-in predictor. Nixtla neuralforecast, which the specification
lists as a source of supervised baselines, is replaced here by our own PatchTST and ridge implementations,
because they share one training loop and one cutoff check with the rank-tuned models.

## Decision

The present project keeps the cutoff rule and the decision metrics as the core of the arena, because the
evidence for them is the strongest in the study. Table 3 shows that the rank IC of the MSE model on its
training dates is 8.00 times its rank IC after the cutoff, which exceeds every difference between
models in Table 1, so the arena scores models only after their cutoff.

The project uses the soft-Spearman loss for the rank-tuned registry model (patchtst-rank), because no loss
reached significance over MSE, while the soft-Spearman loss alone kept usable return forecasts. The pairwise
model had the highest point estimate, 0.031 against 0.028, but its forecasts spread about 12 times wider than
realised returns, which makes them unusable for the forecast endpoint. During the study we changed the
setting rank_loss of the registry model patchtst-rank in config/models.yaml from listnet to spearman, because
Table 1 showed that the ListNet model reached a rank IC of 0.022 with a dispersion ratio of 12.93, while the
soft-Spearman model reached 0.028 with a dispersion ratio of 0.57. The report experiments set the rank loss
of each variant themselves, so this change does not alter any number in the tables.

The project keeps LoRA rank-tuning of Chronos-Bolt as a registry entry, but it does not claim a gain from it,
because the paired t-test of 1.46 (p = 0.15) stays below the critical value of 1.96. The LoRA model raised the
rank IC on the common window from 0.038 to 0.050 with 147 456 trainable weights, which are less than 1% of
the model, but it lowered the net Sharpe ratio on its eligible window from 2.34 to 1.87.

## Limitations

The data is synthetic, and its signal has a known linear form, which favours the ridge baseline and the
momentum rule over models that must discover the form. Three seeds give a seed-to-seed standard deviation of
0.001 to 0.007 in rank IC, which is small next to the bootstrap intervals but rests on few runs. The paired
t-test assumes independent origins, which overlapping signals violate, so the bootstrap intervals in Table 2,
which allow for serial dependence, are the more reliable evidence. The declared cutoffs of public models are
release dates, which can only be later than the true end of their training data, so the rule can exclude
too many origins but never too few. The LoRA Chronos-Bolt model ran with one seed, because each run takes
several minutes on a laptop CPU.

## References

Amazon (2025). Chronos-2. huggingface.co/amazon/chronos-2.

Ansari, A. F. et al. (2024). Chronos: Learning the language of time series. github.com/amazon-science/chronos-forecasting.

Bailey, D. H. and Lopez de Prado, M. (2014). The deflated Sharpe ratio. Journal of Portfolio Management 40(5).

Burges, C. et al. (2005). Learning to rank with gradient descent. ICML.

Cao, Z. et al. (2007). Learning to rank: from pairwise approach to listwise approach. ICML.

Forecast Collapse in TSFMs (2026). alphaxiv.org/abs/2608.14106.

Hu, E. J. et al. (2021). LoRA: Low-rank adaptation of large language models. arXiv:2106.09685.

Nie, Y. et al. (2023). A time series is worth 64 words: long-term forecasting with Transformers. ICLR.

Noguer i Alonso, M. and Franklin (2026). econpapers.repec.org/paper/arxpapers/2606.27100.htm.

Politis, D. N. and Romano, J. P. (1994). The stationary bootstrap. JASA 89(428).

Rahimikia, E. et al. (2025). arXiv:2511.18578.

Shi, Y. et al. (2025). Kronos: a foundation model for the language of financial markets. AAAI.

The command make report of the present project writes Tables 1 to 4 to
$TRADEQ_DATA_DIR/tsfm-arena/reports/report_tables.md, and it writes the same numbers to report.json.
