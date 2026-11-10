# Modeling Decisions Log

Each entry records what was decided, why, and which alternatives were rejected.
Numbers quoted here are pointers into `results/metrics.json`; that file is the source of truth.

---

## Phase 1: Data

### D-001 Use the Freddie Mac SFLLD *sample* files, Release 48
**Decision.** 50,000 randomly sampled loans per vintage, 1999–2025, plus the partial 2026
vintage (25,000 loans). Performance runs through June 2026.
**Why.** Each vintage is a simple random sample of the full dataset, so default rates
are unbiased estimates of the population rates. The data fits on a laptop (sizes, runtime and peak
memory in `phase1_convert`), so anyone can reproduce the results.
**Rejected.** *Full Standard Dataset* (~56M loans): it would add little precision for
this purpose and make the project unreproducible on normal hardware. *Fannie Mae data*:
similar content, but Freddie publishes actual-loss fields that Phase 4 needs.
**Limitation.** GSE conforming loans only. No jumbo, subprime or portfolio loans, so
default rates are lower than for a bank's whole mortgage book.

### D-002 Convert once to typed parquet with strict casting
**Decision.** Polars lazy scan, every column read as text first and then cast with
`strict=True` to the dtype in the R48 layout. Output is zstd parquet, one file per vintage and table.
**Why.** Strict casting turns any layout drift (a new column, a shifted field) into a
hard failure instead of silent nulls. Streaming keeps peak memory near 1 GB.
**Rejected.** *Chunked pandas*: slower and easy to get dtypes wrong. *DuckDB*: equally
valid; polars was chosen so a single library covers I/O and feature engineering.

### D-003 Default event definition
**Decision.** The first month with any of:
(a) delinquency status ≥ 3 (90+ days past due),
(b) status `RA` (REO acquired), or
(c) zero balance code 02/03/09/15 (third-party sale, short sale/charge-off, REO
disposition, whole-loan sale).
**Why.** 90 DPD is the standard regulatory default trigger. (b) and (c) catch loans
that liquidate before reaching 90 DPD; without them defaults would be understated.
Code 15 is included because Freddie computes Actual Loss for it, which marks it as a
credit disposition.
**Rejected.** *180 DPD*: closer to the charge-off point but leaves fewer events and
delays the signal. *Loss-only events* (02/03/09): would treat every cured 90+ DPD loan
as non-default and push those losses into LGD instead of PD.
**Note.** Default is **absorbing**: after the first default a loan leaves the PD
population, even if it later cures. Cures are handled in LGD (Phase 4).

### D-004 Exclude 90+ DPD months during forbearance (`default_definition: ex_forbearance`)
**Decision.** A 90+ DPD month does not trigger default while the loan is in forbearance
(`borrower_assist = F`), was granted a payment deferral that period (`payment_deferral = C`),
or is flagged as disaster-related delinquency (`disaster_flag = Y`). `RA` and
default-type zero balance codes always count.
**Why.** Under COVID forbearance, missed payments still appear as contractual
delinquency in the data, but most of those loans cured and losses were minimal. Under
the strict definition, 12-month default rates for snapshots taken in 2019 and 2020 look
like the 2009 crisis peak, which contradicts realized losses
(`phase1.by_snapshot_year[*].pd_spec_def` vs `.pd`). A PD model trained on that
artefact would link default risk to the wrong drivers. The decision rests on this
data evidence alone. (An earlier draft also cited 2020 interagency guidance on past-due
reporting for loans in pandemic forbearance; that statement is not among the indexed
sources and could not be verified here, so it is not relied on.)
**Rejected.** *Strict spec (any 90+ DPD)*: kept as a sensitivity label (`target_spec`)
and reported side by side. *Dropping 2020–21 entirely*: throws away real defaults that
happened outside forbearance.
**Limitation.** The forbearance and disaster fields are populated only from **2014‑01**
in this data. Earlier events (e.g. Hurricane Katrina in 2005) cannot be identified, so
pre-2014 rates follow the strict definition.

### D-005 Prepayment vs maturity
**Decision.** Zero balance code 01 counts as a **prepayment** only if more than one
scheduled payment remained in the prior month. Otherwise it is a scheduled payoff (censored).
**Why.** Code 01 covers both voluntary payoff and maturity. Only the voluntary kind
belongs in a prepayment hazard, which CECL needs to shorten contractual life.

### D-006 Censoring
**Decision.** Codes 16 (reperforming-loan securitization) and 96 (defect settlement),
matured loans, and loans still active at the cutoff are right-censored.
**Why.** None of these is a credit outcome or a voluntary prepayment. Treating code 16
as default or prepay would mislabel loans that are still paying.

### D-007 12-month PD dataset: yearly snapshots by loan age
**Decision.** One snapshot per loan at loan ages 0, 12, 24, ..., taken only if the loan
is active and has not yet defaulted. Target = 1 if the first default falls in
(snapshot, snapshot + 12 months]. A snapshot is used only if its 12-month window closes
by the performance cutoff, so no label is truncated. A loan that prepays inside the
window without defaulting gets target 0.
**Why.** A 12-month default rate measured from origination is too rare in post-crisis
vintages to estimate a scorecard (`phase1.by_vintage[*].default_12m_from_orig`). Yearly
snapshots match how bank 12-month PD and CECL inputs work: they score loans as they age
and pick up behavioral information such as prior delinquency.
**Rejected.** *Origination-only*: too few events. *Monthly snapshots*: 12x the rows of
nearly duplicate, highly correlated observations, which inflates apparent sample size.
*Censoring prepayers inside the window*: biases PD upward; the CECL hazard (Phase 3)
treats prepayment as a competing risk properly.
**Planned check (Phase 2).** One loan contributes several snapshots, so standard
errors will be clustered by loan.

### D-008 Out-of-time split (to be implemented in Phase 2)
**Decision.** Train on vintages 1999–2014; out-of-time test on 2015–2019. Vintages
2020+ are excluded from PD development. A robustness check cuts by calendar date of the
snapshot as well.
**Why.** The test vintages were originated after all training vintages, so the scorecard
is judged on loans it could not have seen. The training set includes a full credit cycle
(the 2005–2008 vintages).
**Known weakness.** Training snapshots from 2011–2014 vintages continue into calendar
years that overlap the test period, so the two sets share some macro history. The
calendar-time split measures how much that matters.

### D-009 National macro drivers from the Fed's own historic series
**Decision.** National unemployment, HPI and mortgage rate come from
`2026_Final_Historic_Domestic.csv`, the same file family as the DFAST scenarios.
**Why.** The DFAST HPI is the Z.1 owner-occupied real estate price index, not
Case-Shiller or FHFA. Training on the scenario's own definition removes basis risk when
scenarios are fed into the model.

### D-010 As-of date 2025-12-31 for CECL and DFAST
**Why.** It matches the jump-off quarter of the 2026 supervisory scenarios exactly, and
leaves Jan–Jun 2026 actuals for a genuine out-of-sample backtest of the projections.

### D-011 Data-quality findings accepted as-is
- 202 loans have gaps in their monthly history (1,122 missing loan-months out of 76.0M).
  They are kept; events are timed by `period`, not by row position.
- 16 origination records (2025–2026 vintages) have no performance rows yet. They are
  excluded from outcomes, which are built from performance data.
- The loss identity (UPB + accrued interest + expenses + recoveries = Actual Loss, with
  recoveries stored as negatives) holds to within $1 for every disposition that has loss
  data (`phase1.totals.loss_identity_pass` = `dispositions_with_loss`).

---

## Phase 2: 12-month PD scorecard

### D-012 Samples: fit / in-time holdout / out-of-time
**Decision.** Fit on 80% of *loans* from vintages 1999–2014. The other 20% of those loans
form an in-time holdout. Out-of-time (OOT) = vintages 2015–2019. The holdout is split by
loan (a hash of `loan_id` with the seed), never by row.
**Why.** Snapshots of the same loan are correlated; splitting rows would put a loan's
2008 snapshot in fit and its 2009 snapshot in holdout, overstating holdout performance.
The holdout is used only for LightGBM early stopping; OOT is never used for any choice.

### D-013 Features: only what is known at the snapshot date
**Decision.** Origination attributes; loan state at the snapshot (age, delinquency,
paydown, modification, any prior 30+ DPD); mark-to-market LTV from the state FMHPI
**lagged 2 months**; origination rate spread to PMMS (SATO); refinance incentive
(note rate − PMMS at the snapshot).
**Excluded and why.** Freddie's ELTV, VantageScore and valuation method exist only in
recent years, so the model would learn the era rather than the risk. Forbearance flags
start in 2014 for the same reason. Seller name has no economic meaning and would break
for new sellers. MSA and zip3 are too granular for a scorecard; state carries geography.
**Territories.** PR, GU and VI have no state FMHPI and use the national index.

### D-014 Binning: monotonic WoE with optbinning
**Decision.** Up to 8 bins, at least 3% of the sample per bin, monotonic WoE trend
(direction chosen by the solver), except loan age, which is allowed one peak
(default risk rises in the first years of seasoning, then falls). Missing values get
their own bin.
**Why.** Monotone bins are explainable ("higher LTV, more risk"), stable over time, and
make the points table defensible. **Rejected:** unconstrained bins, which fit noise and
produce reversals a validator would challenge.

### D-015 Feature selection
**Decision.** In order: (1) drop IV < 0.02; (2) walk features from highest IV, dropping
any whose WoE has |correlation| > 0.70 with an already-kept feature; (3) backward
elimination on the logistic fit, removing first any **wrong-signed** coefficient
(must be negative under optbinning's WoE convention), then any p-value > 0.01 with
**loan-clustered** standard errors. Every drop and its reason is in
`phase2.scorecard.dropped`.
**Why clustered SEs.** A loan contributes up to ~20 snapshots; ordinary SEs treat them
as independent and overstate significance.

### D-016 Challenger: LightGBM on raw features
**Decision.** Same features, gradient boosting with monotone constraints on the
variables with an economic direction (FICO −, LTV/CLTV/DTI/MTM LTV/delinquency/SATO +),
early stopping on the holdout, SHAP explanations on an OOT sample.
**Role.** It measures how much signal the scorecard leaves unused. It is not the
production model: interpretability and stability under SR 26-2 favour the scorecard
unless the uplift is material.

### D-017 Validation metrics
Gini and KS on fit, holdout and OOT; a 95% Gini CI from a loan-clustered bootstrap;
calibration by decile; actual-to-predicted ratio; score PSI and per-feature CSI between
fit and OOT; OOT Gini by vintage and by snapshot year.
**Robustness:** (a) a calendar split (train only on snapshots ≤ 2013, test on OOT
snapshots ≥ 2015), which removes the calendar overlap noted in D-008; (b) an
origination-only scorecard without behavioral variables; (c) OOT restricted to loans
current at the snapshot; (d) OOT on the strict (spec) default label.

### D-018 Minimum bin size exception for rare, high-risk states (bug fix)
**What happened.** In the first run (`phase2_v1` in metrics.json), current delinquency
(`delq_num`) and the modification flag had IV = 0. Delinquent and modified loans are
~1% of snapshots, below the 3% minimum bin size, so the binner could not isolate them
and merged everything into one bin.
**Decision.** Per-variable minimum bin size of 0.1% for `delq_num` and `modified` only,
applied at **both** the pre-binning and the final binning step. (The first fix set only
the final bin size; the 2% pre-binning floor still merged the rare values. The regression
test now uses the real pre-binning setting and caught it.) The 3% floor stays for
everything else, because it prevents noisy bins.
**Why.** A 30-DPD loan is many times riskier than a current one
(`results/tables` / Phase 2 diagnostics). A mortgage scorecard without a delinquency
characteristic would fail effective challenge. A regression test now pins this.

### D-019 Exclude original note rate and channel (era proxies)
**Evidence (training data only).** `results/tables/pd_stability_screen_train.csv`
compares 1999–2008 with 2009–2014 training vintages. The original rate shifts because
market rates fell, not because borrowers changed. Channel code `T` ("TPO not specified")
exists only before 2009, a disclosure change.
**Decision.** Exclude `orig_rate` (its risk content stays in SATO, the spread to the
market rate at origination) and `channel`, for both the scorecard and the challenger.
**Why not decide on OOT stability.** First-run OOT CSI also flagged them, but choosing
features with OOT data would leak test information into model selection; the
training-only screen is the evidence of record.
**Kept despite shifting.** `hpi_change`, `mtm_ltv` and `loan_age` also move across eras,
but those shifts are the economic cycle the model is meant to capture.

### D-020 Recalibrate PD levels on a recent window (scorecard ranking kept)
**Problem.** The scorecard ranks well out of time but over-predicts in benign years:
it learned the average default rate of a crisis-heavy training period.
**Decision.** Logistic recalibration, logit(PD_new) = a + b·logit(PD_scorecard), fitted on
OOT snapshots from 2015–2017 (outcomes observed by end-2018) and evaluated only on
snapshots from 2019 onward, whose outcome windows start after the calibration windows end.
**Why.** It fixes level and slope without touching rank order, using only information a
bank would have had at the time.
**Rejected.** Recalibrating on the whole OOT sample and reporting on the same sample:
it would look perfect by construction.
**Scope.** This is for the stand-alone 12-month PD. CECL lifetime PD comes from the
macro-driven hazard model (Phase 3), which addresses the same through-the-cycle vs
point-in-time issue structurally.

---

## Phase 3: Lifetime PD and prepayment (competing-risk hazard)

### D-021 Discrete-time multinomial logit hazard on the loan-month panel
**Decision.** Each month a loan at risk either survives, defaults or prepays:
h_k = exp(x·b_k) / (1 + Σ exp(x·b_j)). Lifetime curves follow as
S(t) = Π(1 − h_d − h_p), marginal PD(t) = S(t−1)·h_d(t).
**Why.** Monthly data and time-varying macro inputs map naturally to a discrete-time
hazard. A single multinomial model makes default and prepayment true competing risks
(probabilities sum to ≤ 1 and a prepaid loan can't default), which CECL needs because
contractual life must be shortened by expected prepayments.
**Rejected.** *Two separate logits*: probabilities can exceed 1 jointly. *Cox model*:
continuous time with a nonparametric baseline that can't be projected beyond observed
ages. *Roll-rate (Markov) model*: richer, but needs a delinquency-transition matrix per
state and macro regime, which is too much for the sample size per cell.

### D-022 Only projectable drivers
**Decision.** Covariates: loan age (piecewise linear), origination risk (FICO, CLTV, DTI,
SATO, loan size, purpose, occupancy, single borrower), and three drivers with scenario
paths: national unemployment (level and 4-quarter change) and mortgage rate from the
Fed's historic file, and mark-to-market LTV from state FMHPI. Loan-level dynamic inputs
are lagged one month; unemployment one quarter; HPI three months; the refinance
incentive two months.
**Why.** CECL and DFAST run the model forward under a scenario. A covariate with no
scenario path (current delinquency, forbearance) can't be projected; loans already in
those states at the forecast date get an overlay (Phase 5, D-026).
**Note.** Unlike the Phase 2 scorecard, this model is scenario-conditional, so it uses
realized macro values during fitting. That is not lookahead: at forecast time, the
scenario supplies the path.

### D-023 Choice-based sampling with constant correction
**Decision.** Fit on 25% of loans, keeping every default and prepay month and 5% of
survive months; add ln(0.05) to both outcome constants.
**Why.** 76M loan-months is too large to fit with clustered errors on a laptop. Under
outcome-based sampling a multinomial logit's slopes stay consistent and only the
outcome-specific constants shift, by ln(1/r) (Manski–Lerman; Prentice–Pyke). A unit test
recovers known coefficients from a 5%-sampled synthetic panel.

### D-024 Exclude March 2020 – December 2021 from estimation
**Evidence.** Including those months collapsed the unemployment effect: +2 pts of
unemployment raised predicted defaults by only ~9% in the production fit. Unemployment
jumped ~9 points while forbearance and stimulus prevented the defaults that history
implies.
**Decision.** Exclude the COVID policy period from estimation; keep it in the backtest
and report it separately (`phase3.backtest_covid_only`).
**Why.** It is a policy regime, not a credit cycle. A model that learned "unemployment
doesn't matter" from it would understate stress losses, the main use of this model.
**Implication for CECL.** If such a policy response recurs, the model will overstate
losses. That is a qualitative-adjustment (overlay) matter, not something to bake into
coefficients.

### D-025 No forbearance covariate
**Decision.** The lagged forbearance flag is not a regressor.
**Why.** (1) It can't be projected: in every forecast it would be zero. (2) Its meaning
changed: before 2020 it marked individual hardship (strongly predictive of default);
in COVID it was blanket relief that mostly cured. In the first fit its coefficient
was the largest in the model and drove a spurious 2020 default spike.
**Disclosure.** This design change was made after seeing the out-of-time backtest. It
is justified on projectability grounds, which hold independently of the backtest.

### D-026 Forbearance history and delinquency at the forecast date are overlays
**Evidence.** In 2022–2025, loans with any forbearance history are ~5% of loan-months
but produce about half the out-of-time defaults; for loans never in forbearance the
development model's post-COVID actual/predicted is close to 1 in every vintage group
(`phase3.backtest_post_covid_never_forbearance_by_vintage`).
**Decision.** The hazard model covers loans current and never in forbearance at the
forecast date. Loans delinquent or with forbearance history at the as-of date get
empirically calibrated multipliers in Phase 5.
**Sign check note.** Investor occupancy has a negative default coefficient after
controls, against the usual prior. In GSE data investor loans carry higher FICO and
lower LTV requirements; the conditional effect is plausible but flagged in
`phase3.sign_checks_prod`.

---

## Phase 4: LGD and EAD

### D-027 Two-part LGD on resolved defaults through 2021
**Decision.** E[LGD] = (1 − P(cure | default)) × E[severity | liquidation]. Development
uses defaults through December 2021; later defaults are too often unresolved, and
dropping unresolved cases would bias LGD toward fast (usually cheap) resolutions.
**Why two parts.** About half of defaults cure (Phase 1). A single LGD regression on a
target that is exactly zero for half the sample fits the zeros and the losses badly;
the cure path also responds to different drivers (equity, servicing) than severity does.
**Rejected.** *Severity only* (LGD = severity): ignores cures and roughly doubles
expected loss. *Tobit / zero-inflated beta*: more assumptions, no gain in this sample.

### D-028 Code 16 (reperforming-loan sale) counts as a cure
**Evidence.** 70% of defaulted loans sold as code 16 were current at their last report,
vs 36% for code 96. **Decision.** Count 16 as cured (no credit loss to the holder at
that point); exclude 96 (defects). Switch: `lgd.zb16_treatment: exclude`.

### D-029 Severity on principal, UPB at default as the denominator
**Decision.** Severity = (Actual Loss − delinquent accrued interest) / UPB at the default
month, clipped to [0, 1] for the fractional model (clipping shares in `phase4.severity_clipping`).
**Why.** Accrued interest on nonaccrual loans is reversed against income, not reserved
(an election available under ASC 326 when write-off is timely). UPB at default is the EAD
concept; Freddie's "UPB removed at zero balance" is tiny for some charge-offs and gave
severities in the hundreds.
**Not discounted.** CECL permits non-DCF methods; Freddie losses are nominal. A
note-rate-discounted severity is reported as a sensitivity (`phase4.discounting`).

### D-030 Forward house-price change as an LGD driver
**Evidence.** Without it, cure and severity were nearly flat across the cycle: benign-year
cures were under-predicted and severities over-predicted (2000–2006, 2014+).
**Decision.** Add the state HPI change over the 12 months after default.
**Why legitimate.** LGD here is scenario-conditional, like the hazard: under CECL or DFAST
the scenario supplies the HPI path. It also gives LGD the stress sensitivity a downturn
model needs (`phase4.sensitivities_prod`).

### D-031 Servicing-regime indicator; unemployment removed from the cure model
**Evidence.** After adding forward HPI, cures from 2014 on were still under-predicted, a
timing that matches the 2014 federal servicing rules on loss mitigation and the 2017 GSE
Flex Modification. Unemployment change had a *positive* effect on cures, confounded with
the 2009–2010 modification programs.
**Decision.** Indicator for defaults from 2013 on (set to 1 in all projections); remove
unemployment from the cure model. LGD's macro channel runs through house prices only.
**Disclosure.** Both changes, and moving the out-of-time split from 2012 to 2016 (so the
indicator can be estimated), were made after seeing the first out-of-time results.
**Residual.** LGD remains conservative in 2017–2021 and under-predicts the 2008 peak;
both reported (`phase4.test_oot`, `phase4.by_default_year`), neither patched with further
indicators. See D-041 for the later loan-size correction, which made it more conservative.

### D-032 EAD = scheduled amortization balance
**Evidence.** Actual UPB at default vs the balance scheduled 12 months earlier:
UPB-weighted ratio ≈ 1.00 (`phase4.ead`). Defaulting borrowers stop paying ~3 months
before 90 DPD, which barely moves a 30-year amortization.
**Decision.** EAD = scheduled balance at the projected default month (no undrawn
commitment, no CCF). Curtailments (partial prepayments) are ignored, which is slightly conservative.

---

## Phase 5: CECL allowance

### D-033 Three probability-weighted scenarios built from the Fed's 2026 scenarios
**Decision.** Baseline = the Fed's 2026 supervisory baseline. Downside = baseline + 50% of
the distance to the severely adverse scenario. Upside = the mirror image (baseline − 50%),
with unemployment floored at 3%. Weights 50 / 30 / 20.
**Why.** CECL needs a *reasonable and supportable* range, not a tail event; the full
severely adverse is a capital stress test (Phase 6) and is shown as a sensitivity.
Weights lean to the downside because credit losses are convex in the economy:
a symmetric spread of scenarios produces an asymmetric spread of losses.
**Rejected.** *Single baseline*: ignores that convexity. *Vendor scenarios* (Moody's
etc.): paid, and not reproducible from public data.
**Judgment, disclosed.** The Fed publishes no upside; the mirror construction and the
weights are management judgment, shown with alternatives in the sensitivity table.

### D-034 Reasonable-and-supportable period: 8 quarters
**Why.** It covers the period over which the Fed scenario has meaningful shape (the
unemployment peak arrives in year 2 of the severely adverse path) and is a common
industry choice. Sensitivity: 4 and 12 quarters.

### D-035 Reversion: straight-line over 4 quarters to the historical loss rate
**Decision.** Over the 4 quarters after the R&S period the monthly loss blends linearly
from the model to the historical loss rate (annual credit loss / average outstanding UPB,
averaged over 2000–2021 resolved defaults) applied to the surviving balance; after that,
the historical rate only. Macro inputs revert to 1999–2025 averages over the same
4 quarters (this drives survival/prepayment).
**Why.** It follows the specification and ASC 326's requirement to revert to historical
loss information when forecasts are no longer reasonable and supportable.
**Big judgment, disclosed.** The historical window includes the 2008 crisis, so the
reversion is the largest single driver of the allowance (`phase5.sensitivity`): it raises
losses after year 3 in benign scenarios and lowers them in the stressed one. The alternative "input reversion" (macro reverts,
model continues) and other historical windows are shown side by side.
**Long-run window.** 1999–2025 (the models' development period), not 1976+, whose 1980s
mortgage rates (15%+) would choke off prepayment and lengthen lives artificially.

### D-036 Overlays for delinquency and history at the as-of date
**Decision.** Segment multipliers on the default hazard for loans that are 30 DPD, 60 DPD,
delinquent in forbearance, or current with forbearance/prior-default history, applied for
12 months then fading to 1 over 12 more. Calibrated on year-ends 2016–2018 and 2021–2024
(COVID windows excluded) as each segment's actual/predicted 12-month default rate
*divided by* the clean segment's actual/predicted, so the hazard's own level bias isn't
counted twice. Loans already in default (90+ DPD outside forbearance, or REO) default in
month 1 with certainty and take LGD only.
**Why.** The hazard has no delinquency input by design (D-022); these states are known at
the as-of date and are the strongest short-term predictors.

### D-037 Undiscounted, sample-portfolio allowance
The ACL is the undiscounted lifetime expected credit loss (non-DCF method, D-029) on the
Freddie Mac *sample* loans active at 2025-12-31. Dollar amounts describe that sample; the
ratio to UPB is the meaningful result. No qualitative (Q-factor) adjustment is quantified;
the COVID experience (D-024) is the documented example of where one would be considered.

### D-041 Loan size in real terms in the LGD model (correction found in Phase 6)
**What happened.** Under the severely adverse scenario the LGD came out at about a third
of the 2008–2010 realized LGD, although the scenario's house-price fall is at least as
severe and today's mark-to-market LTVs are similar to Dec 2007. The cause was nominal
loan size: severity falls with log(UPB) (fixed foreclosure costs), and loans grow with
house prices, so the model read "today's loans are big, hence low severity". It was an era
proxy, the same failure as the Phase 2 note rate.
**Decision.** Use UPB deflated by the national FMHPI at default ("real" loan size) in both
LGD parts.
**Trade-off, disclosed.** The nominal version fitted 2017–2021 better; the real version is
more conservative there (`phase4.test_oot`) but stable across eras. For a model used for
stress testing, stability across eras wins.
**Not changed.** The hazard model also uses nominal log(UPB), with small coefficients and a
clean out-of-time backtest; listed as a known limitation rather than refitted.

---

## Phase 6: DFAST

### D-038 Supervisory projection setup
**Decision.** Project the Dec-2025 portfolio over 2026Q1–2028Q1 under the Fed's 2026
baseline and severely adverse paths, as published (no R&S period, no reversion, no
weighting). Loss rate on the Fed's definition: cumulative 9-quarter loss divided by the
average balance over the horizon. Same engine, models and overlays as CECL.
**Benchmark.** Fed 2026 results, Table 9: first-lien domestic loss rate under severely
adverse, all participating banks. Differences in portfolio (GSE conforming, MI coverage)
and in model design mean the benchmark is a range check, not a target.

### D-039 Crisis replay as the stress-engine validation
**Decision.** Run the engine on the Dec-2007 portfolio with the *realized* 2008–2010 macro
path and compare with the losses those loans actually produced.
**Why.** It's the only available test of the full chain (hazard × LGD × EAD × overlays)
in a real severe downturn.
**Caveat.** The models were estimated on data that includes 2008–2010, so this checks
aggregation and calibration under stress, not out-of-sample forecasting.

### D-040 Downturn LGD and the servicing-regime assumption
**Evidence.** Severely adverse LGD remains below the 2008–2010 realized LGD. Switching off
the post-2013 servicing indicator (D-031) raises it materially
(`phase6.downturn_lgd.sa_without_servicing_regime`).
**Decision.** Report both. The main projection keeps the indicator (the observed regime);
the no-servicing case is the downturn-LGD sensitivity a validator would ask for, because
post-2013 cure rates have never been observed in a severe recession.

---

## Phase 7: LLM reporting layer

### D-042 Local open-source stack only
**Decision.** Ollama (installed as a global Homebrew service) serving `qwen3:8b` for text
and `nomic-embed-text` for embeddings; FAISS exact inner-product search; temperature 0 and a
fixed seed.
**Why.** Nothing leaves the machine (in a bank: confidential data, no third-party processing),
the run is reproducible, and there are no API costs. Exact search because the corpus is small.
**Rejected.** Hosted LLM APIs (data leaves the perimeter); approximate indexes (no need at
this corpus size).

### D-043 The LLM never produces numbers: guardrail + citation check
**Decision.** Code formats every figure into a FACTS block; the LLM only writes narrative.
Each draft must pass (1) a number guardrail: every figure, at the precision shown, must
equal a value in metrics.json, config.yaml or the supplied regulatory passages, after
fraction→percent and $→$M/$B conversions (identifiers like SR 26-2 and bare years excluded);
(2) a citation check: every [doc p.N] must be a passage actually supplied. Failures are fed
back for regeneration up to 3 times; a section that still fails is rejected, not published.
**Why.** SR 26-2 places generative AI outside its scope but expects governance and controls
for such tools; a deterministic check is a control a validator can test (`tests/test_guardrail.py`).

### D-044 Evaluation design
**Decision.** 52 questions over the indexed documents. Every key fact is verified to appear
on its gold page before the eval runs. Metrics: retrieval hit@1, hit@5 and MRR (gold document
and page), key-fact recall in the answer, faithfulness (share of answer sentences whose
embedding similarity to some retrieved sentence is ≥ 0.75), numeric grounding (every number
in the answer appears in the retrieved context) and citation validity.
**Limitation.** Embedding-similarity faithfulness is a proxy, not a human judgment; it can
pass paraphrased errors and fail correct paraphrases.

### D-045 Traceability matrix
**Decision.** `eval/requirements.yaml` maps each regulatory requirement (with source page) to
implementation, pytest tests and a metric. The script runs pytest for live test status,
retrieves the requirement's passage as evidence, and records the developer's own assessment
(Met / Partially met) separately from automated status.

### D-046 Dense retrieval kept; hybrid BM25 rejected
**Evidence.** On the 52-question set, dense-only retrieval beat dense + BM25 with
reciprocal-rank fusion on hit@1, hit@5 and MRR (`phase7_eval.retrieval_ablation`). BM25 rewards
term frequency, which further favours the 545-page Fed model document.
**Decision.** Keep dense retrieval. Stop there: trying more variants against the same 52
questions would be tuning on the test set.
**Known limitation.** One very large document crowds out shorter ones (retrieved evidence matches
the cited page for only about half of the traceability requirements). The structural fix is
per-document balancing or splitting, not done.

### D-047 The guardrail is necessary but not sufficient: human review is the control
**Evidence.** The first draft passed the guardrail in every section but contained inverted
interpretations, fabricated regulatory support and misattribution (`docs/mdd/review_log.md`).
A number check verifies provenance, not meaning, and coincidental numeric matches can slip through.
**Decision.** Tighten the guardrail for documentation (section-scoped facts only), add reading notes,
and require developer review of every draft, logged in `docs/mdd/review_log.md`. Remaining
limitations are documented rather than engineered away: the guardrail does not check
the direction of comparisons, the attribution of claims, or whether a cited passage supports
the sentence that cites it.

---

## Phase 8: Packaging

### D-048 Reports rendered from metrics; reproducibility tested on a clean copy
**Decision.** README.md and the model development document are templates rendered from
`results/metrics.json` (`scripts/build_reports.py`); an unresolved placeholder fails the build.
Reproducibility is tested by copying only code, config and templates to an empty folder (raw data
linked, nothing derived), running `make all`, and comparing every numeric value in phases 1–6 of
the two metrics files (`scripts/compare_metrics.py`).
**What the test found.** The first clean run failed: `src/cecl/data/` was missing. The `.gitignore`
pattern `data/` ignores *every* directory named `data`, so the data-pipeline package would never
have been committed and a fresh clone could not run. Patterns are now root-anchored (`/data/`,
`/private/`).
