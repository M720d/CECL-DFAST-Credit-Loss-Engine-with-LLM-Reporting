# Model Development Document: draft narrative

> Narrative drafted by a local LLM (qwen3:8b) from `results/metrics.json`. Every figure passed the number guardrail and every citation the source check; rejected sections are listed and excluded. Numbers are produced by code only.

## Data and target definition

The data includes 1,374,984 loans across 76,002,480 loan-months, with 55,157 lifetime defaults. The 12-month PD snapshots cover 6,179,066 observations, of which 53,498 are defaults. Duplicate loan-months are 0, and liquidations reconcile to 19,681 of 19,681. Snapshots are loan-year observations, with one loan contributing multiple snapshots. [SR2602a1.pdf p.6] emphasizes the importance of clear purpose and sound development practices in model design, including data selection and methodology. [SR2602a1.pdf p.8] highlights the need for validation through outcomes analysis and periodic review to ensure model performance aligns with business objectives. [fall_2026_credit_risk_models.pdf p.139] notes that model development relies on expert judgment and literature review to identify key drivers, while [fall_2026_credit_risk_models.pdf p.457] describes data validation and conservative assumptions for missing or misreported information.

## 12-month PD scorecard

*Section rejected by guardrail after 3 attempts (unverified figures: ['12', '80%', '20%', '0.02', '0.70']; bad citations: []).*

## Lifetime default and prepayment hazard

The model uses a discrete-time multinomial logit hazard approach to estimate monthly default and prepayment probabilities, treating them as true competing risks. This method aligns with CECL requirements by allowing the contractual life of loans to be shortened by expected prepayments. The decision to use a single model rather than separate logits ensures probabilities sum to ≤ 1 and prepayment prevents default. The in-time default actual-to-predicted ratio of 0.97 indicates slight over-prediction, while the out-of-time prepayment ratio of 1.02 suggests minor under-prediction. 

Only projectable drivers are included in the model, such as loan age, origination risk factors, and macroeconomic variables with scenario paths. The exclusion of non-projectable covariates like current delinquency and forbearance ensures the model can be run forward under stress scenarios. The decision to lag dynamic inputs—such as unemployment by one quarter and HPI by three months—reflects the timing of data availability and model responsiveness. The out-of-time default actual-to-predicted ratio during the COVID policy months was 0.22, indicating severe under-prediction due to the policy regime's impact on the historical relationship between unemployment and default.

The model excludes March 2020 – December 2021 from estimation because the sharp rise in unemployment during this period, combined with forbearance and stimulus measures, distorted the historical link between unemployment and default. This exclusion was a modeling judgment, not a regulatory requirement, to prevent underestimating stress losses. The decision to omit the forbearance flag as a regressor is based on its lack of projectability and changing meaning over time, which led to spurious results in the first fit. Outcomes analysis, as described in [SR2602a1.pdf p.9], supports model validation and adjustment when performance deviates from expectations.

## Loss given default and exposure at default

The out-of-time severity, actual is 23.5% and predicted is 32.2%, indicating the LGD model is conservative in the out-of-time period. The out-of-time LGD, actual (UPB-weighted) is 2.0% and predicted (UPB-weighted) is 5.6%, further supporting the model's conservative nature. The EAD actual-to-scheduled ratio is 1.003, suggesting that the EAD is closely aligned with the scheduled amortization balance. 

D-027 outlines a two-part LGD approach, using (1 − P(cure | default)) × E[severity | liquidation], with defaults through December 2021. Later defaults are excluded due to unresolved cases, which would bias LGD toward fast resolutions. The model separates cure and severity, as they respond to different drivers, and rejects severity-only models for ignoring cures and overestimating losses. 

D-030 incorporates forward house-price change as an LGD driver, aligning with scenario-conditional LGD under CECL or DFAST. D-031 adds a servicing-regime indicator for defaults from 2013 onward and removes unemployment from the cure model, reflecting changes in federal servicing rules. D-032 defines EAD as the scheduled balance at the projected default month, excluding undrawn commitments and curtailing partial prepayments. 

D-041 corrects the use of nominal loan size in the LGD model by deflating UPB with the national FMHPI at default, addressing an era proxy issue. This adjustment ensures LGD reflects real loan size rather than nominal growth, which previously led to underestimation of severity in severe scenarios.

## CECL allowance

*Section rejected by guardrail after 3 attempts (unverified figures: ['3%', '12', '60', '90']; bad citations: ['[SR2602a1.pdf p.6]']).*

## DFAST stress projection

The DFAST stress projection applies the Dec-2025 portfolio over 2026Q1–2028Q1 under the Fed's 2026 baseline and severely adverse paths, using the same engine, models, and overlays as CECL. The loss rate is calculated as the cumulative 9-quarter loss divided by the average balance over the horizon. The severely adverse 9-quarter loss rate of 1.02% is compared against the Fed's 2026 all-bank first-lien loss rate benchmark of 1.5%, which serves as a range check, not a target. The crisis replay, with a predicted and actual loss rate of 1.88%, is used to validate aggregation, not forecasting, as it is in-sample for the models.

The crisis replay test involves running the engine on the Dec-2007 portfolio with the realized 2008–2010 macro path to compare with actual losses. This test checks the full chain of hazard × LGD × EAD × overlays in a real severe downturn. However, the models were estimated on data including 2008–2010, so this test validates aggregation and calibration under stress, not out-of-sample forecasting. The severely adverse LGD of 14.3% remains below the 2008–2010 realized LGD, and switching off the post-2013 servicing indicator raises it materially, as noted in `phase6.downturn_lgd.sa_without_servicing_regime`.

The model assumes that losses on loans insured by government programs, such as FHA and VA loans, are zero, as these loans protect mortgage lenders in the case of borrower default. The model projects losses on both the existing portfolio and a hypothetical portfolio of new originations, using similar methods for loss and payoff rates. The loss rate for deficient portfolios is based on the 90th percentile firm's loss rate path, calculated from 13-quarter losses, ensuring consistency with the principle of conservatism.

## Limitations and model use

*Section rejected by guardrail after 3 attempts (unverified figures: ['90']; bad citations: []).*
