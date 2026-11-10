# Model Development Document: draft narrative

> Narrative drafted by a local LLM (qwen3:8b) from `results/metrics.json`. Every figure passed the number guardrail and every citation the source check; rejected sections are listed and excluded. Numbers are produced by code only.

## Data and target definition

The data includes 1,374,984 loans across 76,002,480 loan-months, with 55,157 lifetime defaults using the defined default event. 12-month PD snapshots cover 6,179,066 loans, of which 53,498 defaulted. Duplicate loan-months are 0, and liquidations reconcile to the loss identity with 19,681 of 19,681. The Freddie Mac SFLLD *sample* files, Release 48, are used with 50,000 randomly sampled loans per vintage, 1999–2025, plus 25,000 loans from the partial 2026 vintage. Performance runs through June 2026. This approach ensures unbiased default rate estimates and reproducibility on standard hardware.  

Default events are defined by the first occurrence of delinquency ≥ 3, status `RA`, or zero balance codes 02/03/09/15. This captures liquidations before 90 DPD and aligns with Freddie’s Actual Loss computation. Excluding 90+ DPD months during forbearance, payment deferral, or disaster flags avoids overestimating default rates during periods of temporary relief, as seen in 2019 and 2020. The 12-month PD dataset uses yearly snapshots by loan age, with targets based on the first default within the 12-month window. This mirrors how banks score PD and CECL inputs, ensuring alignment with operational practices.  

The model development process emphasizes conceptual soundness, data selection, and methodological choices, as outlined in [SR2602a1.pdf p.6] and [SR2602a1.pdf p.8]. Outcomes analysis is used to compare model outputs to real-world defaults, ensuring alignment with business objectives. The Board relies on expert judgment and literature review to identify key drivers, ensuring model variables are relevant and reduce over-fitting risks. Data quality is managed through validation and conservative assumptions where deficiencies persist, as described in [fall_2026_credit_risk_models.pdf p.457].

## 12-month PD scorecard

The 12-month PD scorecard features 15 variables and achieves an out-of-time Gini of 0.778 and an out-of-time KS of 0.632. The LightGBM challenger model, using the same features, achieves a higher out-of-time Gini of 0.791, indicating it captures additional signal. The scorecard’s PSI between development and out-of-time is 0.072, suggesting moderate stability. The out-of-time actual-to-predicted default ratio is 0.54, reflecting the model’s calibration in this period.

The scorecard is trained on 80% of loans from vintages 1999–2014, with 20% held out in-time and used for LightGBM early stopping. Out-of-time data includes vintages 2015–2019. Features are selected by dropping IV < 0.02, removing highly correlated WoE variables, and applying backward elimination with loan-clustered standard errors. This approach aligns with [SR2602a1.pdf p.6] on model design and testing, ensuring robustness and validity.

The LightGBM challenger is used to assess the scorecard’s signal capture, with monotone constraints on economically relevant variables and SHAP explanations. Features like `orig_rate` and `channel` are excluded due to their reliance on era proxies, which are unstable across vintages. Recalibration of PD levels is performed on OOT data from 2015–2017 to address over-prediction in benign years, without altering the scorecard’s ranking. This approach supports [SR2602a1.pdf p.9] on outcomes analysis and model adjustment.

## Lifetime default and prepayment hazard

The model uses a discrete-time multinomial logit hazard approach to estimate monthly default and prepayment probabilities, with lifetime default and prepayment curves derived from the survival function and marginal PD. The in-time default actual-to-predicted ratio is 0.97, while out-of-time prepayment is 1.02, indicating reasonable calibration. The out-of-time default actual-to-predicted ratio during the COVID policy months is 0.22, suggesting significant model underperformance during that period. The model excludes March 2020 – December 2021 from estimation due to the unique policy regime, which distorted the unemployment effect, as evidenced by a +2 points unemployment increase raising predicted defaults by only ~9% in the production fit.

The model incorporates only projectable drivers, including loan age, origination risk factors, and macroeconomic variables with defined scenario paths. Loan-level inputs are lagged one month, while unemployment is lagged one quarter and HPI three months. The decision to exclude the forbearance flag as a regressor is based on its inability to be projected and its changing meaning over time, which led to a spurious 2020 default spike in the first fit. This design choice is justified by projectability grounds and was informed by out-of-time backtest results.

The model’s performance is validated through outcomes analysis, which compares model outputs to real-world outcomes, as outlined in [SR2602a1.pdf p.9]. Regular model testing, including out-of-sample and out-of-time testing, ensures the model performs as intended. The exclusion of the COVID policy period from estimation aligns with regulatory expectations to avoid model underperformance during stress scenarios, as noted in [SR2602a1.pdf p.6]. The model’s design supports CECL and DFAST requirements by allowing for forward-looking scenario analysis.

## Loss given default and exposure at default

The model incorporates out-of-time severity and LGD metrics, with actual severity at 23.5% and predicted at 32.2%, while actual LGD (UPB-weighted) is 2.0% and predicted at 5.6%. The EAD actual-to-scheduled ratio is 1.003, reflecting the use of scheduled amortization balance as EAD, consistent with defaulting borrowers stopping payments ~3 months before 90 DPD.  

LGD is modeled as a two-part process, distinguishing between cures and severity, with E[LGD] = (1 − P(cure | default)) × E[severity | liquidation]. This approach accounts for the different drivers influencing cure and severity, such as equity and servicing, and avoids biases from unresolved defaults. Forward house-price change is included as an LGD driver, aligning with scenario-based modeling under CECL and DFAST.  

Servicing-regime indicators are used for defaults from 2013 onward, reflecting changes in loss mitigation policies, while unemployment is excluded from the cure model due to confounding effects from prior modification programs. Loan size is adjusted for real terms using the national FMHPI at default to correct for nominal size misinterpretation in severe scenarios.  

The model adheres to regulatory expectations from [acl_policy_statement_2023.pdf p.5], which emphasizes fair value of collateral and adjustments for costs to sell, and [acl_policy_statement_2023.pdf p.6], which outlines collateral maintenance practical expedients and zero ACL scenarios when collateral exceeds amortized cost.

## CECL allowance

The CECL allowance is calculated using a probability-weighted approach based on three scenarios: baseline, downside, and upside, with weights of 50%, 30%, and 20%, respectively. These scenarios are derived from the Fed's 2026 supervisory scenarios, with the baseline representing the Fed's 2026 supervisory baseline, the downside adjusted by adding 50% of the distance to the severely adverse scenario, and the upside as the mirror image of the baseline, with unemployment floored at 3%. This approach aligns with the need for a reasonable and supportable range, as outlined in [acl_policy_statement_2023.pdf p.4].  

The reasonable and supportable period is set at 8 quarters, reflecting the period over which the Fed scenario has meaningful shape and aligning with common industry practice. Reversion to historical loss information occurs over 4 quarters, with the monthly loss blending linearly from the model to the historical loss rate. This reversion method follows the specification in [acl_policy_statement_2023.pdf p.4], which allows for a straight-line basis or any systematic, rational method.  

Delinquency and history overlays are applied to the default hazard for specific loan segments, including loans 30 DPD, 60 DPD, delinquent in forbearance, or current with forbearance/prior-default history. These overlays are calibrated using actual/predicted 12-month default rates from 2016–2018 and 2021–2024, excluding the COVID windows. Loans already in default (90+ DPD outside forbearance, or REO) default in month 1 with certainty. This approach ensures the hazard reflects current conditions without double-counting bias, as described in [acl_policy_statement_2023.pdf p.4].

## DFAST stress projection

The DFAST stress projection applies the Fed's 2026 baseline and severely adverse paths to the Dec-2025 portfolio over 2026Q1–2028Q1, using the same engine, models, and overlays as CECL. The loss rate is calculated as the cumulative 9-quarter loss divided by the average balance over the horizon. The severely adverse 9-quarter loss rate is 1.02%, compared to the baseline 0.21%, and the Fed's 2026 all-bank first-lien loss rate benchmark is 1.5%. The crisis replay predicted and actual loss rates are both 1.88%, indicating a severe downturn scenario.

The crisis replay validation tests the model's performance by applying the realized 2008–2010 macro path to the Dec-2007 portfolio and comparing the results with actual losses. This provides a real-world check of the hazard × LGD × EAD × overlays chain under severe economic stress. However, the models were estimated on data including 2008–2010, so this test focuses on aggregation and calibration rather than out-of-sample forecasting.

The severely adverse LGD is 14.3%, which remains below the 2008–2010 realized LGD. Disabling the post-2013 servicing indicator increases the LGD materially, as noted in `phase6.downturn_lgd.sa_without_servicing_regime`. Both scenarios are reported, with the main projection retaining the observed servicing regime. The no-servicing case is included to assess downturn-LGD sensitivity, as post-2013 cure rates have not been observed in a severe recession.

## Limitations and model use

The model's calibration in overlay years shows a clean-loan actual-to-predicted ratio of 0.48, indicating a significant underprediction of defaults. For the Jan-Jun 2026 period, actual defaults were 1,858, while the model predicted 2,667, highlighting a notable gap in predictive accuracy. This discrepancy may reflect model limitations in capturing real-world default behavior, particularly during periods of economic stress or policy intervention.

The decision to exclude 90+ DPD months during forbearance aligns with regulatory expectations [SR2602a1.pdf p.7], acknowledging that missed payments during such periods do not equate to defaults. This exclusion is based on observed outcomes during the 2019–2020 period, where default rates under a strict definition were disproportionately high, contradicting actual losses. The model's reversion to historical loss rates over four quarters [D-035] further reflects a balance between forward-looking forecasts and historical data, as required by ASC 326.

The model's use of real loan size in LGD calculations [D-041] addresses an earlier flaw where nominal loan size led to an underestimation of severity. This adjustment ensures that LGD estimates are more reflective of actual risk, particularly in severely adverse scenarios. The inclusion of both the observed servicing regime and a no-servicing case [D-040] provides transparency and supports validation efforts, as outlined in regulatory guidance [SR2602a1.pdf p.9].
