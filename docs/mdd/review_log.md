# Human review log: LLM-drafted MDD narrative

The number guardrail and citation check (D-043) are automated controls. They verify that every
figure traces to code and every citation to a supplied passage. They do **not** verify meaning.
Every LLM draft is therefore reviewed by the developer before use. This log records what review found.

## Review 1: draft v1 (`mdd_draft_v1_rejected_on_review.md`)

Automated result: 7 of 7 sections passed the guardrail (6 on the first attempt).
Human review: **rejected**. Errors found that the automated checks cannot detect:

| Section | Error | Category |
|---|---|---|
| Limitations | Read an actual-to-predicted ratio below 1 as *under*-prediction; it is over-prediction | Inverted meaning |
| Limitations | Claimed the forbearance treatment "aligns with regulatory expectations" citing SR 26-2 p.7, which does not discuss forbearance | Fabricated regulatory support |
| CECL | Described the upside as the mirror image of the *baseline*; it is the mirror of the downside | Factual error |
| Data | Called PD snapshots "loans" | Wrong unit of analysis |
| Data | Described the Federal Reserve's own model practice (from a retrieved Fed document) as this model's practice | Misattribution |
| Hazard | A figure copied from the decisions log ("~9%") passed because it coincidentally matched an unrelated value in metrics/config | Guardrail false negative |
| LGD | "Adheres to" a policy passage on collateral practical expedients the model does not use | Overclaim |
| PD | Called a PSI below 0.1 "moderate stability" | Mis-reading of a convention |

Changes made after review 1:
1. Guardrail for MDD sections restricted to the section's own FACTS, reading notes and cited passages
   (not all of metrics/config), removing coincidental matches.
2. Code-generated READING NOTES added to each section (e.g. how to read actual/predicted, how the
   upside is built, that Fed documents describe the Fed's models).
3. Prompt rules: cite a source only for what it literally says; never present Fed practice as ours.

## Review 2: draft v2 (`mdd_draft.md`)

Automated result: 4 of 7 sections passed; 3 were **rejected** after 3 attempts. Most rejections are
false positives: section-scoped matching blocked legitimate figures that appear in the decisions log
(e.g. the 80/20 fit/holdout split) but not in that section's FACTS.

Errors still present in the passed sections:

| Section | Error | Category |
|---|---|---|
| Hazard | Described a COVID-period actual-to-predicted of 0.22 as "under-prediction" despite the reading note; it is over-prediction | Inverted meaning (repeat) |
| DFAST | Presented the Federal Reserve's model assumptions (zero losses on FHA/VA loans, new originations, a 90th-percentile rule) as this model's practice, despite the explicit prompt rule | Misattribution (repeat) |
| LGD | Garbled description of curtailments ("curtailing partial prepayments") | Wording |

## Conclusion

- The guardrail reliably enforces **numeric provenance**. Its scope is a trade-off: permissive matching
  lets coincidental figures through (review 1); strict matching rejects legitimate ones (review 2).
- It cannot enforce **meaning**: direction of comparisons, attribution, or whether a citation supports
  its sentence. Prompt rules and reading notes reduced but did not remove these errors with a local 8B model.
- **Use policy:** the LLM draft is an input to a human-written MDD, never published as is. Per the
  project owner's decision, the guardrail is documented as is rather than engineered further.
