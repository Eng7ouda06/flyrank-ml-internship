# Capstone Report — CTR / Engagement Opportunity Scoring

- **Author:** Mahmoud Abdelwahab Shaaban
- **Lane:** CTR / Engagement Opportunity Scoring
- **Repo:** https://github.com/Eng7ouda06/flyrank-ml-internship
- **Date:** 2026-09-26

## 0. Abstract

Given a portfolio of content pages, which ones are worth reviewing first because they are
already showing early signs of decline? Using an anonymized 30,000-page, 32-client slice of the
FlyRank ML Internship dataset — 90-day search performance plus a 30-day-vs-prior-30-day trend
label — I built a Logistic Regression ranker and evaluated it against a transparent CTR-gap
baseline rule on an identical, client-grouped holdout, so neither model nor baseline is graded
on clients it has already memorized. Features are restricted to information knowable before the
label window — no trend columns, no last-30-day metrics — and the evaluation harness itself is
checked by swapping in a naive random split and by deliberately reintroducing a leaky feature to
confirm it responds correctly. On seven held-out clients the model reaches precision@20 = 0.75
versus the baseline rule's 0.50 (base rate 0.51), a measured, directional improvement that
shrinks, not grows, once the naive split is replaced with the honest one. The output is a
ranked, reason-coded review queue meant to shorten a human's triage list — decision-support for
a content strategist, not an autonomous refresh system, and not yet validated on any client
outside this sample.

## 1. Problem framing

**Unit of analysis:** a single content page (`content_id`), scored within a client's portfolio.
**Output:** a probability (`decline_probability`) that ranks pages, plus a reason code and a
suggested action label (`refresh_priority`, `refresh_and_monitor`, `review_candidate`,
`monitor`, or `human_review_only`).
**Action a human takes:** a content strategist uses the ranked queue to decide which pages to
manually review first for a possible refresh, this month, out of a portfolio too large to review
in full.
**Cost of a wrong call:** a false positive wastes review time on a page that was fine; a false
negative lets a genuinely declining page go unreviewed for another cycle. Neither is severe on
its own — this is why the output is a prioritized queue for a human, not an automated action.
**Why ML helps here:** the transparent baseline rule (CTR gap vs. position tier) only reads a
90-day snapshot and has no way to distinguish "already declining" from "having one weak
window." A model trained against a real forward label (last 30 vs. prior 30 days) can, in
principle, do better — but only if it's validated in a way that proves it isn't just
memorizing client identity.

## 2. Data safety

**Data used:** `data/raw/content_refresh_anonymized.csv` — the anonymized starter slice of the
FlyRank ML Internship dataset, 30,000 content pieces across 32 pseudonymous clients. This is
explicitly the teaching-scale sample, not FlyRank's full production warehouse (that release —
341,701 pages / 469.9M impressions — is described in FlyRank's own March 2026 research paper);
every number in this report comes from the 30,000-row slice, not the full release.

**Columns deliberately excluded, and why:**
- `trend_pct` / `trend_direction` — these are literally what the label (`is_declining_label`)
  is built from. Using them as features would leak the answer directly.
- `impressions_last_30d`, `clicks_last_30d`, `sessions_last_30d`, and their `prev_30d` twins —
  same reason: they're the raw ingredients `trend_pct` is computed from, so even though they
  aren't the label column itself, they hand the model most of the information needed to
  reconstruct it.
- `client_id` — used only to group the train/test split. Never a model feature.

**Leakage audit performed:** confirmed all of the above are absent from the feature matrix,
then deliberately reintroduced `trend_pct` as a feature on the same client-grouped split. Its
coefficient dwarfed every honest feature by two orders of magnitude and precision@20 jumped to
1.00 — the harness responded exactly as a real leak should make it respond. Removing the column
again reproduced the honest 0.586 AUC exactly, confirming the test is trustworthy in both
directions.

**Client-identifying content:** none. Every `content_id` and `client_id` in this repo is a
pseudonymous hash; no client names, domains, URLs, or raw queries appear anywhere under `work/`.

## 3. Baseline

A transparent rule, not a model: expected CTR by position tier (top 3 / page 1 / striking
distance / page 3–5 / deep), computed from the same 90-day totals used everywhere else. A page
is flagged when its actual CTR falls under 50% of its tier's typical CTR, gated by a
1,000-impression floor and position < 20 so the rule only fires where it has enough volume to
mean something.

It's a fair comparison because it's evaluated on the *exact same rows, same split, same
metrics* as the model (section 5) — not on its own separately-selected queue. On the
client-grouped holdout: precision@20 = 0.50, precision@50 = 0.56, precision@100 = 0.54, ROC AUC
= 0.520 — barely better than the 0.511 base rate, because the rule was never built to predict
the *future*; it flags a CTR gap in the current window only.

## 4. Model / analysis

**Method:** Logistic Regression (standardized features, `class_weight="balanced"`) as the
primary, readable classifier — chosen because the task is binary classification with an
observed label, and coefficients can be sanity-checked against the honest features. A Random
Forest is trained alongside as a comparison and feature-importance cross-check.

**Target, in one sentence:** `is_declining_label` = 1 if `trend_direction == "down"` — whether a
page's last 30 days of impressions were meaningfully weaker than the 30 days before that.

**Feature list (21 numeric + 3 categorical), and what was left out on purpose:**
`impressions_90d, clicks_90d, sessions_90d, pageviews_90d, users_90d, engaged_sessions_90d,
ai_sessions_90d, scroll_events_90d, days_with_impressions, days_with_sessions,
content_age_days, days_since_last_update, ctr, avg_position, engagement_rate, scroll_rate,
ai_traffic_pct, search_volume, competition, cpc, word_count, char_count` plus categorical
`content_type, main_intent, competition_level`. Left out on purpose: `trend_pct`,
`trend_direction`, all six `*_last_30d`/`*_prev_30d` columns, and `client_id` (see section 2).
Missingness is flagged (`*_missing` columns) rather than silently zero-filled.

## 5. Evaluation

**Split:** client-grouped, 80/20 (`GroupShuffleSplit`, seed 42) — 7 clients held out entirely
from training, so the model is judged on clients it has never seen a single page from. Chosen
over a random row split because pages from the same client share a CMS, editorial voice, and
existing authority; a random split lets the model partly memorize client identity instead of
learning a page-level signal — which section 5's own before/after comparison confirms actually
happens here.

**Model vs. baseline, same split, same metrics (base rate 0.511):**

| | precision@20 | precision@50 | precision@100 | ROC AUC | avg. precision |
|---|---|---|---|---|---|
| Baseline rule | 0.50 | 0.56 | 0.54 | 0.520 | 0.518 |
| **Logistic Regression** | **0.75** | **0.80** | **0.74** | 0.586 | 0.589 |
| Random Forest | 0.50 | 0.52 | 0.46 | 0.605 | 0.583 |

**Random split control (the "before"):** the same Logistic Regression, same data, evaluated
under a naive random row split instead: precision@20 = 0.90, ROC AUC = 0.682 — 15 and 9.7 points
higher respectively. Every held-out "test" client in that split also has pages in training,
confirming the gap is client memorization, not a stronger model. The 0.75/0.586 grouped numbers
are the ones this report trusts.

**Error analysis:** the model's most confident false positives (predicted declining, actually
flat/growing) concentrate inside a single held-out client — a client-level tell, not a
page-level one. The most confident misses in the other direction are small pages, well under
the 1,000-impression floor — consistent with the same volume-readability problem the baseline
rule has.

## 6. Interpretation

Logistic Regression's largest coefficients are a `users_90d` vs. `sessions_90d` contrast plus
`days_with_impressions`; the Random Forest instead leans hardest on raw `impressions_90d` and
`avg_position`. Only `days_with_impressions` and `word_count` appear in both models' top
features — `users_90d`, `sessions_90d`, `pageviews_90d`, and `engaged_sessions_90d` are
near-duplicates of the same traffic count, so Logistic Regression is likely splitting one real
signal across several correlated coefficients rather than finding four independent effects.

A light K-Means pass (4 clusters, descriptive only) surfaces a genuine negative/surprise result:
one cluster of young, well-positioned, high-volume pages still shows the *highest* decline rate
of any archetype (0.6) — the opposite of what "young and well-ranked" would predict — and I
don't have an explanation for it in this dataset. That's flagged in the playbook as
`monitor_closely`, not force-fit into a tidy story.

**Negative result:** the baseline rule, which reads only the current 90-day snapshot, is
essentially uninformative about *future* decline (AUC 0.520, barely above 0.511 base rate) —
a well-understood "no effect," and the entire reason a validated forward-looking model was
worth building at all.

## 7. Recommendation

The full-portfolio queue routes 55.0% of pages to `human_review_only` (below the validation
floor), 19.5% to `monitor`, 10.0% to `refresh_and_monitor`, 9.6% to `refresh_priority`, and 5.9%
to `review_candidate`. A FlyRank editor would open the queue, skip anything already flagged
`human_review_only`, and work down the `refresh_priority` rows first — checking client
concentration in any batch before acting on it, since a single client accounted for 74% of this
queue's top 50 rows in testing.

**Confidence and limits, stated plainly:** precision@20 = 0.75 is a *measured, directional*
improvement over a coin-flip baseline, on a 30k-row teaching sample, not a guarantee for any
specific new client. It is decision-support for triage, not evidence that refreshing a flagged
page will improve it — no causal experiment sits behind this.

## 8. Reproducibility

- Repo: https://github.com/Eng7ouda06/flyrank-ml-internship
- Seed: 42, used identically in every notebook (`GroupShuffleSplit`, `train_test_split`,
  `LogisticRegression`, `RandomForestClassifier`, `KMeans`).
- Environment: `requirements.txt` (`pandas>=2.2`, `numpy>=1.26`, `scikit-learn>=1.4`,
  `matplotlib>=3.8`).
- To re-run everything from a fresh clone: `git clone
  https://github.com/Eng7ouda06/flyrank-ml-internship.git`, open
  `work/notebooks/capstone.ipynb` via its Colab badge, Runtime → Run all. It clones the repo,
  rebuilds the honest features, refits baseline/model/random forest under the client-grouped
  split, and regenerates both figures from scratch.
- Held-out evaluation is checkable, not taken on faith: the exact cell that builds the
  client-grouped split lives in `work/notebooks/capstone.ipynb` (and identically in
  `w05_model.ipynb`/`w06_validation_audit.ipynb`), and the metrics it produces are committed at
  `work/outputs/capstone_metrics.json` (and `w05`/`w06`/`w07`'s own JSONs).

## 9. Acknowledgments & data credit

Built on the [FlyRank ML Internship dataset](https://flyrank.ai).

---

> **Claims checklist before submitting:** observed / measured / directional / decision-support
> language everywhere · no causal claims without an experiment or causal design · no "predicted
> Google's algorithm" · no client-identifying details · numbers in this report match a fresh
> re-run. ✅ all satisfied above — base rate (0.511) is reported next to every precision@K/AUC
> figure in section 5.
