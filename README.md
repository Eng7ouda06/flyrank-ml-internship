# Capstone: Ranking Pages That Under-Capture Clicks

**Lane:** CTR / Engagement Opportunity Scoring

- **Deployed paper:** https://eng7ouda06.github.io/flyrank-ml-internship/
- **Capstone notebook:** [`work/notebooks/capstone.ipynb`](work/notebooks/capstone.ipynb)
- **Paper locator:** [`submission/paper_url.txt`](submission/paper_url.txt)

**Question:** Which already-visible pages earn fewer clicks than peers at the same ranking position, and which should a content team review first?

**Result (observed, starter release, client-holdout validation):** the model reached ROC AUC 0.82 against 0.735 for a prior-CTR baseline. Precision in its top 5% was 0.867 against 0.62. The top 10% of pages by score had an 80% under-capture rate against 37% overall.

**Recommendations:** each top-ranked page gets a first action: metadata review, refresh review, content depth review, or monitor.

**Honest framing:** this is decision support built on the starter slice. It shows associations, not causal effects, and says nothing about Google's algorithm. Data is anonymized and public-safe, with no client names, URLs or queries.

---
