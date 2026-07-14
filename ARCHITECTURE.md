# Architecture

This document covers how data moves through the system at both **training time** and **inference time**, and why a few specific design decisions were made.

---

## 1. System Overview

Two independent pipelines share one artifact contract:

```
TRAINING (offline, notebook)              INFERENCE (online, API)
──────────────────────────────           ──────────────────────────────
Raw CSV                                   Raw JSON (single customer)
  │                                          │
  ▼                                          ▼
Clean + encode (pd.get_dummies)           One-hot encode (pd.get_dummies)
  │                                          │
  ▼                                          ▼
Train/val/test split                      Reindex to feature_names.json ◄── (see §3)
  │                                          │
  ▼                                          ▼
Train XGBoost + tune threshold            predict_proba()
  │                                          │
  ▼                                          ▼
Save 3 artifacts ─────────────────────►   Load 3 artifacts at startup
  churn_model_final.json                    │
  feature_names.json                        ▼
  model_config.json                      Apply threshold + SHAP + cost math
                                             │
                                             ▼
                                          JSON response
```

The **three artifacts** are the entire contract between offline training and the online API. Nothing else crosses that boundary — the API never re-derives anything the notebook already computed (thresholds, cost multipliers, column order).

---

## 2. Training-Time Data Flow

1. **Load** `WA_Fn-UseC_-Telco-Customer-Churn.csv` (7,043 rows raw).
2. **Clean**: coerce `TotalCharges` to numeric (it ships as a string with blank entries for new customers), impute or drop blanks → 7,032 usable rows.
3. **Assign per-customer costs**: `FN_cost = MonthlyCharges × 12`, `FP_cost = 50` (flat). These live as extra columns during training so cost-aware sample weighting and cost-based evaluation can reference the correct customer, but are dropped from `X` before fitting — the model never sees its own cost labels as a feature.
4. **Encode**: `pd.get_dummies(df, columns=cat_cols, drop_first=True)` on 15 categorical columns → 30 total feature columns.
5. **Split**: stratified 60/20/20 → train (4,219) / validation (1,406) / test (1,407). Stratification matters here because churn is imbalanced (~27% positive); an unstratified split risks a validation or test fold with a meaningfully different churn rate than train, which would silently bias the threshold-selection step in §4.
6. **Train baseline** `XGBClassifier` with default hyperparameters — this becomes the reference point every later step is measured against.
7. **Threshold sweep** (§4 below) on the **validation** split only.
8. **Hyperparameter search**: `RandomizedSearchCV`, 40 candidates, 5-fold stratified CV, scored on average precision (chosen over ROC-AUC because it's more sensitive to performance on the minority/positive class in an imbalanced setting).
9. **Re-sweep threshold** on the tuned model, this time under a capacity constraint (§4).
10. **Final, one-time evaluation** on the **test** split, at the threshold chosen in step 9 — the test set is touched exactly once, after every modeling decision is already locked in.
11. **Serialize**: model → `churn_model_final.json` (XGBoost's native JSON format, not pickle — portable across XGBoost versions and language-agnostic to read back). Column order → `feature_names.json`. Threshold + cost constants → `model_config.json`.

---

## 3. Why Inference-Time Encoding Isn't Just "Run get_dummies Again"

This is the single most fragile part of the system if handled naively, so it's worth explaining precisely.

`pd.get_dummies()` looks at **all rows present in the DataFrame it's called on** and creates one column per category *actually observed in that batch*. During training, the batch is 7,032 rows spanning every category — so every dummy column that could exist, does.

At inference time, the API receives **one row** — one customer. If that customer's `InternetService` is `"DSL"`, calling `get_dummies()` on that single row will never produce an `InternetService_Fiber optic` or `InternetService_No` column at all, because those categories simply aren't present in a batch of size one. Feed that directly into the model and you get a shape mismatch or, worse, a silently misaligned prediction if you're not checking shapes.

**Fix implemented in `app/model.py`:**

```python
df_enc = pd.get_dummies(df, columns=cat_cols)
df_enc = df_enc.reindex(columns=self.feature_names, fill_value=0)
```

`reindex` against the exact training column list (loaded from `feature_names.json`) guarantees:
- Every column the model expects exists, in the exact order the model was trained on.
- Any dummy column not produced by this single row (because that category wasn't the one selected) is correctly filled with `0` — which is the mathematically correct encoding for "this customer is not in that category."
- Extra columns that shouldn't exist are silently dropped.

This is why `feature_names.json` is a required artifact, not an optional convenience — the API is architecturally incapable of correct predictions without it.

---

## 4. Threshold Selection Logic

Two distinct sweeps happen in the training notebook, answering two different questions.

### Sweep A — Unconstrained cost-minimum
*"Ignoring operational limits, what threshold minimizes total dollar cost?"*

```python
for t in thresholds:
    preds = probs >= t
    fn = missed churners at this cutoff
    fp = false alarms at this cutoff
    cost = fn * fn_cost + fp * fp_cost
optimal_t = threshold minimizing cost
```

Answer: **0.04**. Flags 79.4% of customers. Cheapest possible policy on paper, undeployable in practice — no retention team can act on 4 in 5 customers.

### Sweep B — Capacity-constrained cost-minimum
*"Given we can only act on ~30% of customers, what threshold minimizes cost within that limit?"*

Same sweep, but each candidate threshold is only considered valid if the resulting flagged percentage is ≤ the capacity cap. The minimum-cost threshold *within that feasible set* is selected.

Answer at 30% cap: **0.39** — this is the threshold shipped in `model_config.json` and used by the live API. It is deliberately *not* the same number as Sweep A's answer; the README documents both so a reader can see the reasoning, not just the final number.

Both sweeps run exclusively on the **validation** split. The **test** split is used exactly once, at the very end, purely to report an honest, unbiased estimate of real-world performance at the already-chosen threshold.

---

## 5. Request Lifecycle (Inference)

```
1. Client → POST /predict  (raw JSON, 19 fields)
2. FastAPI validates against Pydantic schema (schemas.py)
   → malformed/out-of-range fields rejected with 422 before reaching the model
3. ChurnModel._encode(raw)
   a. wrap raw dict in a 1-row DataFrame
   b. pd.get_dummies on the 15 categorical columns
   c. reindex to feature_names.json, fill_value=0   ← see §3
4. model.predict_proba(X) → churn_probability
5. churn_probability >= threshold_used (0.39) → will_churn_flag
6. fn_cost_if_missed = raw.MonthlyCharges * fn_cost_multiplier   ← per-customer, computed live
7. explainer.shap_values(X) → top 3 |impact|-ranked features
8. Assemble ChurnPrediction response → return JSON
```

Steps 3–7 all happen against a single in-memory model and a single `TreeExplainer`, both instantiated once at process startup (`ChurnModel.__init__`) — not re-loaded per request. This matters because `TreeExplainer` construction has nontrivial overhead; doing it per-request would make the API meaningfully slower under load for no benefit, since the explainer is stateless with respect to any individual prediction.

---

## 6. Why XGBoost's Native JSON, Not Pickle

`model.save_model('churn_model_final.json')` (XGBoost's built-in serialization) was used instead of `pickle`/`joblib` for two reasons:

1. **Version portability.** A pickled `XGBClassifier` embeds the exact Python/sklearn/xgboost class definitions at save time; loading it with a different library version can silently break or throw obscure errors. XGBoost's native JSON format is versioned and explicitly designed to load across XGBoost releases.
2. **Language/tooling agnostic.** The JSON file can, in principle, be inspected or loaded by non-Python tooling; a pickle cannot.

The tradeoff: native JSON only serializes the booster itself, not arbitrary Python-side state — which is precisely why `feature_names.json` and `model_config.json` exist as separate artifacts rather than being bundled as attributes on a pickled object.

---

## 7. Deployment Topology

```
┌───────────────────────────────────────────┐
│  Docker container                          │
│  ┌───────────────────────────────────────┐│
│  │ Uvicorn (ASGI server)                  ││
│  │  └── FastAPI app                        ││
│  │       ├── GET  /            → dashboard ││
│  │       ├── GET  /health      → liveness  ││
│  │       └── POST /predict     → inference ││
│  │  ChurnModel loaded once at container    ││
│  │  startup, held in memory for the        ││
│  │  container's lifetime                   ││
│  └───────────────────────────────────────┘│
│  Artifacts baked into image at build time: │
│    churn_model_final.json                  │
│    feature_names.json                       │
│    model_config.json                         │
└───────────────────────────────────────────┘
```

Single-container, single-process deployment — appropriate for a portfolio/demo deployment on Render/Railway/HF Spaces. A production version handling real traffic would separate the model-serving process from the web tier and add horizontal scaling, but that's out of scope for what this project is demonstrating (see [README → Limitations](./README.md#limitations--future-work)).