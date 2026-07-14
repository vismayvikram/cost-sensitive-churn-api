# Cost-Sensitive Churn Prediction API

> A churn classifier that optimizes for **total expected dollar loss**, not accuracy or F1 — because missing a churner and false-alarming on a loyal customer do not cost a business the same amount.


---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Why This Is Different](#why-this-is-different)
- [Results](#results)
- [Project Architecture](#project-architecture)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [API Reference](#api-reference)
- [Interactive Dashboard](#interactive-dashboard)
- [Project Structure](#project-structure)
- [Running Locally](#running-locally)
- [Running with Docker](#running-with-docker)
- [Limitations & Future Work](#limitations--future-work)
- [Tech Stack](#tech-stack)

---

## Problem Statement

Subscription-based businesses — telecom, SaaS, streaming, gyms — live or die by retention. Acquiring a new customer typically costs **5–25× more** than retaining an existing one. If a company can predict who's about to churn, it can intervene — a discount, a call, an upgrade — before they leave.

Most tutorial projects stop at "train a classifier, report accuracy." They miss something that matters more in production:

| Error Type | What Happened | Business Cost |
|---|---|---|
| **False Negative** — missed churner | Model says "stay," customer leaves | Full remaining customer value, lost |
| **False Positive** — false alarm | Model says "churn," loyal customer flagged | Cost of one unneeded retention offer (~$50) |

Accuracy and F1 both silently assume these two mistakes are equally bad. They aren't — missing a churner is dramatically more expensive than over-flagging a loyal one. **A model chosen by F1 is not necessarily the model that costs a business the least money.**

---

## Why This Is Different

Rule-based flags ("hasn't logged in 30 days → flag") miss multivariate interaction effects — declining usage **+** a recent complaint **+** an upcoming renewal is a different risk than any signal alone. ML captures that. But a good model on its own isn't the differentiator here — **the cost-sensitive decision layer on top of it is.**

This project:

1. Assigns a real dollar cost to each error type, per customer (`FN_cost = MonthlyCharges × 12`, `FP_cost = $50` flat).
2. Sweeps the classification threshold and measures **total expected cost**, not accuracy, at every point.
3. Discovers the pure cost-minimum is operationally undeployable (flags 79% of the customer base) — so it re-solves the problem **under a realistic capacity constraint** (retention team can only act on ~30% of customers).
4. Ships the constrained-optimal threshold (**0.39**) to a live API, so every prediction is a cost-aware decision, not just a probability.

That progression — theoretical optimum → real-world constraint → deployable decision — is the actual engineering contribution of this project.

---

## Results

All numbers below are from the held-out **test set** (1,407 customers), using thresholds selected on a separate **validation set** (1,406 customers) to avoid leakage.

### Threshold tuning alone, no retraining

| Threshold | Basis | Accuracy | F1 | AUC | Expected Cost |
|---|---|---|---|---|---|
| 0.50 (default) | — | 79.96% | 0.575 | 0.844 | $152,623.80 |
| **0.04** | unconstrained cost-minimum | 46.62% | 0.496 | 0.844 | **$39,557.40** |

Same model, same probabilities, zero retraining — just moving the decision boundary — cuts expected cost by **74.1%** ($113,066.40). This is the core proof that threshold ≠ accuracy-optimal ≠ cost-optimal.

Sanity check on why: at threshold 0.04, the model flags **79.4%** of all customers. That's the mathematically cheapest policy — but no retention team can act on 4 out of every 5 customers. The unconstrained answer is real, but not deployable as-is.

### Adding a capacity constraint (the deployable answer)

| Capacity Cap | Best Threshold | Test Cost | % of Customers Flagged |
|---|---|---|---|
| 10% | 0.68 | $225,526.80 | 10.1% |
| 20% | 0.50 | $152,623.80 | 20.6% |
| **30%** | **0.39** | **$94,887.40** | **32.0%** |
| 50% | 0.20 | $47,595.80 | 49.9% |
| Unconstrained | 0.04 | $39,557.40 | 79.4% |

At a 30%-of-base capacity constraint — a defensible assumption for a real retention team — the cost-optimal threshold is **0.39**, saving **~38% vs. the default 0.50** ($152,623.80 → $94,887.40) while staying operationally realistic.

### Final production model (tuned via `RandomizedSearchCV`, threshold 0.39)

| Configuration | Threshold | AUC | Expected Cost |
|---|---|---|---|
| Tuned XGBoost, default cutoff | 0.50 | 0.835 | $38,404.40 |
| **Tuned XGBoost, deployed cutoff** | **0.39** | **0.835** | **$34,667.00** |

**→ $3,737.40 saved per test-set pass (9.7%), on the exact model shipped in this repo.**

### A negative result worth keeping (this is a feature, not a gap)

Cost-sensitivity can be applied at training time too, via `sample_weight` (weighting churners by `FN_cost` during fitting) instead of just tuning the threshold after the fact. Tested and included for comparison:

| Approach | Threshold | AUC | Expected Cost |
|---|---|---|---|
| Post-hoc threshold tuning only | 0.04 | 0.844 | $39,557.40 |
| Training-time cost weighting (`sample_weight`) | 0.18 | 0.803 | $40,492.80 |

In this dataset, simple post-hoc threshold tuning **outperformed** training-time reweighting — the weighted model's AUC dropped (0.844 → 0.803) without a corresponding cost improvement. Reporting this honestly, rather than hiding the approach that "lost," is intentional: it shows the decision to ship threshold-tuning over reweighting was evidence-based, not just the first thing that worked.

---

## Project Architecture

See **[architecture.md](./architecture.md)** for the full data flow, encoding pipeline, and request lifecycle diagrams.

```
┌────────────────────────────────────────────────────────┐
│  Browser UI (index.html)                                │
│  - 19-field customer input form, grouped by category     │
│  - Auto-calculated TotalCharges (tenure × MonthlyCharges)│
│  - Risk gauge, cost comparison, SHAP explanation cards    │
│  - Collapsible raw JSON response viewer                  │
└─────────────────────┬────────────────────────────────────┘
                       │ POST /predict (JSON)
┌─────────────────────▼────────────────────────────────────┐
│  FastAPI Application (app/main.py)                        │
│  - Pydantic request validation (schemas.py)                │
│  - Serves dashboard at GET /                                │
└─────────────────────┬────────────────────────────────────┘
                       │
┌─────────────────────▼────────────────────────────────────┐
│  ChurnModel (app/model.py)                                 │
│  - Loads XGBoost model, feature list, cost config at startup│
│  - One-hot encodes raw input, reindexes to training columns │
│  - predict_proba → churn probability                        │
│  - Applies capacity-constrained cost-optimal threshold (0.39)│
│  - SHAP TreeExplainer → top 3 per-customer reasons           │
└──────────────────────────────────────────────────────────┘
```

---

## Dataset

**IBM Telco Customer Churn** — [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

- 7,032 customers after cleaning (rows with blank `TotalCharges` handled)
- 19 raw input features: demographics, subscribed services, contract, billing, payment
- Binary target: `Churn`
- One-hot encoded (`pd.get_dummies(drop_first=True)`) → **30 model features**
- Split: 4,219 train / 1,406 validation / 1,407 test (60/20/20, stratified)

---

## Methodology

### 1. Cost assignment (per customer, not a flat global number)

```python
FN_cost = MonthlyCharges × 12   # estimated annual value lost if a churner is missed
FP_cost = $50                    # flat cost of one retention offer
```

`FN_cost` ranges **$219–$1,425** across the dataset (mean $777.58) — a $25/month customer and a $115/month customer are not equally expensive to lose, and the model's threshold decision accounts for that per-customer, not as one global average.

### 2. Threshold optimization

Thresholds swept `0.01 → 0.99` in 0.01 steps on the **validation set**; expected cost computed at each point; minimum selected. Repeated under capacity constraints (10/20/30/50%/unconstrained) to find the deployable optimum. Final decision threshold (**0.39**) validated once, at the end, on the held-out **test set** — never used to choose the threshold itself, to avoid leaking test information into the decision.

### 3. Model selection

`XGBClassifier`, hyperparameters tuned via `RandomizedSearchCV` (40 candidates × 5-fold stratified CV, scored on average precision). Final params: `max_depth=3, learning_rate=0.1, n_estimators=100, subsample=0.8, colsample_bytree=0.7, min_child_weight=5, gamma=0.5`.

### 4. Explainability

`shap.TreeExplainer` for both global feature importance and per-customer decision breakdowns. **SHAP values are in log-odds space**, not probability space — a value of `+0.58` is a log-odds contribution, not "+58% churn probability." Every SHAP-carrying output in this project (API response, dashboard) states this explicitly rather than implying false precision.

---

## API Reference

### `POST /predict`

**Request**

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 2,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.5,
  "TotalCharges": 171.0
}
```

**Response**

```json
{
  "churn_probability": 0.9784,
  "will_churn_flag": true,
  "threshold_used": 0.39,
  "fn_cost_if_missed": 1026.0,
  "fp_cost_if_wrong": 50.0,
  "top_reasons": [
    { "feature": "tenure", "value": 2.0, "impact": 0.5758, "direction": "increases" },
    { "feature": "InternetService_Fiber optic", "value": 1.0, "impact": 0.4965, "direction": "increases" },
    { "feature": "PaymentMethod_Electronic check", "value": 1.0, "impact": 0.2285, "direction": "increases" }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `churn_probability` | float | Raw XGBoost output (0.0–1.0) |
| `will_churn_flag` | bool | `true` if `churn_probability ≥ threshold_used` |
| `threshold_used` | float | Capacity-constrained cost-optimal threshold (0.39) |
| `fn_cost_if_missed` | float | `MonthlyCharges × 12` — this customer's specific FN cost |
| `fp_cost_if_wrong` | float | Flat $50 retention-offer cost |
| `top_reasons[]` | array | Top 3 SHAP contributions (log-odds space — see Methodology) |

### `GET /` — serves the dashboard · `GET /health` — container health check

---

## Interactive Dashboard

Served at `/`. Preset demo profiles (High/Moderate/Low risk), all 19 fields grouped by category, auto-computed read-only `TotalCharges`, a risk gauge, a side-by-side FN-vs-FP cost comparison, plain-language recommendation banner, SHAP driver cards, and a collapsible raw-JSON viewer to prove it's live-API-backed rather than a static mock.

---

## Project Structure

```
churn-api/
├── app/
│   ├── main.py
│   ├── model.py
│   ├── schemas.py
│   └── templates/index.html
├── churn_model_final.json
├── feature_names.json
├── model_config.json
├── eda.ipynb
├── cost_sensitive_training.ipynb
├── shap_summary.png
├── shap_waterfall_example.png
├── cost_vs_threshold.png
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── requirements.txt
├── Dockerfile
├── README.md
└── architecture.md
```

---

## Running Locally

```bash
cd churn-api
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Dashboard: `http://127.0.0.1:8000/` · Swagger docs: `http://127.0.0.1:8000/docs`

## Running with Docker

```bash
cd churn-api
docker build -t churn-detector-api .
docker run -p 8000:8000 churn-detector-api
```

---

## Limitations & Future Work

- **Cost assumptions are fixed and simplified.** `FN_cost = MonthlyCharges × 12` assumes exactly one year of remaining lifetime and ignores discounting, contract-length effects, and acquisition cost. `FP_cost = $50` is a flat assumption, not tied to actual offer economics. A production version would pull both from real billing/CRM data.
- **SHAP correlated-feature attribution.** `tenure`, `MonthlyCharges`, and `TotalCharges` are correlated by construction; Shapley's fair-credit split can attribute the "same" underlying signal to different features across similar customers. Directionally reliable, not exact-decomposition reliable.
- **Static dataset, no drift monitoring.** Real churn drivers shift over time (pricing changes, competitor moves, macro conditions); this model reflects one historical snapshot and has no retraining/monitoring pipeline.
- **Training-time cost-weighting underperformed threshold tuning here** (see Results) — worth revisiting with a custom XGBoost objective function directly optimizing expected cost, rather than `sample_weight` as a proxy.
- **No authentication/rate limiting** on the API — fine for a portfolio demo, not for production traffic.

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | XGBoost |
| Explainability | SHAP (`TreeExplainer`) |
| API | FastAPI + Pydantic |
| Server | Uvicorn (ASGI) |
| Frontend | Vanilla HTML/CSS/JS |
| Containerization | Docker |
| Language | Python 3.11 |