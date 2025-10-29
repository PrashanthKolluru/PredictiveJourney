# 📊 Growth Analytics Toolkit

End-to-end marketing analytics & experimentation framework, built in Python.

This repo contains two core capabilities:

1. **Bayesian Experiment Uplift Modeling (PyMC)**  
   Quantifies true incremental lift from an A/B test with full uncertainty, not just a p-value.

2. **Multi-Touch Attribution via Markov Chains**  
   Reconstructs user journeys across channels, measures each channel’s incremental contribution, and estimates ROAS-like efficiency.

Together, they answer:
- “Should we ship this experiment to 100% rollout?”  
- “Which channels are actually driving conversions efficiently?”


---

## 🚀 What This Project Does

### 1. Bayesian Experiment Uplift (A/B Test Analysis)
This module estimates how much the treatment variant actually improves conversion vs control — and how confident we are in that improvement.

It:
- Loads experiment data with per-user assignment (`treatment` vs `control`) and outcome (`converted` 0/1).
- Compresses those into group-level stats (how many in each arm, how many converted).
- Fits a **Bayesian logistic model in PyMC** to estimate:
  - baseline conversion rate for control
  - conversion rate for treatment
  - uplift (difference between them)
- Samples the full posterior distribution to get:
  - Mean uplift
  - 95% credible interval
  - Probability that uplift is positive

It then writes a machine-friendly summary to `ab_summary.json`.

This goes beyond a t-test or p-value. You get:  
> “There’s an X% chance the treatment really improves conversion.”

That maps directly to rollout decisions.


### 2. Markov Chain Multi-Touch Attribution
This module measures how much each marketing channel contributes to conversion, accounting for the *entire journey*, not just first touch or last touch.

It:
- Builds an ordered journey of touches for each customer (email → social → search → …).
- Constructs an absorbing Markov chain where:
  - every path starts at `"START"`
  - ends in either `"CONVERSION"` or `"NULL"` (no purchase)
- Estimates transition probabilities between channels.
- Computes the baseline probability that a user eventually converts.

Then it runs a **removal effect analysis**:
- For each channel, remove it from the system and recompute the conversion probability.
- The drop in conversion probability is that channel’s incremental value (its lift).
- Normalize those lifts to get each channel’s attribution share (what % of conversions it truly drove).

Finally, it joins spend to estimate ROAS-like efficiency:
- How many attributed conversions per $1 of spend for each channel?

It writes results to:
- `markov_attribution.parquet` – channel-level metrics
- `markov_summary.json` – dashboard/exec summary


---

## 🧠 Why These Two Pieces Belong Together

These notebooks form a lightweight growth analytics stack:

- **Bayesian Uplift Model →** “Does this tactic work if we turn it on?”  
  (causal lift from controlled experiments)

- **Markov Attribution Model →** “Where should we spend money to get more of those conversions?”  
  (incremental contribution of channels in the wild)

Used together:
1. You prove something *works* (via Bayesian lift).
2. You understand *which channels are worth scaling* (via attribution + ROAS-like efficiency).


---

## 📂 Key Inputs and Outputs

### Inputs (from ETL layer)
- `ab_experiment.parquet`  
  Contains one row per user with:
  - `customer_id`
  - `treatment` (0 = control, 1 = treatment)
  - `converted` (0/1)

- `sessions.parquet`  
  Per-customer ordered marketing touchpoints (email, social, search, ads, direct, etc).

- `weekly_sales.parquet`  
  Aggregated conversions across the business for the time window. Used to scale attribution back to business-level “total conversions”.

- `channel_spend.parquet`  
  Spend by channel in the same period. Used to compute ROAS-like efficiency.

> These are all expected under `etl/data/outputs/`.

---

### Outputs (for BI / dashboards / exec review)

#### `ab_summary.json`
Single-source-of-truth for experiment decisioning. Includes:
- `control_rate_mean` and 95% credible interval  
- `treatment_rate_mean` and 95% credible interval  
- `uplift_mean` and 95% credible interval  
- `prob_uplift_positive`

#### `markov_attribution.parquet`
Channel-level attribution table with (per channel):
- `channel`
- `lift` (removal effect: how much conversion rate drops if we delete that channel)
- `attribution_share` (% of incremental conversions this channel deserves)
- `attributed_conversions` (share × total conversions)
- `spend`
- `roas_like` (conversions per $1)

#### `markov_summary.json`
Compact summary for dashboards / alerts / exec email:
- Baseline conversion probability of the Markov chain
- Total conversions in the period
- For each channel:
  - incremental contribution
  - attribution share
  - spend
  - ROAS-like “conversions per dollar”
