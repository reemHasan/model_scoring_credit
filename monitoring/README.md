---
title: Monitoring Loan Scoring Api
emoji: 👀
colorFrom: purple
colorTo: indigo
sdk: docker
pinned: false
license: mit
---
# Loan Scoring — Monitoring Dashboard

Streamlit dashboard for monitoring the Home Credit Scoring API in production.

## What it shows
- KPIs: total calls, error rate, avg latency, rejection rate
- API latency and inference time over time
- Status code distribution (when errors exist)
- Predicted score and class drift over time
- Evidently AI drift and data summary reports

## How to run locally
```bash
streamlit run dashboard.py --server.runOnSave true
```