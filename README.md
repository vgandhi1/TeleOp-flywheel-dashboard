# 🤖 TeleOp Flywheel Dashboard

> **Attribute exact dollar value to human-in-the-loop robotic teleoperation data — and surface operator quality at per-session granularity.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-5.20+-3F4F75?style=flat-square&logo=plotly&logoColor=white)](https://plotly.com)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Tests](https://img.shields.io/badge/Tests-11%20passing-22c55e?style=flat-square&logo=pytest&logoColor=white)](#testing)

---

## The Problem

Humanoid robot programs need massive volumes of high-quality teleoperation data. A single 10,000-episode dataset can cost **$200K–$500K** in operator labor.

Most programs track **quantity** — but have no system to measure **quality** at collection time.

```
Without quality-aware cost attribution, a program can spend
60% of its teleoperation budget generating data that ML
researchers immediately discard.
```

| Symptom | Impact |
|---|---|
| Operators evaluated on throughput (sessions/hour) | Incentivizes speed over quality |
| High-latency sessions (>120ms) corrupt kinematic trajectories | Operator still paid in full |
| No real-time quality signal | ML researchers spend 2–3 days/week auditing raw data |

---

## The Solution

The **TeleOp Flywheel Dashboard** is a data product that:

1. Ingests per-session teleoperation logs (CSV)
2. Computes **Cost per High-Value Hour** and a **Data Quality Score** per session and operator
3. Surfaces a leaderboard, bottleneck tracker, and executive KPI summary

---

## Dashboard Preview

### 📊 Executive Summary
Five fleet-level KPIs at a glance: high-value hours collected, total operator spend, cost per high-value hour, global success rate, and average data quality score. Includes a DQS histogram, cost-by-task bar chart, and a gauge showing the % of sessions meeting the high-value threshold.

### 🏆 Operator Leaderboard
Color-coded table (green / yellow / red) ranked by Data Quality Score with per-operator session counts, HV hours, cost efficiency, success rate, latency, and intervention averages. Paired with a bubble chart of the top 10 operators.

### 🔍 Bottleneck Tracker
Session-level scatter plot of network latency vs. DQS with a 120ms threshold line — the infrastructure investment case. Includes failure rate by latency bucket and a dollar estimate of revenue lost to high-latency sessions.

---

## North Star Metrics

### Cost per High-Value Hour (CPHVH)

$$\text{CPHVH} = \frac{\text{Total Operator Cost (\$)}}{\text{Hours of High-Value Data Collected}}$$

A session is **High-Value** if **all three** conditions hold:

| Condition | Threshold |
|---|---|
| Session succeeded | `success_flag == True` |
| Network latency | `< 120 ms` |
| Operator interventions | `≤ 3` |

### Data Quality Score (DQS) — 0 to 100

| Component | Weight | Formula |
|---|---|---|
| Network Quality | 40% | `min(100, max(0, 100 − (latency − 80) × 1.5))` |
| Task Completion | 40% | `100 if success else 0` |
| Operator Precision | 20% | `max(0, 100 − interventions × 10)` |

**Quality tiers:**
- `DQS ≥ 80` — ✅ High Quality
- `60 ≤ DQS < 80` — ⚠️ Marginal
- `DQS < 60` — ❌ Low Quality

---

## Project Structure

```
teleop-flywheel-dashboard/
├── README.md
├── requirements.txt
├── data/
│   └── sessions.csv           # 1,000-row synthetic dataset (auto-generated)
├── src/
│   ├── data_generator.py      # Synthetic session data generator
│   ├── metrics.py             # CPHVH, DQS, leaderboard computation
│   └── app.py                 # Streamlit dashboard (3 tabs)
├── tests/
│   └── test_metrics.py        # 11 unit tests
└── docs/
    └── screenshots/
```

---

## Quick Start

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd teleop-flywheel-dashboard

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Pre-generate the synthetic dataset
python src/data_generator.py

# 5. Launch the dashboard
streamlit run src/app.py
```

Then open **http://localhost:8501** in your browser.

> The data generator runs automatically on first launch if `data/sessions.csv` is missing — no manual step required.

---

## Synthetic Data Design

| Parameter | Detail |
|---|---|
| **Operators** | 20 (`OP_001`–`OP_020`); top 6 are high-skill |
| **Tasks** | Bin Picking, Assembly, Cable Routing, Part Insertion, Tray Loading |
| **Latency** | Bimodal: 70% reliable WiFi (μ=60ms, σ=20ms), 30% congested (μ=180ms, σ=40ms) |
| **Duration** | Uniform 600–3,600 seconds |
| **Hourly rate** | Uniform $22–$35/hr, seeded per operator |
| **Sessions** | 1,000 rows, `seed=42` for reproducibility |

---

## Testing

```bash
pytest tests/ -v
```

```
test_perfect_session_is_high_value          PASSED
test_high_latency_not_high_value            PASSED
test_failed_session_not_high_value          PASSED
test_excessive_interventions_not_high_value PASSED
test_dqs_perfect_session                    PASSED
test_session_cost_calculation               PASSED
test_dqs_zero_for_worst_case                PASSED
test_quality_tier_high                      PASSED
test_quality_tier_low                       PASSED
test_compute_kpis_returns_expected_keys     PASSED
test_cphvh_infinity_when_no_hv_sessions     PASSED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
11 passed in 0.28s
```

---

## Deployment

### Streamlit Community Cloud (free, public demo)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New App**
3. Set **Main file:** `src/app.py`
4. Deploy — synthetic data auto-generates on first run; no database required

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python src/data_generator.py
EXPOSE 8501
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t teleop-dashboard .
docker run -p 8501:8501 teleop-dashboard
```

---

## Extension Roadmap

| Priority | Feature |
|---|---|
| P1 | Live CSV ingestion with file-watch polling |
| P1 | Per-operator DQS trend line by week |
| P2 | `st.toast` alerts when fleet DQS drops below threshold |
| P2 | Task-type deep dive with latency & intervention distributions |
| P3 | PostgreSQL backend replacing CSV |
| P3 | One-click PDF summary report for program lead weekly review |

---

## Who This Is For

| Persona | Key Question | Dashboard View |
|---|---|---|
| **Data Ops Manager** | Who are my top quality operators? | Executive KPIs + Leaderboard |
| **ML Research Engineer** | What fraction of today's sessions clear my latency threshold? | Bottleneck Tracker + DQS distribution |
| **Robotics Program Lead** | What's my blended cost per high-value hour this week? | Executive KPIs + trend |

---

*Built with Python · Streamlit · Pandas · Plotly — May 2026*
