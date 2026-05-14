# TeleOp Flywheel Dashboard
## Product Requirements & Technical Execution Document

> **Project Type:** PM/Data Product · **Stack:** Python · Streamlit · Pandas · Plotly  
> **Audience:** Data Ops Manager, ML Researcher, Robotics Program Lead  
> **Strategic Purpose:** Attribute exact monetary value to human-in-the-loop robotic teleoperation data and surface operator quality at a per-session granularity.

---

## Table of Contents

1. [Product Context & Problem Statement](#1-product-context--problem-statement)
2. [User Personas](#2-user-personas)
3. [North Star Metrics — Definition & Business Logic](#3-north-star-metrics--definition--business-logic)
4. [Repository Structure](#4-repository-structure)
5. [Phase 1 — Synthetic Data Generator](#5-phase-1--synthetic-data-generator)
6. [Phase 2 — Core Metric Functions](#6-phase-2--core-metric-functions)
7. [Phase 3 — Streamlit Dashboard](#7-phase-3--streamlit-dashboard)
8. [Phase 4 — Deployment & Packaging](#8-phase-4--deployment--packaging)
9. [Acceptance Criteria & Testing](#9-acceptance-criteria--testing)
10. [Extension Roadmap](#10-extension-roadmap)

---

## 1. Product Context & Problem Statement

### Background

Humanoid and collaborative robot programs require massive volumes of high-quality teleoperation data to train imitation-learning and Vision-Language-Action (VLA) models. A single demonstration dataset of 10,000 episodes can cost $200K–$500K in operator labor. Today, most programs track **quantity** (episodes completed, hours logged) but have no system to measure **quality** at collection time.

### Core Problem

> **Without quality-aware cost attribution, a program can spend 60% of its teleoperation budget generating data that ML researchers immediately discard.**

Symptoms of this problem:
- Operators are evaluated on throughput (sessions/hour), incentivizing speed over quality.
- High-latency sessions (>150 ms network round-trip) produce corrupted kinematic trajectories that fail downstream action-cloning filters — but the operator is still paid in full.
- ML researchers spend 2–3 days per week manually auditing and filtering raw episode data before it can enter training pipelines.
- Program leads have no real-time signal that a specific task type, operator cohort, or network zone is underperforming.

### Solution

The **TeleOp Flywheel Dashboard** is an internal data product that:
1. Ingests per-session teleoperation logs (CSV or database query).
2. Computes **Cost per High-Value Hour** and a composite **Data Quality Score** per operator and session.
3. Surfaces a real-time leaderboard, bottleneck tracker, and executive summary for program leadership.

---

## 2. User Personas

### Persona A — Data Operations Manager ("Ops Lead")

| Attribute | Detail |
|---|---|
| **Goal** | Maximize high-quality episode output per dollar of operator spend |
| **Pain today** | Reviews operator timesheets weekly; has no quality signal until ML audit |
| **Key questions** | Who are my top 5 quality operators? Which task type is costing the most per usable hour? |
| **Dashboard view** | Executive KPI summary + Operator Leaderboard |
| **Decision this unlocks** | Shift operator scheduling toward high-performers; retrain or reassign low-quality operators |

### Persona B — ML Research Engineer ("Researcher")

| Attribute | Detail |
|---|---|
| **Goal** | Maximize clean training data entering the pipeline per sprint |
| **Pain today** | Manually inspects episode files; no upstream quality signal before download |
| **Key questions** | What fraction of today's sessions will clear my latency threshold? Which task type has the worst kinematics quality? |
| **Dashboard view** | Bottleneck Tracker scatter plot + Data Quality Score distribution |
| **Decision this unlocks** | Flag sessions for re-collection before ML audit; communicate infrastructure needs to platform engineering |

### Persona C — Robotics Program Lead ("Program Lead")

| Attribute | Detail |
|---|---|
| **Goal** | Deliver episode targets on time and within budget for model training milestones |
| **Key questions** | What is my blended cost per high-value hour this week? Am I on track for the training data milestone? |
| **Dashboard view** | Executive KPI summary + trend over time |
| **Decision this unlocks** | Budget reforecast; network infrastructure investment prioritization |

---

## 3. North Star Metrics — Definition & Business Logic

### Metric 1 — Cost per High-Value Hour (CPHVH)

**Definition:**
$$\text{CPHVH} = \frac{\text{Total Operator Cost (\$)}}{\text{Hours of High-Value Data Collected}}$$

**What counts as "High-Value"?**
A session is classified as **High-Value** if ALL of the following are true:
- `success_flag == True`
- `network_latency_ms < 120` (configurable threshold; above this, kinematics are considered degraded)
- `interventions_count <= 3` (configurable; more than 3 operator corrections signals task complexity or operator error)

**Why this matters:**
If an operator logs 8 hours but only 3 hours meet all three criteria, their effective cost per high-value hour is 2.67× their raw hourly rate. This metric makes the true cost of low-quality data legible to finance and program leadership.

**Python formula:**
```python
def high_value_flag(row, latency_threshold=120, intervention_threshold=3):
    return (
        row['success_flag'] and
        row['network_latency_ms'] < latency_threshold and
        row['interventions_count'] <= intervention_threshold
    )

def cost_per_high_value_hour(df):
    df['session_cost'] = (df['session_duration_seconds'] / 3600) * df['operator_hourly_rate']
    df['is_high_value'] = df.apply(high_value_flag, axis=1)
    total_cost = df['session_cost'].sum()
    hv_hours = df.loc[df['is_high_value'], 'session_duration_seconds'].sum() / 3600
    return total_cost / hv_hours if hv_hours > 0 else float('inf')
```

---

### Metric 2 — Data Quality Score (DQS)

**Definition:** A composite score from 0–100 per session, weighted across three sub-components.

| Component | Weight | Logic |
|---|---|---|
| **Network Quality** | 40% | Penalize latency above 80 ms. Score = max(0, 100 − ((latency − 80) × 1.5)) |
| **Task Completion** | 40% | Binary: `success_flag` = 100 pts, else 0 |
| **Operator Precision** | 20% | Score = max(0, 100 − (interventions_count × 10)) |

**DQS thresholds for leaderboard coloring:**
- `DQS >= 80`: ✅ High Quality (green)
- `60 <= DQS < 80`: ⚠️ Marginal (yellow)
- `DQS < 60`: ❌ Low Quality (red)

**Python formula:**
```python
def data_quality_score(row):
    latency_score = max(0, 100 - ((row['network_latency_ms'] - 80) * 1.5))
    completion_score = 100 if row['success_flag'] else 0
    precision_score = max(0, 100 - (row['interventions_count'] * 10))
    return round(
        (latency_score * 0.40) +
        (completion_score * 0.40) +
        (precision_score * 0.20),
        2
    )
```

---

## 4. Repository Structure

```
teleop-flywheel-dashboard/
├── README.md                  # This PRD document
├── requirements.txt
├── data/
│   └── sessions.csv           # Generated synthetic dataset (gitignored if large)
├── src/
│   ├── data_generator.py      # Phase 1: Synthetic session data generator
│   ├── metrics.py             # Phase 2: Core metric computation functions
│   └── app.py                 # Phase 3: Streamlit dashboard
├── tests/
│   └── test_metrics.py        # Unit tests for metric functions
└── docs/
    └── screenshots/           # Dashboard screenshots for README
```

---

## 5. Phase 1 — Synthetic Data Generator

**File:** `src/data_generator.py`

**Purpose:** Generate a realistic 1,000-row CSV representing teleoperation sessions across a simulated operator pool, task mix, and network condition distribution.

### Design Decisions

- **Operators:** 20 operators (`OP_001`–`OP_020`) with varying skill profiles. High-skill operators have lower intervention counts and higher success rates.
- **Task types:** `["Bin Picking", "Assembly", "Cable Routing", "Part Insertion", "Tray Loading"]` — chosen to reflect real dexterous manipulation task families.
- **Network latency:** Bimodal distribution simulating reliable WiFi (mean 60 ms, σ 20 ms) and congested conditions (mean 180 ms, σ 40 ms). 30% of sessions hit the congested distribution to reflect realistic factory floor conditions.
- **Session duration:** Uniform 600–3600 seconds (10 min to 1 hour).
- **Operator hourly rate:** Uniform $22–$35/hr, seeded per operator for consistency.

### Full Implementation

```python
# src/data_generator.py
import pandas as pd
import numpy as np
import os

def generate_teleop_sessions(n=1000, seed=42, output_path="data/sessions.csv"):
    rng = np.random.default_rng(seed)

    # --- Operator pool ---
    n_operators = 20
    operator_ids = [f"OP_{str(i).zfill(3)}" for i in range(1, n_operators + 1)]
    operator_hourly_rates = {op: round(rng.uniform(22, 35), 2) for op in operator_ids}
    # Skill tiers: top 6 are high-skill (lower interventions, higher success probability)
    high_skill_ops = operator_ids[:6]

    task_types = ["Bin Picking", "Assembly", "Cable Routing", "Part Insertion", "Tray Loading"]

    records = []
    for i in range(n):
        session_id = f"SES_{str(i+1).zfill(5)}"
        operator_id = rng.choice(operator_ids)
        is_high_skill = operator_id in high_skill_ops
        task_type = rng.choice(task_types)

        duration = int(rng.uniform(600, 3600))

        # Bimodal latency: 70% reliable, 30% congested
        if rng.random() < 0.70:
            latency = round(rng.normal(60, 20))
        else:
            latency = round(rng.normal(180, 40))
        latency = max(20, latency)  # floor at 20ms

        # Interventions: high-skill operators intervene less
        if is_high_skill:
            interventions = int(rng.poisson(1.5))
        else:
            interventions = int(rng.poisson(4.0))

        # Success: influenced by skill, latency, and task type
        base_success_prob = 0.85 if is_high_skill else 0.60
        if latency > 150:
            base_success_prob -= 0.30
        if task_type in ["Cable Routing", "Part Insertion"]:
            base_success_prob -= 0.10
        success = bool(rng.random() < max(0.05, base_success_prob))

        hourly_rate = operator_hourly_rates[operator_id]

        records.append({
            "session_id": session_id,
            "operator_id": operator_id,
            "task_type": task_type,
            "session_duration_seconds": duration,
            "network_latency_ms": latency,
            "interventions_count": interventions,
            "success_flag": success,
            "operator_hourly_rate": hourly_rate,
        })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} sessions → {output_path}")
    return df


if __name__ == "__main__":
    generate_teleop_sessions()
```

---

## 6. Phase 2 — Core Metric Functions

**File:** `src/metrics.py`

```python
# src/metrics.py
import pandas as pd
import numpy as np

# --- Configurable thresholds ---
LATENCY_THRESHOLD_MS = 120
INTERVENTION_THRESHOLD = 3
LATENCY_SCORE_FLOOR_MS = 80
LATENCY_PENALTY_RATE = 1.5

def apply_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add computed columns to session DataFrame."""
    df = df.copy()

    # Session cost ($)
    df['session_cost_usd'] = (
        df['session_duration_seconds'] / 3600
    ) * df['operator_hourly_rate']

    # High-value flag
    df['is_high_value'] = (
        df['success_flag'] &
        (df['network_latency_ms'] < LATENCY_THRESHOLD_MS) &
        (df['interventions_count'] <= INTERVENTION_THRESHOLD)
    )

    # Data Quality Score sub-components
    df['latency_score'] = df['network_latency_ms'].apply(
        lambda x: max(0, 100 - ((x - LATENCY_SCORE_FLOOR_MS) * LATENCY_PENALTY_RATE))
    )
    df['completion_score'] = df['success_flag'].apply(lambda x: 100 if x else 0)
    df['precision_score'] = df['interventions_count'].apply(
        lambda x: max(0, 100 - (x * 10))
    )
    df['data_quality_score'] = round(
        (df['latency_score'] * 0.40) +
        (df['completion_score'] * 0.40) +
        (df['precision_score'] * 0.20),
        2
    )

    # Quality tier label
    df['quality_tier'] = pd.cut(
        df['data_quality_score'],
        bins=[-1, 60, 80, 101],
        labels=['Low', 'Marginal', 'High']
    )

    return df


def compute_kpis(df: pd.DataFrame) -> dict:
    """Compute fleet-level KPI summary."""
    total_hv_hours = df.loc[df['is_high_value'], 'session_duration_seconds'].sum() / 3600
    total_cost = df['session_cost_usd'].sum()
    success_rate = df['success_flag'].mean() * 100
    avg_dqs = df['data_quality_score'].mean()
    cphvh = total_cost / total_hv_hours if total_hv_hours > 0 else float('inf')

    return {
        "total_hv_hours": round(total_hv_hours, 1),
        "total_cost_usd": round(total_cost, 2),
        "cost_per_hv_hour": round(cphvh, 2),
        "global_success_rate_pct": round(success_rate, 1),
        "avg_data_quality_score": round(avg_dqs, 1),
        "total_sessions": len(df),
        "high_value_sessions": int(df['is_high_value'].sum()),
    }


def operator_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate operator performance for leaderboard view."""
    grp = df.groupby('operator_id').agg(
        sessions=('session_id', 'count'),
        avg_dqs=('data_quality_score', 'mean'),
        hv_hours=('session_duration_seconds',
                  lambda x: x[df.loc[x.index, 'is_high_value']].sum() / 3600),
        total_cost=('session_cost_usd', 'sum'),
        success_rate=('success_flag', 'mean'),
        avg_latency=('network_latency_ms', 'mean'),
        avg_interventions=('interventions_count', 'mean'),
    ).reset_index()

    grp['cphvh'] = (grp['total_cost'] / grp['hv_hours']).replace([float('inf')], 9999)
    grp['avg_dqs'] = grp['avg_dqs'].round(1)
    grp['hv_hours'] = grp['hv_hours'].round(2)
    grp['success_rate'] = (grp['success_rate'] * 100).round(1)
    grp['avg_latency'] = grp['avg_latency'].round(1)
    grp['avg_interventions'] = grp['avg_interventions'].round(2)
    grp['cphvh'] = grp['cphvh'].round(2)

    return grp.sort_values('avg_dqs', ascending=False).reset_index(drop=True)
```

---

## 7. Phase 3 — Streamlit Dashboard

**File:** `src/app.py`

The dashboard is organized into three views rendered as Streamlit tabs.

### Setup: `requirements.txt`

```
streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.26.0
plotly>=5.20.0
```

### Full Implementation

```python
# src/app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from data_generator import generate_teleop_sessions
from metrics import apply_metrics, compute_kpis, operator_leaderboard

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TeleOp Flywheel Dashboard",
    page_icon="🤖",
    layout="wide",
)

# ── Data loading ─────────────────────────────────────────────────────────────
DATA_PATH = "data/sessions.csv"

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        generate_teleop_sessions(output_path=DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    return apply_metrics(df)

df_raw = load_data()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.title("🔧 Filters")
task_filter = st.sidebar.multiselect(
    "Task Type", options=sorted(df_raw['task_type'].unique()),
    default=list(df_raw['task_type'].unique())
)
latency_max = st.sidebar.slider(
    "Max Network Latency (ms)", min_value=20, max_value=300,
    value=300, step=10
)

df = df_raw[
    (df_raw['task_type'].isin(task_filter)) &
    (df_raw['network_latency_ms'] <= latency_max)
]

kpis = compute_kpis(df)
leaderboard = operator_leaderboard(df)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🤖 TeleOp Flywheel Dashboard")
st.caption("Real-time quality attribution for human-in-the-loop robotic data collection")

tab1, tab2, tab3 = st.tabs([
    "📊 Executive Summary",
    "🏆 Operator Leaderboard",
    "🔍 Bottleneck Tracker"
])

# ── TAB 1: Executive Summary ──────────────────────────────────────────────────
with tab1:
    st.subheader("Fleet-Level KPIs")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🕐 High-Value Hours", f"{kpis['total_hv_hours']:,.1f} hrs")
    col2.metric("💵 Total Operator Cost", f"${kpis['total_cost_usd']:,.0f}")
    col3.metric("⚡ Cost / HV Hour", f"${kpis['cost_per_hv_hour']:,.2f}")
    col4.metric("✅ Global Success Rate", f"{kpis['global_success_rate_pct']}%")
    col5.metric("⭐ Avg Quality Score", f"{kpis['avg_data_quality_score']}/100")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Data Quality Score Distribution")
        fig_hist = px.histogram(
            df, x='data_quality_score', nbins=30,
            color='quality_tier',
            color_discrete_map={"High": "#22c55e", "Marginal": "#f59e0b", "Low": "#ef4444"},
            labels={'data_quality_score': 'DQS', 'count': 'Sessions'},
            category_orders={"quality_tier": ["High", "Marginal", "Low"]}
        )
        fig_hist.update_layout(bargap=0.1, height=350)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        st.subheader("Cost per HV Hour by Task Type")
        task_stats = df.groupby('task_type').apply(
            lambda g: pd.Series({
                'cphvh': g['session_cost_usd'].sum() /
                         max(g.loc[g['is_high_value'], 'session_duration_seconds'].sum() / 3600, 0.01)
            })
        ).reset_index()
        fig_bar = px.bar(
            task_stats.sort_values('cphvh', ascending=True),
            x='cphvh', y='task_type', orientation='h',
            labels={'cphvh': 'Cost / HV Hour ($)', 'task_type': 'Task Type'},
            color='cphvh', color_continuous_scale='RdYlGn_r'
        )
        fig_bar.update_layout(height=350, coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Session Volume: High-Value vs. Total")
    hv_count = int(kpis['high_value_sessions'])
    total_count = int(kpis['total_sessions'])
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round((hv_count / total_count) * 100, 1),
        delta={'reference': 70, 'suffix': '%'},
        title={'text': "% Sessions High-Value (target: 70%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#22c55e"},
            'steps': [
                {'range': [0, 50], 'color': "#fee2e2"},
                {'range': [50, 70], 'color': "#fef3c7"},
                {'range': [70, 100], 'color': "#dcfce7"},
            ],
            'threshold': {'line': {'color': "black", 'width': 3}, 'value': 70}
        }
    ))
    fig_gauge.update_layout(height=280)
    st.plotly_chart(fig_gauge, use_container_width=True)


# ── TAB 2: Operator Leaderboard ───────────────────────────────────────────────
with tab2:
    st.subheader("Operator Quality Leaderboard")
    st.caption("Ranked by Data Quality Score (DQS). Green = High Quality, Yellow = Marginal, Red = Low Quality.")

    def color_dqs(val):
        if val >= 80:
            return 'background-color: #dcfce7; color: #166534'
        elif val >= 60:
            return 'background-color: #fef9c3; color: #854d0e'
        else:
            return 'background-color: #fee2e2; color: #991b1b'

    display_cols = {
        'operator_id': 'Operator',
        'sessions': 'Sessions',
        'avg_dqs': 'Avg DQS',
        'hv_hours': 'HV Hours',
        'cphvh': 'Cost/HV Hr ($)',
        'success_rate': 'Success %',
        'avg_latency': 'Avg Latency (ms)',
        'avg_interventions': 'Avg Interventions',
    }

    lb_display = leaderboard[list(display_cols.keys())].rename(columns=display_cols)
    lb_display.insert(0, 'Rank', range(1, len(lb_display) + 1))

    styled = lb_display.style.applymap(color_dqs, subset=['Avg DQS'])
    st.dataframe(styled, use_container_width=True, height=500)

    st.divider()
    st.subheader("Top 10 Operators — HV Hours vs. DQS")
    top10 = leaderboard.head(10)
    fig_bubble = px.scatter(
        top10, x='hv_hours', y='avg_dqs',
        size='sessions', color='cphvh',
        hover_name='operator_id',
        color_continuous_scale='RdYlGn_r',
        labels={
            'hv_hours': 'High-Value Hours',
            'avg_dqs': 'Avg Data Quality Score',
            'cphvh': 'Cost/HV Hour ($)',
            'sessions': 'Sessions'
        },
        title="Bubble size = # sessions | Color = Cost/HV Hour (green = cheaper)"
    )
    fig_bubble.update_layout(height=400)
    st.plotly_chart(fig_bubble, use_container_width=True)


# ── TAB 3: Bottleneck Tracker ─────────────────────────────────────────────────
with tab3:
    st.subheader("Network Latency vs. Session Outcome")
    st.caption(
        "Each point is one teleoperation session. "
        "High-latency sessions cluster strongly in failure outcomes — "
        "this chart is the infrastructure investment case."
    )

    fig_scatter = px.scatter(
        df,
        x='network_latency_ms',
        y='data_quality_score',
        color='success_flag',
        symbol='task_type',
        size='interventions_count',
        hover_data=['session_id', 'operator_id', 'session_cost_usd'],
        color_discrete_map={True: '#22c55e', False: '#ef4444'},
        labels={
            'network_latency_ms': 'Network Latency (ms)',
            'data_quality_score': 'Data Quality Score',
            'success_flag': 'Session Succeeded',
        },
        title="Latency vs. Data Quality Score — colored by success outcome"
    )
    # Threshold reference line
    fig_scatter.add_vline(
        x=120, line_dash="dash", line_color="orange",
        annotation_text="Latency Threshold (120ms)",
        annotation_position="top right"
    )
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()
    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Failure Rate by Latency Bucket")
        df['latency_bucket'] = pd.cut(
            df['network_latency_ms'],
            bins=[0, 60, 90, 120, 160, 300],
            labels=['<60ms', '60–90ms', '90–120ms', '120–160ms', '>160ms']
        )
        fail_by_bucket = df.groupby('latency_bucket').agg(
            failure_rate=('success_flag', lambda x: (1 - x.mean()) * 100),
            session_count=('session_id', 'count')
        ).reset_index()
        fig_fail = px.bar(
            fail_by_bucket,
            x='latency_bucket', y='failure_rate',
            color='failure_rate',
            color_continuous_scale='RdYlGn_r',
            labels={'latency_bucket': 'Latency Bucket', 'failure_rate': 'Failure Rate (%)'},
        )
        fig_fail.update_layout(height=320, coloraxis_showscale=False)
        st.plotly_chart(fig_fail, use_container_width=True)

    with col_d:
        st.subheader("Revenue Lost to High Latency")
        high_latency_cost = df.loc[
            df['network_latency_ms'] >= 120, 'session_cost_usd'
        ].sum()
        total_cost_all = df['session_cost_usd'].sum()
        st.metric(
            "Cost of High-Latency Sessions",
            f"${high_latency_cost:,.0f}",
            delta=f"{round((high_latency_cost / total_cost_all) * 100, 1)}% of total spend",
            delta_color="inverse"
        )
        st.caption(
            "Sessions with latency ≥ 120ms represent operator hours paid "
            "that yield data ML researchers cannot use for training."
        )
        st.metric(
            "Recoverable HV Hours if Latency Fixed",
            f"{df.loc[df['network_latency_ms'] >= 120, 'session_duration_seconds'].sum() / 3600:,.1f} hrs"
        )
```

### Running the App

```bash
# Install dependencies
pip install -r requirements.txt

# Generate data (auto-runs if missing, but can also run manually)
python src/data_generator.py

# Launch dashboard
streamlit run src/app.py
```

---

## 8. Phase 4 — Deployment & Packaging

### Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run src/app.py
```

### Streamlit Community Cloud (public portfolio demo)

1. Push repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → New App → select repo → set **Main file:** `src/app.py`.
3. Add `requirements.txt` at repo root.
4. Deploy. The app auto-generates synthetic data on first run; no database required.

### Docker (optional, for internal deployment)

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

---

## 9. Acceptance Criteria & Testing

### Unit Tests — `tests/test_metrics.py`

```python
import pandas as pd
import pytest
from src.metrics import apply_metrics, compute_kpis

def base_session(**overrides):
    defaults = {
        "session_id": "SES_00001",
        "operator_id": "OP_001",
        "task_type": "Assembly",
        "session_duration_seconds": 3600,
        "network_latency_ms": 60,
        "interventions_count": 1,
        "success_flag": True,
        "operator_hourly_rate": 25.0,
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])

def test_perfect_session_is_high_value():
    df = apply_metrics(base_session())
    assert df['is_high_value'].iloc[0] == True

def test_high_latency_not_high_value():
    df = apply_metrics(base_session(network_latency_ms=130))
    assert df['is_high_value'].iloc[0] == False

def test_failed_session_not_high_value():
    df = apply_metrics(base_session(success_flag=False))
    assert df['is_high_value'].iloc[0] == False

def test_excessive_interventions_not_high_value():
    df = apply_metrics(base_session(interventions_count=5))
    assert df['is_high_value'].iloc[0] == False

def test_dqs_perfect_session():
    df = apply_metrics(base_session())
    assert df['data_quality_score'].iloc[0] > 90

def test_session_cost_calculation():
    df = apply_metrics(base_session(session_duration_seconds=3600, operator_hourly_rate=25.0))
    assert abs(df['session_cost_usd'].iloc[0] - 25.0) < 0.01
```

Run with: `pytest tests/ -v`

### Dashboard Acceptance Checklist

- [ ] Executive Summary tab loads with 5 KPI metrics visible
- [ ] Gauge chart reflects correct % high-value sessions
- [ ] Task Type bar chart renders without division-by-zero errors
- [ ] Leaderboard tab: operators sorted by DQS descending with correct color coding
- [ ] Bottleneck Tracker: vertical threshold line at 120ms renders on scatter plot
- [ ] Sidebar filters propagate to all three views
- [ ] All plots render correctly with filtered data subsets

---

## 10. Extension Roadmap

| Priority | Feature | Description |
|---|---|---|
| P1 | Live CSV ingestion | Replace static file with polling/file-watch on operator upload folder |
| P1 | Per-operator trend over time | Line chart of DQS trend by week for each operator |
| P2 | Alert thresholds | Streamlit `st.toast` alerts when fleet DQS drops below configurable threshold |
| P2 | Task-type deep dive | Drilldown page showing per-task latency, intervention distributions |
| P3 | PostgreSQL backend | Replace CSV with live query against teleoperation log database |
| P3 | Export to PDF | One-click PDF summary report for program lead weekly review |
| P3 | Operator benchmarking | Percentile rank vs. operator cohort, with historical trend |

---

*Document version 1.0 — May 2026*
