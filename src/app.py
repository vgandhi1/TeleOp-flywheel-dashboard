import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from data_generator import generate_teleop_sessions
from metrics import apply_metrics, compute_kpis, operator_leaderboard

st.set_page_config(
    page_title="TeleOp Flywheel Dashboard",
    page_icon="🤖",
    layout="wide",
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sessions.csv")


@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        generate_teleop_sessions(output_path=DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    return apply_metrics(df)


df_raw = load_data()

# ── Sidebar filters ────────────────────────────────────────────────────────────
st.sidebar.title("🔧 Filters")
task_filter = st.sidebar.multiselect(
    "Task Type",
    options=sorted(df_raw['task_type'].unique()),
    default=list(df_raw['task_type'].unique()),
)
latency_max = st.sidebar.slider(
    "Max Network Latency (ms)",
    min_value=20,
    max_value=300,
    value=300,
    step=10,
)

df = df_raw[
    (df_raw['task_type'].isin(task_filter)) &
    (df_raw['network_latency_ms'] <= latency_max)
]

kpis = compute_kpis(df)
leaderboard = operator_leaderboard(df)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🤖 TeleOp Flywheel Dashboard")
st.caption("Real-time quality attribution for human-in-the-loop robotic data collection")

tab1, tab2, tab3 = st.tabs([
    "📊 Executive Summary",
    "🏆 Operator Leaderboard",
    "🔍 Bottleneck Tracker",
])

# ── TAB 1: Executive Summary ───────────────────────────────────────────────────
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
            df,
            x='data_quality_score',
            nbins=30,
            color='quality_tier',
            color_discrete_map={"High": "#22c55e", "Marginal": "#f59e0b", "Low": "#ef4444"},
            labels={'data_quality_score': 'DQS', 'count': 'Sessions'},
            category_orders={"quality_tier": ["High", "Marginal", "Low"]},
        )
        fig_hist.update_layout(bargap=0.1, height=350)
        st.plotly_chart(fig_hist, width='stretch')

    with col_b:
        st.subheader("Cost per HV Hour by Task Type")
        task_stats = df.groupby('task_type').apply(
            lambda g: pd.Series({
                'cphvh': g['session_cost_usd'].sum() /
                         max(g.loc[g['is_high_value'], 'session_duration_seconds'].sum() / 3600, 0.01)
            }),
            include_groups=False,
        ).reset_index()
        fig_bar = px.bar(
            task_stats.sort_values('cphvh', ascending=True),
            x='cphvh',
            y='task_type',
            orientation='h',
            labels={'cphvh': 'Cost / HV Hour ($)', 'task_type': 'Task Type'},
            color='cphvh',
            color_continuous_scale='RdYlGn_r',
        )
        fig_bar.update_layout(height=350, coloraxis_showscale=False)
        st.plotly_chart(fig_bar, width='stretch')

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
            'threshold': {'line': {'color': "black", 'width': 3}, 'value': 70},
        },
    ))
    fig_gauge.update_layout(height=280)
    st.plotly_chart(fig_gauge, width='stretch')


# ── TAB 2: Operator Leaderboard ────────────────────────────────────────────────
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

    styled = lb_display.style.map(color_dqs, subset=['Avg DQS'])
    st.dataframe(styled, width='stretch', height=500)

    st.divider()
    st.subheader("Top 10 Operators — HV Hours vs. DQS")
    top10 = leaderboard.head(10)
    fig_bubble = px.scatter(
        top10,
        x='hv_hours',
        y='avg_dqs',
        size='sessions',
        color='cphvh',
        hover_name='operator_id',
        color_continuous_scale='RdYlGn_r',
        labels={
            'hv_hours': 'High-Value Hours',
            'avg_dqs': 'Avg Data Quality Score',
            'cphvh': 'Cost/HV Hour ($)',
            'sessions': 'Sessions',
        },
        title="Bubble size = # sessions | Color = Cost/HV Hour (green = cheaper)",
    )
    fig_bubble.update_layout(height=400)
    st.plotly_chart(fig_bubble, width='stretch')


# ── TAB 3: Bottleneck Tracker ──────────────────────────────────────────────────
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
        title="Latency vs. Data Quality Score — colored by success outcome",
    )
    fig_scatter.add_vline(
        x=120,
        line_dash="dash",
        line_color="orange",
        annotation_text="Latency Threshold (120ms)",
        annotation_position="top right",
    )
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, width='stretch')

    st.divider()
    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Failure Rate by Latency Bucket")
        df_bt = df.copy()
        df_bt['latency_bucket'] = pd.cut(
            df_bt['network_latency_ms'],
            bins=[0, 60, 90, 120, 160, 300],
            labels=['<60ms', '60–90ms', '90–120ms', '120–160ms', '>160ms'],
        )
        fail_by_bucket = df_bt.groupby('latency_bucket', observed=True).agg(
            failure_rate=('success_flag', lambda x: (1 - x.mean()) * 100),
            session_count=('session_id', 'count'),
        ).reset_index()
        fig_fail = px.bar(
            fail_by_bucket,
            x='latency_bucket',
            y='failure_rate',
            color='failure_rate',
            color_continuous_scale='RdYlGn_r',
            labels={'latency_bucket': 'Latency Bucket', 'failure_rate': 'Failure Rate (%)'},
        )
        fig_fail.update_layout(height=320, coloraxis_showscale=False)
        st.plotly_chart(fig_fail, width='stretch')

    with col_d:
        st.subheader("Revenue Lost to High Latency")
        high_latency_cost = df.loc[df['network_latency_ms'] >= 120, 'session_cost_usd'].sum()
        total_cost_all = df['session_cost_usd'].sum()
        st.metric(
            "Cost of High-Latency Sessions",
            f"${high_latency_cost:,.0f}",
            delta=f"{round((high_latency_cost / total_cost_all) * 100, 1)}% of total spend",
            delta_color="inverse",
        )
        st.caption(
            "Sessions with latency ≥ 120ms represent operator hours paid "
            "that yield data ML researchers cannot use for training."
        )
        st.metric(
            "Recoverable HV Hours if Latency Fixed",
            f"{df.loc[df['network_latency_ms'] >= 120, 'session_duration_seconds'].sum() / 3600:,.1f} hrs",
        )
