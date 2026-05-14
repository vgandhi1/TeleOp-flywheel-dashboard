import pandas as pd
import numpy as np

# Configurable thresholds
LATENCY_THRESHOLD_MS = 120
INTERVENTION_THRESHOLD = 3
LATENCY_SCORE_FLOOR_MS = 80
LATENCY_PENALTY_RATE = 1.5


def apply_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add computed cost and quality columns to a session DataFrame."""
    df = df.copy()

    df['session_cost_usd'] = (
        df['session_duration_seconds'] / 3600
    ) * df['operator_hourly_rate']

    df['is_high_value'] = (
        df['success_flag'] &
        (df['network_latency_ms'] < LATENCY_THRESHOLD_MS) &
        (df['interventions_count'] <= INTERVENTION_THRESHOLD)
    )

    df['latency_score'] = df['network_latency_ms'].apply(
        lambda x: min(100, max(0, 100 - ((x - LATENCY_SCORE_FLOOR_MS) * LATENCY_PENALTY_RATE)))
    )
    df['completion_score'] = df['success_flag'].apply(lambda x: 100 if x else 0)
    df['precision_score'] = df['interventions_count'].apply(
        lambda x: max(0, 100 - (x * 10))
    )
    df['data_quality_score'] = round(
        (df['latency_score'] * 0.40) +
        (df['completion_score'] * 0.40) +
        (df['precision_score'] * 0.20),
        2,
    )

    df['quality_tier'] = pd.cut(
        df['data_quality_score'],
        bins=[-1, 60, 80, 101],
        labels=['Low', 'Marginal', 'High'],
    )

    return df


def compute_kpis(df: pd.DataFrame) -> dict:
    """Compute fleet-level KPI summary from an enriched session DataFrame."""
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
    """Aggregate per-operator performance metrics for the leaderboard view."""
    hv_mask = df['is_high_value']
    grp = df.groupby('operator_id').agg(
        sessions=('session_id', 'count'),
        avg_dqs=('data_quality_score', 'mean'),
        total_cost=('session_cost_usd', 'sum'),
        success_rate=('success_flag', 'mean'),
        avg_latency=('network_latency_ms', 'mean'),
        avg_interventions=('interventions_count', 'mean'),
    ).reset_index()

    hv_hours_per_op = (
        df[hv_mask]
        .groupby('operator_id')['session_duration_seconds']
        .sum()
        .div(3600)
        .rename('hv_hours')
    )
    grp = grp.merge(hv_hours_per_op, on='operator_id', how='left')
    grp['hv_hours'] = grp['hv_hours'].fillna(0.0)

    grp['cphvh'] = (grp['total_cost'] / grp['hv_hours']).replace([float('inf')], 9999)
    grp['avg_dqs'] = grp['avg_dqs'].round(1)
    grp['hv_hours'] = grp['hv_hours'].round(2)
    grp['success_rate'] = (grp['success_rate'] * 100).round(1)
    grp['avg_latency'] = grp['avg_latency'].round(1)
    grp['avg_interventions'] = grp['avg_interventions'].round(2)
    grp['cphvh'] = grp['cphvh'].round(2)

    return grp.sort_values('avg_dqs', ascending=False).reset_index(drop=True)
