import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from metrics import apply_metrics, compute_kpis


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
    assert bool(df['is_high_value'].iloc[0]) is True


def test_high_latency_not_high_value():
    df = apply_metrics(base_session(network_latency_ms=130))
    assert bool(df['is_high_value'].iloc[0]) is False


def test_failed_session_not_high_value():
    df = apply_metrics(base_session(success_flag=False))
    assert bool(df['is_high_value'].iloc[0]) is False


def test_excessive_interventions_not_high_value():
    df = apply_metrics(base_session(interventions_count=5))
    assert bool(df['is_high_value'].iloc[0]) is False


def test_dqs_perfect_session():
    df = apply_metrics(base_session())
    assert df['data_quality_score'].iloc[0] > 90


def test_session_cost_calculation():
    df = apply_metrics(base_session(session_duration_seconds=3600, operator_hourly_rate=25.0))
    assert abs(df['session_cost_usd'].iloc[0] - 25.0) < 0.01


def test_dqs_zero_for_worst_case():
    """A failed session with extreme latency and many interventions should score low."""
    df = apply_metrics(base_session(
        success_flag=False,
        network_latency_ms=300,
        interventions_count=10,
    ))
    assert df['data_quality_score'].iloc[0] == 0.0


def test_quality_tier_high():
    df = apply_metrics(base_session(network_latency_ms=60, interventions_count=0))
    assert str(df['quality_tier'].iloc[0]) == 'High'


def test_quality_tier_low():
    df = apply_metrics(base_session(success_flag=False, network_latency_ms=250, interventions_count=10))
    assert str(df['quality_tier'].iloc[0]) == 'Low'


def test_compute_kpis_returns_expected_keys():
    df = apply_metrics(base_session())
    kpis = compute_kpis(df)
    expected_keys = {
        'total_hv_hours', 'total_cost_usd', 'cost_per_hv_hour',
        'global_success_rate_pct', 'avg_data_quality_score',
        'total_sessions', 'high_value_sessions',
    }
    assert expected_keys == set(kpis.keys())


def test_cphvh_infinity_when_no_hv_sessions():
    """If no session is high-value, CPHVH should be inf."""
    df = apply_metrics(base_session(success_flag=False))
    kpis = compute_kpis(df)
    assert kpis['cost_per_hv_hour'] == float('inf')
