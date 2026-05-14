import pandas as pd
import numpy as np
import os


def generate_teleop_sessions(n=1000, seed=42, output_path="data/sessions.csv"):
    rng = np.random.default_rng(seed)

    n_operators = 20
    operator_ids = [f"OP_{str(i).zfill(3)}" for i in range(1, n_operators + 1)]
    operator_hourly_rates = {op: round(rng.uniform(22, 35), 2) for op in operator_ids}
    # Top 6 operators are high-skill: lower interventions, higher success probability
    high_skill_ops = operator_ids[:6]

    task_types = ["Bin Picking", "Assembly", "Cable Routing", "Part Insertion", "Tray Loading"]

    records = []
    for i in range(n):
        session_id = f"SES_{str(i + 1).zfill(5)}"
        operator_id = rng.choice(operator_ids)
        is_high_skill = operator_id in high_skill_ops
        task_type = rng.choice(task_types)

        duration = int(rng.uniform(600, 3600))

        # Bimodal latency: 70% reliable WiFi, 30% congested factory floor
        if rng.random() < 0.70:
            latency = round(rng.normal(60, 20))
        else:
            latency = round(rng.normal(180, 40))
        latency = max(20, latency)

        if is_high_skill:
            interventions = int(rng.poisson(1.5))
        else:
            interventions = int(rng.poisson(4.0))

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
