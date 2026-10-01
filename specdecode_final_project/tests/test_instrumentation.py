import csv

from specdecode.instrumentation import save_steps_csv, snapshot_runtime, step_to_dict
from specdecode.speculative import StepResult


def make_step():
    return StepResult(
        step_index=0,
        k_requested=4,
        num_accepted=3,
        new_token_ids=[7, 8, 9, 10],
        draft_time_s=0.012,
        verify_time_s=0.008,
        step_time_s=0.025,
        policy_time_s=0.0002,
        device_type="cpu",
        draft_entropy_mean=1.2,
        draft_entropy_max=1.8,
        draft_entropy_last=1.4,
    )


def test_cpu_snapshot_has_no_gpu_metrics():
    s = snapshot_runtime("cpu")
    assert s.device_type == "cpu"
    assert s.gpu_utilization_pct is None
    assert s.gpu_mem_allocated_mb is None


def test_step_to_dict_contains_detailed_experiment_fields():
    row = step_to_dict(make_step())
    required = {
        "k_requested", "num_accepted", "num_rejected", "acceptance_rate",
        "draft_time_s", "verify_time_s", "step_time_s", "policy_time_s",
        "draft_entropy_mean", "draft_entropy_max", "draft_entropy_last",
        "draft_ms_per_token", "verify_ms_per_token", "draft_verify_ratio",
    }
    assert required.issubset(row)
    assert row["acceptance_rate"] == 0.75


def test_save_steps_csv_roundtrip(tmp_path):
    path = tmp_path / "steps.csv"
    save_steps_csv([make_step()], str(path))
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["num_accepted"] == "3"
    assert rows[0]["k_requested"] == "4"
