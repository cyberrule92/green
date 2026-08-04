"""Fine-tuning: dataset validation, sizing, carbon scheduling and the payback gate.

The invariant that matters most here is the same one ``model_onboarding`` pins:
a fine-tuned adapter is **not routable until measured**. Fine-tuning exists
precisely because quality is expected to change, so the direction and size of
that change is the one thing that must not be assumed.

Second in importance is the carbon accounting. A training run is the largest
single compute event this system performs, so its cost has to be metered rather
than estimated, and the deferral has to be justified by a real forecast rather
than asserted.

No network, no Docker, no GPU.
"""
from __future__ import annotations

import json

import pytest

import finetuning as ft


BASE = {
    "id": "local-vgpu-full",
    "model_id": "Qwen2.5-1.5B-Instruct",
    "vllm_model_id": "Qwen/Qwen2.5-1.5B-Instruct",
    "parameter_count_b": 1.5,
    "accuracy_baseline": 0.92,
    "latency_ms_p50": 225,
    "power_tdp_w": 225,
    "device_share": 0.255,
    "max_model_len": 2048,
}


def _resources(free_mb: float, *, measured: bool = True, total_mb: float = 21547.0):
    from model_onboarding import HostResources

    r = HostResources(
        gpu_name="test",
        vram_total_mb=total_mb,
        vram_used_mb=total_mb - free_mb,
        vram_free_mb=free_mb,
        vram_reserve_mb=1024.0,
        disk_free_gb=100.0,
        disk_total_gb=240.0,
        disk_reserve_gb=20.0,
    )
    if not measured:
        r.vram_basis = "unavailable"
    return r


def _records(n, *, rating=1, prefix="p"):
    return [{"prompt": f"{prefix}{i}", "response": f"answer {i}", "rating": rating} for i in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset validation
# ─────────────────────────────────────────────────────────────────────────────


def test_only_upvoted_pairs_become_training_signal():
    """A down-vote says what NOT to say, which supervised tuning cannot use."""
    rows, stats = ft.build_dataset(_records(3, rating=1) + _records(2, rating=-1, prefix="n"), source="feedback")
    assert len(rows) == 3
    assert stats.reasons["not_upvoted"] == 2


def test_duplicates_are_dropped():
    dup = [{"prompt": "same", "response": "same", "rating": 1}] * 4
    rows, stats = ft.build_dataset(dup, source="feedback")
    assert len(rows) == 1
    assert stats.reasons["duplicate"] == 3


def test_degradation_placeholder_is_never_trained_on():
    """Training on it would teach the model to apologise for being unavailable."""
    rows, stats = ft.build_dataset(
        [{"prompt": "hi", "response": "The AI inference backend is currently starting up or under high load.",
          "rating": 1}],
        source="feedback",
    )
    assert rows == []
    assert stats.reasons["degradation_placeholder"] == 1


def test_empty_sides_are_rejected():
    rows, stats = ft.build_dataset(
        [{"prompt": "", "response": "x", "rating": 1}, {"prompt": "y", "response": "  ", "rating": 1}],
        source="feedback",
    )
    assert rows == []
    assert stats.reasons["empty_prompt"] == 1 and stats.reasons["empty_response"] == 1


def test_dataset_below_the_floor_is_not_usable():
    _, stats = ft.build_dataset(_records(ft.MIN_TRAINING_SAMPLES - 1), source="feedback")
    assert stats.usable is False


def test_dataset_at_the_floor_is_usable():
    _, stats = ft.build_dataset(_records(ft.MIN_TRAINING_SAMPLES), source="feedback")
    assert stats.usable is True


def test_rejections_are_itemised_not_just_counted():
    """"We trained on 40 of your 900 rows" has to be visible before the GPU spins up."""
    _, stats = ft.build_dataset(
        _records(2) + _records(1, rating=-1, prefix="n") + [{"prompt": "", "response": "z", "rating": 1}],
        source="feedback",
    )
    assert stats.rejected == 2
    assert set(stats.reasons) == {"not_upvoted", "empty_prompt"}


def test_write_dataset_round_trips(tmp_path):
    rows = [{"prompt": "a", "response": "b"}, {"prompt": "c", "response": "d"}]
    path = ft.write_dataset(rows, tmp_path / "sub" / "train.jsonl")
    lines = [json.loads(l) for l in open(path, encoding="utf-8")]
    assert lines == rows


# ─────────────────────────────────────────────────────────────────────────────
# Sizing
# ─────────────────────────────────────────────────────────────────────────────


def test_qlora_needs_less_vram_than_lora():
    """4-bit base weights are what make a 1.5B trainable beside live inference."""
    q = ft.plan_finetune(base_entry=BASE, samples=500, resources=_resources(20000), method="qlora")
    l = ft.plan_finetune(base_entry=BASE, samples=500, resources=_resources(20000), method="lora")
    assert q.est_vram_mb < l.est_vram_mb


def test_plan_refuses_when_the_slice_is_full():
    """Training competes with live inference; it must not evict it by surprise."""
    plan = ft.plan_finetune(base_entry=BASE, samples=500, resources=_resources(1200))
    assert not plan.fits
    assert "competes with live inference" in plan.reason


def test_plan_refuses_when_vram_is_unmeasured():
    plan = ft.plan_finetune(base_entry=BASE, samples=500, resources=_resources(20000, measured=False))
    assert not plan.fits
    assert "could not be measured" in plan.reason


def test_plan_refuses_a_base_with_no_parameter_count():
    plan = ft.plan_finetune(base_entry={"id": "x", "vllm_model_id": "y"}, samples=500, resources=_resources(20000))
    assert not plan.fits
    assert "parameter count" in plan.reason


def test_plan_refuses_a_base_with_no_repo():
    entry = dict(BASE); entry.pop("vllm_model_id")
    plan = ft.plan_finetune(base_entry=entry, samples=500, resources=_resources(20000))
    assert not plan.fits
    assert "repo id" in plan.reason


def test_lora_rank_is_capped():
    plan = ft.plan_finetune(base_entry=BASE, samples=500, resources=_resources(20000), lora_rank=99999)
    assert plan.lora_rank == ft.MAX_LORA_RANK


def test_duration_is_labelled_an_estimate_not_a_measurement():
    """The metered value from the real run is what gets stored; this is a guide."""
    plan = ft.plan_finetune(base_entry=BASE, samples=500, resources=_resources(20000))
    assert plan.fits
    assert plan.duration_basis == "estimated_from_throughput_assumption"
    assert plan.est_duration_s > 0


def test_more_samples_cost_more_carbon():
    small = ft.plan_finetune(base_entry=BASE, samples=300, resources=_resources(20000), grid_ci=400)
    big = ft.plan_finetune(base_entry=BASE, samples=3000, resources=_resources(20000), grid_ci=400)
    assert big.est_carbon_g > small.est_carbon_g


# ─────────────────────────────────────────────────────────────────────────────
# Carbon-aware scheduling — the feature's actual sustainability mechanism
# ─────────────────────────────────────────────────────────────────────────────


def test_clean_grid_runs_immediately():
    out = ft.choose_window(None, duration_s=3600, current_ci=120, defer_above_ci=300)
    assert out["defer"] is False
    assert "at or below" in out["reason"]


def test_dirty_grid_defers_to_the_forecast_window():
    out = ft.choose_window(
        lambda duration_hours: {"start": "2026-07-30T02:00:00Z", "carbon_intensity": 180.0},
        duration_s=7200, current_ci=520, defer_above_ci=300,
    )
    assert out["defer"] is True
    assert out["start_ci"] == 180.0
    assert out["expected_saving_pct"] == pytest.approx(65.4, abs=0.2)


def test_deferral_without_a_forecast_is_refused():
    """Deferring with no window is an open-ended wait, not a carbon saving."""
    out = ft.choose_window(None, duration_s=3600, current_ci=520, defer_above_ci=300)
    assert out["defer"] is False
    assert "no forecast window" in out["reason"]


def test_a_broken_forecast_does_not_block_the_job():
    def boom(duration_hours):
        raise RuntimeError("forecast down")

    out = ft.choose_window(boom, duration_s=3600, current_ci=520, defer_above_ci=300)
    assert out["defer"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Service guards and the measurement gate
# ─────────────────────────────────────────────────────────────────────────────


class _StubZoo:
    def __init__(self, models):
        self._models = [dict(m) for m in models]

    def list_models(self):
        return [dict(m) for m in self._models]

    def get_model(self, mid):
        for m in self._models:
            if m.get("id") == mid:
                return dict(m)
        return None

    def register_model(self, entry):
        self._models = [m for m in self._models if m.get("id") != entry.get("id")]
        self._models.append(dict(entry))


@pytest.fixture
def svc(tmp_path, monkeypatch):
    monkeypatch.setenv("FINETUNE_ENABLED", "true")
    return ft.FineTuningService(
        _StubZoo([BASE]),
        feedback_records=lambda: _records(500),
        grid_ci=lambda: 250.0,
        state_path=tmp_path / "ft.json",
        work_dir=tmp_path / "work",
    )


def test_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("FINETUNE_ENABLED", raising=False)
    s = ft.FineTuningService(_StubZoo([BASE]), state_path=tmp_path / "f.json", work_dir=tmp_path)
    with pytest.raises(PermissionError):
        s.submit("local-vgpu-full")


def test_submit_refuses_without_a_trainer_image(svc, monkeypatch):
    """No image ships with a PEFT toolchain; say so rather than pretending."""
    monkeypatch.delenv("MODEL_FINETUNE_IMAGE", raising=False)
    with pytest.raises(RuntimeError) as exc:
        svc.submit("local-vgpu-full")
    assert "MODEL_FINETUNE_IMAGE" in str(exc.value) or "docker socket" in str(exc.value)


def test_registered_adapter_is_not_routable(svc):
    job = ft.FineTuneJob(job_id="j1", base_model_id="local-vgpu-full", adapter_id="ft-test")
    job.plan = {"lora_rank": 16, "method": "qlora"}
    job.dataset = {"samples": 400}
    job.carbon_g = 120.0
    job.duration_s = 5400.0
    job.adapter_path = "/w/adapter"
    entry = svc.register_adapter(job)
    assert entry["available"] is False
    assert entry["accuracy_basis"] == "unmeasured"
    assert entry["lora_adapter"] is True
    assert entry["base_model_id"] == "local-vgpu-full"
    assert entry["training_carbon_g"] == 120.0


def test_adapter_does_not_inherit_the_base_accuracy(svc):
    """The whole point is that quality changed; inheriting 0.92 would assert it did not."""
    job = ft.FineTuneJob(job_id="j2", base_model_id="local-vgpu-full", adapter_id="ft-test2")
    job.plan = {"lora_rank": 8, "method": "qlora"}
    entry = svc.register_adapter(job)
    assert entry["accuracy_baseline"] == 0.0
    assert BASE["accuracy_baseline"] == 0.92


def test_measuring_an_unserved_adapter_is_refused(svc):
    job = ft.FineTuneJob(job_id="j3", base_model_id="local-vgpu-full", adapter_id="ft-test3")
    job.plan = {"lora_rank": 16}
    svc.register_adapter(job)
    with pytest.raises(ValueError) as exc:
        svc.apply_measurement("ft-test3", {"accuracy": 0.8, "latency_ms_p50": 200}, basis="b")
    assert "no live endpoint" in str(exc.value)


def test_measurement_records_carbon_payback(svc):
    job = ft.FineTuneJob(job_id="j4", base_model_id="local-vgpu-full", adapter_id="ft-test4")
    job.plan = {"lora_rank": 16}
    job.carbon_g = 100.0
    entry = svc.register_adapter(job)
    entry["vllm_endpoint_url"] = "http://x:8050/v1"
    svc.zoo.register_model(entry)
    out = svc.apply_measurement(
        "ft-test4",
        {"accuracy": 0.85, "latency_ms_p50": 210,
         "carbon_g_per_request": 0.010, "incumbent_carbon_g_per_request": 0.015},
        basis="measured:eval-01",
    )
    assert out["available"] is True
    assert out["training_payback"]["pays_back"] is True
    assert out["training_payback"]["requests_to_payback"] == 20000


def test_a_finetune_that_is_not_cheaper_is_reported_as_never_paying_back(svc):
    job = ft.FineTuneJob(job_id="j5", base_model_id="local-vgpu-full", adapter_id="ft-test5")
    job.plan = {"lora_rank": 16}
    job.carbon_g = 100.0
    entry = svc.register_adapter(job)
    entry["vllm_endpoint_url"] = "http://x:8050/v1"
    svc.zoo.register_model(entry)
    out = svc.apply_measurement(
        "ft-test5",
        {"accuracy": 0.85, "latency_ms_p50": 210,
         "carbon_g_per_request": 0.020, "incumbent_carbon_g_per_request": 0.015},
        basis="measured:eval-02",
    )
    assert out["training_payback"]["pays_back"] is False


def test_measuring_a_non_adapter_is_refused(svc):
    with pytest.raises(ValueError) as exc:
        svc.apply_measurement("local-vgpu-full", {"accuracy": 0.8, "latency_ms_p50": 100}, basis="b")
    assert "not a fine-tuned adapter" in str(exc.value)


def test_interrupted_jobs_do_not_come_back_looking_alive(tmp_path):
    state = tmp_path / "ft.json"
    state.write_text(json.dumps({"jobs": [
        {"job_id": "z1", "base_model_id": "local-vgpu-full", "adapter_id": "ft-z", "state": "training"}
    ]}), encoding="utf-8")
    s = ft.FineTuningService(_StubZoo([BASE]), state_path=state, work_dir=tmp_path)
    job = s.get_job("z1")
    assert job.state == "failed"
    assert "interrupted" in job.error
