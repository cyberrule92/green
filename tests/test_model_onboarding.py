"""Onboarding: sizing arithmetic, plan preference order, and the measurement gate.

The gate is the part that matters. CSS ranks on ``accuracy_baseline`` and
``latency_ms_p50``, and the shipped zoo declares ``local-vgpu-full`` at 0.92
while it measured 0.793 in practice. An onboarding pipeline that
let a downloaded model declare its own accuracy would put that same kind of
fiction into the router automatically, so the tests below pin the invariant that
a newly registered model is **not routable** until measured figures arrive.

No network and no Docker: the HF catalog and the container runtime are the two
things this module talks to, and both are passed in or stubbed.
"""
from __future__ import annotations

import json

import pytest

import model_onboarding as mo


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

# Qwen2.5-1.5B-Instruct's real shape — GQA with 2 KV heads against 12 query
# heads, which is exactly the case a naive estimate gets 6x wrong.
QWEN15_ARCH = {
    "num_hidden_layers": 28,
    "hidden_size": 1536,
    "num_attention_heads": 12,
    "num_key_value_heads": 2,
    "max_position_embeddings": 32768,
}


def _resources(vram_free_mb: float, *, total_mb: float = 21547.0, disk_free_gb: float = 80.0) -> mo.HostResources:
    return mo.HostResources(
        gpu_name="test",
        vram_total_mb=total_mb,
        vram_used_mb=total_mb - vram_free_mb,
        vram_free_mb=vram_free_mb,
        vram_reserve_mb=1024.0,
        disk_free_gb=disk_free_gb,
        disk_total_gb=240.0,
        disk_reserve_gb=20.0,
    )


def _summary(repo_id: str = "Qwen/Qwen2.5-1.5B-Instruct", params_b: float | None = 1.5, **extra):
    out = {
        "repo_id": repo_id,
        "parameter_count_b": params_b,
        "config": {},
        "quant_format": None,
    }
    out.update(extra)
    return out


DONOR = {
    "id": "local-vgpu-full",
    "hardware": "vgpu",
    "hardware_class": "gpu",
    "region": "local",
    "region_label": "On-Premises vGPU",
    "power_tdp_w": 225,
    "hardware_efficiency": 0.75,
    "pue": 1.3,
    "mfg_carbon_kg": 143,
    "device_lifetime_years": 5,
    "device_utilization": 0.35,
    "device_share": 0.255,
    "annual_inference_volume": 100000,
    "region_carbon_multiplier": 1.0,
    "grid_zone": "local",
    "cost_units": 0.6,
    "parameter_count_b": 1.5,
}


# ─────────────────────────────────────────────────────────────────────────────
# Sizing arithmetic
# ─────────────────────────────────────────────────────────────────────────────


def test_kv_cache_honours_grouped_query_attention():
    """GQA models must be sized on KV heads, not attention heads.

    2 x 28 layers x 2 kv_heads x 128 head_dim x 2 bytes = 28,672 B/token.
    At 2048 tokens x 4 sequences that is 224 MiB exactly.
    """
    kv = mo.estimate_kv_cache_mb(QWEN15_ARCH, max_model_len=2048, concurrency=4)
    assert kv == pytest.approx(224.0, rel=1e-6)


def test_kv_cache_would_be_six_times_larger_without_gqa():
    no_gqa = dict(QWEN15_ARCH)
    no_gqa["num_key_value_heads"] = no_gqa["num_attention_heads"]
    naive = mo.estimate_kv_cache_mb(no_gqa, 2048, 4)
    correct = mo.estimate_kv_cache_mb(QWEN15_ARCH, 2048, 4)
    assert naive / correct == pytest.approx(6.0)


def test_kv_cache_returns_zero_on_unusable_config():
    """A config we cannot read must not silently size as if the cache were free."""
    assert mo.estimate_kv_cache_mb({}, 2048, 4) == 0.0
    assert mo.estimate_kv_cache_mb({"num_hidden_layers": "?"}, 2048, 4) == 0.0


def test_four_bit_weights_are_roughly_a_quarter_of_fp16():
    _, fp16_w, _ = mo.estimate_vram_mb(7.0, "fp16", QWEN15_ARCH, 2048)
    _, awq_w, _ = mo.estimate_vram_mb(7.0, "awq", QWEN15_ARCH, 2048)
    assert fp16_w / awq_w == pytest.approx(2.0 / 0.56, rel=1e-6)
    assert 3.4 < fp16_w / awq_w < 3.7


# ─────────────────────────────────────────────────────────────────────────────
# MIG-aware probing
# ─────────────────────────────────────────────────────────────────────────────

# Real nvidia-smi output from the deployment host (driver 570.172.08). The board
# reports 24576 MiB; the MIG instance a container can use is 21547 MiB.
MIG_SMI_TEXT = """
|   0  NVIDIA H100L-2-24C             On  |   00000000:03:00.0 Off |                   On |
| N/A   N/A    P0            N/A  /  N/A  |   14913MiB /  24576MiB |     N/A      Default |

+-----------------------------------------------------------------------------------------+
| MIG devices:                                                                            |
+------------------+----------------------------------+-----------+-----------------------+
|  0    0   0   0  |           14913MiB / 21547MiB    | 32      0 |  2   0    2    0    2 |
|                  |                12MiB /  4096MiB  |           |                       |
+------------------+----------------------------------+-----------+-----------------------+
"""


def test_mig_instance_capacity_is_preferred_over_board_capacity(monkeypatch):
    """Sizing against the board over-promises by ~3 GB on this host.

    Over-promising VRAM is how every engine on a shared slice wedges at once, so
    the MIG instance figure has to win.
    """
    monkeypatch.setattr(mo, "_nvidia_smi", lambda args, timeout=10.0: MIG_SMI_TEXT)
    parsed = mo._probe_mig_memory()
    assert parsed == (21547.0, 14913.0)


def test_mig_parse_takes_memory_row_not_bar1_row(monkeypatch):
    """The BAR1 row matches the same "NMiB / NMiB" shape and must not win."""
    monkeypatch.setattr(mo, "_nvidia_smi", lambda args, timeout=10.0: MIG_SMI_TEXT)
    total, _ = mo._probe_mig_memory()
    assert total != 4096.0


def test_unparsable_mig_output_degrades_with_a_visible_basis(monkeypatch):
    """Falling back to the board figure is allowed; hiding that it happened is not."""
    monkeypatch.setattr(
        mo,
        "_nvidia_smi",
        lambda args, timeout=10.0: (
            "NVIDIA H100L-2-24C, 24576, 14913, 6660, Enabled" if args else "no MIG table here"
        ),
    )
    name, total, used, free, basis, mig_mode = mo.probe_gpu()
    assert total == 24576.0
    assert basis == "gpu_board_mig_unparsed"
    assert mig_mode == "Enabled"
    # The board total minus used would say 9663; memory.free is MIG-aware and
    # says 6660. Admission depends on free, so the reported value must survive.
    assert free == 6660.0


def test_probe_is_safe_without_a_gpu(monkeypatch):
    monkeypatch.setattr(mo, "_nvidia_smi", lambda args, timeout=10.0: None)
    name, total, used, free, basis, _ = mo.probe_gpu()
    assert (name, total, used, free, basis) == (None, 0.0, 0.0, 0.0, "unavailable")


# ─────────────────────────────────────────────────────────────────────────────
# Quantization format detection
# ─────────────────────────────────────────────────────────────────────────────


def test_declared_quant_method_beats_the_repo_name():
    fmt = mo.detect_quant_format(
        "someone/Model-AWQ", {"quantization_config": {"quant_method": "gptq"}}
    )
    assert fmt == "gptq"


def test_ambiguous_name_tokens_are_not_resolved_to_a_method():
    """"int4" says the bit width, not the method, and AWQ/GPTQ are not swappable."""
    assert mo.detect_quant_format("someone/Model-int4") == "unspecified-int4"
    assert mo.detect_quant_format("someone/Model-8bit") == "unspecified-8bit"


def test_unquantized_repo_reports_no_format():
    assert mo.detect_quant_format("Qwen/Qwen2.5-1.5B-Instruct") is None


def test_parameter_count_falls_back_to_the_repo_name():
    assert mo.infer_parameter_count_b(_summary("org/Foo-7B", params_b=None)) == 7.0
    assert mo.infer_parameter_count_b(_summary("org/Foo", params_b=None)) is None


# ─────────────────────────────────────────────────────────────────────────────
# Plan preference order
# ─────────────────────────────────────────────────────────────────────────────


def test_prequantized_sibling_wins_because_its_carbon_was_already_paid():
    plan = mo.plan_quantization(
        summary=_summary("org/Big-7B", params_b=7.0),
        arch=QWEN15_ARCH,
        resources=_resources(8000.0),
        prequantized=[{"repo_id": "community/Big-7B-AWQ", "quant_format": "awq"}],
    )
    assert plan.fits
    assert plan.source == "prequantized"
    assert plan.serve_repo_id == "community/Big-7B-AWQ"
    assert "already paid" in plan.reason


def test_fp16_preferred_for_a_small_model_that_fits_with_slack():
    """No quality risk is worth taking for a marginal VRAM saving."""
    plan = mo.plan_quantization(
        summary=_summary(params_b=1.5),
        arch=QWEN15_ARCH,
        resources=_resources(20000.0),
    )
    assert plan.fits
    assert (plan.quant_format, plan.source) == ("fp16", "native")
    assert "--quantization" not in " ".join(plan.vllm_args)


def test_inflight_bitsandbytes_when_nothing_upstream_fits():
    """Load-time 4-bit costs throughput, not carbon — the right fallback."""
    plan = mo.plan_quantization(
        summary=_summary("org/Big-7B", params_b=7.0),
        arch=QWEN15_ARCH,
        resources=_resources(8000.0),
    )
    assert plan.fits
    assert (plan.quant_format, plan.source) == ("bitsandbytes", "inflight")
    assert "--quantization=bitsandbytes" in plan.vllm_args


def test_local_quantization_is_never_chosen_implicitly():
    """It is the most carbon-expensive operation here; it needs an explicit ask."""
    plan = mo.plan_quantization(
        summary=_summary("org/Big-7B", params_b=7.0),
        arch=QWEN15_ARCH,
        resources=_resources(6000.0),
        allow_local_quantize=False,
        local_quantizer_available=True,
    )
    assert plan.source != "local_quantize"


def test_local_quantization_refused_without_a_toolchain_and_says_so():
    plan = mo.plan_quantization(
        summary=_summary("org/Huge-70B", params_b=70.0),
        arch=QWEN15_ARCH,
        resources=_resources(6000.0),
        allow_local_quantize=True,
        local_quantizer_available=False,
    )
    assert not plan.fits
    assert any("no local AWQ toolchain" in r.get("reason", "") for r in plan.rejected)


def test_oversized_model_is_refused_with_every_option_explained():
    plan = mo.plan_quantization(
        summary=_summary("org/Huge-70B", params_b=70.0),
        arch=QWEN15_ARCH,
        resources=_resources(6000.0),
    )
    assert not plan.fits
    assert plan.rejected, "a refusal with no explanation is not actionable"
    assert all("reason" in r for r in plan.rejected)


def test_unknown_parameter_count_is_refused_rather_than_guessed():
    plan = mo.plan_quantization(
        summary=_summary("org/Mystery", params_b=None),
        arch=QWEN15_ARCH,
        resources=_resources(20000.0),
    )
    assert not plan.fits
    assert plan.parameter_basis == "unknown"
    assert "guess" in plan.reason


def test_disk_shortage_refuses_even_when_vram_is_plentiful():
    plan = mo.plan_quantization(
        summary=_summary("org/Big-7B", params_b=7.0),
        arch=QWEN15_ARCH,
        resources=_resources(20000.0, disk_free_gb=21.0),
    )
    assert not plan.fits
    assert any("disk" in r.get("reason", "") for r in plan.rejected)


def test_gpu_memory_utilization_is_a_fraction_of_the_whole_device():
    """vLLM reads this as a share of the device, not of what is free.

    On a slice already holding other containers, a naive 0.9 would claim memory
    the residents own.
    """
    plan = mo.plan_quantization(
        summary=_summary(params_b=1.5),
        arch=QWEN15_ARCH,
        resources=_resources(6000.0, total_mb=21547.0),
    )
    assert plan.fits
    assert 0.0 < plan.gpu_memory_utilization < 0.5
    expected = min(0.95, (plan.est_vram_mb * 1.12) / 21547.0)
    assert plan.gpu_memory_utilization == pytest.approx(expected, rel=1e-6)


def test_unsizable_declared_format_is_not_claimed_unservable():
    """vLLM serves more formats than this planner selects; say that honestly."""
    plan = mo.plan_quantization(
        summary=_summary("org/Model-GGUF", params_b=7.0, config={"quantization_config": {"quant_method": "gguf"}}),
        arch=QWEN15_ARCH,
        resources=_resources(20000.0),
    )
    note = " ".join(r.get("reason", "") for r in plan.rejected)
    assert "cannot size" in note
    assert "vLLM may well serve it" in note


# ─────────────────────────────────────────────────────────────────────────────
# The measurement gate
# ─────────────────────────────────────────────────────────────────────────────


def _entry(plan_kwargs=None):
    plan = mo.plan_quantization(
        summary=_summary(params_b=1.5),
        arch=QWEN15_ARCH,
        resources=_resources(20000.0),
        **(plan_kwargs or {}),
    )
    return mo.build_zoo_entry(
        "onboard-test",
        _summary(params_b=1.5),
        plan,
        donor=DONOR,
        endpoint_env="VLLM_ONBOARD_TEST_URL",
        endpoint_url=None,
    )


def test_new_entry_is_not_routable():
    """The central invariant: an unmeasured model cannot be a CSS candidate.

    ``routing_policies.rank_routing_candidates`` filters on ``available``, so
    ``available: False`` is what actually keeps it out of the ranking.
    """
    entry = _entry()
    assert entry["available"] is False
    assert entry["accuracy_basis"] == "unmeasured"
    assert entry["latency_basis"] == "unmeasured"
    assert entry["accuracy_baseline"] == 0.0


def test_new_entry_declares_no_verbosity_prior():
    """expected_output_tokens feeds CSS's carbon term; an invented one would skew it."""
    assert "expected_output_tokens" not in _entry()


def test_device_properties_are_inherited_and_labelled():
    """TDP and embodied carbon describe the board, which really is shared."""
    entry = _entry()
    assert entry["power_tdp_w"] == DONOR["power_tdp_w"]
    assert entry["mfg_carbon_kg"] == DONOR["mfg_carbon_kg"]
    assert entry["device_share"] == DONOR["device_share"]
    assert entry["device_fields_basis"] == "inherited:local-vgpu-full"


def test_measurement_makes_it_routable_and_records_provenance():
    patch = mo.measurement_patch(
        {"accuracy": 0.81, "latency_ms_p50": 240, "latency_ms_p95": 610, "expected_output_tokens": 96, "samples": 450},
        basis="measured:onboard-bench-01",
    )
    assert patch["available"] is True
    assert patch["accuracy_baseline"] == 0.81
    assert patch["accuracy_basis"] == "measured:onboard-bench-01"
    assert patch["expected_output_tokens_basis"] == "measured:onboard-bench-01"


def test_measurement_requires_both_accuracy_and_latency():
    with pytest.raises(ValueError):
        mo.measurement_patch({"accuracy": 0.8}, basis="b")
    with pytest.raises(ValueError):
        mo.measurement_patch({"latency_ms_p50": 100}, basis="b")


def test_accuracy_must_be_a_pass_rate():
    """Guards the units confusion that would put 81.0 where 0.81 belongs."""
    with pytest.raises(ValueError):
        mo.measurement_patch({"accuracy": 81.0, "latency_ms_p50": 100}, basis="b")


def test_measurement_without_observed_length_leaves_the_prior_alone():
    patch = mo.measurement_patch({"accuracy": 0.7, "latency_ms_p50": 100}, basis="b")
    assert "expected_output_tokens" not in patch


# ─────────────────────────────────────────────────────────────────────────────
# Quantization payback
# ─────────────────────────────────────────────────────────────────────────────


def test_payback_counts_requests_to_break_even():
    out = mo.estimate_payback(100.0, 0.010, 0.015)
    assert out["pays_back"] is True
    assert out["requests_to_payback"] == 20000


def test_payback_reports_a_rung_that_never_pays_back():
    out = mo.estimate_payback(100.0, 0.020, 0.015)
    assert out["pays_back"] is False
    assert "never recovered" in out["detail"]


def test_payback_is_none_until_both_rungs_are_measured():
    assert mo.estimate_payback(100.0, None, 0.015) is None
    assert mo.estimate_payback(100.0, 0.010, None) is None
    assert mo.estimate_payback(0.0, 0.010, 0.015) is None


# ─────────────────────────────────────────────────────────────────────────────
# Service-level guards (no network, no docker)
# ─────────────────────────────────────────────────────────────────────────────


class _StubZoo:
    def __init__(self, models):
        self._models = list(models)

    def list_models(self):
        return [dict(m) for m in self._models]

    def get_model(self, model_id):
        for m in self._models:
            if m.get("id") == model_id:
                return dict(m)
        return None

    def register_model(self, entry):
        self._models = [m for m in self._models if m.get("id") != entry.get("id")]
        self._models.append(dict(entry))


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_ONBOARD_ENABLED", "true")
    return mo.ModelOnboardingService(
        _StubZoo([DONOR]),
        state_path=tmp_path / "jobs.json",
        cache_dir=tmp_path / "cache",
    )


def test_onboarding_is_off_unless_enabled(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_ONBOARD_ENABLED", raising=False)
    svc = mo.ModelOnboardingService(_StubZoo([DONOR]), state_path=tmp_path / "j.json", cache_dir=tmp_path)
    with pytest.raises(PermissionError):
        svc.start("org/Foo-7B")


def test_trust_remote_code_needs_its_own_deliberate_opt_in(service, monkeypatch):
    """It is arbitrary code execution from a third-party repo."""
    monkeypatch.delenv("MODEL_ONBOARD_ALLOW_TRUST_REMOTE_CODE", raising=False)
    with pytest.raises(PermissionError) as exc:
        service.start("org/Foo-7B", trust_remote_code=True)
    assert "arbitrary code" in str(exc.value)


def test_measuring_an_unserved_model_is_refused(service):
    """Figures measured against no endpoint describe nothing."""
    service.zoo.register_model({"id": "onboard-x", "onboarded": True, "available": False})
    with pytest.raises(ValueError) as exc:
        service.apply_measurement("onboard-x", {"accuracy": 0.8, "latency_ms_p50": 100}, basis="b")
    assert "no live endpoint" in str(exc.value)


def test_static_models_are_not_editable_through_this_pipeline(service):
    with pytest.raises(ValueError) as exc:
        service.apply_measurement("local-vgpu-full", {"accuracy": 0.8, "latency_ms_p50": 100}, basis="b")
    assert "not onboarded by this pipeline" in str(exc.value)


def test_static_models_are_not_servable_through_this_pipeline(service):
    with pytest.raises(ValueError) as exc:
        service.serve("local-vgpu-full")
    assert "served by compose" in str(exc.value)


def test_unserving_clears_availability_before_the_container_goes(service):
    """A model whose backend is gone must stop being a CSS candidate."""
    service.zoo.register_model(
        {"id": "onboard-x", "onboarded": True, "available": True, "vllm_endpoint_url": "http://x:8010/v1"}
    )
    service.unserve("onboard-x")
    entry = service.zoo.get_model("onboard-x")
    assert entry["available"] is False
    assert entry["vllm_endpoint_url"] is None


def test_donor_must_share_the_hardware(service):
    with pytest.raises(ValueError):
        service._pick_donor("no-such-model")


def test_interrupted_jobs_do_not_come_back_looking_alive(tmp_path, monkeypatch):
    """The thread died with the process; reporting 'downloading' forever is a lie."""
    state = tmp_path / "jobs.json"
    state.write_text(
        json.dumps(
            {"jobs": [{"job_id": "j1", "repo_id": "org/Foo", "model_id": "onboard-foo", "state": "downloading"}]}
        ),
        encoding="utf-8",
    )
    svc = mo.ModelOnboardingService(_StubZoo([DONOR]), state_path=state, cache_dir=tmp_path)
    job = svc.get_job("j1")
    assert job.state == "failed"
    assert "interrupted" in job.error


def test_serving_manager_reports_why_it_is_unavailable(tmp_path):
    """Docker's socket is not mounted by default; say so instead of failing opaquely."""
    mgr = mo.ServingManager(mo.DockerClient(socket_path=str(tmp_path / "nope.sock")))
    cap = mgr.capability()
    assert cap["enabled"] is False
    assert "not mounted" in cap["reason"]


# ─────────────────────────────────────────────────────────────────────────────
# Failing closed when VRAM cannot be measured
#
# The API container ships without nvidia-smi and the metrics sidecar defaults to
# DISABLE_GPU_METRICS=1, so "no GPU reading available" is the *default* state of
# a fresh deployment — not an edge case.
# ─────────────────────────────────────────────────────────────────────────────


def test_unmeasured_vram_yields_no_budget():
    """Capacity without occupancy is not a budget."""
    res = _resources(20000.0)
    res.vram_basis = "unavailable"
    assert res.vram_measured is False
    assert res.vram_budget_mb == 0.0


def test_planner_refuses_with_the_remedy_not_a_misleading_zero():
    res = _resources(20000.0)
    res.vram_basis = "unavailable"
    plan = mo.plan_quantization(summary=_summary(params_b=1.5), arch=QWEN15_ARCH, resources=res)
    assert not plan.fits
    assert "could not be measured" in plan.reason
    # "0 MB free" would read as "the GPU is full" and send an operator hunting
    # for a problem that does not exist.
    assert "only 0 MB free" not in plan.reason
    assert "DISABLE_GPU_METRICS" in plan.reason


def test_metrics_sidecar_is_used_when_nvidia_smi_is_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(mo, "_nvidia_smi", lambda args, timeout=10.0: None)
    res = mo.probe_resources(
        tmp_path,
        lambda: {
            "system_gpu_TotalMemory": 21547.0,
            "system_gpu_UsedMemory": 14888.0,
            "system_gpu_MemoryFree": 6659.0,
        },
    )
    assert res.vram_basis == "metrics_sidecar"
    assert res.vram_measured is True
    assert res.vram_total_mb == 21547.0


def test_sidecar_reporting_zeros_does_not_count_as_a_measurement(monkeypatch, tmp_path):
    """DISABLE_GPU_METRICS=1 makes every field 0.0; that is absence, not an empty GPU."""
    monkeypatch.setattr(mo, "_nvidia_smi", lambda args, timeout=10.0: None)
    res = mo.probe_resources(
        tmp_path,
        lambda: {"system_gpu_TotalMemory": 0.0, "system_gpu_UsedMemory": 0.0, "system_gpu_MemoryFree": 0.0},
    )
    assert res.vram_basis == "unavailable"
    assert res.vram_measured is False


def test_a_broken_sidecar_does_not_break_planning(monkeypatch, tmp_path):
    monkeypatch.setattr(mo, "_nvidia_smi", lambda args, timeout=10.0: None)

    def boom():
        raise RuntimeError("sidecar down")

    res = mo.probe_resources(tmp_path, boom)
    assert res.vram_basis == "unavailable"


def test_nvidia_smi_wins_over_the_sidecar(monkeypatch, tmp_path):
    """The direct reading is MIG-aware; the sidecar's is not."""
    monkeypatch.setattr(mo, "_nvidia_smi", lambda args, timeout=10.0: MIG_SMI_TEXT if not args else
                        "NVIDIA H100L-2-24C, 24576, 14913, 6660, Enabled")
    res = mo.probe_resources(tmp_path, lambda: {"system_gpu_TotalMemory": 999.0, "system_gpu_MemoryFree": 999.0})
    assert res.vram_basis == "mig_instance"
    assert res.vram_total_mb == 21547.0


def test_insufficient_permissions_is_treated_as_no_reading(monkeypatch):
    """A MIG board denies memory queries to unprivileged containers.

    nvidia-smi still exits 0 and prints "[Insufficient Permissions]" in the
    numeric columns. A partial reading is worse than none, because it would be
    sized against.
    """
    monkeypatch.setattr(
        mo,
        "_nvidia_smi",
        lambda args, timeout=10.0: "NVIDIA H100L-2-24C, [Insufficient Permissions], "
        "[Insufficient Permissions], [Insufficient Permissions], Enabled",
    )
    name, total, used, free, basis, _ = mo.probe_gpu()
    assert basis == "unavailable"
    assert (total, used, free) == (0.0, 0.0, 0.0)


def test_sidecar_free_is_taken_as_reported_not_recomputed(monkeypatch, tmp_path):
    """total - used = 9688 on this board; the MIG-aware free is 6660."""
    monkeypatch.setattr(mo, "_nvidia_smi", lambda args, timeout=10.0: None)
    res = mo.probe_resources(
        tmp_path,
        lambda: {
            "system_gpu_TotalMemory": 24576.0,
            "system_gpu_UsedMemory": 14888.0,
            "system_gpu_MemoryFree": 6660.0,
        },
    )
    assert res.vram_free_mb == 6660.0
    assert res.vram_budget_mb == 6660.0 - 1024.0
