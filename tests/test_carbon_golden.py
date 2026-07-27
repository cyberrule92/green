"""Golden values for the carbon arithmetic.

These numbers are what the product's central claim rests on, so they are pinned
against a fixture zoo with round parameters (``tests/fixtures/model_zoo_min.json``)
chosen so every expectation below can be checked by hand:

    tdp_w = 100 W,  pue = 2.0,  hardware_efficiency = 0.5,  duration = 9 s

    energy_kwh = (tdp_w x duration x pue) / (he_eff x 1000 x 3600)
               = (100 x 9 x 2.0) / (0.5 x 3_600_000)
               = 1800 / 1_800_000
               = 0.001 kWh

    op_carbon_g = energy_kwh x grid_ci x region_multiplier
                = 0.001 x 1000 x 1.0
                = 1.0 g

The MoE model halves the effective HE (all_to_all_overhead_ratio = 0.5), so it
costs exactly 2x; the regional model carries a 3x region multiplier, so it costs
exactly 3x. Any change to those relationships is a change to the carbon model and
should be a deliberate, reviewed diff -- not a surprise.
"""
from __future__ import annotations

import pytest

# 9 s at the fixture's parameters yields exactly 0.001 kWh.
DURATION_S = 9.0
GRID_CI = 1000.0
EXPECTED_ENERGY_KWH = 0.001
EXPECTED_OP_G_DENSE = 1.0


class TestOperationalCarbon:
    def test_dense_model_golden_value(self, zoo):
        out = zoo.compute_operational_carbon(
            "test-dense", grid_carbon_g_per_kwh=GRID_CI, inference_duration_s=DURATION_S
        )
        assert out["energy_kwh"] == pytest.approx(EXPECTED_ENERGY_KWH, rel=1e-9)
        assert out["op_carbon_g"] == pytest.approx(EXPECTED_OP_G_DENSE, rel=1e-9)
        assert out["he_effective"] == pytest.approx(0.5, rel=1e-9)

    def test_moe_all_to_all_penalty_halves_efficiency(self, zoo):
        """all_to_all_overhead_ratio=0.5 halves HE, so carbon exactly doubles."""
        out = zoo.compute_operational_carbon(
            "test-moe", grid_carbon_g_per_kwh=GRID_CI, inference_duration_s=DURATION_S
        )
        assert out["he_effective"] == pytest.approx(0.25, rel=1e-9)
        assert out["op_carbon_g"] == pytest.approx(2 * EXPECTED_OP_G_DENSE, rel=1e-9)

    def test_region_multiplier_scales_linearly(self, zoo):
        out = zoo.compute_operational_carbon(
            "test-regional", grid_carbon_g_per_kwh=GRID_CI, inference_duration_s=DURATION_S
        )
        assert out["op_carbon_g"] == pytest.approx(3 * EXPECTED_OP_G_DENSE, rel=1e-9)

    def test_unknown_model_returns_error_rather_than_raising(self, zoo):
        out = zoo.compute_operational_carbon(
            "nope", grid_carbon_g_per_kwh=GRID_CI, inference_duration_s=DURATION_S
        )
        assert out["total_carbon_g"] == 0.0
        assert "error" in out


class TestEmbodiedCarbon:
    def test_rate_is_mfg_carbon_amortised_over_device_seconds(self, zoo):
        """rate = (mfg_kg x 1000 x share) / (lifetime_years x SECONDS_PER_YEAR x util).

        The fixture uses share=1, util=1, lifetime=1 year, mfg=100 kg, so the rate
        is exactly 100_000 g / one year of seconds."""
        from model_zoo import _SECONDS_PER_YEAR

        out = zoo.compute_embodied_rate("test-dense")
        assert out["emb_rate_g_per_device_s"] == pytest.approx(
            100_000.0 / _SECONDS_PER_YEAR, rel=1e-12
        )

    def test_embodied_carbon_is_rate_times_duration(self, zoo):
        rate = zoo.compute_embodied_rate("test-dense")["emb_rate_g_per_device_s"]
        out = zoo.compute_embodied_carbon("test-dense", inference_duration_s=DURATION_S)
        # compute_embodied_carbon rounds to 8dp, so compare at that granularity
        # rather than with a tight relative tolerance.
        assert out["emb_carbon_g"] == pytest.approx(rate * DURATION_S, abs=1e-8)

    def test_embodied_carbon_is_not_a_constant(self, zoo):
        """Regression guard for the bug documented in compute_embodied_rate's
        docstring, where the rate's denominator cancelled the duration it was
        multiplied by and every request was billed the same amount."""
        short = zoo.compute_embodied_carbon("test-dense", inference_duration_s=1.0)
        long = zoo.compute_embodied_carbon("test-dense", inference_duration_s=100.0)
        assert long["emb_carbon_g"] == pytest.approx(100 * short["emb_carbon_g"], abs=1e-6)
        # The substance of the guard: it scales, rather than being flat.
        assert long["emb_carbon_g"] > short["emb_carbon_g"] * 50


class TestTotalAndRequestCarbon:
    def test_total_is_operational_plus_embodied(self, zoo):
        total = zoo.compute_total_carbon(
            "test-dense", grid_carbon_g_per_kwh=GRID_CI, inference_duration_s=DURATION_S
        )
        assert total["total_carbon_g"] == pytest.approx(
            total["operational"]["op_carbon_g"] + total["embodied"]["emb_carbon_g"], rel=1e-9
        )

    def test_request_carbon_matches_operational_for_same_inputs(self, zoo):
        """compute_request_carbon (ex-post) and compute_operational_carbon
        (ex-ante) must agree on the operational term for identical inputs --
        they are the same physics, billed at different times."""
        ante = zoo.compute_operational_carbon(
            "test-dense", grid_carbon_g_per_kwh=GRID_CI, inference_duration_s=DURATION_S
        )
        post = zoo.compute_request_carbon(
            "test-dense", grid_carbon_g_per_kwh=GRID_CI, measured_duration_s=DURATION_S
        )
        assert post["op_carbon_g"] == pytest.approx(ante["op_carbon_g"], rel=1e-9)

    def test_per_output_token_rate_is_none_when_no_tokens(self, zoo):
        """None, not carbon/1 -- booking a whole leg against a single token would
        write a nonsense rate into the ledger (see the field's comment)."""
        out = zoo.compute_request_carbon(
            "test-dense", grid_carbon_g_per_kwh=GRID_CI,
            measured_duration_s=DURATION_S, output_tokens=0,
        )
        assert out["ug_per_output_token"] is None

    def test_per_output_token_rate_when_tokens_present(self, zoo):
        out = zoo.compute_request_carbon(
            "test-dense", grid_carbon_g_per_kwh=GRID_CI,
            measured_duration_s=DURATION_S, output_tokens=100,
        )
        assert out["ug_per_output_token"] == pytest.approx(
            (out["total_carbon_g"] / 100) * 1e6, rel=1e-6
        )


class TestFlopFieldsDoNotAffectCarbon:
    """Intended and documented behaviour.

    Operational carbon is ``power x time`` against a measured duration. FLOP
    counts and token counts are computed and returned as reporting metadata but
    feed no carbon number -- multiplying by a FLOP estimate as well would
    double-count the same work, and the LLMCarbon FLOP form is the alternative
    for when no measured duration exists, which is never the case here. See
    ``compute_operational_carbon``'s docstring and the ``_field_notes`` block in
    config/model_zoo.json.

    These tests pin that contract. If someone later decides FLOPs *should* drive
    the number, that is a modelling change and these should fail loudly.
    """

    def test_token_count_does_not_change_operational_carbon(self, zoo):
        few = zoo.compute_operational_carbon(
            "test-dense", grid_carbon_g_per_kwh=GRID_CI,
            inference_duration_s=DURATION_S, token_count=1,
        )
        many = zoo.compute_operational_carbon(
            "test-dense", grid_carbon_g_per_kwh=GRID_CI,
            inference_duration_s=DURATION_S, token_count=100_000,
        )
        assert few["op_carbon_g"] == pytest.approx(many["op_carbon_g"], rel=1e-12)
        # ...even though the reported FLOP total does scale with tokens.
        assert many["total_flops"] == 100_000 * few["total_flops"]

    def test_sparse_flop_path_is_reported_but_does_not_change_carbon(self, zoo):
        """The MoE branch selects flop_count_per_token_sparse for reporting; the
        only thing that moves carbon is the separate he_effective penalty."""
        out = zoo.compute_operational_carbon(
            "test-moe", grid_carbon_g_per_kwh=GRID_CI, inference_duration_s=DURATION_S
        )
        assert out["sparse_path"] is True
        assert out["flop_per_token"] == 2_000_000_000  # the sparse count
        # Carbon is explained entirely by he_effective (0.25), not by FLOPs:
        expected = (100.0 * DURATION_S * 2.0) / (0.25 * 1000.0 * 3600.0) * GRID_CI
        assert out["op_carbon_g"] == pytest.approx(expected, rel=1e-9)
