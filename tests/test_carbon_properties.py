"""Invariants that should survive any reformulation of the carbon model.

Where test_carbon_golden.py pins *specific numbers* against a fixture, this file
pins *relationships* -- the things that must stay true whatever the formula is.
It also asserts bounds on the shipped config/model_zoo.json, so a bad entry (from
a hand edit or the model-zoo auto-updater) fails a test rather than a division.
"""
from __future__ import annotations

import json

import pytest

GRID_CI = 1000.0


class TestCarbonInvariants:
    def test_zero_duration_costs_nothing(self, zoo):
        out = zoo.compute_total_carbon(
            "test-dense", grid_carbon_g_per_kwh=GRID_CI, inference_duration_s=0.0
        )
        assert out["total_carbon_g"] == 0.0

    @pytest.mark.parametrize("model_id", ["test-dense", "test-moe", "test-regional"])
    def test_carbon_is_strictly_increasing_in_duration(self, zoo, model_id):
        prev = -1.0
        for duration in (0.5, 1.0, 5.0, 30.0):
            out = zoo.compute_operational_carbon(
                model_id, grid_carbon_g_per_kwh=GRID_CI, inference_duration_s=duration
            )
            assert out["op_carbon_g"] > prev
            prev = out["op_carbon_g"]

    @pytest.mark.parametrize("model_id", ["test-dense", "test-moe", "test-regional"])
    def test_carbon_is_strictly_increasing_in_grid_intensity(self, zoo, model_id):
        prev = -1.0
        for ci in (10.0, 100.0, 400.0, 900.0):
            out = zoo.compute_operational_carbon(
                model_id, grid_carbon_g_per_kwh=ci, inference_duration_s=5.0
            )
            assert out["op_carbon_g"] > prev
            prev = out["op_carbon_g"]

    def test_carbon_is_linear_in_grid_intensity(self, zoo):
        """Doubling grid CI must exactly double operational carbon -- the whole
        premise of routing to a cleaner grid depends on this being linear."""
        low = zoo.compute_operational_carbon(
            "test-dense", grid_carbon_g_per_kwh=250.0, inference_duration_s=5.0
        )
        high = zoo.compute_operational_carbon(
            "test-dense", grid_carbon_g_per_kwh=500.0, inference_duration_s=5.0
        )
        assert high["op_carbon_g"] == pytest.approx(2 * low["op_carbon_g"], rel=1e-9)

    def test_negative_duration_is_clamped_not_negative(self, zoo):
        """A clock skew must never produce a negative carbon credit."""
        out = zoo.compute_request_carbon(
            "test-dense", grid_carbon_g_per_kwh=GRID_CI, measured_duration_s=-5.0
        )
        assert out["total_carbon_g"] >= 0.0

    def test_total_equals_operational_plus_embodied(self, zoo):
        out = zoo.compute_total_carbon(
            "test-dense", grid_carbon_g_per_kwh=GRID_CI, inference_duration_s=3.0
        )
        assert out["total_carbon_g"] == pytest.approx(
            out["operational"]["op_carbon_g"] + out["embodied"]["emb_carbon_g"], abs=1e-8
        )


class TestDegenerateHardwareEfficiency:
    """A model whose all_to_all_overhead_ratio is 1.0 drives he_effective to zero.

    he_effective is a divisor in both carbon functions. compute_request_carbon
    always clamped it; compute_operational_carbon did not, and raised
    ZeroDivisionError for such an entry while its sibling priced it fine. No
    shipped model triggers this, but register_model and the model-zoo
    auto-updater can add entries that would. Both now clamp to 0.05.
    """

    def test_request_carbon_survives_degenerate_efficiency(self, zoo):
        out = zoo.compute_request_carbon(
            "test-degenerate-he", grid_carbon_g_per_kwh=GRID_CI, measured_duration_s=9.0
        )
        assert out["he_effective"] == pytest.approx(0.05, rel=1e-9)
        assert out["op_carbon_g"] > 0

    def test_operational_carbon_survives_degenerate_efficiency(self, zoo):
        out = zoo.compute_operational_carbon(
            "test-degenerate-he", grid_carbon_g_per_kwh=GRID_CI, inference_duration_s=9.0
        )
        assert out["he_effective"] == pytest.approx(0.05, rel=1e-9)
        assert out["op_carbon_g"] > 0

    def test_the_two_functions_agree_on_a_degenerate_entry(self, zoo):
        """The point of the clamp: neither function may consider an entry
        priceable that the other does not."""
        ante = zoo.compute_operational_carbon(
            "test-degenerate-he", grid_carbon_g_per_kwh=GRID_CI, inference_duration_s=9.0
        )
        post = zoo.compute_request_carbon(
            "test-degenerate-he", grid_carbon_g_per_kwh=GRID_CI, measured_duration_s=9.0
        )
        assert ante["op_carbon_g"] == pytest.approx(post["op_carbon_g"], rel=1e-9)


class TestShippedZooBounds:
    """The real config/model_zoo.json must stay inside sane physical bounds.

    Deliberately separate from the golden tests: those use a fixture so retuning
    a real model does not churn them, while these guard the data that actually
    ships.
    """

    @pytest.fixture(scope="class")
    def models(self, shipped_zoo_path):
        return json.loads(shipped_zoo_path.read_text(encoding="utf-8"))["models"]

    def test_model_ids_are_unique(self, models):
        """get_model() keys on 'id' and returns the first match, so a duplicate
        would silently shadow an entry."""
        ids = [m.get("id") for m in models]
        assert len(ids) == len(set(ids)), "duplicate model ids in shipped zoo"

    def test_every_model_has_an_id(self, models):
        assert all(m.get("id") for m in models)

    @pytest.mark.parametrize("field", ["hardware_efficiency", "power_tdp_w", "pue"])
    def test_positive_physical_parameters(self, models, field):
        for m in models:
            value = m.get(field)
            if value is None:
                continue
            assert value > 0, f"{m.get('id')}: {field} must be > 0, got {value}"

    def test_hardware_efficiency_is_a_fraction(self, models):
        for m in models:
            he = m.get("hardware_efficiency")
            if he is not None:
                assert 0 < he <= 1, f"{m.get('id')}: hardware_efficiency out of range: {he}"

    def test_all_to_all_overhead_cannot_zero_out_efficiency(self, models):
        """he_effective = he x (1 - all_to_all_overhead_ratio); a ratio of 1.0
        makes that zero and both carbon functions divide by it."""
        for m in models:
            ratio = m.get("all_to_all_overhead_ratio", 0.0)
            assert 0 <= ratio < 1, f"{m.get('id')}: all_to_all_overhead_ratio out of range: {ratio}"

    def test_device_utilization_and_share_are_fractions(self, models):
        for m in models:
            for field in ("device_utilization", "device_share"):
                value = m.get(field)
                if value is not None:
                    assert 0 < value <= 1, f"{m.get('id')}: {field} out of range: {value}"

    def test_every_shipped_model_can_be_priced(self, shipped_zoo_path):
        """End-to-end: every entry in the shipped zoo produces a finite,
        non-negative carbon number without raising."""
        from model_zoo import ModelZooService

        zoo = ModelZooService(zoo_path=shipped_zoo_path)
        for model in zoo.list_models():
            out = zoo.compute_request_carbon(
                model["id"], grid_carbon_g_per_kwh=400.0, measured_duration_s=2.0
            )
            assert out["total_carbon_g"] >= 0
            assert out["total_carbon_g"] < 1000, f"{model['id']}: implausible carbon"

    def test_ex_ante_and_ex_post_agree_for_every_shipped_model(self, shipped_zoo_path):
        """CSS ranks candidates with compute_operational_carbon and bills with
        compute_request_carbon. If they disagree, the system chooses on one
        number and reports another."""
        from model_zoo import ModelZooService

        zoo = ModelZooService(zoo_path=shipped_zoo_path)
        for model in zoo.list_models():
            ante = zoo.compute_operational_carbon(
                model["id"], grid_carbon_g_per_kwh=400.0, inference_duration_s=2.0
            )
            post = zoo.compute_request_carbon(
                model["id"], grid_carbon_g_per_kwh=400.0, measured_duration_s=2.0
            )
            assert ante["op_carbon_g"] == pytest.approx(post["op_carbon_g"], rel=1e-9), model["id"]
