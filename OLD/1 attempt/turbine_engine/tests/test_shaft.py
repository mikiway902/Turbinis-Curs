"""Focused tests for the standalone architecture shaft power balances.

These tests check the explicit per-architecture closure arithmetic, sign
retention, efficiency placement, validation, immutability, and the current
single-shaft cycle anchor.  They do not evaluate an architecture-specific cycle
and do not claim physical validation of any topology.
"""

from __future__ import annotations

import math
import subprocess
import sys
import unittest
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import turbine_engine  # noqa: E402
import turbine_engine.shaft as shaft_module  # noqa: E402

from turbine_engine.architecture import Architecture  # noqa: E402
from turbine_engine.cycle import (  # noqa: E402
    CycleConfig,
    CycleState,
    calculate_cycle_point,
)
from turbine_engine.shaft import (  # noqa: E402
    ShaftPowerBalance,
    evaluate_shaft_power_balance,
)


def single_shaft_single_cc(
    *,
    compressor_gas_powers: tuple[float, ...] = (150_000.0,),
    turbine_gas_powers: tuple[float, ...] = (300_000.0,),
    mechanical_efficiencies: tuple[float, ...] = (0.9,),
    required_gearbox_input_power: float = 110_000.0,
    gearbox_efficiency: float = 0.98,
) -> ShaftPowerBalance:
    return evaluate_shaft_power_balance(
        Architecture.SINGLE_SHAFT_SINGLE_CC,
        compressor_gas_powers=compressor_gas_powers,
        turbine_gas_powers=turbine_gas_powers,
        mechanical_efficiencies=mechanical_efficiencies,
        required_gearbox_input_power=required_gearbox_input_power,
        gearbox_efficiency=gearbox_efficiency,
    )


class SyntheticClosureTests(unittest.TestCase):
    def test_single_shaft_dual_cc_sums_two_compressor_loads_on_main(self) -> None:
        result = evaluate_shaft_power_balance(
            Architecture.SINGLE_SHAFT_DUAL_CC,
            compressor_gas_powers=(100_000.0, 60_000.0),
            turbine_gas_powers=(250_000.0,),
            mechanical_efficiencies=(0.98,),
            required_gearbox_input_power=80_000.0,
            gearbox_efficiency=0.98,
        )

        self.assertEqual(result.turbine_mechanical_powers, (245_000.0,))
        self.assertEqual(result.compressor_gas_power_total, 160_000.0)
        self.assertEqual(result.main_net_power, 85_000.0)
        self.assertEqual(result.actual_gearbox_input_power, 85_000.0)
        self.assertEqual(result.main_residual, 5_000.0)
        self.assertEqual(result.gearbox_output_power, 83_300.0)
        self.assertIsNone(result.gas_generator_net_power)
        self.assertIsNone(result.gas_generator_residual)
        self.assertIsNone(result.free_power_turbine_mechanical_power)
        self.assertIsNone(result.free_power_turbine_residual)

    def test_single_shaft_single_cc_uses_one_compressor_and_one_turbine(self) -> None:
        result = single_shaft_single_cc()

        self.assertEqual(result.turbine_mechanical_powers, (270_000.0,))
        self.assertEqual(result.main_net_power, 120_000.0)
        self.assertEqual(result.actual_gearbox_input_power, 120_000.0)
        self.assertEqual(result.main_residual, 10_000.0)
        self.assertEqual(result.gearbox_output_power, 117_600.0)
        self.assertIsNone(result.gas_generator_residual)
        self.assertIsNone(result.free_power_turbine_residual)

    def test_free_power_turbine_separates_gg_and_output_shafts(self) -> None:
        result = evaluate_shaft_power_balance(
            Architecture.FREE_POWER_TURBINE,
            compressor_gas_powers=(90_000.0,),
            turbine_gas_powers=(130_000.0, 120_000.0),
            mechanical_efficiencies=(0.98, 0.99),
            required_gearbox_input_power=110_000.0,
            gearbox_efficiency=0.98,
        )

        self.assertEqual(
            result.turbine_mechanical_powers, (127_400.0, 118_800.0)
        )
        self.assertIsNone(result.main_net_power)
        self.assertIsNone(result.main_residual)
        self.assertEqual(result.gas_generator_net_power, 37_400.0)
        self.assertEqual(result.gas_generator_residual, 37_400.0)
        self.assertEqual(result.free_power_turbine_mechanical_power, 118_800.0)
        self.assertEqual(result.actual_gearbox_input_power, 118_800.0)
        self.assertEqual(result.free_power_turbine_residual, 8_800.0)
        self.assertEqual(result.gearbox_output_power, 116_424.0)

    def test_free_power_turbine_residuals_are_independent(self) -> None:
        result = evaluate_shaft_power_balance(
            Architecture.FREE_POWER_TURBINE,
            compressor_gas_powers=(200_000.0,),
            turbine_gas_powers=(130_000.0, 100_000.0),
            mechanical_efficiencies=(0.98, 0.99),
            required_gearbox_input_power=110_000.0,
            gearbox_efficiency=0.98,
        )

        self.assertEqual(result.gas_generator_net_power, -72_600.0)
        self.assertEqual(result.gas_generator_residual, -72_600.0)
        self.assertEqual(result.free_power_turbine_mechanical_power, 99_000.0)
        self.assertEqual(result.actual_gearbox_input_power, 99_000.0)
        self.assertEqual(result.free_power_turbine_residual, -11_000.0)
        self.assertEqual(result.gearbox_output_power, 97_020.0)

    def test_insufficient_turbine_power_is_retained_as_negative_values(self) -> None:
        result = single_shaft_single_cc(
            compressor_gas_powers=(200_000.0,),
            turbine_gas_powers=(100_000.0,),
            mechanical_efficiencies=(0.9,),
        )

        self.assertEqual(result.turbine_mechanical_powers, (90_000.0,))
        self.assertEqual(result.main_net_power, -110_000.0)
        self.assertEqual(result.actual_gearbox_input_power, -110_000.0)
        self.assertEqual(result.main_residual, -220_000.0)
        self.assertEqual(result.gearbox_output_power, -107_800.0)

    def test_zero_power_is_a_valid_no_load_balance(self) -> None:
        result = single_shaft_single_cc(
            compressor_gas_powers=(0.0,),
            turbine_gas_powers=(0.0,),
            mechanical_efficiencies=(1.0,),
            required_gearbox_input_power=0.0,
            gearbox_efficiency=1.0,
        )

        self.assertEqual(result.actual_gearbox_input_power, 0.0)
        self.assertEqual(result.main_residual, 0.0)
        self.assertEqual(result.gearbox_output_power, 0.0)


class EfficiencyPlacementTests(unittest.TestCase):
    def test_mechanical_efficiency_is_applied_once_to_turbine_gas_power(self) -> None:
        result = single_shaft_single_cc(
            compressor_gas_powers=(150_000.0,),
            turbine_gas_powers=(300_000.0,),
            mechanical_efficiencies=(0.9,),
        )

        self.assertEqual(result.turbine_mechanical_powers, (270_000.0,))
        self.assertNotEqual(
            result.turbine_mechanical_powers[0], 300_000.0 * 0.81
        )
        self.assertEqual(result.main_residual, 10_000.0)

    def test_gearbox_efficiency_changes_only_downstream_output(self) -> None:
        base = single_shaft_single_cc(gearbox_efficiency=0.98)

        for gearbox_efficiency in (0.5, 0.75, 1.0):
            with self.subTest(gearbox_efficiency=gearbox_efficiency):
                changed = replace(
                    base, gearbox_efficiency=gearbox_efficiency
                )
                self.assertEqual(
                    changed.actual_gearbox_input_power,
                    base.actual_gearbox_input_power,
                )
                self.assertEqual(changed.main_residual, base.main_residual)
                self.assertEqual(
                    changed.turbine_mechanical_powers,
                    base.turbine_mechanical_powers,
                )
                self.assertEqual(
                    changed.gearbox_output_power,
                    changed.actual_gearbox_input_power * gearbox_efficiency,
                )

    def test_free_power_turbine_output_uses_fpt_not_gg_or_target(self) -> None:
        result = evaluate_shaft_power_balance(
            Architecture.FREE_POWER_TURBINE,
            compressor_gas_powers=(90_000.0,),
            turbine_gas_powers=(130_000.0, 120_000.0),
            mechanical_efficiencies=(0.98, 0.99),
            required_gearbox_input_power=110_000.0,
            gearbox_efficiency=0.5,
        )

        self.assertEqual(result.actual_gearbox_input_power, 118_800.0)
        self.assertEqual(result.gearbox_output_power, 59_400.0)
        self.assertNotEqual(
            result.gearbox_output_power, 110_000.0 * 0.5
        )


class MassFlowBoundaryTests(unittest.TestCase):
    def test_mass_flow_is_resolved_externally_and_specific_work_is_not_equated(
        self,
    ) -> None:
        # The same mass-flow-weighted gas powers can come from different
        # specific-work / mass-flow pairs.  The evaluator accepts only W and
        # must return the same balance for both pairs.
        first = evaluate_shaft_power_balance(
            Architecture.SINGLE_SHAFT_SINGLE_CC,
            compressor_gas_powers=(250_000.0 * 0.6,),
            turbine_gas_powers=(500_000.0 * 0.6,),
            mechanical_efficiencies=(0.9,),
            required_gearbox_input_power=110_000.0,
            gearbox_efficiency=0.98,
        )
        second = evaluate_shaft_power_balance(
            Architecture.SINGLE_SHAFT_SINGLE_CC,
            compressor_gas_powers=(500_000.0 * 0.3,),
            turbine_gas_powers=(1_000_000.0 * 0.3,),
            mechanical_efficiencies=(0.9,),
            required_gearbox_input_power=110_000.0,
            gearbox_efficiency=0.98,
        )

        self.assertEqual(first, second)
        self.assertEqual(first.actual_gearbox_input_power, 120_000.0)
        self.assertEqual(first.main_residual, 10_000.0)


class ValidationTests(unittest.TestCase):
    def test_rejects_non_architecture_member(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_shaft_power_balance(
                "SINGLE_SHAFT_SINGLE_CC",  # type: ignore[arg-type]
                compressor_gas_powers=(150_000.0,),
                turbine_gas_powers=(300_000.0,),
                mechanical_efficiencies=(0.9,),
                required_gearbox_input_power=110_000.0,
                gearbox_efficiency=0.98,
            )

    def test_rejects_wrong_component_counts_per_architecture(self) -> None:
        cases = (
            (
                Architecture.SINGLE_SHAFT_DUAL_CC,
                (100_000.0,),
                (250_000.0,),
                (0.98,),
            ),
            (
                Architecture.SINGLE_SHAFT_DUAL_CC,
                (100_000.0, 60_000.0),
                (250_000.0, 10_000.0),
                (0.98,),
            ),
            (
                Architecture.SINGLE_SHAFT_DUAL_CC,
                (100_000.0, 60_000.0),
                (250_000.0,),
                (0.98, 0.98),
            ),
            (
                Architecture.SINGLE_SHAFT_SINGLE_CC,
                (150_000.0, 10_000.0),
                (300_000.0,),
                (0.9,),
            ),
            (
                Architecture.SINGLE_SHAFT_SINGLE_CC,
                (150_000.0,),
                (300_000.0,),
                (0.9, 0.9),
            ),
            (
                Architecture.FREE_POWER_TURBINE,
                (90_000.0, 10_000.0),
                (130_000.0, 120_000.0),
                (0.98, 0.99),
            ),
            (
                Architecture.FREE_POWER_TURBINE,
                (90_000.0,),
                (130_000.0,),
                (0.98,),
            ),
            (
                Architecture.FREE_POWER_TURBINE,
                (90_000.0,),
                (130_000.0, 120_000.0),
                (0.98,),
            ),
        )
        for architecture, compressors, turbines, efficiencies in cases:
            with self.subTest(
                architecture=architecture,
                compressors=compressors,
                turbines=turbines,
                efficiencies=efficiencies,
            ):
                with self.assertRaises(ValueError):
                    evaluate_shaft_power_balance(
                        architecture,
                        compressor_gas_powers=compressors,
                        turbine_gas_powers=turbines,
                        mechanical_efficiencies=efficiencies,
                        required_gearbox_input_power=110_000.0,
                        gearbox_efficiency=0.98,
                    )

    def test_rejects_non_iterable_or_non_numeric_powers(self) -> None:
        cases = (
            {"compressor_gas_powers": 150_000.0},
            {"compressor_gas_powers": "150000"},
            {"turbine_gas_powers": 300_000.0},
            {"turbine_gas_powers": ("bad",)},
            {"mechanical_efficiencies": 0.9},
            {"mechanical_efficiencies": "0.9"},
            {"mechanical_efficiencies": ("bad",)},
        )
        defaults = {
            "compressor_gas_powers": (150_000.0,),
            "turbine_gas_powers": (300_000.0,),
            "mechanical_efficiencies": (0.9,),
            "required_gearbox_input_power": 110_000.0,
            "gearbox_efficiency": 0.98,
        }
        for override in cases:
            with self.subTest(override=override):
                kwargs = {**defaults, **override}
                with self.assertRaises(ValueError):
                    evaluate_shaft_power_balance(
                        Architecture.SINGLE_SHAFT_SINGLE_CC,
                        **kwargs,
                    )

    def test_rejects_non_finite_component_powers(self) -> None:
        for bad in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    single_shaft_single_cc(
                        compressor_gas_powers=(bad,)
                    )
                with self.assertRaises(ValueError):
                    evaluate_shaft_power_balance(
                        Architecture.FREE_POWER_TURBINE,
                        compressor_gas_powers=(90_000.0,),
                        turbine_gas_powers=(130_000.0, bad),
                        mechanical_efficiencies=(0.98, 0.99),
                        required_gearbox_input_power=110_000.0,
                        gearbox_efficiency=0.98,
                    )

    def test_rejects_non_finite_derived_residuals_from_finite_inputs(self) -> None:
        cases = (
            (
                Architecture.SINGLE_SHAFT_SINGLE_CC,
                dict(
                    compressor_gas_powers=(1e308,),
                    turbine_gas_powers=(0.0,),
                    mechanical_efficiencies=(1.0,),
                    required_gearbox_input_power=1e308,
                    gearbox_efficiency=1.0,
                ),
            ),
            (
                Architecture.FREE_POWER_TURBINE,
                dict(
                    compressor_gas_powers=(1e308,),
                    turbine_gas_powers=(-1e308, 0.0),
                    mechanical_efficiencies=(1.0, 1.0),
                    required_gearbox_input_power=0.0,
                    gearbox_efficiency=1.0,
                ),
            ),
        )
        for architecture, kwargs in cases:
            with self.subTest(architecture=architecture):
                with self.assertRaisesRegex(ValueError, "non-finite derived"):
                    evaluate_shaft_power_balance(architecture, **kwargs)

    def test_rejects_invalid_mechanical_and_gearbox_efficiencies(self) -> None:
        for bad in (0.0, -0.1, 1.0000001, float("nan"), float("inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    single_shaft_single_cc(
                        mechanical_efficiencies=(bad,)
                    )
                with self.assertRaises(ValueError):
                    single_shaft_single_cc(gearbox_efficiency=bad)

    def test_rejects_negative_or_non_finite_required_gearbox_input(self) -> None:
        for bad in (-1.0, float("nan"), float("inf"), "bad"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    single_shaft_single_cc(
                        required_gearbox_input_power=bad
                    )


class ResultConsistencyTests(unittest.TestCase):
    def test_derived_values_are_properties_not_forgeable_fields(self) -> None:
        result = single_shaft_single_cc()

        self.assertEqual(
            {field.name for field in fields(result)},
            {
                "architecture",
                "compressor_gas_powers",
                "turbine_gas_powers",
                "mechanical_efficiencies",
                "required_gearbox_input_power",
                "gearbox_efficiency",
            },
        )
        field_names = {field.name for field in fields(result)}
        for name in (
            "actual_gearbox_input_power",
            "gearbox_output_power",
            "main_net_power",
            "main_residual",
            "turbine_mechanical_powers",
        ):
            self.assertNotIn(name, field_names)

    def test_replace_recomputes_derived_values_consistently(self) -> None:
        result = single_shaft_single_cc()
        changed = replace(result, compressor_gas_powers=(300_000.0,))

        self.assertEqual(changed.compressor_gas_powers, (300_000.0,))
        self.assertEqual(changed.main_net_power, -30_000.0)
        self.assertEqual(changed.actual_gearbox_input_power, -30_000.0)
        self.assertEqual(changed.main_residual, -140_000.0)
        self.assertEqual(changed.gearbox_output_power, -29_400.0)

    def test_replace_cannot_bypass_input_validation(self) -> None:
        result = single_shaft_single_cc()
        with self.assertRaises(ValueError):
            replace(result, compressor_gas_powers=(float("nan"),))
        with self.assertRaises(ValueError):
            replace(result, mechanical_efficiencies=(0.0,))
        with self.assertRaises(ValueError):
            replace(result, required_gearbox_input_power=-1.0)

    def test_result_and_tuple_inputs_are_immutable(self) -> None:
        result = single_shaft_single_cc()

        self.assertIsInstance(result.compressor_gas_powers, tuple)
        self.assertIsInstance(result.turbine_gas_powers, tuple)
        self.assertIsInstance(result.mechanical_efficiencies, tuple)
        with self.assertRaises(FrozenInstanceError):
            result.compressor_gas_powers = ()  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            result.main_residual = 0.0  # type: ignore[misc]


class CycleAnchorTests(unittest.TestCase):
    def test_current_cycle_common_shaft_anchor_closes_at_110kw(self) -> None:
        config = CycleConfig()
        cycle = calculate_cycle_point(
            CycleState(pi=11.0, T_g=1250.0, N=110_000.0), config
        )
        mass_flow = cycle.G
        gas_generator_mass_fraction = cycle.g_t + cycle.g_return

        compressor_gas_power = cycle.L_k * mass_flow
        turbine_gas_power = (
            cycle.L_t * gas_generator_mass_fraction * mass_flow
        )
        result = evaluate_shaft_power_balance(
            Architecture.SINGLE_SHAFT_SINGLE_CC,
            compressor_gas_powers=(compressor_gas_power,),
            turbine_gas_powers=(turbine_gas_power,),
            mechanical_efficiencies=(config.eta_mech_turbine,),
            required_gearbox_input_power=cycle.N,
            gearbox_efficiency=config.eta_reducer,
        )

        self.assertAlmostEqual(
            result.turbine_mechanical_powers[0]
            - result.compressor_gas_powers[0],
            cycle.N,
            places=6,
        )
        self.assertAlmostEqual(
            result.actual_gearbox_input_power, 110_000.0, places=6
        )
        self.assertAlmostEqual(result.main_residual, 0.0, places=6)
        self.assertAlmostEqual(
            result.gearbox_output_power, 107_800.0, places=6
        )
        self.assertEqual(
            result.gearbox_efficiency, config.eta_reducer
        )


class PublicAndImportTests(unittest.TestCase):
    def test_package_exports_the_shaft_balance(self) -> None:
        self.assertIs(turbine_engine.ShaftPowerBalance, ShaftPowerBalance)
        self.assertIs(
            turbine_engine.evaluate_shaft_power_balance,
            evaluate_shaft_power_balance,
        )

    def test_shaft_import_does_not_load_notebook_or_optimizer_paths(self) -> None:
        code = (
            "import sys\n"
            "import turbine_engine\n"
            "from turbine_engine import Architecture\n"
            "from turbine_engine.shaft import evaluate_shaft_power_balance\n"
            "result = evaluate_shaft_power_balance(\n"
            "    Architecture.SINGLE_SHAFT_SINGLE_CC,\n"
            "    compressor_gas_powers=(150000.0,),\n"
            "    turbine_gas_powers=(300000.0,),\n"
            "    mechanical_efficiencies=(0.9,),\n"
            "    required_gearbox_input_power=110000.0,\n"
            "    gearbox_efficiency=0.98,\n"
            ")\n"
            "assert result.main_residual == 10000.0\n"
            "assert 'cycle_link' not in sys.modules, sorted(sys.modules)\n"
            "assert 'optimizer_core' not in sys.modules, sorted(sys.modules)\n"
            "print(result.gearbox_output_power)\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"stdout={completed.stdout!r}\nstderr={completed.stderr!r}",
        )

    def test_module_documents_scope_and_mass_flow_boundary(self) -> None:
        doc = shaft_module.__doc__ or ""
        lowered = doc.lower()
        self.assertIn("not wired", lowered)
        self.assertIn("mass flow is resolved", lowered)
        self.assertIn("residuals are", lowered)
        self.assertIn("single_shaft_dual_cc", lowered)
        self.assertIn("free_power_turbine", lowered)


if __name__ == "__main__":
    unittest.main()
