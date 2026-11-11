"""Integration tests for the explicit architecture-cycle RAW adapter.

These tests exercise the real 0D architecture cycles for all three topologies,
both combustor closures, strict JSON schema rejection, explicit failure
mapping, deterministic dataset recovery/reproduction, and the bounded smoke
CLI.  They are software/contract tests, not a physical validation.
"""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import math
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from turbine_engine.architecture import Architecture  # noqa: E402
from turbine_engine.architecture_cycle import (  # noqa: E402
    ARCHITECTURE_CYCLE_LIMITATIONS,
    ARCHITECTURE_CYCLE_TEXTBOOK_LIMITATIONS,
    ARCHITECTURE_GAS_PROPERTY_BASIS_TEXTBOOK,
    AlphaBasis,
    ArchitectureCycleError,
    ArchitectureCycleInput,
    ArchitectureCyclePhysicalError,
    ExternalCombustorClosure,
    ExternalCombustorState,
    LegacyCombustorClosure,
    PolynomialGasModel,
)
from turbine_engine.architecture_raw import (  # noqa: E402
    ARCHITECTURE_RAW_MODEL_VARIANT,
    ARCHITECTURE_RAW_TEXTBOOK_MODEL_VARIANT,
    COMPRESSOR_DRY_AIR_ALPHA,
    INPUT_SCHEMA_VERSION,
    INPUT_SCHEMA_VERSION_V2,
    OUTPUT_SCHEMA_VERSION,
    OUTPUT_SCHEMA_VERSION_V2,
    ArchitectureRawSchemaError,
    architecture_from_payload,
    evaluate_architecture_payload,
    evaluate_architecture_record,
    input_to_payload,
    model_variant_for_payload,
    payload_to_input,
)
from turbine_engine.constraints import (  # noqa: E402
    annotate_candidate,
    rho_065_profile,
)
from turbine_engine.cycle import (  # noqa: E402
    CycleSolverError,
    calc_cp_kerosene,
    calc_k_kerosene,
)
from turbine_engine.dataset import (  # noqa: E402
    CHECKPOINT_NAME,
    JOURNAL_NAME,
    MANIFEST_NAME,
    DatasetWriter,
    RunConfigMismatchError,
    load_manifest,
    read_records,
    reproduce_candidate,
)
from turbine_engine.diagnostics import (  # noqa: E402
    NON_PHYSICAL_STATE,
    NUMERICAL_FAILURE,
    SolverDiagnosticsError,
)
from turbine_engine.raw import (  # noqa: E402
    CalculationStatus,
    ConstraintStatus,
    canonical_json,
    evaluate_candidate,
    make_run_id,
)


GAS = PolynomialGasModel(
    R=287.4,
    l0=14.61,
    q_lhv_kj=43150.0,
    eta_combustion=0.98,
    T_ref=288.0,
    alpha_basis=AlphaBasis.FUEL_BASED,
)

TARGET = 110_000.0
GEARBOX = 0.98
P_T_OUT = 150_000.0
RUN_CONFIG = {
    "model_variant": ARCHITECTURE_RAW_MODEL_VARIANT,
    "evaluator_version": "architecture-raw-test-v1",
    "preset": "unit-test",
}
RUN_ID = make_run_id(RUN_CONFIG)


def request_for(
    architecture: Architecture,
    *,
    ratios: tuple[float, ...] | None = None,
    turbine_etas: tuple[float, ...] | None = None,
    combustor: LegacyCombustorClosure | ExternalCombustorClosure | None = None,
    gas_model: PolynomialGasModel = GAS,
    p_t_out: float = P_T_OUT,
    target: float = TARGET,
) -> ArchitectureCycleInput:
    if ratios is None:
        ratios = {
            Architecture.SINGLE_SHAFT_SINGLE_CC: (11.0,),
            Architecture.SINGLE_SHAFT_DUAL_CC: (
                math.sqrt(11.0),
                math.sqrt(11.0),
            ),
            Architecture.FREE_POWER_TURBINE: (8.0,),
        }[architecture]
    if turbine_etas is None:
        turbine_etas = {
            Architecture.SINGLE_SHAFT_SINGLE_CC: (0.90,),
            Architecture.SINGLE_SHAFT_DUAL_CC: (0.90,),
            Architecture.FREE_POWER_TURBINE: (0.90, 0.90),
        }[architecture]
    if combustor is None:
        combustor = LegacyCombustorClosure(1250.0, 0.96)
    return ArchitectureCycleInput(
        architecture=architecture,
        inlet_total_temperature=288.0,
        inlet_total_pressure=101_300.0,
        compressor_pressure_ratios=ratios,
        compressor_polytropic_efficiency=0.85,
        combustor=combustor,
        turbine_outlet_total_pressure=p_t_out,
        turbine_polytropic_efficiencies=turbine_etas,
        turbine_mechanical_efficiencies=(0.99,) * len(turbine_etas),
        target_gearbox_input_power=target,
        gearbox_efficiency=GEARBOX,
        gas_model=gas_model,
    )


def external_closure() -> ExternalCombustorClosure:
    temperature = 1200.0
    return ExternalCombustorClosure(
        ExternalCombustorState(
            T_total=temperature,
            p_total=900_000.0,
            fuel_air_ratio=0.017,
            cp=calc_cp_kerosene(temperature, 4.0),
            k=calc_k_kerosene(temperature, 4.0, GAS.R),
        )
    )


def nonphysical_request() -> ArchitectureCycleInput:
    request = request_for(
        Architecture.SINGLE_SHAFT_SINGLE_CC, ratios=(30.0,)
    )
    p_gas = request.inlet_total_pressure * 30.0 * 0.96
    return replace(
        request, turbine_outlet_total_pressure=p_gas * 0.999
    )


def payload_for(
    architecture: Architecture = Architecture.SINGLE_SHAFT_SINGLE_CC,
    **kwargs,
) -> dict:
    return input_to_payload(request_for(architecture, **kwargs))


def source_request_for(
    architecture: Architecture = Architecture.SINGLE_SHAFT_SINGLE_CC,
    **kwargs,
) -> ArchitectureCycleInput:
    return replace(
        request_for(architecture, **kwargs),
        gas_property_basis=ARCHITECTURE_GAS_PROPERTY_BASIS_TEXTBOOK,
    )


def source_payload_for(
    architecture: Architecture = Architecture.SINGLE_SHAFT_SINGLE_CC,
    **kwargs,
) -> dict:
    return input_to_payload(source_request_for(architecture, **kwargs))


def promote_source(payload: dict) -> dict:
    """Promote a legacy JSON fixture to explicit v2 without typed construction."""

    source = copy.deepcopy(payload)
    source["schema_version"] = INPUT_SCHEMA_VERSION_V2
    source["gas_property_basis"] = (
        ARCHITECTURE_GAS_PROPERTY_BASIS_TEXTBOOK
    )
    return source


class SchemaAndRoundtripTests(unittest.TestCase):
    def test_both_combustor_closures_roundtrip_for_all_architectures(self):
        for architecture in Architecture:
            for name, combustor in (
                ("legacy", LegacyCombustorClosure(1250.0, 0.96)),
                ("external", external_closure()),
            ):
                with self.subTest(architecture=architecture.value, closure=name):
                    request = request_for(architecture, combustor=combustor)
                    payload = input_to_payload(request)
                    self.assertEqual(payload_to_input(payload), request)
                    self.assertEqual(
                        input_to_payload(payload_to_input(payload)), payload
                    )
                    canonical_json(payload)

    def test_payload_shape_is_explicit_and_versioned(self):
        legacy = payload_for()
        self.assertEqual(legacy["schema_version"], INPUT_SCHEMA_VERSION)
        self.assertEqual(legacy["combustor"]["kind"], "legacy")
        self.assertEqual(
            set(legacy["gas_model"]),
            {
                "R",
                "l0",
                "q_lhv_kj",
                "eta_combustion",
                "T_ref",
                "alpha_basis",
                "model_variant",
            },
        )
        self.assertEqual(legacy["gas_model"]["alpha_basis"], "fuel_based")

        external = payload_for(combustor=external_closure())
        self.assertEqual(external["combustor"]["kind"], "external")
        self.assertEqual(
            set(external["combustor"]["state"]),
            {"T_total", "p_total", "fuel_air_ratio", "cp", "k"},
        )
        self.assertNotIn("gas_property_basis", legacy)
        self.assertEqual(
            set(legacy),
            {
                "schema_version",
                "architecture",
                "inlet_total_temperature",
                "inlet_total_pressure",
                "compressor_pressure_ratios",
                "compressor_polytropic_efficiency",
                "combustor",
                "turbine_outlet_total_pressure",
                "turbine_polytropic_efficiencies",
                "turbine_mechanical_efficiencies",
                "target_gearbox_input_power",
                "gearbox_efficiency",
                "gas_model",
            },
        )

    def test_source_v2_roundtrip_for_all_architectures(self):
        for architecture in Architecture:
            with self.subTest(architecture=architecture.value):
                request = source_request_for(architecture)
                payload = input_to_payload(request)
                self.assertEqual(
                    payload["schema_version"], INPUT_SCHEMA_VERSION_V2
                )
                self.assertEqual(
                    payload["gas_property_basis"],
                    ARCHITECTURE_GAS_PROPERTY_BASIS_TEXTBOOK,
                )
                canonical_json(payload)
                self.assertEqual(payload_to_input(payload), request)
                self.assertEqual(
                    input_to_payload(payload_to_input(payload)), payload
                )

        legacy = payload_for()
        legacy["gas_property_basis"] = (
            ARCHITECTURE_GAS_PROPERTY_BASIS_TEXTBOOK
        )
        with self.assertRaises(ArchitectureRawSchemaError):
            payload_to_input(legacy)

    def test_v2_schema_requires_a_valid_string_basis(self):
        missing = source_payload_for()
        missing.pop("gas_property_basis")
        non_string = source_payload_for()
        non_string["gas_property_basis"] = 1
        unknown = source_payload_for()
        unknown["gas_property_basis"] = "guess"
        extra = source_payload_for()
        extra["another_method"] = "textbook_true_cp"
        for name, payload in (
            ("missing", missing),
            ("non_string", non_string),
            ("unknown", unknown),
            ("extra", extra),
        ):
            with self.subTest(case=name):
                with self.assertRaises(ArchitectureRawSchemaError):
                    payload_to_input(payload)
                self.assertIs(
                    evaluate_architecture_payload(payload).status,
                    CalculationStatus.MODEL_ERROR,
                )

    def test_strict_schema_rejects_bad_inputs(self):
        base = payload_for()

        def mutated(**changes):
            value = copy.deepcopy(base)
            value.update(changes)
            return value

        cases = []
        missing = copy.deepcopy(base)
        missing.pop("gas_model")
        cases.append(("missing_top", missing))
        cases.append(("extra_top", mutated(unexpected=1)))
        cases.append(
            ("bad_architecture", mutated(architecture="TURBOFAN"))
        )
        cases.append(
            ("bool_number", mutated(inlet_total_temperature=True))
        )
        cases.append(
            ("numeric_string", mutated(inlet_total_temperature="288.0"))
        )
        cases.append(
            ("null_number", mutated(inlet_total_temperature=None))
        )
        cases.append(
            ("nonfinite_number", mutated(inlet_total_temperature=math.nan))
        )
        cases.append(
            (
                "tuple_array",
                mutated(compressor_pressure_ratios=(3.3,)),
            )
        )
        bad_gas = copy.deepcopy(base)
        bad_gas["gas_model"]["extra"] = 1
        cases.append(("extra_gas_key", bad_gas))
        bad_basis = copy.deepcopy(base)
        bad_basis["gas_model"]["alpha_basis"] = "guess"
        cases.append(("bad_alpha_basis", bad_basis))
        no_basis = copy.deepcopy(base)
        no_basis["gas_model"].pop("alpha_basis")
        cases.append(("missing_alpha_basis_no_default", no_basis))
        bad_kind = copy.deepcopy(base)
        bad_kind["combustor"]["kind"] = "magic"
        cases.append(("bad_combustor_kind", bad_kind))
        extra_legacy = copy.deepcopy(base)
        extra_legacy["combustor"]["extra"] = 1
        cases.append(("extra_legacy_key", extra_legacy))
        bad_external = payload_for(combustor=external_closure())
        bad_external["combustor"]["state"].pop("k")
        cases.append(("missing_external_state", bad_external))
        cases.append(
            ("bad_schema_version", mutated(schema_version="v0"))
        )

        for name, payload in cases:
            with self.subTest(case=name):
                with self.assertRaises(ArchitectureRawSchemaError):
                    payload_to_input(payload)

    def test_schema_reports_architecture_cardinality_mismatch(self):
        dual = payload_for(Architecture.SINGLE_SHAFT_DUAL_CC)
        dual["compressor_pressure_ratios"] = [11.0]
        with self.assertRaises(ArchitectureRawSchemaError):
            payload_to_input(dual)

        free = payload_for(Architecture.FREE_POWER_TURBINE)
        free["turbine_polytropic_efficiencies"] = [0.9]
        with self.assertRaises(ArchitectureRawSchemaError):
            payload_to_input(free)

    def test_invalid_nested_values_are_schema_errors_and_recover(self):
        invalid_r = payload_for()
        invalid_r["gas_model"]["R"] = 0.0
        with self.assertRaises(ArchitectureRawSchemaError):
            payload_to_input(invalid_r)
        outcome = evaluate_architecture_payload(invalid_r)
        self.assertIs(outcome.status, CalculationStatus.MODEL_ERROR)
        self.assertEqual(outcome.failure_stage, "schema")
        record = evaluate_candidate(
            evaluate_architecture_payload,
            invalid_r,
            architecture=Architecture.SINGLE_SHAFT_SINGLE_CC.value,
            model_variant=ARCHITECTURE_RAW_MODEL_VARIANT,
            run_id=RUN_ID,
        )
        self.assertIs(
            record.calculation_status, CalculationStatus.MODEL_ERROR
        )
        self.assertEqual(record.failure_stage, "schema")

        invalid_legacy = payload_for()
        invalid_legacy["combustor"]["total_pressure_recovery"] = 1.5
        with self.assertRaises(ArchitectureRawSchemaError):
            payload_to_input(invalid_legacy)

        invalid_external = payload_for(combustor=external_closure())
        invalid_external["combustor"]["state"]["k"] = 1.0
        with self.assertRaises(ArchitectureRawSchemaError):
            payload_to_input(invalid_external)

        recovered = payload_for()
        self.assertEqual(
            payload_to_input(recovered),
            request_for(Architecture.SINGLE_SHAFT_SINGLE_CC),
        )
        self.assertIs(
            evaluate_architecture_payload(recovered).status,
            CalculationStatus.SUCCESS,
        )

    def test_number_conversion_is_exact_and_overflow_is_schema_error(self):
        exact = payload_for()
        exact["inlet_total_temperature"] = 2**53
        parsed = payload_to_input(exact)
        self.assertEqual(parsed.inlet_total_temperature, float(2**53))

        inexact = payload_for()
        inexact["inlet_total_temperature"] = 2**53 + 1
        with self.assertRaises(ArchitectureRawSchemaError):
            payload_to_input(inexact)
        outcome = evaluate_architecture_payload(inexact)
        self.assertIs(outcome.status, CalculationStatus.MODEL_ERROR)
        self.assertEqual(outcome.failure_stage, "schema")

        huge = payload_for()
        huge["inlet_total_temperature"] = 10**400
        with self.assertRaises(ArchitectureRawSchemaError) as context:
            payload_to_input(huge)
        self.assertIn("too large", str(context.exception))

        huge_array = payload_for()
        huge_array["compressor_pressure_ratios"] = [10**400]
        with self.assertRaises(ArchitectureRawSchemaError):
            payload_to_input(huge_array)

    def test_architecture_from_payload(self):
        payload = payload_for(Architecture.FREE_POWER_TURBINE)
        self.assertIs(
            architecture_from_payload(payload),
            Architecture.FREE_POWER_TURBINE,
        )
        with self.assertRaises(ArchitectureRawSchemaError):
            architecture_from_payload({"schema_version": INPUT_SCHEMA_VERSION})
        with self.assertRaises(ArchitectureRawSchemaError):
            architecture_from_payload({"architecture": "UNKNOWN"})


class EvaluatorTests(unittest.TestCase):
    def test_all_three_architectures_return_complete_causal_state(self):
        for architecture in Architecture:
            with self.subTest(architecture=architecture.value):
                outcome = evaluate_architecture_payload(
                    payload_for(architecture)
                )
                self.assertIs(outcome.status, CalculationStatus.SUCCESS)
                self.assertIsNone(outcome.failure_stage)
                outputs = outcome.outputs
                intermediates = outcome.intermediates
                self.assertIsNotNone(outputs)
                self.assertIsNotNone(intermediates)
                canonical_json(outputs)
                canonical_json(intermediates)

                self.assertEqual(
                    outputs["schema_version"], OUTPUT_SCHEMA_VERSION
                )
                self.assertEqual(
                    outputs["architecture"], architecture.value
                )
                self.assertAlmostEqual(
                    outputs["actual_gearbox_input_power"], TARGET, delta=1e-6
                )
                self.assertAlmostEqual(
                    outputs["gearbox_output_power"],
                    TARGET * GEARBOX,
                    delta=1e-6,
                )
                self.assertIn(
                    "NOT_PHYSICALLY_VALIDATED", outputs["validation_scope"]
                )
                self.assertIn(
                    "NOT_PHYSICALLY_VALIDATED", outputs["limitations"]
                )
                self.assertTrue(
                    any(
                        marker.startswith("MISSING_COMPRESSOR_PHYSICS")
                        for marker in outputs["validation_scope"]
                    )
                )
                self.assertEqual(
                    outputs["compressor"]["outlet"],
                    intermediates["compressor"]["outlet"],
                )
                self.assertIn(
                    "actual_gearbox_input_power", outputs["shaft"]
                )
                self.assertEqual(
                    outputs["gas_model"]["alpha_basis"], "fuel_based"
                )

                residuals = outputs["residuals"]
                self.assertIn("compressor_pressure", residuals)
                self.assertIn("mass_balance", residuals)
                self.assertIn("target_power", residuals)
                self.assertIn("turbine_0_pressure", residuals)
                for value in residuals.values():
                    self.assertTrue(math.isfinite(value))

                compressor = intermediates["compressor"]
                self.assertEqual(
                    compressor["outlet"], compressor["station_states"][-1]
                )
                self.assertEqual(
                    compressor["inlet"]["mass_flow"],
                    outputs["air_mass_flow"],
                )
                self.assertEqual(
                    intermediates["combustor"]["outlet"]["mass_flow"],
                    outputs["gas_mass_flow"],
                )
                self.assertEqual(
                    intermediates["turbines"][0]["inlet"]["mass_flow"],
                    outputs["gas_mass_flow"],
                )
                self.assertEqual(
                    len(intermediates["turbines"]),
                    len(intermediates["shaft"]["turbine_gas_powers"]),
                )

                # These are derived properties; dataclasses.asdict() omits them.
                shaft = intermediates["shaft"]
                for key in (
                    "turbine_mechanical_powers",
                    "compressor_gas_power_total",
                    "turbine_mechanical_power_total",
                    "actual_gearbox_input_power",
                    "gearbox_output_power",
                ):
                    self.assertIn(key, shaft)
                self.assertAlmostEqual(
                    shaft["actual_gearbox_input_power"],
                    outputs["actual_gearbox_input_power"],
                    delta=1e-12,
                )

                roles = intermediates["turbine_roles"]
                if architecture is Architecture.FREE_POWER_TURBINE:
                    self.assertEqual(
                        roles,
                        {
                            "gas_generator": 0,
                            "free_power": 1,
                            "main": None,
                        },
                    )
                    self.assertIn("shaft_gg", residuals)
                    self.assertIn("shaft_fpt", residuals)
                    self.assertIsNotNone(shaft["gas_generator_residual"])
                    self.assertIsNotNone(
                        shaft["free_power_turbine_residual"]
                    )
                else:
                    self.assertEqual(
                        roles,
                        {
                            "gas_generator": None,
                            "free_power": None,
                            "main": 0,
                        },
                    )
                    self.assertIn("shaft_main", residuals)
                    self.assertIsNotNone(shaft["main_residual"])

    def test_source_outputs_are_v2_and_carry_complete_process_metadata(self):
        for architecture in Architecture:
            with self.subTest(architecture=architecture.value):
                outcome = evaluate_architecture_payload(
                    source_payload_for(architecture)
                )
                self.assertIs(outcome.status, CalculationStatus.SUCCESS)
                outputs = outcome.outputs
                intermediates = outcome.intermediates
                canonical_json(outputs)
                canonical_json(intermediates)
                for record in (outputs, intermediates):
                    self.assertEqual(
                        record["schema_version"], OUTPUT_SCHEMA_VERSION_V2
                    )
                    self.assertEqual(
                        record["gas_property_basis"],
                        ARCHITECTURE_GAS_PROPERTY_BASIS_TEXTBOOK,
                    )
                    self.assertEqual(
                        record["validation_scope"],
                        list(ARCHITECTURE_CYCLE_TEXTBOOK_LIMITATIONS),
                    )
                self.assertEqual(
                    outcome.diagnostics["input_schema_version"],
                    INPUT_SCHEMA_VERSION_V2,
                )
                self.assertEqual(
                    outcome.diagnostics["schema_version"],
                    OUTPUT_SCHEMA_VERSION_V2,
                )
                self.assertEqual(
                    outputs["model_variant"],
                    "uncooled_no_bleed_0d_textbook_process_mean",
                )

                compressor = outputs["compressor"]
                self.assertEqual(
                    len(compressor["process_properties"]),
                    len(compressor["station_states"]),
                )
                for process in compressor["process_properties"]:
                    self.assertEqual(
                        set(process),
                        {
                            "T_in",
                            "T_out",
                            "T_mean",
                            "cp",
                            "k",
                            "pressure_ratio",
                            "eta_polytropic",
                            "gas_constant_R",
                            "alpha",
                            "method",
                            "fixed_point_residual_K",
                            "tolerance_K",
                        },
                    )
                    self.assertEqual(
                        process["alpha"], COMPRESSOR_DRY_AIR_ALPHA
                    )
                    self.assertEqual(process["alpha"], "dry_air")
                    for value in process.values():
                        if isinstance(value, float):
                            self.assertTrue(math.isfinite(value))

                for turbine in outputs["turbines"]:
                    process = turbine["process_properties"]
                    self.assertIsNotNone(process)
                    self.assertEqual(
                        set(process),
                        {
                            "T_in",
                            "T_out",
                            "T_mean",
                            "cp",
                            "k",
                            "alpha",
                            "gas_constant_R",
                            "method",
                            "tolerance_K",
                            "temperature_residual_K",
                            "pressure_formula_temperature_residual_K",
                            "requested_specific_work",
                            "achieved_specific_work",
                            "work_relative_residual",
                        },
                    )
                    self.assertTrue(math.isfinite(process["alpha"]))
                    work = (
                        process["requested_specific_work"],
                        process["achieved_specific_work"],
                        process["work_relative_residual"],
                    )
                    if work[0] is None:
                        self.assertEqual(work, (None, None, None))
                    else:
                        self.assertGreater(work[0], 0.0)
                        self.assertTrue(math.isfinite(work[1]))
                        self.assertTrue(math.isfinite(work[2]))

    def test_legacy_and_external_closures_are_preserved(self):
        legacy = evaluate_architecture_payload(payload_for())
        self.assertEqual(legacy.outputs["combustor_closure"],
                         "legacy_calc_g_m_no_bleed")
        self.assertIsNotNone(legacy.intermediates["combustor"]["alpha"])
        self.assertEqual(
            legacy.intermediates["combustor"]["alpha_basis"],
            "fuel_based",
        )

        external = evaluate_architecture_payload(
            payload_for(combustor=external_closure())
        )
        self.assertIs(external.status, CalculationStatus.SUCCESS)
        self.assertEqual(
            external.outputs["combustor_closure"], "external_state"
        )
        self.assertEqual(
            external.intermediates["combustor"]["closure"],
            "external_state",
        )
        self.assertIsNone(external.intermediates["combustor"]["alpha"])
        self.assertIsNone(external.intermediates["combustor"]["heat_input"])
        self.assertEqual(
            external.intermediates["combustor"]["outlet"]["T_total"],
            1200.0,
        )
        self.assertEqual(
            external.intermediates["combustor"]["outlet"]["p_total"],
            900_000.0,
        )

    def test_empirical_alpha_basis_is_explicit(self):
        empirical = replace(
            GAS, alpha_basis=AlphaBasis.EMPIRICAL_POLYNOMIAL
        )
        outcome = evaluate_architecture_payload(
            payload_for(gas_model=empirical)
        )
        self.assertIs(outcome.status, CalculationStatus.SUCCESS)
        self.assertEqual(
            outcome.intermediates["gas_model"]["alpha_basis"],
            "empirical_polynomial",
        )
        self.assertEqual(
            outcome.intermediates["combustor"]["alpha_basis"],
            "empirical_polynomial",
        )

    def test_incompatible_source_fixtures_are_model_error_with_payload(self):
        external = promote_source(
            payload_for(combustor=external_closure())
        )
        empirical = promote_source(
            payload_for(
                gas_model=replace(
                    GAS, alpha_basis=AlphaBasis.EMPIRICAL_POLYNOMIAL
                )
            )
        )
        for name, payload in (
            ("external", external),
            ("empirical", empirical),
        ):
            with self.subTest(case=name):
                self.assertEqual(
                    model_variant_for_payload(payload),
                    ARCHITECTURE_RAW_TEXTBOOK_MODEL_VARIANT,
                )
                outcome = evaluate_architecture_payload(payload)
                self.assertIs(
                    outcome.status, CalculationStatus.MODEL_ERROR
                )
                self.assertEqual(outcome.failure_stage, "schema")
                self.assertEqual(
                    outcome.last_valid_state["input"], payload
                )
                self.assertEqual(
                    outcome.diagnostics["input_schema_version"],
                    INPUT_SCHEMA_VERSION_V2,
                )

    def test_model_variant_derivation_and_source_record_label_guard(self):
        self.assertEqual(
            model_variant_for_payload(payload_for()),
            ARCHITECTURE_RAW_MODEL_VARIANT,
        )
        source_payload = source_payload_for()
        self.assertEqual(
            model_variant_for_payload(source_payload),
            ARCHITECTURE_RAW_TEXTBOOK_MODEL_VARIANT,
        )

        wrong = evaluate_candidate(
            evaluate_architecture_payload,
            source_payload,
            architecture=Architecture.SINGLE_SHAFT_SINGLE_CC.value,
            model_variant=ARCHITECTURE_RAW_MODEL_VARIANT,
            run_id=RUN_ID,
        )
        outcome = evaluate_architecture_record(wrong)
        self.assertIs(outcome.status, CalculationStatus.MODEL_ERROR)
        self.assertEqual(
            outcome.failure_stage, "model_variant_label_mismatch"
        )
        self.assertEqual(
            outcome.diagnostics["payload_model_variant"],
            ARCHITECTURE_RAW_TEXTBOOK_MODEL_VARIANT,
        )

        correct = evaluate_candidate(
            evaluate_architecture_payload,
            source_payload,
            architecture=Architecture.SINGLE_SHAFT_SINGLE_CC.value,
            model_variant=ARCHITECTURE_RAW_TEXTBOOK_MODEL_VARIANT,
            run_id=RUN_ID,
        )
        self.assertIs(
            evaluate_architecture_record(correct).status,
            CalculationStatus.SUCCESS,
        )

    def test_source_numerical_failure_keeps_status_and_schema(self):
        payload = source_payload_for()
        with mock.patch(
            "turbine_engine.architecture_raw.calculate_architecture_cycle",
            side_effect=CycleSolverError("source iteration did not close"),
        ):
            outcome = evaluate_architecture_payload(payload)
        self.assertIs(
            outcome.status, CalculationStatus.NUMERICAL_FAILURE
        )
        self.assertEqual(outcome.failure_stage, "cycle_solver")
        self.assertEqual(
            outcome.diagnostics["input_schema_version"],
            INPUT_SCHEMA_VERSION_V2,
        )
        self.assertEqual(
            outcome.last_valid_state["input"], payload
        )

    def test_nonphysical_point_maps_to_non_physical_state(self):
        payload = input_to_payload(nonphysical_request())
        outcome = evaluate_architecture_payload(payload)
        self.assertIs(
            outcome.status, CalculationStatus.NON_PHYSICAL_STATE
        )
        self.assertEqual(
            outcome.failure_stage, "architecture_cycle_physical"
        )
        self.assertIn("input", outcome.last_valid_state)
        self.assertEqual(
            outcome.last_valid_state["input"]["architecture"],
            Architecture.SINGLE_SHAFT_SINGLE_CC.value,
        )

        low_combustor = evaluate_architecture_payload(
            payload_for(
                combustor=LegacyCombustorClosure(600.0, 0.96)
            )
        )
        self.assertIs(
            low_combustor.status, CalculationStatus.NON_PHYSICAL_STATE
        )

    def test_invalid_schema_maps_to_model_error(self):
        payload = payload_for()
        payload["inlet_total_temperature"] = "288"
        outcome = evaluate_architecture_payload(payload)
        self.assertIs(outcome.status, CalculationStatus.MODEL_ERROR)
        self.assertEqual(outcome.failure_stage, "schema")

    def test_typed_numerical_failures_keep_structured_state(self):
        payload = payload_for()
        with mock.patch(
            "turbine_engine.architecture_raw.calculate_architecture_cycle",
            side_effect=CycleSolverError("compressor iteration did not close"),
        ):
            outcome = evaluate_architecture_payload(payload)
        self.assertIs(
            outcome.status, CalculationStatus.NUMERICAL_FAILURE
        )
        self.assertEqual(outcome.failure_stage, "cycle_solver")
        self.assertIn("input", outcome.last_valid_state)

        solver_error = SolverDiagnosticsError(
            "no valid stage evaluation",
            stage="baseline.распределить_π",
            reason="pi_secant_iteration_exhausted",
            last_valid_state={"pi": 8.0, "temperature": math.nan},
        )
        with mock.patch(
            "turbine_engine.architecture_raw.calculate_architecture_cycle",
            side_effect=solver_error,
        ):
            outcome = evaluate_architecture_payload(payload)
        self.assertIs(
            outcome.status, CalculationStatus.NUMERICAL_FAILURE
        )
        self.assertEqual(
            outcome.failure_stage, "baseline.распределить_π"
        )
        self.assertEqual(
            outcome.failure_reason, "pi_secant_iteration_exhausted"
        )
        self.assertEqual(
            outcome.last_valid_state["solver_last_valid_state"][
                "temperature"
            ],
            None,
        )
        self.assertEqual(
            outcome.diagnostics["solver"]["reason"],
            "pi_secant_iteration_exhausted",
        )

        with mock.patch(
            "turbine_engine.architecture_raw.calculate_architecture_cycle",
            side_effect=ArchitectureCyclePhysicalError("no positive power"),
        ):
            outcome = evaluate_architecture_payload(payload)
        self.assertIs(
            outcome.status, CalculationStatus.NON_PHYSICAL_STATE
        )

        with mock.patch(
            "turbine_engine.architecture_raw.calculate_architecture_cycle",
            side_effect=ArchitectureCycleError("bad configuration"),
        ):
            outcome = evaluate_architecture_payload(payload)
        self.assertIs(outcome.status, CalculationStatus.MODEL_ERROR)
        self.assertEqual(
            outcome.failure_stage, "architecture_cycle_config"
        )

    def test_solver_diagnostics_status_is_preserved(self):
        payload = payload_for()
        cases = (
            (NUMERICAL_FAILURE, CalculationStatus.NUMERICAL_FAILURE),
            (NON_PHYSICAL_STATE, CalculationStatus.NON_PHYSICAL_STATE),
        )
        for declared, expected in cases:
            with self.subTest(declared=declared):
                error = SolverDiagnosticsError(
                    "typed solver outcome",
                    stage="solver.stage",
                    reason="typed_reason",
                    status=declared,
                    last_valid_state={"x": 1.0},
                )
                with mock.patch(
                    "turbine_engine.architecture_raw."
                    "calculate_architecture_cycle",
                    side_effect=error,
                ):
                    outcome = evaluate_architecture_payload(payload)
                self.assertIs(outcome.status, expected)
                self.assertEqual(outcome.failure_stage, "solver.stage")
                self.assertEqual(outcome.failure_reason, "typed_reason")
                self.assertEqual(
                    outcome.diagnostics["solver"]["status"], declared
                )

    def test_actual_path_typed_failures_keep_numeric_status(self):
        payload = payload_for()
        cycle_error = CycleSolverError("calc_g_m denominator is zero")
        with mock.patch.object(
            PolynomialGasModel, "fuel_air_ratio", side_effect=cycle_error
        ):
            outcome = evaluate_architecture_payload(payload)
        self.assertIs(
            outcome.status, CalculationStatus.NUMERICAL_FAILURE
        )
        self.assertEqual(outcome.failure_stage, "cycle_solver")
        self.assertEqual(
            outcome.failure_reason, "calc_g_m denominator is zero"
        )
        self.assertIn("input", outcome.last_valid_state)
        self.assertEqual(
            outcome.last_valid_state["input"]["architecture"],
            Architecture.SINGLE_SHAFT_SINGLE_CC.value,
        )

        solver_error = SolverDiagnosticsError(
            "gas-property solve did not close",
            stage="architecture_cycle.gas_cp_k",
            reason="gas_property_iteration_exhausted",
            last_valid_state={"T_g": 1250.0, "alpha": 4.0},
        )
        with mock.patch.object(
            PolynomialGasModel, "gas_cp_k", side_effect=solver_error
        ):
            outcome = evaluate_architecture_payload(payload)
        self.assertIs(
            outcome.status, CalculationStatus.NUMERICAL_FAILURE
        )
        self.assertEqual(
            outcome.failure_stage, "architecture_cycle.gas_cp_k"
        )
        self.assertEqual(
            outcome.failure_reason, "gas_property_iteration_exhausted"
        )
        self.assertEqual(
            outcome.last_valid_state["solver_last_valid_state"],
            {"T_g": 1250.0, "alpha": 4.0},
        )
        self.assertEqual(
            outcome.diagnostics["solver"]["status"], NUMERICAL_FAILURE
        )

    def test_actual_path_unexpected_runtime_error_is_not_nonphysical(self):
        payload = payload_for()
        with mock.patch.object(
            PolynomialGasModel,
            "fuel_air_ratio",
            side_effect=RuntimeError("unexpected implementation failure"),
        ):
            with self.assertRaises(RuntimeError):
                evaluate_architecture_payload(payload)
            record = evaluate_candidate(
                evaluate_architecture_payload,
                payload,
                architecture=Architecture.SINGLE_SHAFT_SINGLE_CC.value,
                model_variant=ARCHITECTURE_RAW_MODEL_VARIANT,
                run_id=RUN_ID,
            )
        self.assertIs(
            record.calculation_status, CalculationStatus.MODEL_ERROR
        )
        self.assertEqual(record.failure_stage, "evaluator_exception")
        self.assertEqual(
            record.diagnostics["exception_type"], "RuntimeError"
        )

    def test_unexpected_error_is_left_visible_to_generic_raw(self):
        payload = payload_for()
        with mock.patch(
            "turbine_engine.architecture_raw.calculate_architecture_cycle",
            side_effect=ValueError("unexpected adapter bug"),
        ):
            with self.assertRaises(ValueError):
                evaluate_architecture_payload(payload)
            record = evaluate_candidate(
                evaluate_architecture_payload,
                payload,
                architecture=Architecture.SINGLE_SHAFT_SINGLE_CC.value,
                model_variant=ARCHITECTURE_RAW_MODEL_VARIANT,
                run_id=RUN_ID,
            )
        self.assertIs(
            record.calculation_status, CalculationStatus.MODEL_ERROR
        )
        self.assertEqual(record.failure_stage, "evaluator_exception")
        self.assertEqual(
            record.diagnostics["exception_type"], "ValueError"
        )

    def test_architecture_label_mismatch_is_model_error(self):
        payload = payload_for(Architecture.SINGLE_SHAFT_SINGLE_CC)
        record = evaluate_candidate(
            evaluate_architecture_payload,
            payload,
            architecture=Architecture.FREE_POWER_TURBINE.value,
            model_variant=ARCHITECTURE_RAW_MODEL_VARIANT,
            run_id=RUN_ID,
        )
        outcome = evaluate_architecture_record(record)
        self.assertIs(outcome.status, CalculationStatus.MODEL_ERROR)
        self.assertEqual(
            outcome.failure_stage, "architecture_label_mismatch"
        )
        self.assertEqual(
            outcome.diagnostics["payload_architecture"],
            Architecture.SINGLE_SHAFT_SINGLE_CC.value,
        )

        correct = evaluate_candidate(
            evaluate_architecture_payload,
            payload,
            architecture=Architecture.SINGLE_SHAFT_SINGLE_CC.value,
            model_variant=ARCHITECTURE_RAW_MODEL_VARIANT,
            run_id=RUN_ID,
        )
        self.assertIs(
            evaluate_architecture_record(correct).status,
            CalculationStatus.SUCCESS,
        )

    def test_original_payload_is_preserved(self):
        payload = payload_for(Architecture.SINGLE_SHAFT_DUAL_CC)
        original = copy.deepcopy(payload)
        record = evaluate_candidate(
            evaluate_architecture_payload,
            payload,
            architecture=Architecture.SINGLE_SHAFT_DUAL_CC.value,
            model_variant=ARCHITECTURE_RAW_MODEL_VARIANT,
            run_id=RUN_ID,
        )
        self.assertEqual(record.payload, original)
        self.assertEqual(payload, original)
        self.assertEqual(record.outputs["architecture"],
                         Architecture.SINGLE_SHAFT_DUAL_CC.value)


class DatasetIntegrationTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.root = Path(self._temporary.name)
        self.payloads = [
            payload_for(Architecture.SINGLE_SHAFT_SINGLE_CC),
            payload_for(
                Architecture.SINGLE_SHAFT_SINGLE_CC,
                combustor=external_closure(),
            ),
            payload_for(Architecture.SINGLE_SHAFT_DUAL_CC),
            payload_for(Architecture.FREE_POWER_TURBINE),
            input_to_payload(nonphysical_request()),
        ]

    def _append_all(self, writer, payloads, start_index=0):
        for offset, payload in enumerate(payloads):
            writer.evaluate_and_append(
                evaluate_architecture_payload,
                payload,
                architecture=architecture_from_payload(payload).value,
                sampling_method="fixture",
                sampling_metadata={"index": start_index + offset},
            )

    def test_interrupted_resume_matches_uninterrupted(self):
        uninterrupted = self.root / "uninterrupted"
        writer = DatasetWriter.create(
            uninterrupted, run_config=RUN_CONFIG
        )
        self._append_all(writer, self.payloads)
        writer.close()

        interrupted = self.root / "interrupted"
        writer = DatasetWriter.create(interrupted, run_config=RUN_CONFIG)
        self._append_all(writer, self.payloads[:2])
        saved_manifest = (interrupted / MANIFEST_NAME).read_bytes()
        saved_checkpoint = (interrupted / CHECKPOINT_NAME).read_bytes()
        self._append_all(writer, self.payloads[2:3], start_index=2)
        # Simulate a crash after the journal line but before cursor commits.
        (interrupted / MANIFEST_NAME).write_bytes(saved_manifest)
        (interrupted / CHECKPOINT_NAME).write_bytes(saved_checkpoint)
        writer.close(finalize=False)

        resumed = DatasetWriter.open(
            interrupted, expected_run_config=RUN_CONFIG
        )
        self.assertEqual(resumed.record_count, 3)
        self._append_all(resumed, self.payloads[3:], start_index=3)
        resumed.close()

        for name in (JOURNAL_NAME, MANIFEST_NAME, CHECKPOINT_NAME):
            self.assertEqual(
                (interrupted / name).read_bytes(),
                (uninterrupted / name).read_bytes(),
                name,
            )
        self.assertEqual(
            [record.to_dict() for record in read_records(interrupted)],
            [record.to_dict() for record in read_records(uninterrupted)],
        )

    def test_every_stored_record_reproduces(self):
        destination = self.root / "reproduce"
        writer = DatasetWriter.create(
            destination, run_config=RUN_CONFIG
        )
        self._append_all(writer, self.payloads)
        writer.close()

        records = read_records(destination)
        self.assertEqual(len(records), len(self.payloads))
        for record in records:
            with self.subTest(candidate=record.candidate_id):
                result = reproduce_candidate(
                    destination,
                    record.candidate_id,
                    evaluate_architecture_payload,
                    rtol=1e-12,
                    atol=1e-9,
                )
                self.assertTrue(result.matches, result.differences)

    def test_missing_rho_annotation_is_not_checked_and_raw_is_unchanged(self):
        destination = self.root / "rho"
        writer = DatasetWriter.create(
            destination, run_config=RUN_CONFIG
        )
        self._append_all(writer, self.payloads[:2])
        writer.close()

        records = read_records(destination)
        before = [record.to_dict() for record in records]
        annotations = [
            annotate_candidate(record, rho_065_profile())
            for record in records
        ]
        for annotation in annotations:
            self.assertIs(
                annotation.constraint_status, ConstraintStatus.NOT_CHECKED
            )
            self.assertIsNone(annotation.feasible)
            self.assertEqual(len(annotation.checks), 1)
            self.assertFalse(annotation.checks[0].checked)
            self.assertEqual(annotation.checks[0].reason, "missing")
        self.assertEqual(
            [record.to_dict() for record in records], before
        )
        self.assertTrue(
            all(
                record.calculation_status is CalculationStatus.SUCCESS
                for record in records
            )
        )

    def test_resume_rejects_incompatible_configuration(self):
        destination = self.root / "mismatch"
        writer = DatasetWriter.create(
            destination, run_config=RUN_CONFIG
        )
        writer.close()
        with self.assertRaises(RunConfigMismatchError):
            DatasetWriter.open(
                destination,
                expected_run_config={
                    "model_variant": "different-variant",
                    "evaluator_version": "architecture-raw-test-v1",
                    "preset": "unit-test",
                },
            )
        with self.assertRaises(RunConfigMismatchError):
            DatasetWriter.open(
                destination,
                expected_run_config=RUN_CONFIG,
                expected_evaluator_version="different-version",
            )


class SmokeCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "scripts" / "architecture_smoke.py"
        spec = importlib.util.spec_from_file_location(
            "architecture_smoke", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.smoke = module

    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.root = Path(self._temporary.name)

    def test_help_exits_zero(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as context:
                self.smoke.main(["--help"])
        self.assertEqual(context.exception.code, 0)

    def test_manual_run_is_bounded_deterministic_and_resumable(self):
        destination = self.root / "manual"
        summary = self.smoke.run_smoke(destination, preset="manual")
        self.assertIs(summary["physically_validated"], False)
        self.assertEqual(
            summary["validation_status"], "NOT_PHYSICALLY_VALIDATED"
        )
        self.assertEqual(
            summary["stage_rho"]["status"], "NOT_CHECKED"
        )
        self.assertEqual(
            summary["rho_annotation"]["status"], "NOT_CHECKED"
        )
        self.assertEqual(summary["points_attempted"], 13)
        self.assertEqual(summary["record_count"], 13)
        self.assertEqual(summary["records_appended"], 13)
        self.assertGreaterEqual(
            summary["status_counts"].get("SUCCESS", 0), 1
        )
        self.assertGreaterEqual(
            summary["status_counts"].get("NON_PHYSICAL_STATE", 0), 1
        )
        self.assertTrue(summary["reproduce"]["matches"])
        self.assertEqual(
            summary["journal_size_bytes"],
            (destination / JOURNAL_NAME).stat().st_size,
        )
        journal = (destination / JOURNAL_NAME).read_bytes()
        manifest = (destination / MANIFEST_NAME).read_bytes()

        with self.assertRaises(self.smoke.ArchitectureSmokeError):
            self.smoke.run_smoke(destination, preset="manual")

        resumed = self.smoke.run_smoke(
            destination, preset="manual", resume=True
        )
        self.assertEqual(resumed["records_appended"], 0)
        self.assertEqual(resumed["record_count"], 13)
        self.assertEqual(
            (destination / JOURNAL_NAME).read_bytes(), journal
        )
        self.assertEqual(
            (destination / MANIFEST_NAME).read_bytes(), manifest
        )

        with self.assertRaises(RunConfigMismatchError):
            self.smoke.run_smoke(
                destination,
                preset="manual",
                resume=True,
                evaluator_version="incompatible",
            )

    def test_run_config_contains_fixture_plan_identity(self):
        payloads = self.smoke.manual_payloads()
        config = self.smoke.make_run_config("manual")
        self.assertEqual(config["payload_plan_count"], len(payloads))
        self.assertEqual(
            config["payload_plan_sha256"],
            self.smoke.canonical_hash(payloads),
        )
        self.assertEqual(
            config,
            self.smoke.make_run_config("manual", payloads=payloads),
        )

    def test_resume_rejects_changed_fixture_plan_same_version(self):
        destination = self.root / "changed-plan"
        self.smoke.run_smoke(destination, preset="manual")

        original = self.smoke.payloads_for_preset

        def changed_plan(preset):
            payloads = original(preset)
            payloads[0] = copy.deepcopy(payloads[0])
            payloads[0]["target_gearbox_input_power"] = 109_000.0
            return payloads

        with mock.patch.object(
            self.smoke, "payloads_for_preset", new=changed_plan
        ):
            with self.assertRaises(RunConfigMismatchError):
                self.smoke.run_smoke(
                    destination, preset="manual", resume=True
                )

    def test_interrupted_run_leaves_incomplete_manifest_and_resumes(self):
        for exc_type in (KeyboardInterrupt, RuntimeError):
            with self.subTest(exc=exc_type.__name__):
                interrupted = self.root / f"interrupted-{exc_type.__name__}"
                uninterrupted = (
                    self.root / f"uninterrupted-{exc_type.__name__}"
                )
                original = self.smoke.architecture_from_payload
                calls = {"count": 0}

                def interrupted_lookup(payload):
                    if calls["count"] == 3:
                        raise exc_type("simulated interruption")
                    calls["count"] += 1
                    return original(payload)

                with mock.patch.object(
                    self.smoke,
                    "architecture_from_payload",
                    new=interrupted_lookup,
                ):
                    with self.assertRaises(exc_type):
                        self.smoke.run_smoke(
                            interrupted, preset="manual"
                        )

                manifest = load_manifest(interrupted)
                self.assertFalse(manifest["complete"])
                self.assertEqual(manifest["record_count"], 3)
                self.assertEqual(
                    len(
                        (interrupted / JOURNAL_NAME)
                        .read_bytes()
                        .splitlines()
                    ),
                    3,
                )

                self.smoke.run_smoke(
                    interrupted, preset="manual", resume=True
                )
                self.smoke.run_smoke(uninterrupted, preset="manual")
                for name in (JOURNAL_NAME, MANIFEST_NAME, CHECKPOINT_NAME):
                    self.assertEqual(
                        (interrupted / name).read_bytes(),
                        (uninterrupted / name).read_bytes(),
                        name,
                    )

    def test_grid_preset_covers_all_architectures(self):
        summary = self.smoke.run_smoke(
            self.root / "grid", preset="grid"
        )
        self.assertEqual(summary["points_attempted"], 13)
        self.assertEqual(
            {record["architecture"] for record in summary["records"]},
            {architecture.value for architecture in Architecture},
        )
        self.assertGreaterEqual(
            summary["status_counts"].get("NON_PHYSICAL_STATE", 0), 1
        )
        self.assertTrue(summary["reproduce"]["matches"])

    def test_source_cli_manual_run_keeps_all_points_and_actual_method(self):
        destination = self.root / "source-cli"
        with contextlib.redirect_stdout(io.StringIO()):
            code = self.smoke.main(
                [
                    "--destination",
                    str(destination),
                    "--preset",
                    "manual",
                    "--gas-property-basis",
                    "textbook_true_cp",
                    "--quiet",
                ]
            )
        self.assertEqual(code, 0)

        summary = self.smoke.run_smoke(
            destination,
            preset="manual",
            resume=True,
            gas_property_basis="textbook_true_cp",
        )
        self.assertEqual(summary["records_appended"], 0)
        self.assertEqual(summary["points_attempted"], 13)
        self.assertEqual(summary["record_count"], 13)
        self.assertEqual(summary["gas_property_basis"], "textbook_true_cp")
        self.assertEqual(
            summary["model_variant"],
            ARCHITECTURE_RAW_TEXTBOOK_MODEL_VARIANT,
        )
        self.assertEqual(
            summary["validation_scope"],
            list(ARCHITECTURE_CYCLE_TEXTBOOK_LIMITATIONS),
        )
        # Measured fixture outcome, not an invented success claim: the four
        # external/empirical fixtures cannot use the textbook source method.
        self.assertEqual(
            summary["status_counts"],
            {"MODEL_ERROR": 4, "NON_PHYSICAL_STATE": 2, "SUCCESS": 7},
        )
        self.assertEqual(
            sum(summary["status_counts"].values()),
            summary["points_attempted"],
        )
        self.assertTrue(summary["reproduce"]["matches"])
        self.assertEqual(summary["stage_rho"]["status"], "NOT_CHECKED")
        self.assertEqual(
            summary["rho_annotation"]["status"], "NOT_CHECKED"
        )

        records = read_records(destination)
        self.assertEqual(len(records), 13)
        self.assertEqual(
            {record.architecture for record in records},
            {architecture.value for architecture in Architecture},
        )
        self.assertEqual(
            {record.model_variant for record in records},
            {ARCHITECTURE_RAW_TEXTBOOK_MODEL_VARIANT},
        )
        for record in records:
            if record.calculation_status is CalculationStatus.SUCCESS:
                self.assertEqual(
                    record.outputs["schema_version"],
                    OUTPUT_SCHEMA_VERSION_V2,
                )
                self.assertEqual(
                    record.outputs["gas_property_basis"],
                    ARCHITECTURE_GAS_PROPERTY_BASIS_TEXTBOOK,
                )
            elif record.calculation_status is CalculationStatus.MODEL_ERROR:
                self.assertEqual(record.failure_stage, "schema")
                self.assertIsNotNone(record.last_valid_state)

        with self.assertRaises(RunConfigMismatchError):
            self.smoke.run_smoke(
                destination, preset="manual", resume=True
            )

    def test_source_interrupted_resume_matches_uninterrupted(self):
        interrupted = self.root / "source-interrupted"
        uninterrupted = self.root / "source-uninterrupted"
        original = self.smoke.architecture_from_payload
        calls = {"count": 0}

        def interrupted_lookup(payload):
            if calls["count"] == 3:
                raise RuntimeError("simulated source interruption")
            calls["count"] += 1
            return original(payload)

        with mock.patch.object(
            self.smoke,
            "architecture_from_payload",
            new=interrupted_lookup,
        ):
            with self.assertRaises(RuntimeError):
                self.smoke.run_smoke(
                    interrupted,
                    preset="manual",
                    gas_property_basis="textbook_true_cp",
                )

        manifest = load_manifest(interrupted)
        self.assertFalse(manifest["complete"])
        self.assertEqual(manifest["record_count"], 3)

        self.smoke.run_smoke(
            interrupted,
            preset="manual",
            resume=True,
            gas_property_basis="textbook_true_cp",
        )
        self.smoke.run_smoke(
            uninterrupted,
            preset="manual",
            gas_property_basis="textbook_true_cp",
        )
        for name in (JOURNAL_NAME, MANIFEST_NAME, CHECKPOINT_NAME):
            self.assertEqual(
                (interrupted / name).read_bytes(),
                (uninterrupted / name).read_bytes(),
                name,
            )

    def test_cli_main_creates_a_dataset(self):
        destination = self.root / "cli"
        with contextlib.redirect_stdout(io.StringIO()):
            code = self.smoke.main(
                [
                    "--destination",
                    str(destination),
                    "--preset",
                    "manual",
                    "--quiet",
                ]
            )
        self.assertEqual(code, 0)
        records = read_records(destination)
        self.assertEqual(len(records), 13)
        self.assertTrue(
            any(
                record.calculation_status
                is CalculationStatus.NON_PHYSICAL_STATE
                for record in records
            )
        )


if __name__ == "__main__":
    unittest.main()
