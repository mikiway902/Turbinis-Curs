"""Focused tests for the station-state and engine-topology contracts.

These tests check validation, immutability, explicit topology values, and the
absence of import/construction side effects.  They do not evaluate a cycle and
do not claim physical validation of any architecture.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path
from types import MappingProxyType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import turbine_engine  # noqa: E402

from turbine_engine.architecture import (  # noqa: E402
    ARCHITECTURES,
    Architecture,
    ArchitectureDescriptor,
    EngineArchitecture,
    ShaftGroup,
    StationState,
)

VALID_STATION = {
    "T_total": 1250.0,
    "p_total": 1_000_000.0,
    "mass_flow": 0.6,
    "cp": 1150.0,
    "k": 1.33,
}


class StationStateTests(unittest.TestCase):
    def test_valid_state_stores_only_explicit_values(self) -> None:
        state = StationState(**VALID_STATION, identifier="turbine_inlet")

        self.assertEqual(state.T_total, 1250.0)
        self.assertEqual(state.p_total, 1_000_000.0)
        self.assertEqual(state.mass_flow, 0.6)
        self.assertEqual(state.cp, 1150.0)
        self.assertEqual(state.k, 1.33)
        self.assertEqual(state.identifier, "turbine_inlet")
        self.assertEqual(
            asdict(state),
            {**VALID_STATION, "identifier": "turbine_inlet"},
        )

    def test_identifier_is_optional(self) -> None:
        state = StationState(**VALID_STATION)
        self.assertIsNone(state.identifier)

    def test_numeric_values_are_normalised_to_float(self) -> None:
        state = StationState(
            T_total=1250,
            p_total=1_000_000,
            mass_flow=1,
            cp=1150,
            k=1.33,
        )
        self.assertIsInstance(state.T_total, float)
        self.assertIsInstance(state.p_total, float)
        self.assertIsInstance(state.mass_flow, float)
        self.assertIsInstance(state.cp, float)
        self.assertIsInstance(state.k, float)

    def test_state_is_immutable(self) -> None:
        state = StationState(**VALID_STATION)

        with self.assertRaises(FrozenInstanceError):
            state.T_total = 1.0  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            state.identifier = "changed"  # type: ignore[misc]

    def test_rejects_non_positive_or_non_finite_required_values(self) -> None:
        for field in ("T_total", "p_total", "mass_flow", "cp"):
            for value in (0.0, -1.0, float("nan"), float("inf")):
                with self.subTest(field=field, value=value):
                    kwargs = dict(VALID_STATION)
                    kwargs[field] = value
                    with self.assertRaises(ValueError):
                        StationState(**kwargs)

    def test_rejects_invalid_isentropic_exponent(self) -> None:
        for value in (1.0, 0.9, float("nan"), float("inf")):
            with self.subTest(k=value):
                kwargs = dict(VALID_STATION)
                kwargs["k"] = value
                with self.assertRaises(ValueError):
                    StationState(**kwargs)

    def test_rejects_invalid_identifier(self) -> None:
        for value in ("", "   ", 123):
            with self.subTest(identifier=value):
                with self.assertRaises(ValueError):
                    StationState(**VALID_STATION, identifier=value)


class ShaftGroupTests(unittest.TestCase):
    def test_valid_group(self) -> None:
        group = ShaftGroup(
            name="MAIN",
            compressor_units=2,
            turbine_assemblies=1,
            drives_output=True,
        )
        self.assertEqual(group.name, "MAIN")
        self.assertEqual(group.compressor_units, 2)
        self.assertEqual(group.turbine_assemblies, 1)
        self.assertTrue(group.drives_output)

    def test_group_is_immutable(self) -> None:
        group = ShaftGroup("MAIN", 1, 1, True)
        with self.assertRaises(FrozenInstanceError):
            group.name = "OTHER"  # type: ignore[misc]

    def test_rejects_empty_or_invalid_groups(self) -> None:
        cases = (
            {"name": "", "compressor_units": 1, "turbine_assemblies": 1},
            {"name": "MAIN", "compressor_units": 0, "turbine_assemblies": 0},
            {"name": "MAIN", "compressor_units": -1, "turbine_assemblies": 1},
            {"name": "MAIN", "compressor_units": 1, "turbine_assemblies": -1},
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ShaftGroup(drives_output=True, **kwargs)

        with self.assertRaises(ValueError):
            ShaftGroup("MAIN", 1, 1, drives_output=1)  # type: ignore[arg-type]


class ArchitectureTopologyTests(unittest.TestCase):
    def test_enum_has_exactly_the_target_members(self) -> None:
        self.assertEqual(
            {member.value for member in Architecture},
            {
                "SINGLE_SHAFT_SINGLE_CC",
                "SINGLE_SHAFT_DUAL_CC",
                "FREE_POWER_TURBINE",
            },
        )

    def test_registry_covers_every_architecture(self) -> None:
        self.assertEqual(set(ARCHITECTURES), set(Architecture))
        for architecture, descriptor in ARCHITECTURES.items():
            with self.subTest(architecture=architecture):
                self.assertIs(descriptor.architecture, architecture)

    def test_single_shaft_single_cc_topology(self) -> None:
        descriptor = ARCHITECTURES[Architecture.SINGLE_SHAFT_SINGLE_CC]

        self.assertEqual(descriptor.compressor_units, 1)
        self.assertEqual(descriptor.combustors, 1)
        self.assertEqual(descriptor.turbine_assemblies, 1)
        self.assertEqual(len(descriptor.shaft_groups), 1)

        group = descriptor.shaft_groups[0]
        self.assertEqual(group.compressor_units, 1)
        self.assertEqual(group.turbine_assemblies, 1)
        self.assertTrue(group.drives_output)

    def test_single_shaft_dual_cc_topology(self) -> None:
        descriptor = ARCHITECTURES[Architecture.SINGLE_SHAFT_DUAL_CC]

        self.assertEqual(descriptor.compressor_units, 2)
        self.assertEqual(descriptor.combustors, 1)
        self.assertEqual(descriptor.turbine_assemblies, 1)
        self.assertEqual(len(descriptor.shaft_groups), 1)

        group = descriptor.shaft_groups[0]
        self.assertEqual(group.compressor_units, 2)
        self.assertEqual(group.turbine_assemblies, 1)
        self.assertTrue(group.drives_output)

    def test_free_power_turbine_has_separate_shaft_groups(self) -> None:
        descriptor = ARCHITECTURES[Architecture.FREE_POWER_TURBINE]

        self.assertEqual(descriptor.compressor_units, 1)
        self.assertEqual(descriptor.combustors, 1)
        self.assertEqual(descriptor.turbine_assemblies, 2)
        self.assertEqual(len(descriptor.shaft_groups), 2)

        output = [g for g in descriptor.shaft_groups if g.drives_output]
        gas_generator = [g for g in descriptor.shaft_groups if not g.drives_output]
        self.assertEqual(len(output), 1)
        self.assertEqual(len(gas_generator), 1)

        self.assertEqual(output[0].name, "FREE_POWER_TURBINE")
        self.assertEqual(output[0].compressor_units, 0)
        self.assertEqual(output[0].turbine_assemblies, 1)

        self.assertEqual(gas_generator[0].name, "GAS_GENERATOR")
        self.assertEqual(gas_generator[0].compressor_units, 1)
        self.assertEqual(gas_generator[0].turbine_assemblies, 1)

    def test_every_architecture_has_one_combustor_and_one_output_group(self) -> None:
        for architecture in Architecture:
            with self.subTest(architecture=architecture):
                descriptor = ARCHITECTURES[architecture]
                self.assertEqual(descriptor.combustors, 1)
                self.assertEqual(
                    sum(group.drives_output for group in descriptor.shaft_groups),
                    1,
                )
                self.assertEqual(
                    sum(
                        group.compressor_units
                        for group in descriptor.shaft_groups
                    ),
                    descriptor.compressor_units,
                )
                self.assertEqual(
                    sum(
                        group.turbine_assemblies
                        for group in descriptor.shaft_groups
                    ),
                    descriptor.turbine_assemblies,
                )

    def test_descriptor_rejects_wrong_component_counts(self) -> None:
        cases = (
            (
                Architecture.SINGLE_SHAFT_SINGLE_CC,
                dict(
                    compressor_units=2,
                    combustors=1,
                    turbine_assemblies=1,
                    shaft_groups=(ShaftGroup("MAIN", 2, 1, True),),
                ),
            ),
            (
                Architecture.SINGLE_SHAFT_SINGLE_CC,
                dict(
                    compressor_units=1,
                    combustors=2,
                    turbine_assemblies=1,
                    shaft_groups=(ShaftGroup("MAIN", 1, 1, True),),
                ),
            ),
            (
                Architecture.SINGLE_SHAFT_SINGLE_CC,
                dict(
                    compressor_units=1,
                    combustors=1,
                    turbine_assemblies=2,
                    shaft_groups=(ShaftGroup("MAIN", 1, 2, True),),
                ),
            ),
            (
                Architecture.SINGLE_SHAFT_DUAL_CC,
                dict(
                    compressor_units=1,
                    combustors=1,
                    turbine_assemblies=1,
                    shaft_groups=(ShaftGroup("MAIN", 1, 1, True),),
                ),
            ),
            (
                Architecture.SINGLE_SHAFT_DUAL_CC,
                dict(
                    compressor_units=2,
                    combustors=2,
                    turbine_assemblies=1,
                    shaft_groups=(ShaftGroup("MAIN", 2, 1, True),),
                ),
            ),
            (
                Architecture.FREE_POWER_TURBINE,
                dict(
                    compressor_units=2,
                    combustors=1,
                    turbine_assemblies=2,
                    shaft_groups=(
                        ShaftGroup("GAS_GENERATOR", 2, 1, False),
                        ShaftGroup("FREE_POWER_TURBINE", 0, 1, True),
                    ),
                ),
            ),
            (
                Architecture.FREE_POWER_TURBINE,
                dict(
                    compressor_units=1,
                    combustors=2,
                    turbine_assemblies=2,
                    shaft_groups=(
                        ShaftGroup("GAS_GENERATOR", 1, 1, False),
                        ShaftGroup("FREE_POWER_TURBINE", 0, 1, True),
                    ),
                ),
            ),
        )
        for architecture, kwargs in cases:
            with self.subTest(architecture=architecture, kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ArchitectureDescriptor(
                        architecture=architecture,
                        **kwargs,
                    )

    def test_descriptor_rejects_wrong_shaft_group_layout(self) -> None:
        cases = (
            # Totals and output count match, but a "single shaft" is split.
            (
                Architecture.SINGLE_SHAFT_SINGLE_CC,
                dict(
                    compressor_units=1,
                    combustors=1,
                    turbine_assemblies=1,
                    shaft_groups=(
                        ShaftGroup("A", 1, 0, True),
                        ShaftGroup("B", 0, 1, False),
                    ),
                ),
            ),
            (
                Architecture.SINGLE_SHAFT_DUAL_CC,
                dict(
                    compressor_units=2,
                    combustors=1,
                    turbine_assemblies=1,
                    shaft_groups=(
                        ShaftGroup("A", 1, 1, True),
                        ShaftGroup("B", 1, 0, False),
                    ),
                ),
            ),
            # Free-power-turbine totals and output count match, but the
            # compressor and free power turbine share the wrong shaft.
            (
                Architecture.FREE_POWER_TURBINE,
                dict(
                    compressor_units=1,
                    combustors=1,
                    turbine_assemblies=2,
                    shaft_groups=(
                        ShaftGroup("GAS_GENERATOR", 0, 2, False),
                        ShaftGroup("FREE_POWER_TURBINE", 1, 0, True),
                    ),
                ),
            ),
            # Correct group names and totals, but the wrong group drives the
            # useful output.
            (
                Architecture.FREE_POWER_TURBINE,
                dict(
                    compressor_units=1,
                    combustors=1,
                    turbine_assemblies=2,
                    shaft_groups=(
                        ShaftGroup("GAS_GENERATOR", 0, 1, True),
                        ShaftGroup("FREE_POWER_TURBINE", 1, 1, False),
                    ),
                ),
            ),
            # Correct free-power-turbine totals collapsed onto one shaft.
            (
                Architecture.FREE_POWER_TURBINE,
                dict(
                    compressor_units=1,
                    combustors=1,
                    turbine_assemblies=2,
                    shaft_groups=(ShaftGroup("MAIN", 1, 2, True),),
                ),
            ),
        )
        for architecture, kwargs in cases:
            with self.subTest(architecture=architecture, kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ArchitectureDescriptor(
                        architecture=architecture,
                        **kwargs,
                    )

    def test_registry_and_descriptors_are_immutable(self) -> None:
        descriptor = ARCHITECTURES[Architecture.SINGLE_SHAFT_SINGLE_CC]

        with self.assertRaises(FrozenInstanceError):
            descriptor.compressor_units = 2  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            descriptor.shaft_groups[0].name = "OTHER"  # type: ignore[misc]
        with self.assertRaises(TypeError):
            ARCHITECTURES[Architecture.SINGLE_SHAFT_SINGLE_CC] = descriptor  # type: ignore[index]

    def test_registry_is_a_read_only_mapping(self) -> None:
        self.assertIsInstance(ARCHITECTURES, MappingProxyType)

    def test_descriptor_rejects_inconsistent_topology(self) -> None:
        cases = (
            # Compressor total does not match the shaft groups.
            dict(
                compressor_units=2,
                combustors=1,
                turbine_assemblies=1,
                shaft_groups=(ShaftGroup("MAIN", 1, 1, True),),
            ),
            # Turbine total does not match the shaft groups.
            dict(
                compressor_units=1,
                combustors=1,
                turbine_assemblies=2,
                shaft_groups=(ShaftGroup("MAIN", 1, 1, True),),
            ),
            # No output shaft group.
            dict(
                compressor_units=1,
                combustors=1,
                turbine_assemblies=1,
                shaft_groups=(ShaftGroup("MAIN", 1, 1, False),),
            ),
            # Two output shaft groups.
            dict(
                compressor_units=2,
                combustors=1,
                turbine_assemblies=2,
                shaft_groups=(
                    ShaftGroup("A", 1, 1, True),
                    ShaftGroup("B", 1, 1, True),
                ),
            ),
            # Duplicate shaft group names.
            dict(
                compressor_units=1,
                combustors=1,
                turbine_assemblies=1,
                shaft_groups=(
                    ShaftGroup("A", 1, 0, True),
                    ShaftGroup("A", 0, 1, False),
                ),
            ),
            # Empty shaft-group list.
            dict(
                compressor_units=1,
                combustors=1,
                turbine_assemblies=1,
                shaft_groups=(),
            ),
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ArchitectureDescriptor(
                        architecture=Architecture.SINGLE_SHAFT_SINGLE_CC,
                        **kwargs,
                    )

    def test_descriptor_rejects_non_architecture_member(self) -> None:
        with self.assertRaises(ValueError):
            ArchitectureDescriptor(
                architecture="SINGLE_SHAFT_SINGLE_CC",  # type: ignore[arg-type]
                compressor_units=1,
                combustors=1,
                turbine_assemblies=1,
                shaft_groups=(ShaftGroup("MAIN", 1, 1, True),),
            )


class PublicContractTests(unittest.TestCase):
    def test_package_exports_the_contract_symbols(self) -> None:
        for name, expected in (
            ("ARCHITECTURES", ARCHITECTURES),
            ("Architecture", Architecture),
            ("EngineArchitecture", EngineArchitecture),
            ("ArchitectureDescriptor", ArchitectureDescriptor),
            ("ShaftGroup", ShaftGroup),
            ("StationState", StationState),
        ):
            with self.subTest(name=name):
                self.assertIs(getattr(turbine_engine, name), expected)

    def test_constructing_contracts_does_not_mutate_inputs_or_registry(self) -> None:
        groups = [ShaftGroup("MAIN", 1, 1, True)]
        groups_before = list(groups)
        registry_before = {
            key: (
                descriptor.compressor_units,
                descriptor.combustors,
                descriptor.turbine_assemblies,
                descriptor.shaft_groups,
            )
            for key, descriptor in ARCHITECTURES.items()
        }

        descriptor = ArchitectureDescriptor(
            architecture=Architecture.SINGLE_SHAFT_SINGLE_CC,
            compressor_units=1,
            combustors=1,
            turbine_assemblies=1,
            shaft_groups=groups,
        )
        StationState(**VALID_STATION)

        self.assertEqual(groups, groups_before)
        self.assertIsInstance(descriptor.shaft_groups, tuple)
        self.assertEqual(
            registry_before,
            {
                key: (
                    item.compressor_units,
                    item.combustors,
                    item.turbine_assemblies,
                    item.shaft_groups,
                )
                for key, item in ARCHITECTURES.items()
            },
        )

    def test_architecture_import_does_not_load_notebook_loader(self) -> None:
        code = (
            "import sys\n"
            "import turbine_engine\n"
            "from turbine_engine import ARCHITECTURES, Architecture, StationState\n"
            "state = StationState(T_total=1250.0, p_total=1e6, mass_flow=0.6, "
            "cp=1150.0, k=1.33)\n"
            "assert state.k > 1.0\n"
            "assert ARCHITECTURES[Architecture.SINGLE_SHAFT_SINGLE_CC]"
            ".compressor_units == 1\n"
            "assert 'cycle_link' not in sys.modules, sorted(sys.modules)\n"
            "assert 'optimizer_core' not in sys.modules, sorted(sys.modules)\n"
            "print(state.T_total)\n"
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


if __name__ == "__main__":
    unittest.main()
