"""Reusable calculation core for Turbinis-Curs.

The package is intentionally independent of notebook execution and optimizer
state.  See :mod:`turbine_engine.architecture` for the validated station-state
and explicit engine-topology contracts,
:mod:`turbine_engine.compressor` for the 0D variable-k compressor unit/train,
:mod:`turbine_engine.combustor` for the scalar combustor heat-addition wrapper,
:mod:`turbine_engine.turbine` for the 0D lumped turbine-expansion unit,
:mod:`turbine_engine.shaft` for the standalone architecture power balances,
:mod:`turbine_engine.architecture_cycle` for the opt-in explicit 0D
architecture reference cycles,
:mod:`turbine_engine.cycle` for the single-point cycle extraction,
:mod:`turbine_engine.gas` for the stage gas-property helper,
:mod:`turbine_engine.stage` for the scalar meanline stage extraction, and
:mod:`turbine_engine.reaction` for the transcribed root reaction relation,
:mod:`turbine_engine.orchestration` for the const-hub three-stage
orchestration, :mod:`turbine_engine.performance` for the turbine aggregate
performance aggregation, :mod:`turbine_engine.expansion` for the turbine
expansion iteration and preliminary sizing,
:mod:`turbine_engine.geometry` for the meridional flowpath geometry, and
:mod:`turbine_engine.baseline` for the conventional multi-stage cell-16
orchestration, including their provenance notes.
"""

from turbine_engine.architecture import (
    ARCHITECTURES,
    Architecture,
    ArchitectureDescriptor,
    EngineArchitecture,
    ShaftGroup,
    StationState,
)
from turbine_engine.architecture_cycle import (
    ARCHITECTURE_CYCLE_LIMITATIONS,
    AlphaBasis,
    ArchitectureCycleError,
    ArchitectureCycleInput,
    ArchitectureCyclePhysicalError,
    ArchitectureCycleResult,
    CombustorClosureResult,
    ExternalCombustorClosure,
    ExternalCombustorState,
    LegacyCombustorClosure,
    PolynomialGasModel,
    calculate_architecture_cycle,
)
from turbine_engine.baseline import (
    BaselineTurbine,
    BaselineTurbineResult,
    calculate_baseline_turbine,
)
from turbine_engine.compressor import (
    CompressorResult,
    compressor_train,
    compressor_unit,
)
from turbine_engine.combustor import (
    CombustorResult,
    combustor_heat_addition,
)
from turbine_engine.cycle import (
    CycleConfig,
    CycleResult,
    CycleSolverError,
    CycleState,
    DEFAULT_CONFIG,
    SOLVER_MAX_ITER,
    SOLVER_TOL,
    calc_alpha,
    calc_compressor,
    calc_cp_kerosene,
    calc_eta_expansion,
    calc_g_m,
    calc_k_kerosene,
    calculate_cycle_point,
)
from turbine_engine.gas import (
    GasPropertyError,
    calc_gas_cp_k,
    calc_h0_for_pressure,
    calc_stage_cp_k,
)
from turbine_engine.expansion import (
    EXPANSION_ITER_MAX,
    EXPANSION_TEMPERATURE_TOL,
    TurbineExpansion,
    calculate_turbine_expansion,
)
from turbine_engine.geometry import (
    WIDTH_ITER_MAX,
    BladeWidths,
    ContourPoint,
    MeridionalGeometry,
    MeridionalGeometryConfig,
    calculate_meridional_geometry,
)
from turbine_engine.reaction import (
    ReactionCalculationError,
    calculate_root_reaction,
)
from turbine_engine.orchestration import (
    ConstHubOrchestration,
    build_const_hub_orchestration,
)
from turbine_engine.performance import (
    GasProperty,
    TurbinePerformance,
    calculate_turbine_performance,
)
from turbine_engine.stage import (
    DEFAULT_STAGE_CONFIG,
    StageCalculationError,
    StageConfig,
    StageResult,
    StageState,
    calculate_stage,
)
from turbine_engine.turbine import (
    TurbineResult,
    turbine_expansion,
    turbine_pressure_ratio_for_work,
)
from turbine_engine.shaft import (
    ShaftPowerBalance,
    evaluate_shaft_power_balance,
)

__all__ = [
    "ARCHITECTURES",
    "ARCHITECTURE_CYCLE_LIMITATIONS",
    "AlphaBasis",
    "Architecture",
    "ArchitectureCycleError",
    "ArchitectureCycleInput",
    "ArchitectureCyclePhysicalError",
    "ArchitectureCycleResult",
    "ArchitectureDescriptor",
    "CombustorClosureResult",
    "EngineArchitecture",
    "ExternalCombustorClosure",
    "ExternalCombustorState",
    "LegacyCombustorClosure",
    "PolynomialGasModel",
    "BaselineTurbine",
    "BaselineTurbineResult",
    "BladeWidths",
    "CompressorResult",
    "CombustorResult",
    "ContourPoint",
    "CycleConfig",
    "CycleResult",
    "CycleSolverError",
    "CycleState",
    "ConstHubOrchestration",
    "DEFAULT_CONFIG",
    "DEFAULT_STAGE_CONFIG",
    "EXPANSION_ITER_MAX",
    "EXPANSION_TEMPERATURE_TOL",
    "GasProperty",
    "GasPropertyError",
    "MeridionalGeometry",
    "MeridionalGeometryConfig",
    "ReactionCalculationError",
    "SOLVER_MAX_ITER",
    "SOLVER_TOL",
    "ShaftGroup",
    "ShaftPowerBalance",
    "StageCalculationError",
    "StageConfig",
    "StageResult",
    "StageState",
    "StationState",
    "TurbineExpansion",
    "TurbinePerformance",
    "TurbineResult",
    "WIDTH_ITER_MAX",
    "calc_alpha",
    "calc_compressor",
    "calc_cp_kerosene",
    "calc_eta_expansion",
    "calc_g_m",
    "calc_gas_cp_k",
    "calc_h0_for_pressure",
    "calc_k_kerosene",
    "calc_stage_cp_k",
    "calculate_architecture_cycle",
    "calculate_baseline_turbine",
    "calculate_cycle_point",
    "calculate_meridional_geometry",
    "calculate_root_reaction",
    "calculate_stage",
    "calculate_turbine_expansion",
    "calculate_turbine_performance",
    "build_const_hub_orchestration",
    "combustor_heat_addition",
    "compressor_train",
    "compressor_unit",
    "evaluate_shaft_power_balance",
    "turbine_expansion",
    "turbine_pressure_ratio_for_work",
]
