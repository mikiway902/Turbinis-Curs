"""Runtime regression anchors for the single-point cycle implementation.

PROVENANCE / STATUS: software-drift diagnostic only.

This is NOT a validated physical baseline. It records what the current cycle
implementation returned for fixed operating points so that the extraction into
``turbine_engine.cycle`` can be checked for unintended output drift. The values
in ``LEGACY_ANCHORS`` were observed with CPython 3.12.10 and numpy 2.5.4 /
scipy 1.18.1 / pandas 3.1.1 on 2026-11-10::

    optimizer_core.cycle_point(model, pi, T_g, 110000.0)

for ``pi`` in 3, 6, 11, 13 and ``T_g`` in 1250 K, 1300 K. The eight points
cover both cooling branches because the current rule is ``T_g > 1250.0``.
``Turbinis-Calc/Отчет_по_промту_2026-11-08.md`` independently records
approximately G = 0.5889, C_e = 0.3245, eta_e = 0.2571 for the pi = 11,
T_g = 1250 K point. Those numbers belong to the *old post-reducer* plane.

POWER-PLANE CHANGE (user-confirmed 2026-11-10)
==============================================
The active target is 110 kW **at the gearbox input, before the reducer**.
Therefore the new anchors are derived from the recorded legacy post-reducer
anchors by the exact relations::

    N_ud_new          = N_ud_legacy / eta_reducer
    N_reducer_out_ud  = N_ud_legacy
    G_new             = G_legacy * eta_reducer
    C_e_new           = C_e_legacy * eta_reducer
    eta_e_new         = eta_e_legacy / eta_reducer
    N_e_ud            = unchanged (built on the downstream gearbox output)

The legacy numbers are retained below and used in an explicitly labelled
differential test. No expected value was changed without this basis. A failure
here means software drift or another intentional model/formula change and must
be investigated and re-approved separately. These tests never invoke an
optimizer search or sweep.

The tolerances are tight enough to catch formula/iteration drift but allow
last-bit differences between platforms. Values that the implementation copies
directly from the arguments, assigns as literals, or selects as a boolean branch
are asserted exactly.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Optimizer"))
sys.path.insert(0, str(ROOT / "Turbinis-Calc"))

from turbine_engine.cycle import CycleConfig, CycleState, calculate_cycle_point  # noqa: E402

# Legacy observed outputs for (T_g, pi) at N = 110000 W. ``d`` mirrors the
# dictionary formerly returned by the notebook ``calc_tvd``. These values are
# the *post-reducer* legacy plane; see ``new_anchor`` for the active pre-reducer
# anchor derivation.
LEGACY_ANCHORS = {
    (1250.0, 3.0): {
        "G": 0.8299009283764727,
        "p_g": 285909.12,
        "p_t": 106365.0,
        "c_t": 210.1420512785724,
        "g_t": 1.0001431894997075,
        "alpha_g": 2.954805841792094,
        "L_k": 129497.4465838522,
        "L_t": 264999.6913590183,
        "T_t": 1005.7568518740208,
        "eta_t_cycle": 0.9105157356246628,
        "N_ud": 132545.94161641915,
        "C_e": 0.6149979486723777,
        "eta_e": 0.1356588195013236,
        "g_ohl": 0.015,
        "g_voz": 0.009,
        "охлаждение": False,
        "d": {
            "g_m": 0.023164388234994747,
            "g_т": 1.0001431894997075,
            "α": 2.954805841792094,
            "L_к": 129497.4465838522,
            "L_т": 264999.6913590183,
            "π_т": 2.4891122291003875,
            "π_с_полн": 1.1338982497466854,
            "η_т": 0.9105157356246628,
            "T_т": 1005.7568518740208,
            "H_с_полн": 35717.132704308235,
            "c_с": 253.90829945331907,
            "N_в_уд": 132545.94161641915,
            "R_уд": 256.2298311507692,
            "N_e_уд": 149627.93035980375,
        },
    },
    (1250.0, 6.0): {
        "G": 0.593043340950115,
        "p_g": 571818.24,
        "p_t": 106365.0,
        "c_t": 193.7002656680893,
        "g_t": 0.9973653052499414,
        "alpha_g": 3.3679939859337447,
        "L_k": 239433.36800940376,
        "L_t": 430294.04495716305,
        "T_t": 849.03207536069,
        "eta_t_cycle": 0.9182094470517357,
        "N_ud": 185483.91391389532,
        "C_e": 0.3855595743628074,
        "eta_e": 0.21638652301790262,
        "g_ohl": 0.015,
        "g_voz": 0.009,
        "охлаждение": False,
        "d": {
            "g_m": 0.02032256291554109,
            "g_т": 0.9973653052499414,
            "α": 3.3679939859337447,
            "L_к": 239433.36800940376,
            "L_т": 430294.04495716305,
            "π_т": 4.976755131886638,
            "π_с_полн": 1.134233019389104,
            "η_т": 0.9182094470517357,
            "T_т": 849.03207536069,
            "H_с_полн": 30211.559425770793,
            "c_с": 233.5205874511202,
            "N_в_уд": 185483.91391389532,
            "R_уд": 235.0070172723922,
            "N_e_уд": 201151.04839872147,
        },
    },
    (1250.0, 11.0): {
        "G": 0.5889342350626282,
        "p_g": 1048333.44,
        "p_t": 106365.0,
        "c_t": 180.09582219113014,
        "g_t": 0.9943384755983069,
        "alpha_g": 3.973413639490531,
        "L_k": 358583.40802120743,
        "L_t": 552874.7217839112,
        "T_t": 730.1309202648472,
        "eta_t_cycle": 0.9246868857757602,
        "N_ud": 186778.07037028918,
        "C_e": 0.3245483371454034,
        "eta_e": 0.2570646223192704,
        "g_ohl": 0.015,
        "g_voz": 0.009,
        "охлаждение": False,
        "d": {
            "g_m": 0.017226061993152738,
            "g_т": 0.9943384755983069,
            "α": 3.973413639490531,
            "L_к": 358583.40802120743,
            "L_т": 552874.7217839112,
            "π_т": 9.121796918343145,
            "π_с_полн": 1.1345133083580776,
            "η_т": 0.9246868857757602,
            "T_т": 730.1309202648472,
            "H_с_полн": 26024.134596369946,
            "c_с": 216.73385279288453,
            "N_в_уд": 186778.07037028918,
            "R_уд": 217.4574134717606,
            "N_e_уд": 201275.23126840655,
        },
    },
    (1250.0, 13.0): {
        "G": 0.6128622380057887,
        "p_g": 1238939.52,
        "p_t": 106365.0,
        "c_t": 176.47255678447968,
        "g_t": 0.9933895285964396,
        "alpha_g": 4.210711992143856,
        "L_k": 395804.79447416775,
        "L_t": 583407.4009985166,
        "T_t": 700.0836211242982,
        "eta_t_cycle": 0.9264278670676623,
        "N_ud": 179485.68728582852,
        "C_e": 0.31870119457541185,
        "eta_e": 0.26178093189697466,
        "g_ohl": 0.015,
        "g_voz": 0.009,
        "охлаждение": False,
        "d": {
            "g_m": 0.01625527222142153,
            "g_т": 0.9933895285964396,
            "α": 4.210711992143856,
            "L_к": 395804.79447416775,
            "L_т": 583407.4009985166,
            "π_т": 10.77958038693306,
            "π_с_полн": 1.1345896186112785,
            "η_т": 0.9264278670676623,
            "T_т": 700.0836211242982,
            "H_с_полн": 24965.083979180796,
            "c_с": 212.27806429874317,
            "N_в_уд": 179485.68728582852,
            "R_уд": 212.78530880378185,
            "N_e_уд": 193671.37453941398,
        },
    },
    (1300.0, 3.0): {
        "G": 0.7880218403375965,
        "p_g": 285909.12,
        "p_t": 106365.0,
        "c_t": 214.23048415198778,
        "g_t": 0.9658550660554973,
        "alpha_g": 2.762167703508414,
        "L_k": 129497.4465838522,
        "L_t": 275826.37567682116,
        "T_t": 1047.3751231643394,
        "eta_t_cycle": 0.91045376489756,
        "N_ud": 139590.03972894317,
        "C_e": 0.602322615303024,
        "eta_e": 0.13851363636853883,
        "g_ohl": 0.05,
        "g_voz": 0.03,
        "охлаждение": True,
        "d": {
            "g_m": 0.024779910934214583,
            "g_т": 0.9658550660554973,
            "α": 2.762167703508414,
            "L_к": 129497.4465838522,
            "L_т": 275826.37567682116,
            "π_т": 2.489527930543618,
            "π_с_полн": 1.1337089113853387,
            "η_т": 0.91045376489756,
            "T_т": 1047.3751231643394,
            "H_с_полн": 37150.13153969802,
            "c_с": 258.95170868166696,
            "N_в_уд": 139590.03972894317,
            "R_уд": 257.8783709543654,
            "N_e_уд": 156781.93112590085,
        },
    },
    (1300.0, 6.0): {
        "G": 0.5582363765393137,
        "p_g": 571818.24,
        "p_t": 106365.0,
        "c_t": 197.57800033919236,
        "g_t": 0.9631642768940177,
        "alpha_g": 3.1218420805460676,
        "L_k": 239433.36800940376,
        "L_t": 448015.9705390654,
        "T_t": 884.9592641825756,
        "eta_t_cycle": 0.9181113162536101,
        "N_ud": 197049.1437371482,
        "C_e": 0.3775271255057941,
        "eta_e": 0.2209904668462026,
        "g_ohl": 0.05,
        "g_voz": 0.03,
        "охлаждение": True,
        "d": {
            "g_m": 0.021924962221769408,
            "g_т": 0.9631642768940177,
            "α": 3.1218420805460676,
            "L_к": 239433.36800940376,
            "L_т": 448015.9705390654,
            "π_т": 4.977543519762538,
            "π_с_полн": 1.1340533694157022,
            "η_т": 0.9181113162536101,
            "T_т": 884.9592641825756,
            "H_с_полн": 31453.790224583812,
            "c_с": 238.27314442751154,
            "N_в_уд": 197049.1437371482,
            "R_уд": 236.64437518861337,
            "N_e_уд": 212825.43541638908,
        },
    },
    (1300.0, 11.0): {
        "G": 0.5451813849710535,
        "p_g": 1048333.44,
        "p_t": 106365.0,
        "c_t": 183.7817579579121,
        "g_t": 0.9602323475233286,
        "alpha_g": 3.6380185470045836,
        "L_k": 358583.40802120743,
        "L_t": 575794.694977662,
        "T_t": 761.5573673479804,
        "eta_t_cycle": 0.9245648517080994,
        "N_ud": 201767.71076994945,
        "C_e": 0.31638586194184265,
        "eta_e": 0.2636966620460628,
        "g_ohl": 0.05,
        "g_voz": 0.03,
        "охлаждение": True,
        "d": {
            "g_m": 0.01881416182846533,
            "g_т": 0.9602323475233286,
            "α": 3.6380185470045836,
            "L_к": 358583.40802120743,
            "L_т": 575794.694977662,
            "π_т": 9.123177439106394,
            "π_с_полн": 1.1343416336111132,
            "η_т": 0.9245648517080994,
            "T_т": 761.5573673479804,
            "H_с_полн": 27114.406034963155,
            "c_с": 221.22726525703942,
            "N_в_уд": 201767.71076994945,
            "R_уд": 219.06639421164428,
            "N_e_уд": 216372.13705072575,
        },
    },
    (1300.0, 13.0): {
        "G": 0.5631312800273744,
        "p_g": 1238939.52,
        "p_t": 106365.0,
        "c_t": 180.1048021650683,
        "g_t": 0.9593131528853021,
        "alpha_g": 3.836913255466473,
        "L_k": 395804.79447416775,
        "L_t": 607632.6516237217,
        "T_t": 730.3471034775142,
        "eta_t_cycle": 0.9263004065437275,
        "N_ud": 195336.3343528933,
        "C_e": 0.30986222091041904,
        "eta_e": 0.26924836292563037,
        "g_ohl": 0.05,
        "g_voz": 0.03,
        "охлаждение": True,
        "d": {
            "g_m": 0.01783888900297309,
            "g_т": 0.9593131528853021,
            "α": 3.836913255466473,
            "L_к": 395804.79447416775,
            "L_т": 607632.6516237217,
            "π_т": 10.781191409823416,
            "π_с_полн": 1.1344200779939886,
            "η_т": 0.9263004065437275,
            "T_т": 730.3471034775142,
            "H_с_полн": 26015.42279899124,
            "c_с": 216.69757301866392,
            "N_в_уд": 195336.3343528933,
            "R_уд": 214.38175918568737,
            "N_e_уд": 209628.45163193913,
        },
    },
}

ETA_REDUCER = 0.98


def new_anchor(legacy: dict) -> dict:
    """Map a recorded legacy post-reducer anchor to the active pre-reducer plane."""

    updated = dict(legacy)
    updated["G"] = legacy["G"] * ETA_REDUCER
    updated["N_ud"] = legacy["N_ud"] / ETA_REDUCER
    updated["N_reducer_out_ud"] = legacy["N_ud"]
    updated["C_e"] = legacy["C_e"] * ETA_REDUCER
    updated["eta_e"] = legacy["eta_e"] / ETA_REDUCER
    d = dict(legacy["d"])
    d["N_в_уд"] = legacy["N_ud"] / ETA_REDUCER
    d["N_ред_уд"] = legacy["N_ud"]
    # N_e_уд is unchanged: it is built on the downstream gearbox output.
    updated["d"] = d
    return updated


ANCHORS = {key: new_anchor(value) for key, value in LEGACY_ANCHORS.items()}


class CyclePointAnchorTests(unittest.TestCase):
    PI = 11.0
    T_G = 1250.0
    N = 110000.0

    # Active pre-reducer anchors derived from the recorded legacy values.
    EXPECTED = {
        "G": 0.5771555503613757,
        "p_g": 1048333.44,
        "p_t": 106365.0,
        "L_k": 358583.40802120743,
        "L_t": 552874.7217839112,
        "eta_t_cycle": 0.9246868857757602,
        "N_ud": 190589.86772478488,
        "N_reducer_out_ud": 186778.07037028918,
        "C_e": 0.3180573704024953,
        "eta_e": 0.2623108391012963,
    }
    LEGACY_EXPECTED = {
        "G": 0.5889342350626282,
        "N_ud": 186778.07037028918,
        "C_e": 0.3245483371454034,
        "eta_e": 0.2570646223192704,
        "N_e_ud": 201275.23126840655,
    }
    RELATIVE_TOLERANCE = 1e-9

    # ``cycle_point`` formerly overwrote these cycle entries. The extracted
    # core must leave them alone (and the public wrapper must remain compatible).
    CYCLE_STATE_KEYS = ("T_г_полн", "охлаждение_лопаток", "g_охл", "g_воз")

    @classmethod
    def setUpClass(cls) -> None:
        # Importing the model requires the declared runtime dependencies. Skip
        # only when one of those is genuinely unavailable; any other import or
        # model failure must surface as an error, not as a skip.
        try:
            import optimizer_core as oc
        except ModuleNotFoundError as exc:
            dependency = (exc.name or "").split(".")[0]
            if dependency in {"numpy", "scipy"}:
                raise unittest.SkipTest(
                    "cycle_point regression anchor needs runtime dependency "
                    f"'{dependency}', which is not installed; install the "
                    "project dependencies to run this test"
                ) from exc
            raise
        cls.oc = oc
        cls.model = oc.load_model()

    def assertFiniteNear(self, actual: float, expected: float) -> None:
        self.assertTrue(math.isfinite(actual), f"expected finite value, got {actual!r}")
        self.assertAlmostEqual(
            actual / expected,
            1.0,
            delta=self.RELATIVE_TOLERANCE,
            msg=f"expected ~{expected!r}, got {actual!r}",
        )

    def assertAnchorPoint(self, actual: dict, expected: dict) -> None:
        """Compare one result dictionary with an observed legacy anchor."""

        self.assertEqual(actual["охлаждение"], expected["охлаждение"])
        self.assertEqual(actual["g_ohl"], expected["g_ohl"])
        self.assertEqual(actual["g_voz"], expected["g_voz"])
        for field, value in expected.items():
            if field in {"охлаждение", "g_ohl", "g_voz", "d"}:
                continue
            with self.subTest(field=field):
                self.assertFiniteNear(actual[field], value)
        for field, value in expected["d"].items():
            with self.subTest(field=f"d.{field}"):
                self.assertFiniteNear(actual["d"][field], value)

    @staticmethod
    def core_as_point(result) -> dict:
        """Map the typed core result onto the legacy result dictionary."""

        return {
            "G": result.G,
            "p_g": result.p_g,
            "p_t": result.p_t,
            "c_t": result.c_t,
            "g_t": result.g_t,
            "alpha_g": result.alpha,
            "L_k": result.L_k,
            "L_t": result.L_t,
            "T_t": result.T_t,
            "eta_t_cycle": result.eta_t,
            "N_ud": result.N_ud,
            "N_reducer_out_ud": result.N_reducer_out_ud,
            "C_e": result.C_e,
            "eta_e": result.eta_e,
            "g_ohl": result.g_cooling,
            "g_voz": result.g_return,
            "охлаждение": result.cooled,
            "d": {
                "g_m": result.g_m,
                "g_т": result.g_t,
                "α": result.alpha,
                "L_к": result.L_k,
                "L_т": result.L_t,
                "π_т": result.pi_t,
                "π_с_полн": result.pi_c_full,
                "η_т": result.eta_t,
                "T_т": result.T_t,
                "H_с_полн": result.H_c_full,
                "c_с": result.c_c,
                "N_в_уд": result.N_ud,
                "N_ред_уд": result.N_reducer_out_ud,
                "R_уд": result.R_ud,
                "N_e_уд": result.N_e_ud,
            },
        }

    def test_cycle_point_is_a_deterministic_diagnostic_anchor(self) -> None:
        result = self.oc.cycle_point(self.model, self.PI, self.T_G, self.N)

        # The returned operating point must echo the requested arguments.
        self.assertEqual(result["pi"], self.PI)
        self.assertEqual(result["T_g"], self.T_G)
        self.assertEqual(result["N"], self.N)

        # Deterministic cycle outputs. Diagnostic drift anchor only.
        for field, expected in self.EXPECTED.items():
            with self.subTest(field=field):
                self.assertFiniteNear(result[field], expected)

        # At exactly T_g = 1250 K the current implementation selects the
        # uncooled branch: охлаждение is ``T_g > 1250.0``, so it is false here.
        self.assertFalse(result["охлаждение"])
        self.assertEqual(result["g_ohl"], 0.015)
        self.assertEqual(result["g_voz"], 0.009)

    def test_legacy_post_reducer_anchor_is_a_labelled_differential(self) -> None:
        """The old numbers remain reproducible only by reversing eta_reducer."""

        result = self.oc.cycle_point(self.model, self.PI, self.T_G, self.N)

        # Active plane -> legacy plane.  This is an explicit differential, not
        # an assertion that the active implementation still uses the old plane.
        self.assertFiniteNear(result["N_ud"] * ETA_REDUCER, self.LEGACY_EXPECTED["N_ud"])
        self.assertFiniteNear(result["G"] / ETA_REDUCER, self.LEGACY_EXPECTED["G"])
        self.assertFiniteNear(result["C_e"] / ETA_REDUCER, self.LEGACY_EXPECTED["C_e"])
        self.assertFiniteNear(result["eta_e"] * ETA_REDUCER, self.LEGACY_EXPECTED["eta_e"])
        self.assertFiniteNear(
            result["N_reducer_out_ud"], self.LEGACY_EXPECTED["N_ud"]
        )
        self.assertFiniteNear(
            result["d"]["N_e_уд"], self.LEGACY_EXPECTED["N_e_ud"]
        )

    def test_cycle_point_matches_grid_anchors(self) -> None:
        for (T_g, pi), expected in ANCHORS.items():
            with self.subTest(T_g=T_g, pi=pi):
                result = self.oc.cycle_point(self.model, pi, T_g, self.N)
                self.assertEqual(result["pi"], pi)
                self.assertEqual(result["T_g"], T_g)
                self.assertEqual(result["N"], self.N)
                self.assertAnchorPoint(result, expected)

    def test_new_core_matches_grid_anchors(self) -> None:
        config = CycleConfig.from_namespace(self.model.cycle)
        for (T_g, pi), expected in ANCHORS.items():
            with self.subTest(T_g=T_g, pi=pi):
                result = calculate_cycle_point(
                    CycleState(pi=pi, T_g=T_g, N=self.N), config
                )
                self.assertEqual(result.pi, pi)
                self.assertEqual(result.T_g, T_g)
                self.assertEqual(result.N, self.N)
                self.assertAnchorPoint(self.core_as_point(result), expected)

    def test_cycle_state_is_restored_after_call(self) -> None:
        for key in self.CYCLE_STATE_KEYS:
            self.assertIn(key, self.model.cycle, "cycle state key disappeared")

        before = {key: self.model.cycle[key] for key in self.CYCLE_STATE_KEYS}
        self.oc.cycle_point(self.model, self.PI, self.T_G, self.N)
        anchored = {key: self.model.cycle[key] for key in self.CYCLE_STATE_KEYS}
        self.assertEqual(anchored, before)

        # Use a point that actually selects the cooled branch so that a missing
        # restore cannot pass trivially (the anchored point happens to match
        # the loaded cycle's default state).
        probe = self.oc.cycle_point(self.model, 9.0, 1300.0, 90000.0)
        probe_state = {
            "T_г_полн": 1300.0,
            "охлаждение_лопаток": True,
            "g_охл": 0.05,
            "g_воз": 0.03,
        }
        self.assertNotEqual(
            before,
            probe_state,
            "restoration probe does not change the loaded cycle state",
        )
        self.assertTrue(probe["охлаждение"])
        self.assertEqual(probe["g_ohl"], 0.05)
        self.assertEqual(probe["g_voz"], 0.03)
        restored = {key: self.model.cycle[key] for key in self.CYCLE_STATE_KEYS}
        self.assertEqual(restored, before)


if __name__ == "__main__":
    unittest.main()
