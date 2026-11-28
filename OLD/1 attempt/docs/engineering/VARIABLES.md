# Variables and units

All production core interfaces use SI unless an identifier or docstring states
otherwise.

| Symbol | Meaning | Unit |
|---|---|---|
| `r` | radius | m |
| `p` | static pressure | Pa |
| `T` | static temperature | K |
| `rho` | gas density | kg/m3 |
| `R` | specific gas constant | J/(kg K) |
| `cp` | specific heat at constant pressure | J/(kg K) |
| `s` | specific entropy relative to a reference | J/(kg K) |
| `c_a`, `c_u` | absolute axial/tangential velocity | m/s |
| `w` | relative velocity | m/s |
| `u` | blade speed | m/s |
| `h0` | stagnation enthalpy | J/kg |
| `G` | mass flow | kg/s |
| `N` | shaft power | W |
| `n` | rotational speed | rpm |
| `alpha` | absolute-flow angle | degree at project interfaces |
| `sigma` | blade stress | MPa in project reports |
| `l_c` | stator/nozzle blade height | m |
| `l_l` / `l_л` | rotor blade height | m |

Angles passed to NumPy trigonometric functions must be converted explicitly to
radians. Stress units must be converted explicitly before combining Pa and MPa.
