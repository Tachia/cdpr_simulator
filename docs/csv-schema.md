# CSV schema and robot-config format

How Phase-2 (`train_from_csv.py`, `compare_models.py`, the Streamlit
upload panel) maps arbitrary CSVs into the canonical layout the
analysis pipeline expects, and how to drive any robot geometry —
small tabletop or heavy industrial — through the simulator.

## Canonical CSV layout

`scripts/run_simulation.py` writes a wide CSV with these columns. Any
analysis CSV that uses the same names works out of the box; CSVs that
use other names are auto-mapped through the alias table below.

| Column | Meaning | Units |
|---|---|---|
| `t` | timestamp | seconds |
| `px`, `py`, `pz` | end-effector position | metres |
| `qx`, `qy`, `qz`, `qw` | end-effector orientation (xyzw quaternion) | unitless |
| `vx`, `vy`, `vz` | end-effector translational velocity | m/s |
| `wx`, `wy`, `wz` | end-effector angular velocity | rad/s |
| `px_ref`, `py_ref`, `pz_ref` | reference (commanded) position | metres |
| `track_err` | Euclidean position-tracking error | metres |
| `L1`, `L2`, … `Lm` | per-cable length | metres |
| `T1`, `T2`, … `Tm` | per-cable tension | newtons |
| `feasible` | 1 if every tension in [t_min, t_max] this step | 0/1 |
| `infeasible_qp` | 1 if the QP allocator itself reported infeasible | 0/1 |

Only `t, px, py, pz` are strictly required. Velocities are
finite-differenced from positions when absent; quaternions default to
the identity orientation; angular velocity defaults to zero.

## Auto-detected aliases

`scripts/_csv_io.py` carries an alias table — a single CSV column
matching any of these names (case-insensitive, with underscores or
spaces) becomes the canonical column. Examples:

| Canonical | Accepted aliases (representative) |
|---|---|
| `t` | `Time`, `Timestamp`, `ts`, `sec`, `seconds` |
| `px` | `x`, `Pos X`, `Position X`, `pos_x`, `x_m` |
| `vx` | `velocity_x`, `vel_x`, `linear_vel_x` |
| `wx` | `omega_x`, `ang_vel_x` |
| `qw` | `quat_w`, `q0` |

For cable-indexed columns the loader requires a **numeric suffix**
(`L1`, `tension_4`, `cable_2`, `force_7` ✓ — `Layer`, `Time`,
`tensionMean` ✗). Accepted prefixes:

* lengths: `L`, `length`, `cable_length`, `cable_len`, `len`
* tensions: `T`, `tension`, `cable_tension`, `cable`, `force`, `tau`

If the auto-mapping misses a column you need, pass an explicit map:

```powershell
python scripts\train_from_csv.py --input data.csv --model mlp `
    --column-map "px=ee_pos_x,py=ee_pos_y,pz=ee_pos_z"
```

## URL inputs

Any CLI / Streamlit input accepting a CSV also accepts an
`http://` / `https://` URL. The file is downloaded once to a stable
temp path (printed to the console) so subsequent runs reuse the
cached copy:

```powershell
python scripts\train_from_csv.py --input https://example.com/data.csv --model pinn
```

## Robot-config JSON format

`scripts/run_simulation.py --robot-config robots/custom.json` accepts
a JSON description of any robot geometry — useful for industrial
("heavy" 50–5000 N) or tabletop ("light" 0.5–50 N) scenarios.

```json
{
    "name": "industrial-heavy-8cable",
    "n_cables": 8,
    "dof": 6,
    "anchors":     [[5.0, 5.0, 4.0], [-5.0, 5.0, 4.0], "..."],
    "attachments": [[0.6, 0.6, 0.3], [-0.6, 0.6, 0.3], "..."],
    "mass": 350.0,
    "inertia": [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]],
    "com": [0.0, 0.0, 0.0],
    "t_min": 50.0,
    "t_max": 5000.0,
    "cable_diameter_m": 0.012
}
```

Field semantics:

| Field | Type | Notes |
|---|---|---|
| `name` | str | cosmetic label |
| `n_cables` | int | must match `len(anchors) == len(attachments)` |
| `dof` | int | 3 for point-mass, 6 for full SE(3) |
| `anchors` | list of `[x,y,z]` | fixed frame attachment points (m) |
| `attachments` | list of `[x,y,z]` | platform body-frame attachment points (m) |
| `mass` | float | platform mass (kg); payload via `--payload-mass` |
| `inertia` | 3×3 list | principal inertia tensor about CoM (kg·m²) |
| `com` | `[x,y,z]` | centre of mass in body frame (m) |
| `t_min`, `t_max` | float | per-cable tension bounds (N) |
| `cable_diameter_m` | float | only used by elastic / sagging models |

Same JSON layout is persisted automatically as the `robot_spec` block
inside the manifest, so Phase-2 replay / RL workflows can rebuild any
geometry without code changes.

## Worked example: heavy industrial CDPR with 50–5000 N tension

```powershell
python scripts\run_simulation.py `
    --robot-config examples\robots\industrial_heavy.json `
    --kind circle --radius 2.0 --duration 30 `
    --controller pd --kp-pos 800 --kp-rot 200 `
    --t-min 50 --t-max 5000 `
    --out out\industrial-circle --open
```

The output CSV + manifest can then drive Phase-2 without any further
configuration:

```powershell
python scripts\compare_models.py `
    --input out\industrial-circle\timeseries.csv `
    --out  out\industrial-compare `
    --models replay mlp pinn ppo sac `
    --epochs 80 --rl-steps 5000
```
