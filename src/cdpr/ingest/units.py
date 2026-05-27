r"""Unit conversion and coordinate-frame normalisation.

Two kinds of fix-ups commonly needed before downstream analysis can trust
the data:

1. **Unit conversion.** Lab data routinely arrives in millimetres,
   degrees, kgf, or whatever the instrument vendor preferred; the
   framework's internal convention is SI (m, rad, N, s, kg). The
   :func:`convert_units` helper applies per-channel scale and offset.

2. **Frame normalisation.** The world-frame origin and axes used by a
   motion-capture system rarely match the CDPR base anchor frame.
   :func:`transform_frame` applies a rigid pose
   :math:`(\mathbf{R}_\text{wm}, \mathbf{t}_\text{wm})` to every
   position-like column so that the resulting data lives in the
   CDPR world frame.

Both operations are explicit --- nothing happens unless the user asks for
it --- so a recorded pipeline can reproduce the exact transformation
applied to the raw data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

from cdpr.ingest.containers import ColumnMap

if TYPE_CHECKING:                                           # pragma: no cover
    import pandas as pd


# ---------------------------------------------------------------------------
# Per-channel scale + offset
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ChannelScale:
    """``y_SI = scale * y_raw + offset`` per channel.

    A scalar scale broadcasts across the three / four components of a
    vector channel. An array scale must match the channel's component
    count. ``offset`` defaults to zero.
    """

    scale: float | ArrayLike = 1.0
    offset: float | ArrayLike = 0.0


def convert_units(
    frame: "pd.DataFrame",
    columns: ColumnMap,
    *,
    position: ChannelScale | None = None,
    quaternion: ChannelScale | None = None,
    euler_deg_to_rad: bool = False,
    linear_velocity: ChannelScale | None = None,
    angular_velocity: ChannelScale | None = None,
    cable_lengths: ChannelScale | None = None,
    cable_tensions: ChannelScale | None = None,
    time: ChannelScale | None = None,
) -> tuple["pd.DataFrame", dict[str, object]]:
    """Apply per-channel scale and offset."""
    out = frame.copy()
    applied: dict[str, dict[str, float]] = {}

    def apply_scale(channel_cols: tuple[str, ...] | None, sc: ChannelScale | None) -> None:
        if channel_cols is None or sc is None:
            return
        scale = np.asarray(sc.scale, dtype=np.float64)
        offset = np.asarray(sc.offset, dtype=np.float64)
        if scale.ndim == 0:
            scale = np.full(len(channel_cols), float(scale))
        if offset.ndim == 0:
            offset = np.full(len(channel_cols), float(offset))
        if len(scale) != len(channel_cols) or len(offset) != len(channel_cols):
            raise ValueError(
                f"Scale/offset length must match channel components: "
                f"channels={len(channel_cols)}, scale={len(scale)}, offset={len(offset)}"
            )
        for k, c in enumerate(channel_cols):
            out[c] = out[c].to_numpy(dtype=np.float64) * float(scale[k]) + float(offset[k])
            applied[c] = {"scale": float(scale[k]), "offset": float(offset[k])}

    apply_scale((columns.time,) if columns.time else None, time)
    apply_scale(columns.position, position)
    apply_scale(columns.quaternion, quaternion)
    apply_scale(columns.linear_velocity, linear_velocity)
    apply_scale(columns.angular_velocity, angular_velocity)
    apply_scale(columns.cable_lengths, cable_lengths)
    apply_scale(columns.cable_tensions, cable_tensions)

    if euler_deg_to_rad and columns.euler is not None:
        for c in columns.euler:
            out[c] = np.deg2rad(out[c].to_numpy(dtype=np.float64))
            applied[c] = {"scale": float(np.pi / 180.0), "offset": 0.0}

    return out, {"applied": applied, "euler_deg_to_rad": euler_deg_to_rad}


# ---------------------------------------------------------------------------
# Frame transform
# ---------------------------------------------------------------------------

def transform_frame(
    frame: "pd.DataFrame",
    columns: ColumnMap,
    *,
    rotation: Rotation | None = None,
    translation: ArrayLike = (0.0, 0.0, 0.0),
    transform_velocities: bool = True,
) -> tuple["pd.DataFrame", dict[str, object]]:
    r"""Rigidly transform every position-like channel into a new world frame.

    The supplied ``rotation`` and ``translation`` define the *new* world
    frame in coordinates of the *current* world frame: a point with raw
    coordinates :math:`\mathbf{x}` lands at
    :math:`\mathbf{R}\mathbf{x} + \mathbf{t}`. Velocities are rotated but
    not translated; angular velocities are rotated.

    Quaternion-valued channels are post-multiplied by the inverse rotation
    so that orientation expressed in the new frame remains a valid
    rotation in that frame.
    """
    R = rotation if rotation is not None else Rotation.identity()
    t = np.asarray(translation, dtype=np.float64).reshape(3)
    out = frame.copy()
    touched: list[str] = []

    def rotate_vector_columns(cols: tuple[str, str, str] | None,
                              translate: bool = False) -> None:
        if cols is None:
            return
        v = out[list(cols)].to_numpy(dtype=np.float64)
        v_new = R.apply(v)
        if translate:
            v_new = v_new + t
        for k, c in enumerate(cols):
            out[c] = v_new[:, k]
            touched.append(c)

    rotate_vector_columns(columns.position, translate=True)
    if transform_velocities:
        rotate_vector_columns(columns.linear_velocity, translate=False)
        rotate_vector_columns(columns.angular_velocity, translate=False)

    if columns.quaternion is not None:
        q_raw = out[list(columns.quaternion)].to_numpy(dtype=np.float64)
        rots = Rotation.from_quat(q_raw)
        new_rots = R * rots
        q_new = new_rots.as_quat()
        for k, c in enumerate(columns.quaternion):
            out[c] = q_new[:, k]
            touched.append(c)

    return out, {
        "rotation_rotvec": R.as_rotvec().tolist(),
        "translation": t.tolist(),
        "transformed_columns": touched,
    }
