"""XHAND1 tactile helpers for MuJoCo.

Native ``touch`` sensors provide robust scalar normal force.  ``TactileReader``
also reconstructs a 120x3 force field per finger from MuJoCo contacts, with all
forces rotated into each distal ``link2`` frame before accumulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


FINGERS = ("thumb", "index", "mid", "ring", "pinky")


@dataclass(frozen=True)
class FingerSummary:
    active_taxels: int
    max_normal_force: float
    net_force_link: np.ndarray
    peak_force_link: float


class TactileReader:
    """Read native normal force and reconstructed 3-axis force for one hand."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, side: str):
        if side not in ("right", "left"):
            raise ValueError("side 必须为 right 或 left")
        self.model = model
        self.data = data
        self.side = side

        body_names = {
            "thumb": f"{side}_hand_thumb_rota_link2",
            "index": f"{side}_hand_index_rota_link2",
            "mid": f"{side}_hand_mid_link2",
            "ring": f"{side}_hand_ring_link2",
            "pinky": f"{side}_hand_pinky_link2",
        }
        self.body_ids = np.array(
            [model.body(body_names[finger]).id for finger in FINGERS], dtype=int
        )
        self.weld_ids = model.body_weldid[self.body_ids].copy()
        self.site_ids = np.array(
            [
                [
                    model.site(f"{side}_tactile_{finger}_{point:03d}_site").id
                    for point in range(1, 121)
                ]
                for finger in FINGERS
            ],
            dtype=int,
        )
        sensor_ids = np.array(
            [
                [
                    model.sensor(f"{side}_tactile_{finger}_{point:03d}").id
                    for point in range(1, 121)
                ]
                for finger in FINGERS
            ],
            dtype=int,
        )
        if np.any(model.sensor_dim[sensor_ids] != 1):
            raise ValueError("XHAND touch sensor 应为标量")
        self.sensor_addresses = model.sensor_adr[sensor_ids].copy()

    def normal_forces(self) -> np.ndarray:
        """Return native touch output with shape ``(5, 120)`` in newtons."""

        return self.data.sensordata[self.sensor_addresses].copy()

    def taxel_forces_link(self, max_assignment_distance: float = 0.006) -> np.ndarray:
        """Return contact force per taxel in distal-link coordinates.

        Shape is ``(5, 120, 3)`` and finger order is ``FINGERS``.  Each contact
        is assigned once to the nearest measurement point on the involved distal
        link.  ``mj_contactForce`` is first rotated contact->world, its sign is
        corrected for geom1/geom2, and then it is rotated world->link.
        """

        forces = np.zeros((len(FINGERS), 120, 3), dtype=np.float64)
        contact_force = np.zeros(6, dtype=np.float64)

        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            if contact.efc_address < 0 or contact.geom1 < 0 or contact.geom2 < 0:
                continue

            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]
            weld1 = self.model.body_weldid[body1]
            weld2 = self.model.body_weldid[body2]
            matched = np.flatnonzero((self.weld_ids == weld1) | (self.weld_ids == weld2))
            if not matched.size:
                continue

            mujoco.mj_contactForce(self.model, self.data, contact_index, contact_force)
            frame = contact.frame.reshape(3, 3)
            force_world_on_geom2 = frame.T @ contact_force[:3]

            for finger_index in matched:
                finger_weld = self.weld_ids[finger_index]
                if finger_weld == weld1 and finger_weld == weld2:
                    continue
                force_world = (
                    -force_world_on_geom2 if finger_weld == weld1 else force_world_on_geom2
                )
                body_id = self.body_ids[finger_index]
                rotation_link_to_world = self.data.xmat[body_id].reshape(3, 3)
                force_link = rotation_link_to_world.T @ force_world

                sites_world = self.data.site_xpos[self.site_ids[finger_index]]
                distances = np.linalg.norm(sites_world - contact.pos, axis=1)
                taxel_index = int(np.argmin(distances))
                if distances[taxel_index] <= max_assignment_distance:
                    forces[finger_index, taxel_index] += force_link

        return forces

    def taxel_forces_sensor(self, max_assignment_distance: float = 0.006) -> np.ndarray:
        """Return reconstructed forces in the delivered sensor-axis convention."""

        link_forces = self.taxel_forces_link(max_assignment_distance)
        sensor_forces = link_forces.copy()

        # T16 transform: link=(sensor_z, sensor_x, sensor_y), therefore
        # sensor=(link_y, link_z, link_x).  It applies to four non-thumb fingers.
        sensor_forces[1:, :, 0] = link_forces[1:, :, 1]
        sensor_forces[1:, :, 1] = link_forces[1:, :, 2]
        sensor_forces[1:, :, 2] = link_forces[1:, :, 0]

        # T30 right keeps axes.  The delivered left transform mirrors Y.
        if self.side == "left":
            sensor_forces[0, :, 1] *= -1
        return sensor_forces

    def summary(self) -> dict[str, FingerSummary]:
        normal = self.normal_forces()
        forces = self.taxel_forces_link()
        result: dict[str, FingerSummary] = {}
        for finger_index, finger in enumerate(FINGERS):
            force_norms = np.linalg.norm(forces[finger_index], axis=1)
            result[finger] = FingerSummary(
                active_taxels=int(np.count_nonzero(normal[finger_index] > 1e-8)),
                max_normal_force=float(np.max(normal[finger_index], initial=0.0)),
                net_force_link=forces[finger_index].sum(axis=0),
                peak_force_link=float(np.max(force_norms, initial=0.0)),
            )
        return result

