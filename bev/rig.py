"""The camera rig and every coordinate transform the pipeline uses.

Frames (all right-handed, metres):
  world   the simulator's frame (x east-ish, y north-ish, z up)
  ego     origin on the ground under the ego's centre; x forward, y left, z up (nuScenes ego / BEVFormer "lidar" frame)
  camera  OpenCV: x right, y down, z forward (along the optical axis)
  image   pixels: u right, v down, (0, 0) the top-left corner

A camera is (name, yaw offset from the ego's heading, horizontal field of view, mount position in ego). BEVFormer's
`lidar2img` is our `ego2img` = K @ [R | t]: 4 x 4, mapping homogeneous ego points to (u*z, v*z, z, 1).
"""
import math
from dataclasses import dataclass

import numpy as np

# nuScenes' 6-camera layout (yaw offsets, rounded) and its fields of view: ~70 deg, the back camera ~110 deg
NUSCENES_LIKE = (("CAM_FRONT", 0.0, 70.0), ("CAM_FRONT_LEFT", 55.0, 70.0), ("CAM_FRONT_RIGHT", -55.0, 70.0),
                 ("CAM_BACK", 180.0, 110.0), ("CAM_BACK_LEFT", 110.0, 70.0), ("CAM_BACK_RIGHT", -110.0, 70.0))


@dataclass
class Rig:
    width: int = 176          # nuScenes images are 1600 x 900; BEVFormer-tiny uses 800 x 450; this rig ~1/9 of that
    height: int = 96
    mount_z: float = 1.6
    mount_r: float = 0.5      # cameras sit on a 0.5 m circle around the ego's centre, facing out
    cams: tuple = NUSCENES_LIKE

    @property
    def n(self):
        return len(self.cams)

    def K(self, i):
        f = (self.width / 2) / math.tan(math.radians(self.cams[i][2]) / 2)
        return np.array([[f, 0.0, self.width / 2], [0.0, f, self.height / 2], [0.0, 0.0, 1.0]])

    def vfov(self, i):
        """Vertical field of view in degrees (what VTK / pyvista call the view angle)."""
        return math.degrees(2 * math.atan(math.tan(math.radians(self.cams[i][2]) / 2) * self.height / self.width))

    def mount(self, i):
        a = math.radians(self.cams[i][1])
        return np.array([self.mount_r * math.cos(a), self.mount_r * math.sin(a), self.mount_z])

    def R(self, i):
        """Rotation ego -> camera: rows are the camera's x (right), y (down), z (forward) axes written in ego coordinates."""
        a = math.radians(self.cams[i][1])
        fwd = np.array([math.cos(a), math.sin(a), 0.0])
        right = np.array([math.sin(a), -math.cos(a), 0.0])
        return np.stack([right, np.array([0.0, 0.0, -1.0]), fwd])

    def ego2cam(self, i):
        T = np.eye(4)
        T[:3, :3] = self.R(i)
        T[:3, 3] = -self.R(i) @ self.mount(i)
        return T

    def ego2img(self):
        """[n_cams, 4, 4]: BEVFormer's lidar2img."""
        out = []
        for i in range(self.n):
            K4 = np.eye(4)
            K4[:3, :3] = self.K(i)
            out.append(K4 @ self.ego2cam(i))
        return np.stack(out)

    def project(self, P):
        """Ego points [..., 3] -> pixels [n_cams, ..., 2], depth [n_cams, ...], valid [n_cams, ...] (in front and inside the image)."""
        P = np.asarray(P, float)
        h = np.concatenate([P, np.ones(P.shape[:-1] + (1,))], -1)
        q = np.einsum("cij,...j->c...i", self.ego2img(), h)
        z = q[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = q[..., :2] / np.maximum(z, 1e-5)[..., None]
        valid = (z > 0.1) & (uv[..., 0] >= 0) & (uv[..., 0] < self.width) & (uv[..., 1] >= 0) & (uv[..., 1] < self.height)
        return uv, z, valid

    def world_eye_target(self, i, pose):
        """Where the renderer puts camera i for an ego at world pose (x, y, yaw): (eye, target) in world coordinates."""
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        e2w = lambda p: np.array([x + c * p[0] - s * p[1], y + s * p[0] + c * p[1], p[2]])  # noqa: E731
        m = self.mount(i)
        return e2w(m), e2w(m + self.R(i)[2] * 10.0)


def world_to_ego(xy, pose):
    """World points [..., 2] -> ego frame for an ego at world pose (x, y, yaw)."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    d = np.asarray(xy, float) - np.array([x, y])
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)


def ego_motion(pose_prev, pose_now):
    """The 2-D rigid transform taking a point in the previous ego frame to the current one: (R [2, 2], t [2])."""
    xp, yp, ap = pose_prev
    xn, yn, an = pose_now
    da = ap - an
    R = np.array([[math.cos(da), -math.sin(da)], [math.sin(da), math.cos(da)]])
    t = world_to_ego(np.array([xp, yp]), pose_now)
    return R, t


def box_corners(box):
    """[x, y, z (centre), l, w, h, yaw] in ego -> the 8 corners [8, 3]."""
    x, y, z, l, w, h, yaw = box[:7]
    c, s = math.cos(yaw), math.sin(yaw)
    xs = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * l / 2
    ys = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * w / 2
    zs = np.array([-1, -1, -1, -1, 1, 1, 1, 1]) * h / 2
    return np.stack([x + c * xs - s * ys, y + s * xs + c * ys, z + zs], 1)
