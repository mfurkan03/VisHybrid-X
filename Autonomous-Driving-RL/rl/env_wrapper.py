"""
MetaDrive RL Environment Wrapper
=================================
Produces the same 4-channel observation format as the IL training pipeline:
  channel 0 : inverted depth  (closer = higher value)
  channels 1-3 : RGB

Ego state [total_speed, last_steer, heading_delta] is returned alongside the
image as a separate (EGO_DIM,) array, matching the IL model's input API.

Observation : tuple( np.ndarray (4, H, H) float32,
                     np.ndarray (EGO_DIM,) float32 )
Action      : np.ndarray (2,) → [steering, throttle]
"""
import sys
sys.modules['xformers'] = None  # prevent xformers CPU crash (must be first)

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera

# Import IL depth model and ego-state utilities from the parent repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.models import DepthEstimationModel, extract_ego_state, EGO_DIM

from rl.rewards import compute_reward, RewardConfig


class MetaDriveRLWrapper:
    """
    Gym-like wrapper around MetaDriveEnv.

    Observation: tuple(img_np (4, image_size, image_size), ego_np (EGO_DIM,))
    Action     : np.ndarray (2,)  →  [steering ∈ [-1,1], throttle ∈ [-1,1]]
    """

    ACT_DIM = 2

    def __init__(
        self,
        reward_cfg: RewardConfig = None,
        env_config: dict = None,
        show_perception: bool = False,
        image_size: int = 84,
        dpt_path: str = None,
    ):
        self.reward_cfg    = reward_cfg or RewardConfig()
        self.show_perception = show_perception
        self.image_size    = image_size
        self.OBS_SHAPE     = (4, image_size, image_size)

        default_cfg = {
            "use_render": False,
            "show_interface": False,
            # image_observation must be True — otherwise MetaDrive never
            # initialises the RGB sensor and the model runs blind.
            "image_observation": True,
            "sensors": {
                "rgb": (RGBCamera, 196, 196),
            },
            "vehicle_config": {
                "image_source": "rgb",
                "lidar": {"num_lasers": 0, "distance": 0},
                "side_detector": {"num_lasers": 0},
                "lane_line_detector": {"num_lasers": 0},
            },
            "image_on_cuda": False,
            "start_seed": 0,
            "num_scenarios": 50,
            "traffic_density": 0.1,
            "decision_repeat": 5,
            "horizon": 400,
        }
        if env_config:
            default_cfg.update(env_config)

        self.env = MetaDriveEnv(default_cfg)

        # IL-correct depth model: uses inverted depth so closer = higher value.
        self.depth_model = DepthEstimationModel(finetuned_path=dpt_path)

        self._prev_route   = 0.0
        self._prev_action  = np.zeros(self.ACT_DIM, dtype=np.float32)
        self.last_steer    = 0.0
        self._step_count   = 0

        # Fallback obs (returned on sensor failure)
        self._last_img_np  = np.zeros((4, image_size, image_size), dtype=np.float32)
        self._last_ego_np  = np.zeros(EGO_DIM, dtype=np.float32)

    # ── Gym-like interface ────────────────────────────────────────────────────

    def reset(self, seed: int | None = None):
        kwargs = {} if seed is None else {"seed": seed}
        raw_obs, _ = self.env.reset(**kwargs)
        self._prev_route  = 0.0
        self._prev_action = np.zeros(self.ACT_DIM, dtype=np.float32)
        self.last_steer   = 0.0
        self._step_count  = 0
        return self._get_obs(raw_obs)

    def step(self, action: np.ndarray):
        raw_obs, _, terminated, truncated, info = self.env.step(action)

        speed = 0.0
        try:
            speed = self.env.vehicle.speed_km_h
        except Exception:
            pass

        reward, reward_details = compute_reward(
            info, action, self._prev_route, speed, self.reward_cfg, self._prev_action
        )
        self._prev_route  = info.get("route_completion", 0.0)
        self._prev_action = action.copy()
        self.last_steer   = float(action[0])

        obs  = self._get_obs(raw_obs)
        done = terminated or truncated

        info["reward_details"] = reward_details
        info["speed_km_h"]     = speed

        return obs, reward, done, info

    def close(self):
        self.env.close()
        if self.show_perception:
            cv2.destroyAllWindows()

    # ── Observation construction ──────────────────────────────────────────────

    def _extract_rgb(self, raw_obs) -> np.ndarray | None:
        """
        Pull a uint8 RGB (H, W, 3) array from MetaDrive's raw observation dict.
        MetaDrive's RGBCamera already outputs RGB — no colour-channel swap needed.
        Returns None on failure (caller falls back to sensor direct-read).
        """
        if raw_obs is None:
            return None
        try:
            if not (isinstance(raw_obs, dict) and "image" in raw_obs):
                return None
            img = raw_obs["image"]
            # MetaDrive image shape: (H, W, C, stack) float [0,1]
            if img.ndim == 4:
                img = img[:, :, :, -1]   # take the latest frame
            if img.ndim != 3:
                return None
            rgb = img[..., :3]
            if rgb.max() <= 1.0:
                return (rgb * 255).astype(np.uint8)
            return rgb.astype(np.uint8)
        except Exception:
            return None

    def _get_obs(self, raw_obs=None):
        """
        Build observation tuple (img_np, ego_np).

        img_np  : (4, image_size, image_size) float32
                  ch0 = inverted depth (IL-correct)
                  ch1-3 = RGB [0,1]
        ego_np  : (EGO_DIM,) float32 = [total_speed, last_steer, heading_delta]
        """
        rgb_uint8 = self._extract_rgb(raw_obs)

        # Fallback: read directly from the sensor
        if rgb_uint8 is None:
            try:
                rgb_raw = self.env.engine.get_sensor("rgb").perceive(self.env.agent)
                if hasattr(rgb_raw, "get"):
                    rgb_raw = rgb_raw.get()
                rgb_arr = np.array(rgb_raw)
                if rgb_arr.ndim == 3 and rgb_arr.shape[2] >= 3:
                    rgb3 = rgb_arr[..., :3]
                    rgb_uint8 = (rgb3 * 255).astype(np.uint8) if rgb3.max() <= 1.0 else rgb3.astype(np.uint8)
            except Exception as e:
                print(f"[WARNING] RGB sensor read failed: {e}")
                return self._last_img_np, self._last_ego_np

        dev = self.depth_model.device

        # ── Depth: IL pipeline (inverted so closer = higher) ─────────────────
        with torch.no_grad():
            # predict_batch_with_grad expects (N, H, W, 3) uint8 or float32
            depth_raw = self.depth_model.predict_batch_with_grad(rgb_uint8[np.newaxis])
            # depth_raw: (1, 1, H, W) on depth_model.device

        d = depth_raw[0, 0]  # (H, W) tensor on dev
        depth_inv = 1.0 - (d - d.min()) / (d.max() - d.min() + 1e-6)  # (H, W)

        # ── RGB tensor ───────────────────────────────────────────────────────
        rgb_t = torch.from_numpy(rgb_uint8).float().to(dev) / 255.0  # (H, W, 3)
        rgb_t = rgb_t.permute(2, 0, 1).unsqueeze(0)                   # (1, 3, H, W)

        # ── Combine [depth, R, G, B] → (1, 4, H, W) ─────────────────────────
        combined = torch.cat(
            [depth_inv.unsqueeze(0).unsqueeze(0), rgb_t], dim=1
        )  # (1, 4, H, W)

        # ── Resize to image_size ──────────────────────────────────────────────
        if combined.shape[-1] != self.image_size:
            combined = F.interpolate(
                combined,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        img_np = combined.squeeze(0).detach().cpu().float().numpy()  # (4, H, H)

        # ── Ego state ─────────────────────────────────────────────────────────
        try:
            ego_reading = extract_ego_state(self.env.agent, self.last_steer)
            ego_np = ego_reading.ego_model  # (EGO_DIM,) float32
        except Exception:
            ego_np = self._last_ego_np.copy()

        self._step_count += 1

        # ── Optional visualisation (render mode, every 3 steps) ───────────────
        if self.show_perception and self._step_count % 3 == 0:
            sz = (250, 250)
            rgb_disp   = cv2.resize(rgb_uint8, sz)
            depth_vis  = (img_np[0] * 255).astype(np.uint8)
            depth_col  = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
            depth_disp = cv2.resize(depth_col, sz)
            rgb_ch_disp = cv2.resize(
                (img_np[1:4].transpose(1, 2, 0) * 255).astype(np.uint8), sz
            )
            combined_vis = np.hstack([rgb_disp, depth_disp, rgb_ch_disp])
            cv2.imshow("RL View:  RGB  |  Depth  |  RGB (model input)", combined_vis)
            cv2.waitKey(1)

        # Periodic CUDA cache flush
        if self._step_count % 100 == 0:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._last_img_np = img_np
        self._last_ego_np = ego_np
        return img_np, ego_np
