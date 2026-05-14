"""
MetaDrive RL Environment Wrapper
=================================
Produces the same 4-channel observation format as the IL training pipeline,
using the same camera rig (build_cameras) and BGR→RGB conversion as IL:
  channel 0 : inverted depth  (closer = higher value)
  channels 1-3 : RGB

Ego state [total_speed, last_steer, heading_delta] is returned alongside the
image as a separate (EGO_DIM,) array, matching the IL model's input API.

Two modes controlled by use_depth_model:
  True  (default) : full pipeline — DPT depth + RGB → (4, H, H) float32
  False (worker)  : raw RGB only  → (196, 196, 3) uint8
                    DPT runs in the main process on a batch of N images.

Observation (use_depth_model=True)  : tuple( np.ndarray (4, H, H) float32,
                                             np.ndarray (EGO_DIM,) float32 )
Observation (use_depth_model=False) : tuple( np.ndarray (196, 196, 3) uint8,
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

# Import IL depth model, ego-state utilities, and camera builder from the parent repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.models import DepthEstimationModel, extract_ego_state, EGO_DIM
from src.data.cameras import build_cameras

from rl.rewards import compute_reward, RewardConfig


class MetaDriveRLWrapper:
    """
    Gym-like wrapper around MetaDriveEnv.

    use_depth_model=True  (default): standard single-env mode. DPT runs
        inside this wrapper per step. Returns (4, H, W) float32 + ego.
    use_depth_model=False (worker mode): skips DPT entirely. Returns
        (196, 196, 3) uint8 + ego. Used in SubprocVecEnv worker processes;
        the main process handles batched DPT for all N workers at once.
    """

    ACT_DIM = 2

    def __init__(
        self,
        reward_cfg: RewardConfig = None,
        env_config: dict = None,
        show_perception: bool = False,
        image_size: int = 84,
        dpt_path: str = None,
        use_depth_model: bool = True,
        render: bool = False,
    ):
        self.reward_cfg      = reward_cfg or RewardConfig()
        self.show_perception = show_perception
        self.image_size      = image_size
        self.use_depth_model = use_depth_model
        self.render          = render
        self.OBS_SHAPE       = (4, image_size, image_size)

        # Use the same camera rig as IL training so the model sees the same viewpoint.
        _, il_sensors, rgb_cam_names, _ = build_cameras(1)
        self.rgb_name = rgb_cam_names[0]  # "cam_0"

        default_cfg = {
            "use_render": render,
            "show_interface": False,
            # image_observation must be True — otherwise MetaDrive never
            # initialises the RGB sensor and the model runs blind.
            "image_observation": True,
            "sensors": il_sensors,
            "vehicle_config": {
                "image_source": self.rgb_name,
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

        if use_depth_model:
            # IL-correct depth model: uses inverted depth so closer = higher value.
            self.depth_model = DepthEstimationModel(finetuned_path=dpt_path)
        else:
            self.depth_model = None

        self._prev_route   = 0.0
        self._prev_action  = np.zeros(self.ACT_DIM, dtype=np.float32)
        self.last_steer    = 0.0
        self._step_count   = 0

        # Fallback obs shapes differ by mode
        if use_depth_model:
            self._last_img_np = np.zeros((4, image_size, image_size), dtype=np.float32)
        else:
            self._last_img_np = np.zeros((196, 196, 3), dtype=np.uint8)
        self._last_ego_np = np.zeros(EGO_DIM, dtype=np.float32)

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
        if self.show_perception or self.render:
            cv2.destroyAllWindows()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_ego(self) -> np.ndarray:
        """Read ego state [total_speed, last_steer, heading_delta]."""
        try:
            ego_reading = extract_ego_state(self.env.agent, self.last_steer)
            return ego_reading.ego_model   # (EGO_DIM,) float32
        except Exception:
            return self._last_ego_np.copy()

    def _extract_rgb(self) -> np.ndarray | None:
        """
        Pull a uint8 RGB (H, W, 3) array from the IL camera via perceive().
        MetaDrive perceive() returns BGR — we flip to RGB to match IL training.
        Returns None on failure.
        """
        try:
            rgb_raw = self.env.engine.get_sensor(self.rgb_name).perceive(
                to_float=False, new_parent_node=self.env.agent.origin
            )
            if hasattr(rgb_raw, "get"):
                rgb_raw = rgb_raw.get()
            rgb_arr = np.array(rgb_raw, dtype=np.uint8)
            # MetaDrive perceive() returns BGR; convert to RGB to match IL training.
            return rgb_arr[..., ::-1].copy()
        except Exception as e:
            print(f"[WARNING] RGB sensor read failed: {e}")
            return None

    def _extract_raw_rgb(self) -> np.ndarray:
        """
        Return (196, 196, 3) uint8 RGB without running DPT.
        Used in worker mode (use_depth_model=False).
        Falls back to the cached fallback on failure.
        """
        rgb_uint8 = self._extract_rgb()
        if rgb_uint8 is None:
            rgb_uint8 = self._last_img_np.copy()
        return rgb_uint8

    def _get_obs(self, raw_obs=None):
        """
        Build observation tuple (img, ego).

        Worker mode (use_depth_model=False):
            img  : (196, 196, 3) uint8
            ego  : (EGO_DIM,) float32
        Standard mode (use_depth_model=True):
            img  : (4, image_size, image_size) float32
                   ch0 = inverted depth (IL-correct)
                   ch1-3 = RGB [0,1]
            ego  : (EGO_DIM,) float32
        """
        # ── Worker mode: skip DPT, return raw RGB ────────────────────────────
        if not self.use_depth_model:
            rgb_uint8 = self._extract_raw_rgb()
            ego_np    = self._get_ego()
            self._step_count += 1
            self._last_img_np = rgb_uint8
            self._last_ego_np = ego_np

            if self.render and self._step_count % 3 == 0:
                self._show_info_panel(ego_np)

            return rgb_uint8, ego_np

        # ── Standard mode: DPT depth + RGB ───────────────────────────────────
        rgb_uint8 = self._extract_rgb()

        if rgb_uint8 is None:
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
        ego_np = self._get_ego()

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

    def _show_info_panel(self, ego_np: np.ndarray):
        """Draw a CV2 overlay showing steer, throttle, and ego state."""
        steer    = float(self._prev_action[0])
        throttle = float(self._prev_action[1])
        speed, last_steer, heading_delta = (float(ego_np[i]) for i in range(3))

        W, H = 400, 220
        panel = np.zeros((H, W, 3), dtype=np.uint8)

        def _bar(y, label, val, lo, hi, color):
            cv2.putText(panel, f"{label}: {val:+.3f}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            bx, bw = 180, 180
            frac = (val - lo) / max(hi - lo, 1e-6)
            frac = max(0.0, min(1.0, frac))
            cv2.rectangle(panel, (bx, y - 12), (bx + bw, y), (50, 50, 50), -1)
            cv2.rectangle(panel, (bx, y - 12), (bx + int(bw * frac), y), color, -1)

        _bar(30,  "Steer   ",    steer,    -1.0,  1.0, (0, 200, 255))
        _bar(70,  "Throttle",    throttle, -1.0,  1.0, (0, 255, 100))
        _bar(110, "Speed(kph)",  speed,     0.0, 60.0, (255, 180,  50))
        _bar(150, "Heading Delta",   heading_delta, -1.0, 1.0, (200, 100, 255))
        _bar(190, "Last steer",  last_steer,    -1.0, 1.0, (100, 200, 255))

        cv2.imshow("RL Info Panel", panel)
        cv2.waitKey(1)
