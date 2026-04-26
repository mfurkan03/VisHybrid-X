import argparse
import os
import sys
import types

import cv2
import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


try:
    import metadrive  # noqa: F401
except ModuleNotFoundError:
    metadrive_module = types.ModuleType("metadrive")
    metadrive_module.MetaDriveEnv = object
    component_module = types.ModuleType("metadrive.component")
    sensors_module = types.ModuleType("metadrive.component.sensors")
    rgb_camera_module = types.ModuleType("metadrive.component.sensors.rgb_camera")
    rgb_camera_module.RGBCamera = object

    sys.modules["metadrive"] = metadrive_module
    sys.modules["metadrive.component"] = component_module
    sys.modules["metadrive.component.sensors"] = sensors_module
    sys.modules["metadrive.component.sensors.rgb_camera"] = rgb_camera_module


TRAIN_POLICY_IMPORT_ERROR = None

try:
    from src.train_test_policy import DrivingPolicyNet, DepthEstimationModel, get_lane_mask_visual
except ModuleNotFoundError as exc:
    TRAIN_POLICY_IMPORT_ERROR = exc
    DepthEstimationModel = None

    def get_lane_mask_visual(rgb_image, threshold_value=180):
        if rgb_image.max() <= 1.0:
            img_uint8 = (rgb_image * 255.0).astype(np.uint8)
        else:
            img_uint8 = rgb_image.astype(np.uint8)
        h, _ = img_uint8.shape[:2]
        roi_img = img_uint8.copy()
        roi_img[0:int(h * 0.55), :] = 0
        gray = cv2.cvtColor(roi_img, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        return mask

    class DrivingPolicyNet(nn.Module):
        def __init__(self, action_dim=2):
            super().__init__()
            self.steer_conv = nn.Sequential(
                nn.Conv2d(1, 24, kernel_size=5, stride=2), nn.ReLU(),
                nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ReLU(),
                nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ReLU(),
                nn.Conv2d(48, 64, kernel_size=3, stride=1), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            )
            self.steer_fc = nn.Sequential(
                nn.Linear(64 * 3 * 3, 100), nn.ReLU(),
                nn.Linear(100, 50), nn.ReLU(),
                nn.Linear(50, 10), nn.ReLU(),
                nn.Linear(10, 1),
            )
            self.accel_conv = nn.Sequential(
                nn.Conv2d(1, 24, kernel_size=5, stride=2), nn.ReLU(),
                nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ReLU(),
                nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ReLU(),
                nn.Conv2d(48, 64, kernel_size=3, stride=1), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            )
            self.accel_fc = nn.Sequential(
                nn.Linear(64 * 3 * 3, 100), nn.ReLU(),
                nn.Linear(100, 50), nn.ReLU(),
                nn.Linear(50, 10), nn.ReLU(),
                nn.Linear(10, 1),
            )
            self.flatten = nn.Flatten()

        def forward(self, x):
            depth_input = x[:, 0:1, :, :]
            lane_input = x[:, 1:2, :, :]
            s_feat = self.flatten(self.steer_conv(lane_input))
            steer_pred = self.steer_fc(s_feat)
            a_feat = self.flatten(self.accel_conv(depth_input))
            accel_pred = self.accel_fc(a_feat)
            return torch.cat([steer_pred, accel_pred], dim=1)


def parse_args():
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(
        description="Custom Grad-CAM demo for DrivingPolicyNet on a single RGB image.",
        epilog=(
            "Example:\n"
            "python src/gradcam_demo.py --checkpoint_path model.pth "
            "--image_path test.jpg --output_dir outputs --target steer"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--image_path", default=None)
    parser.add_argument("--npz_path", default=None)
    parser.add_argument("--frame_index", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--depth_checkpoint_path", default=None)
    parser.add_argument("--recompute_from_rgb", action="store_true")
    parser.add_argument("--target", required=True, choices=["steer", "accel"])
    args = parser.parse_args()

    if bool(args.image_path) == bool(args.npz_path):
        parser.error("Provide exactly one of --image_path or --npz_path.")
    if args.frame_index < 0:
        parser.error("--frame_index must be >= 0.")
    return args


def load_rgb_image(image_path):
    bgr_image = cv2.imread(image_path)
    if bgr_image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


def load_rgb_and_combined_from_npz(npz_path, frame_index):
    data = np.load(npz_path, allow_pickle=True)

    rgb_key = None
    if "cam_0_rgb" in data.files:
        rgb_key = "cam_0_rgb"
    elif "rgb" in data.files:
        rgb_key = "rgb"
    else:
        rgb_candidates = [k for k in data.files if "rgb" in k.lower()]
        if rgb_candidates:
            rgb_key = rgb_candidates[0]

    if rgb_key is None:
        raise KeyError(f"No RGB array found in {npz_path}. Keys: {data.files}")

    rgb_frames = data[rgb_key]
    if frame_index >= len(rgb_frames):
        raise IndexError(
            f"frame_index {frame_index} is out of range for {npz_path} "
            f"(num_frames={len(rgb_frames)})"
        )

    rgb_image = rgb_frames[frame_index]
    if rgb_image.dtype != np.uint8:
        if rgb_image.max() <= 1.0:
            rgb_image = (rgb_image * 255.0).astype(np.uint8)
        else:
            rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

    combined_key = None
    if "cam_0_combined" in data.files:
        combined_key = "cam_0_combined"
    elif "combined" in data.files:
        combined_key = "combined"

    combined_frame = None
    if combined_key is not None:
        combined_frames = data[combined_key]
        if frame_index >= len(combined_frames):
            raise IndexError(
                f"frame_index {frame_index} is out of range for combined data in {npz_path} "
                f"(num_frames={len(combined_frames)})"
            )
        combined_frame = combined_frames[frame_index].astype(np.float32)

    return rgb_image, rgb_key, combined_frame, combined_key


def normalize_to_uint8(image):
    image = image.astype(np.float32)
    image -= image.min()
    image /= image.max() + 1e-8
    return (image * 255.0).astype(np.uint8)


def make_overlay(base_gray_or_rgb, cam, alpha=0.45):
    cam_uint8 = normalize_to_uint8(cam)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    if base_gray_or_rgb.ndim == 2:
        base_uint8 = normalize_to_uint8(base_gray_or_rgb)
        base_rgb = cv2.cvtColor(base_uint8, cv2.COLOR_GRAY2RGB)
    else:
        base_rgb = base_gray_or_rgb
        if base_rgb.dtype != np.uint8:
            base_rgb = normalize_to_uint8(base_rgb)

    overlay = cv2.addWeighted(base_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0.0)
    return overlay


def save_rgb(path, image_rgb):
    cv2.imwrite(path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def build_explanation_text(input_source, target, target_layer_name, pred_np):
    if target == "steer":
        target_meaning = (
            "This CAM explains the steering output. Higher activation highlights regions "
            "that most increased the predicted steering value."
        )
        base_meaning = (
            "cam_base_overlay.png uses the lane representation as the base image, because "
            "the steering branch operates on the lane channel."
        )
    else:
        target_meaning = (
            "This CAM explains the acceleration output. Higher activation highlights regions "
            "that most increased the predicted acceleration value."
        )
        base_meaning = (
            "cam_base_overlay.png uses the depth representation as the base image, because "
            "the acceleration branch operates on the depth channel."
        )

    lines = [
        "Grad-CAM Output Explanation",
        "",
        f"Input source: {input_source}",
        f"Selected target: {target}",
        f"Used layer: {target_layer_name}",
        f"Predicted steer: {pred_np[0]:.8f}",
        f"Predicted accel: {pred_np[1]:.8f}",
        "",
        target_meaning,
        base_meaning,
        "",
        "Files:",
        "- input_rgb.png: original RGB frame used for visualization.",
        "- lane_mask.png: lane mask representation used by the policy input pipeline.",
        "- depth_map.png: depth channel representation used by the policy input pipeline.",
        "- cam_base_overlay.png: Grad-CAM heatmap over the branch-specific base representation.",
        "- cam_rgb_overlay.png: Grad-CAM heatmap over the original RGB frame for easier human interpretation.",
        "- prediction.txt: raw run metadata and numeric predictions.",
    ]
    return "\n".join(lines) + "\n"


def prepare_input(rgb_image, depth_estimator, device):
    depth_tensor = depth_estimator.predict_single(rgb_image, return_tensor=True).to(device)

    lane_mask = get_lane_mask_visual(rgb_image)
    lane_resized = cv2.resize(lane_mask, (84, 84), interpolation=cv2.INTER_AREA)
    lane_norm = (lane_resized / 255.0).astype(np.float32)
    lane_tensor = torch.from_numpy(lane_norm).to(device).unsqueeze(0).unsqueeze(0)

    input_tensor = torch.cat([depth_tensor, lane_tensor], dim=1)
    if input_tensor.shape != (1, 2, 84, 84):
        raise RuntimeError(f"Unexpected input tensor shape: {tuple(input_tensor.shape)}")

    depth_map_84 = depth_tensor.detach().cpu().numpy()[0, 0]
    lane_map_84 = lane_tensor.detach().cpu().numpy()[0, 0]
    return input_tensor, lane_mask, lane_map_84, depth_map_84


def create_depth_estimator(device, depth_checkpoint_path=None):
    if DepthEstimationModel is None:
        raise RuntimeError(
            "DepthEstimationModel could not be imported because the Depth-Anything-V2 "
            f"code is missing. Original import error: {TRAIN_POLICY_IMPORT_ERROR}"
        )

    depth_estimator = DepthEstimationModel()
    if hasattr(depth_estimator, "device"):
        depth_estimator.device = str(device)
    if hasattr(depth_estimator, "model"):
        depth_estimator.model = depth_estimator.model.to(device).eval()

    if depth_checkpoint_path:
        depth_checkpoint = torch.load(depth_checkpoint_path, map_location=device)
        depth_estimator.model.load_state_dict(depth_checkpoint)
        depth_estimator.model = depth_estimator.model.to(device).eval()

    return depth_estimator


def prepare_input_from_combined(rgb_image, combined_frame, device):
    if combined_frame is None:
        raise ValueError("combined_frame is required for this path.")
    if combined_frame.shape != (2, 84, 84):
        raise RuntimeError(f"Unexpected combined frame shape: {combined_frame.shape}")

    input_tensor = torch.from_numpy(combined_frame).to(device).unsqueeze(0)
    depth_map_84 = combined_frame[0]
    lane_map_84 = combined_frame[1]
    lane_mask = normalize_to_uint8(lane_map_84)
    return input_tensor, lane_mask, lane_map_84, depth_map_84


def select_target_layer(model, target):
    # Legacy architecture (separate steer/accel conv branches)
    if hasattr(model, "steer_conv") and hasattr(model, "accel_conv"):
        if target == "steer":
            return model.steer_conv[8], "model.steer_conv[8]"
        if target == "accel":
            return model.accel_conv[8], "model.accel_conv[8]"

    # Current architecture (depth/lane feature extractors + shared trunk)
    # We keep the target mapping consistent with the original pipeline intent:
    # steer is lane-focused, accel is depth-focused.
    if hasattr(model, "lane_conv") and hasattr(model, "depth_conv"):
        if target == "steer":
            return model.lane_conv[8], "model.lane_conv[8]"
        if target == "accel":
            return model.depth_conv[8], "model.depth_conv[8]"

    raise ValueError(f"Unsupported target: {target}")


def select_target_scalar(pred, target):
    if target == "steer":
        return pred[0, 0]
    if target == "accel":
        return pred[0, 1]
    raise ValueError(f"Unsupported target: {target}")


def compute_gradcam(model, input_tensor, target, target_layer):
    activations = None
    gradients = None

    def forward_hook(module, module_input, module_output):
        nonlocal activations
        activations = module_output

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        pred = model(input_tensor)
        target_scalar = select_target_scalar(pred, target)

        model.zero_grad(set_to_none=True)
        target_scalar.backward()

        if activations is None or gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1)
        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= cam.max() + 1e-8

        return pred.detach(), cam.detach().cpu().numpy()[0]
    finally:
        forward_handle.remove()
        backward_handle.remove()


def run_gradcam_for_rgb(
    model,
    rgb_image,
    input_source,
    target,
    device,
    combined_frame=None,
    depth_estimator=None,
):
    original_h, original_w = rgb_image.shape[:2]

    if combined_frame is not None:
        input_tensor, lane_mask, lane_map_84, depth_map_84 = prepare_input_from_combined(
            rgb_image, combined_frame, device
        )
    else:
        if depth_estimator is None:
            depth_estimator = create_depth_estimator(device)
        input_tensor, lane_mask, lane_map_84, depth_map_84 = prepare_input(
            rgb_image, depth_estimator, device
        )

    target_layer, target_layer_name = select_target_layer(model, target)
    pred, cam_84 = compute_gradcam(model, input_tensor, target, target_layer)

    cam_original = cv2.resize(cam_84, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
    lane_original = cv2.resize(lane_map_84, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    depth_original = cv2.resize(depth_map_84, (original_w, original_h), interpolation=cv2.INTER_CUBIC)

    if target == "steer":
        base_image = lane_original
    else:
        base_image = depth_original

    cam_base_overlay = make_overlay(base_image, cam_original)
    cam_rgb_overlay = make_overlay(rgb_image, cam_original)
    pred_np = pred.cpu().numpy()[0]
    explanation_text = build_explanation_text(
        input_source=input_source,
        target=target,
        target_layer_name=target_layer_name,
        pred_np=pred_np,
    )

    return {
        "input_rgb": rgb_image,
        "lane_mask": normalize_to_uint8(lane_original),
        "depth_map": normalize_to_uint8(depth_original),
        "cam_base_overlay": cam_base_overlay,
        "cam_rgb_overlay": cam_rgb_overlay,
        "pred_np": pred_np,
        "target_layer_name": target_layer_name,
        "explanation_text": explanation_text,
        "input_source": input_source,
        "target": target,
    }


def save_gradcam_outputs(output_dir, checkpoint_path, result):
    os.makedirs(output_dir, exist_ok=True)

    save_rgb(os.path.join(output_dir, "input_rgb.png"), result["input_rgb"])
    cv2.imwrite(os.path.join(output_dir, "lane_mask.png"), result["lane_mask"])
    cv2.imwrite(os.path.join(output_dir, "depth_map.png"), result["depth_map"])
    save_rgb(os.path.join(output_dir, "cam_base_overlay.png"), result["cam_base_overlay"])
    save_rgb(os.path.join(output_dir, "cam_rgb_overlay.png"), result["cam_rgb_overlay"])

    pred_np = result["pred_np"]
    with open(os.path.join(output_dir, "prediction.txt"), "w", encoding="utf-8") as f:
        f.write(f"checkpoint path: {checkpoint_path}\n")
        f.write(f"input source: {result['input_source']}\n")
        f.write(f"selected target: {result['target']}\n")
        f.write(f"predicted steer: {pred_np[0]:.8f}\n")
        f.write(f"predicted accel: {pred_np[1]:.8f}\n")
        f.write(f"used layer: {result['target_layer_name']}\n")

    with open(os.path.join(output_dir, "explanation.txt"), "w", encoding="utf-8") as f:
        f.write(result["explanation_text"])


def main():
    args = parse_args()
    device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.image_path:
        rgb_image = load_rgb_image(args.image_path)
        input_source = args.image_path
        combined_frame = None
    else:
        rgb_image, rgb_key, combined_frame, combined_key = load_rgb_and_combined_from_npz(
            args.npz_path, args.frame_index
        )
        if combined_key and not args.recompute_from_rgb:
            input_source = (
                f"{args.npz_path} [rgb_key={rgb_key}, combined_key={combined_key}, "
                f"frame_index={args.frame_index}]"
            )
        else:
            input_source = (
                f"{args.npz_path} [rgb_key={rgb_key}, frame_index={args.frame_index}, "
                f"recompute_from_rgb=True]"
            )
            combined_frame = None

    model = DrivingPolicyNet().to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    depth_estimator = None
    if combined_frame is None:
        depth_estimator = create_depth_estimator(device, args.depth_checkpoint_path)

    result = run_gradcam_for_rgb(
        model=model,
        rgb_image=rgb_image,
        input_source=input_source,
        target=args.target,
        device=device,
        combined_frame=combined_frame,
        depth_estimator=depth_estimator,
    )
    save_gradcam_outputs(args.output_dir, args.checkpoint_path, result)

    print(f"Saved Grad-CAM outputs to: {args.output_dir}")
    print(f"Prediction steer={result['pred_np'][0]:.6f}, accel={result['pred_np'][1]:.6f}")
    print(f"Target={args.target}, layer={result['target_layer_name']}")


if __name__ == "__main__":
    main()
