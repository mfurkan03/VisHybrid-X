import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.stats import pearsonr
from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera

import cv2
import time
import sys
from pathlib import Path

# --- DEPTH ANYTHING V2 PATH ---
# Depth-Anything-V2 kütüphanesinin bulunduğu klasör
target_folder = Path(__file__).resolve().parent.parent / 'Depth-Anything-V2'
sys.path.append(str(target_folder))
from depth_anything_v2.dpt import DepthAnythingV2

# ==========================================
# 0. YARDIMCI FONKSİYONLAR
# ==========================================
def get_lane_mask_visual(rgb_image, threshold_value=180):
    if rgb_image.max() <= 1.0:
        img_uint8 = (rgb_image * 255.0).astype(np.uint8)
    else:
        img_uint8 = rgb_image.astype(np.uint8)
        
    h, w = img_uint8.shape[:2]
    roi_img = img_uint8.copy()
    roi_img[0:int(h * 0.55), :] = 0 # Gökyüzünü karart
    
    gray = cv2.cvtColor(roi_img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    return mask

# ==========================================
# 1. DEPTH ESTIMATION MODELİ (V2 GÜNCELLEMESİ)
# ==========================================
class DepthEstimationModel:
    def __init__(self, encoder='vits'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # V2 Encoder Ayarları
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        
        self.model = DepthAnythingV2(**model_configs[encoder])
        # Checkpoint yolu
        ckpt_path = f'{target_folder}/checkpoints/depth_anything_v2_{encoder}.pth'
        
        if os.path.exists(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
            print(f"[INFO] DepthAnythingV2 Yüklendi: {ckpt_path}")
        else:
            print(f"[UYARI] Checkpoint yok: {ckpt_path}")

        self.model = self.model.to(self.device).eval()
        self.use_fp16 = (self.device == 'cuda')

    def process_depth_map(self, depth_raw: np.ndarray) -> np.ndarray:
        depth_resized = cv2.resize(depth_raw, (84, 84), interpolation=cv2.INTER_AREA)
        d_min, d_max = depth_resized.min(), depth_resized.max()
        if d_max - d_min > 1e-6:
            depth_norm = (depth_resized - d_min) / (d_max - d_min)
        else:
            depth_norm = depth_resized - d_min
        return np.expand_dims(depth_norm, axis=0).astype(np.float32)

    def predict_single(self, rgb_image: np.ndarray, return_tensor=False):
        # 1. Veri tipini float64'ten float32'ye veya uint8'e çek
        # MetaDrive görüntüsü 0-255 arasındaysa uint8'e zorla
        if rgb_image.dtype != np.uint8:
            if rgb_image.max() <= 1.0:
                rgb_image = (rgb_image * 255).astype(np.uint8)
            else:
                rgb_image = rgb_image.astype(np.uint8)

        # 2. ÖNEMLİ: Depth-Anything-V2 içeride BGR->RGB dönüşümü yapıyor.
        # MetaDrive RGB verdiği için, içeriye BGR gibi kandırıp yolluyoruz 
        # (Böylece içerideki çevrimden sonra tekrar doğru RGB olur)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        if self.use_fp16:
            with torch.cuda.amp.autocast():
                # infer_image zaten içeride işlemleri yapar
                depth_raw = self.model.infer_image(rgb_image)
        else:
            depth_raw = self.model.infer_image(rgb_image)

        if return_tensor:
            return self.process_depth_tensor(torch.from_numpy(depth_raw).to(self.device))
        return self.process_depth_map(depth_raw)

    def process_depth_tensor(self, depth_raw: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        d_tensor = depth_raw.unsqueeze(0).unsqueeze(0)
        depth_resized = F.interpolate(d_tensor, size=(84, 84), mode='bilinear', align_corners=False)
        d_min, d_max = depth_resized.min(), depth_resized.max()
        if d_max - d_min > 1e-6:
            depth_norm = (depth_resized - d_min) / (d_max - d_min)
        else:
            depth_norm = depth_resized - d_min
        return depth_norm.float()

# ==========================================
# 2. POLICY NETWORK (Şerit ve Derinlik Beyni)
# ==========================================
class DrivingPolicyNet(nn.Module):
    def __init__(self, action_dim=2):
        super().__init__()
        # Şerit Takip Bloğu
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
        # Gaz/Fren Bloğu
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
        lane_input  = x[:, 1:2, :, :]
        
        s_feat = self.flatten(self.steer_conv(lane_input))
        steer_pred = self.steer_fc(s_feat)
        
        a_feat = self.flatten(self.accel_conv(depth_input))
        accel_pred = self.accel_fc(a_feat)
        
        return torch.cat([steer_pred, accel_pred], dim=1)

# ==========================================
# 3. DATASET VE METRİKLER
# ==========================================
class MetaDriveDepthDataset(Dataset):
    def __init__(self, data_dir):
        self.files = glob.glob(os.path.join(data_dir, "*.npz"))
        self.depths, self.actions = [], []
        for f in self.files:
            data = np.load(f, allow_pickle=True)
            combined_key = [k for k in data.files if "_combined" in k]
            if combined_key:
                self.depths.extend(data[combined_key[0]])
                self.actions.extend(data["action"])
    def __len__(self): return len(self.actions)
    def __getitem__(self, idx):
        return torch.tensor(self.depths[idx], dtype=torch.float32), torch.tensor(self.actions[idx], dtype=torch.float32)

def compute_offline_metrics(pred_actions, true_actions):
    steering_mse = float(np.mean((pred_actions[:, 0] - true_actions[:, 0]) ** 2))
    accel_mse = float(np.mean((pred_actions[:, 1] - true_actions[:, 1]) ** 2))
    corr, _ = pearsonr(pred_actions[:, 0], true_actions[:, 0])
    direction_acc = float(np.mean(np.sign(pred_actions[:, 1]) == np.sign(true_actions[:, 1])))
    return {"steering_mse": steering_mse, "accel_mse": accel_mse, "steering_corr": float(corr), "direction_acc": direction_acc}

# ==========================================
# 4. EĞİTİM (TRAINING)
# ==========================================
def train_policy(epochs=20, batch_size=64, model_path="policy_model.pth", data_dir="dataset", val_split=0.2):
    print("--- Eğitim Başlıyor ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MetaDriveDepthDataset(data_dir=data_dir)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = DrivingPolicyNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    def custom_driving_loss(pred, target):
        mse = (pred - target) ** 2
        brake_mask = (target[:, 1] < 0.0).float()
        penalty_weight = 1.0 + (brake_mask * 2.0) # Fren anlarında 3x ceza
        mse_weighted = mse.clone()
        mse_weighted[:, 1] = mse[:, 1] * penalty_weight
        return mse_weighted.mean()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        all_pred, all_true = [], []

        for batch_depths, batch_actions in train_loader:
            batch_depths, batch_actions = batch_depths.to(device), batch_actions.to(device)
            optimizer.zero_grad()
            pred = model(batch_depths)
            loss = custom_driving_loss(pred, batch_actions)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            all_pred.append(pred.detach().cpu().numpy())
            all_true.append(batch_actions.cpu().numpy())

        # Validation
        model.eval()
        val_pred, val_true = [], []
        with torch.no_grad():
            for bd, ba in val_loader:
                p = model(bd.to(device))
                val_pred.append(p.cpu().numpy())
                val_true.append(ba.numpy())

        tr_m = compute_offline_metrics(np.concatenate(all_pred), np.concatenate(all_true))
        val_m = compute_offline_metrics(np.concatenate(val_pred), np.concatenate(val_true))

        print(f"Epoch [{epoch+1:02d}/{epochs}] Loss: {train_loss/len(train_loader):.4f} | "
              f"Steer MSE: {tr_m['steering_mse']:.4f}/{val_m['steering_mse']:.4f} | "
              f"Dir Acc: {val_m['direction_acc']:.3f}")

    torch.save(model.state_dict(), model_path)
    print("Model Kaydedildi.")

# ==========================================
# 5. TEST (INFERENCE)
# ==========================================
def test_policy(model_path, num_episodes):
    print("--- Simülasyon Testi Başlıyor (Depth Anything V2) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    depth_estimator = DepthEstimationModel()
    policy_model = DrivingPolicyNet().to(device)
    policy_model.load_state_dict(torch.load(model_path, map_location=device))
    policy_model.eval()

    config = {
        "use_render": True,
        "image_observation": True,
        "sensors": {"rgb": (RGBCamera, 200, 200)},
        "vehicle_config": {"image_source": "rgb"},
        "show_interface": False,
        "image_on_cuda": False,
    }
    env = MetaDriveEnv(config)

    success_flags, route_completions = [], []

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        step_count = 0
        anlik_fps = 0.0
        last_time = time.time()

        while not done:
            step_count += 1
            rgb_sensor = env.engine.get_sensor("rgb")
            rgb_img = rgb_sensor.perceive(env.agent)

            # 1. Derinlik (V2 GPU Inference)
            depth_tensor = depth_estimator.predict_single(rgb_img, return_tensor=True)
            
            # 2. Şerit
            lane_mask = get_lane_mask_visual(rgb_img)
            lane_norm = (cv2.resize(lane_mask, (84, 84)) / 255.0).astype(np.float32)
            lane_tensor = torch.tensor(lane_norm, device=device).unsqueeze(0).unsqueeze(0)

            # 3. Policy Tahmini
            combined_tensor = torch.cat([depth_tensor, lane_tensor], dim=1)
            with torch.no_grad():
                pred_action = policy_model(combined_tensor).cpu().numpy()[0]

            # Görselleştirme
            if step_count % 2 == 0:
                depth_vis = (depth_tensor[0,0].cpu().numpy() * 255).astype(np.uint8)
                cv2.imshow("Depth V2", cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO))
                cv2.imshow("Lane Mask", cv2.resize(lane_mask, (200, 200)))
                cv2.waitKey(1)

            # Action Repeat & Step
            action_repeat = 3
            for _ in range(action_repeat):
                obs, reward, terminated, truncated, info = env.step(pred_action)
                done = terminated or truncated
                if done: break
            
            # FPS Hesaplama
            cur_time = time.time()
            elapsed = cur_time - last_time
            last_time = cur_time
            if elapsed > 0: anlik_fps = 0.9 * anlik_fps + 0.1 * (1.0 / elapsed)
            print(f"EP: {ep+1} | Steer: {pred_action[0]:.2f} | FPS: {anlik_fps:.1f}", end="\r")

        success_flags.append(bool(info.get("arrive_dest", False)))
        route_completions.append(info.get("route_completion", 0.0))
        print(f"\nBölüm {ep+1} Bitti. Başarı: {success_flags[-1]}")

    print(f"\n=== ÖZET ===\nSuccess: {np.mean(success_flags)*100:.1f}%\nRoute: {np.mean(route_completions)*100:.1f}%")
    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test", "all"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--model_path", type=str,default="policy_model.pth")
    args = parser.parse_args()

    if args.mode in ["train", "all"]:
        train_policy(epochs=args.epochs)
    if args.mode in ["test", "all"]:
        test_policy(args.model_path,args.episodes)