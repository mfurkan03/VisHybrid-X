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
# ==========================================
# 1. DEPTH ESTIMATION MODELİ
# ==========================================
class DepthEstimationModel:
    def __init__(self, model_path=None):
        pass

    def predict(self, rgb_image: np.ndarray) -> np.ndarray:
        depth_map = np.mean(rgb_image, axis=2, keepdims=True)
        depth_map = np.transpose(depth_map, (2, 0, 1))
        return (depth_map / 255.0).astype(np.float32)


# ==========================================
# 3. MODEL MİMARİSİ VE DATASET
# ==========================================
class DrivingPolicyNet(nn.Module):
    def __init__(self, action_dim=2):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 3 * 3, 100), nn.ReLU(),  # ← 576
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 10), nn.ReLU(),
            nn.Linear(10, action_dim),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        return self.fc_layers(x)


class MetaDriveDepthDataset(Dataset):
    def __init__(self, data_dir):
        self.files = glob.glob(os.path.join(data_dir, "*.npz"))
        self.depths, self.actions = [], []
        for f in self.files:
            data = np.load(f)
            self.depths.extend(data["depth"])
            self.actions.extend(data["action"])

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.depths[idx], dtype=torch.float32),
            torch.tensor(self.actions[idx], dtype=torch.float32),
        )


# ==========================================
# 4. OFFLİNE METRİK HESAPLAMA
# ==========================================
def compute_offline_metrics(pred_actions: np.ndarray, true_actions: np.ndarray) -> dict:
    """
    pred_actions, true_actions: (N, 2) → [:, 0] steering, [:, 1] acceleration
    """
    steering_mse = float(np.mean((pred_actions[:, 0] - true_actions[:, 0]) ** 2))
    accel_mse    = float(np.mean((pred_actions[:, 1] - true_actions[:, 1]) ** 2))

    # Pearson korelasyon — steering
    corr, _ = pearsonr(pred_actions[:, 0], true_actions[:, 0])
    steering_corr = float(corr)

    # Yönsel doğruluk: işaret (gaz/fren yönü) aynı mı?
    direction_acc = float(
        np.mean(np.sign(pred_actions[:, 1]) == np.sign(true_actions[:, 1]))
    )

    return {
        "steering_mse":   steering_mse,
        "accel_mse":      accel_mse,
        "steering_corr":  steering_corr,
        "direction_acc":  direction_acc,
    }


# ==========================================
# 5. EĞİTİM (TRAINING) — metriklerle
# ==========================================
def train_policy(
    epochs=20,
    batch_size=64,
    model_path="policy_model.pth",
    data_dir="dataset",
    val_split=0.2,
):
    print("--- Eğitim Başlıyor ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan Cihaz: {device}")

    dataset = MetaDriveDepthDataset(data_dir=data_dir)
    if len(dataset) == 0:
        print(f"HATA: '{data_dir}' klasöründe dataset boş!")
        return

    # Train / Validation split
    val_size   = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    print(f"Train: {train_size} | Validation: {val_size} örnek")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)

    model     = DrivingPolicyNet(action_dim=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        # ── TRAIN ──
        model.train()
        train_loss = 0.0
        all_pred, all_true = [], []

        for batch_depths, batch_actions in train_loader:
            batch_depths  = batch_depths.to(device)
            batch_actions = batch_actions.to(device)
            optimizer.zero_grad()
            pred = model(batch_depths)
            loss = criterion(pred, batch_actions)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            all_pred.append(pred.detach().cpu().numpy())
            all_true.append(batch_actions.cpu().numpy())

        all_pred = np.concatenate(all_pred)
        all_true = np.concatenate(all_true)
        train_metrics = compute_offline_metrics(all_pred, all_true)
        avg_train_loss = train_loss / len(train_loader)

        # ── VALİDATION ──
        model.eval()
        val_pred, val_true = [], []
        with torch.no_grad():
            for batch_depths, batch_actions in val_loader:
                pred = model(batch_depths.to(device))
                val_pred.append(pred.cpu().numpy())
                val_true.append(batch_actions.numpy())

        val_pred    = np.concatenate(val_pred)
        val_true    = np.concatenate(val_true)
        val_metrics = compute_offline_metrics(val_pred, val_true)

        # ── LOG ──
        print(
            f"Epoch [{epoch+1:02d}/{epochs}] "
            f"TrainLoss ↓: {avg_train_loss:.4f} | "
            f"Steer MSE (tr/val) ↓: {train_metrics['steering_mse']:.4f}/{val_metrics['steering_mse']:.4f} | "
            f"Accel MSE (tr/val) ↓: {train_metrics['accel_mse']:.4f}/{val_metrics['accel_mse']:.4f} | "
            f"Steer Corr (tr/val) ↑: {train_metrics['steering_corr']:.3f}/{val_metrics['steering_corr']:.3f} | "
            f"Dir Acc (tr/val) ↑: {train_metrics['direction_acc']:.3f}/{val_metrics['direction_acc']:.3f}"
        )

    torch.save(model.state_dict(), model_path)
    print(f"Eğitim tamamlandı! Model kaydedildi: {model_path}\n")


# ==========================================
# 6. TEST (INFERENCE) — online metriklerle
# ==========================================
def test_policy(model_path="policy_model.pth", num_episodes=5):
    print("--- Simülasyon Testi Başlıyor ---")
    if not os.path.exists(model_path):
        print("HATA: Eğitilmiş model bulunamadı!")
        return

    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth_estimator = DepthEstimationModel()

    policy_model = DrivingPolicyNet(action_dim=2).to(device)
    policy_model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    policy_model.eval()

    config = {
        "use_render": True,
        "image_observation": True,
        "sensors": {"rgb": (84, 84)},
        "vehicle_config": {"image_source": "rgb"},
    }
    env = MetaDriveEnv(config)

    # ── Online metrik toplayıcılar ──
    success_flags       = []
    route_completions   = []
    episode_rewards     = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        done           = False
        ep_reward      = 0.0

        with torch.no_grad():
            while not done:
                rgb_img = env.vehicle.sensors["rgb"].perceive(
                    env.vehicle, env.engine.physics_world.dynamic_world, None
                )
                depth_map    = depth_estimator.predict(rgb_img)
                depth_tensor = (
                    torch.tensor(depth_map, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(device)
                )
                pred_action = policy_model(depth_tensor).cpu().numpy()[0]
                obs, reward, terminated, truncated, info = env.step(pred_action)
                ep_reward += reward
                done = terminated or truncated

        # MetaDrive info sözlüğünden metrikleri çek
        arrived          = bool(info.get("arrive_dest", False))
        route_completion = float(info.get("route_completion", 0.0))

        success_flags.append(arrived)
        route_completions.append(route_completion)
        episode_rewards.append(ep_reward)

        print(
            f"  Bölüm {ep+1}/{num_episodes} | "
            f"Başarı: {'✓' if arrived else '✗'} | "
            f"Rota Tamamlama: {route_completion*100:.1f}% | "
            f"Toplam Ödül: {ep_reward:.2f}"
        )

    # ── Özet ──
    print("\n=== Online Metrik Özeti ===")
    print(f"  Success Rate      ↑: {np.mean(success_flags)*100:.1f}%")
    print(f"  Route Completion  ↑: {np.mean(route_completions)*100:.1f}%")
    print(f"  Episode Reward    ↑: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")

    env.close()
    print("Test tamamlandı.\n")


# ==========================================
# ANA ÇALIŞTIRMA BLOĞU
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MetaDrive Imitation Learning Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "test", "all"],
    )
    parser.add_argument("--epochs",       type=int,   default=20)
    parser.add_argument("--data_dir",     type=str,   default="dataset")
    parser.add_argument("--test_episodes",type=int,   default=5,
                        help="Test sırasında çalıştırılacak bölüm sayısı")
    parser.add_argument("--val_split",    type=float, default=0.2,
                        help="Validation için ayrılacak veri oranı (0-1)")
    args = parser.parse_args()

    if args.mode in ["train", "all"]:
        train_policy(epochs=args.epochs, data_dir=args.data_dir, val_split=args.val_split)

    if args.mode in ["test", "all"]:
        test_policy(num_episodes=args.test_episodes)