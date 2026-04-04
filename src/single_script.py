import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from metadrive import MetaDriveEnv
from metadrive.examples import expert
from metadrive.component.sensors.rgb_camera import RGBCamera # Bu importu en üste ekleyin
# ==========================================
# 1. DEPTH ESTIMATION MODELI
# ==========================================
class DepthEstimationModel:
    def __init__(self, model_path=None):
        pass # Kendi modelinizi burada yükleyin

    def predict(self, rgb_image: np.ndarray) -> np.ndarray:
        # Dummy işlem: RGB'yi griye çevirip 1 kanallı hale getirir
        depth_map = np.mean(rgb_image, axis=2, keepdims=True)
        depth_map = np.transpose(depth_map, (2, 0, 1))
        return (depth_map / 255.0).astype(np.float32)

# ==========================================
# 2. VERİ TOPLAMA
# ==========================================
def collect_expert_data(num_episodes=10, save_dir="dataset"):
    os.makedirs(save_dir, exist_ok=True)
    
    config = {
        "use_render": True,  
        "image_observation": True,
        "sensors": {
            "rgb": (RGBCamera, 84, 84), 
        },
        "vehicle_config": dict(image_source="rgb"),
        "show_interface": False, # <--- Düzeltilen satır burası
        "image_on_cuda": False,  
        "preload_models": True
    }
    
    env = MetaDriveEnv(config)
    depth_estimator = DepthEstimationModel()

    total_steps = 0

    for ep in range(num_episodes):
        obs, info = env.reset()
        
        # expert = ExpertPolicy(env.agent)  <--- BU SATIRI SİLDİK
        
        depth_maps = []
        actions = []
        
        done = False
        while not done:
            # Artık 'expert' fonksiyonu import ettiğimiz PPO modelini kullanarak çalışacak
            action = expert(env.agent, deterministic=True) 
            
            # Görüntüyü sensörden al
            rgb_sensor = env.engine.get_sensor("rgb")
            rgb_img = rgb_sensor.perceive(env.agent) 
            
            # Depth Map
            depth_map = depth_estimator.predict(rgb_img)
            
            depth_maps.append(depth_map)
            actions.append(action)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_steps += 1
            
        np.savez_compressed(
            os.path.join(save_dir, f"episode_{ep}.npz"), 
            depth=np.array(depth_maps), 
            action=np.array(actions)
        )
        print(f"Bölüm {ep+1}/{num_episodes} kaydedildi. (Adım: {len(actions)})")

    env.close()
    print(f"Veri toplama tamamlandı! Toplam Adım: {total_steps}\n")

# ==========================================
# 3. MODEL MİMARİSİ VE DATASET
# ==========================================
class DrivingPolicyNet(nn.Module):
    def __init__(self, action_dim=2):
        super(DrivingPolicyNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()
        )
        self.flatten = nn.Flatten()
        # 84x84 giriş için Conv katmanları sonrasında boyut 64*1*1 kalır.
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 1 * 1, 100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 10), nn.ReLU(),
            nn.Linear(10, action_dim)
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
            self.depths.extend(data['depth'])
            self.actions.extend(data['action'])
            
    def __len__(self): return len(self.actions)
    def __getitem__(self, idx):
        return torch.tensor(self.depths[idx], dtype=torch.float32), torch.tensor(self.actions[idx], dtype=torch.float32)

# ==========================================
# 4. EĞİTİM (TRAINING)
# ==========================================
def train_policy(epochs=20, batch_size=64, model_path="policy_model.pth"):
    print("--- Eğitim Başlıyor ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan Cihaz: {device}")
    
    dataset = MetaDriveDepthDataset(data_dir="dataset")
    if len(dataset) == 0:
        print("HATA: Dataset boş! Önce veri toplamalısınız.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = DrivingPolicyNet(action_dim=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_depths, batch_actions in dataloader:
            batch_depths, batch_actions = batch_depths.to(device), batch_actions.to(device)
            
            optimizer.zero_grad()
            pred_actions = model(batch_depths)
            loss = criterion(pred_actions, batch_actions)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Ortalama Loss: {epoch_loss/len(dataloader):.4f}")
        
    torch.save(model.state_dict(), model_path)
    print(f"Eğitim tamamlandı! Model kaydedildi: {model_path}\n")

# ==========================================
# 5. TEST (INFERENCE)
# ==========================================
def test_policy(model_path="policy_model.pth"):
    print("--- Simülasyon Testi Başlıyor ---")
    if not os.path.exists(model_path):
        print("HATA: Eğitilmiş model bulunamadı! Önce eğitim yapmalısınız.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth_estimator = DepthEstimationModel()
    
    policy_model = DrivingPolicyNet(action_dim=2).to(device)
    policy_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    policy_model.eval()
    
    config = {
        "use_render": True, # Test aşamasında görselleştirmeyi açıyoruz
        "image_observation": True,
        "sensors": {"rgb": (84, 84)},
        "vehicle_config": {"image_source": "rgb"}
    }
    env = MetaDriveEnv(config)
    obs, info = env.reset()
    done = False
    
    with torch.no_grad():
        while not done:
            rgb_img = env.vehicle.sensors["rgb"].perceive(env.vehicle, env.engine.physics_world.dynamic_world, None)
            depth_map = depth_estimator.predict(rgb_img)
            
            depth_tensor = torch.tensor(depth_map, dtype=torch.float32).unsqueeze(0).to(device)
            pred_action = policy_model(depth_tensor).cpu().numpy()[0]
            
            obs, reward, terminated, truncated, info = env.step(pred_action)
            done = terminated or truncated

    env.close()
    print("Test tamamlandı.\n")
# ==========================================
# ANA ÇALIŞTIRMA BLOĞU
# ==========================================
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MetaDrive Imitation Learning Pipeline")
    parser.add_argument(
        "--mode", 
        type=str, 
        required=True, 
        choices=["collect", "train", "test", "all"], 
        help="Çalıştırmak istediğiniz modu seçin: collect, train, test veya all"
    )
    parser.add_argument("--episodes", type=int, default=10, help="Veri toplama için bölüm sayısı")
    parser.add_argument("--epochs", type=int, default=20, help="Eğitim için epoch sayısı")
    
    args = parser.parse_args()

    if args.mode in ["collect", "all"]:
        collect_expert_data(num_episodes=args.episodes)
        
    if args.mode in ["train", "all"]:
        train_policy(epochs=args.epochs)
        
    if args.mode in ["test", "all"]:
        test_policy()