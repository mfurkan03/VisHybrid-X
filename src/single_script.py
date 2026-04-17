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
from metadrive.examples import expert
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.depth_camera import DepthCamera


 # --- YENİ EKLENEN GRAFİK KODU ---
import cv2
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/berke/Desktop/Projects/Bitirme/Video-Depth-Anything')
from video_depth_anything.video_depth import VideoDepthAnything
from video_depth_anything.video_depth_stream import VideoDepthAnything as VideoDepthAnythingStream

# LANE MASK
def get_lane_mask_visual(rgb_image, threshold_value=180):
    if rgb_image.max() <= 1.0:
        img_uint8 = (rgb_image * 255.0).astype(np.uint8)
    else:
        img_uint8 = rgb_image.astype(np.uint8)
        
    # 1. ROI (Region of Interest) - Sadece Yere Odaklan
    h, w = img_uint8.shape[:2]
    roi_img = img_uint8.copy()
    
    # Ekranın üst %55'lik kısmını (gökyüzü ve ufuk) simsiyah yap
    roi_img[0:int(h * 0.55), :] = 0
    
    # 2. Şerit Filtreleme
    gray = cv2.cvtColor(roi_img, cv2.COLOR_RGB2GRAY)
    
    # Çözünürlüğü artırdığımız için eşik değerini tekrar 180'e çıkardık ki asfalt siyah kalsın
    _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    
    return mask
# ==========================================
# 1. DEPTH ESTIMATION MODELİ
# ==========================================
class DepthEstimationModel:
    def __init__(self, encoder='vits'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        self.model = VideoDepthAnything(**model_configs[encoder])
        self.stream_model = VideoDepthAnythingStream(**model_configs[encoder])
        ckpt_path = f'/home/berke/Desktop/Projects/Bitirme/Video-Depth-Anything/checkpoints/video_depth_anything_{encoder}.pth'
        if os.path.exists(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=True)
            self.stream_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=True)
            print(f"[INFO] VideoDepthAnything Ağırlığı Yüklendi: {ckpt_path}")
        else:
            print(f"[UYARI] Yükleme başarısız, dosya yok: {ckpt_path}")
        self.model = self.model.to(self.device).eval()
        self.stream_model = self.stream_model.to(self.device).eval()

        # --- FP16 (Yarım Hassasiyet) Geçişi ---
        # VDA'nın iç katmanları (dpt_temporal) bazı yerlerde .float() kullandığından
        # ağırlıkları direkt FP16'ya çeviremiyoruz. Bunun yerine torch.autocast FP16 sarmalayıcısı
        # ile inference sırasında otomatik Tensor Core ivmelendirmesi yapacağız.
        self.use_fp16 = (self.device == 'cuda')
        if self.use_fp16:
            print("[INFO] VideoDepthAnything FP16 autocast modu etkinleştirildi! (Tensor Core ivmeli)")

    def process_depth_map(self, depth_raw: np.ndarray) -> np.ndarray:
        depth_resized = cv2.resize(depth_raw, (84, 84), interpolation=cv2.INTER_AREA)
        # 0-1 Normalization
        d_min, d_max = depth_resized.min(), depth_resized.max()
        if d_max - d_min > 1e-6:
            depth_norm = (depth_resized - d_min) / (d_max - d_min)
        else:
            depth_norm = depth_resized - d_min
        
        depth_map = np.expand_dims(depth_norm, axis=0) # (1, 84, 84)
        return depth_map.astype(np.float32)

    def predict_batch(self, rgb_images: list) -> list:
        frames = np.stack(rgb_images, axis=0)
        if frames.max() <= 1.0:
            frames = (frames * 255.0).astype(np.uint8)
            
        print(f"\n[INFO] Batched frames: {frames.shape}. Depth hesaplanıyor...")
        # input_size=252 works faster for our low res (84x84) images
        depths_raw, _ = self.model.infer_video_depth(frames, target_fps=30, input_size=252, device=self.device)
        print("[INFO] Batch derinlik haritaları üretildi.")
        
        return [self.process_depth_map(d) for d in depths_raw]

    def predict_single(self, rgb_image: np.ndarray, return_tensor=False):
        frame = rgb_image
        if frame.max() <= 1.0:
            frame = (frame * 255.0).astype(np.uint8)
            
        # Stream modeli 32 karelik cache'i hafızasında tutup sadece 1 yeni karenin feature'ını çıkarır. Çok hızlıdır.
        # fp32=True diyerek VDA'nın kendi iç autocast'ini kapatıyoruz çünkü biz burada dıştan sarıyoruz
        if return_tensor:
            if self.use_fp16:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    depth_raw = self.stream_model.infer_video_depth_one(frame, input_size=252, device=self.device, fp32=True, return_tensor=True)
            else:
                depth_raw = self.stream_model.infer_video_depth_one(frame, input_size=252, device=self.device, return_tensor=True)
            return self.process_depth_tensor(depth_raw)
        else:
            depth_raw = self.stream_model.infer_video_depth_one(frame, input_size=252, device=self.device)
            return self.process_depth_map(depth_raw)

    def process_depth_tensor(self, depth_raw: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        # Numpy CPU kopyası yerine saf GPU tensöru işlemi
        # depth_raw shape: (H, W). interpolate için (1, 1, H, W) formatına çeviriyoruz
        d_tensor = depth_raw.unsqueeze(0).unsqueeze(0)
        depth_resized = F.interpolate(d_tensor, size=(84, 84), mode='bilinear', align_corners=False)
        
        # 0-1 Normalizasyonu da yine PyTorch GPU ile donanım düzeyinde halledeceğiz
        d_min, d_max = depth_resized.min(), depth_resized.max()
        if d_max - d_min > 1e-6:
            depth_norm = (depth_resized - d_min) / (d_max - d_min)
        else:
            depth_norm = depth_resized - d_min
            
        return depth_norm.float() # dönüş Tipi Tensor (1, 1, 84, 84) GPU, float32'ye geri çevir (PolicyNet ve normalize için)



# ==========================================
# 2. VERİ TOPLAMA
# ==========================================
def collect_expert_data(num_episodes=50, save_dir="dataset"):
    os.makedirs(save_dir, exist_ok=True)
    print(f"Veriler '{save_dir}' klasörüne kaydediliyor...")

    config = {
        "use_render": False,        
        "image_observation": True,
        # Çözünürlüğü 400x400 yaptık ki şeritleri net bulalım
        "sensors": {"rgb": (RGBCamera, 400, 400)},
        "vehicle_config": dict(image_source="rgb"),
        "show_interface": False,
        "image_on_cuda": False,     
        "preload_models": True,
    }

    env = MetaDriveEnv(config)
    depth_estimator = DepthEstimationModel()
    total_steps = 0

    for ep in range(num_episodes):
        obs, info = env.reset()
        
        # Sadece 2 kanallı verileri ve eylemleri tutacağız (RAM dostu)
        combined_observations = []
        actions = []
        done = False

        while not done:
            # 1. Uzmanın O ANKİ kusursuz kararını al
            expert_action = expert(env.agent, deterministic=True)

            # 2. Arabayı fiziksel olarak savurmak için eylemin bir kopyasını alıp onu boz
            applied_action = expert_action.copy()
            import random
            if total_steps % 10 == 0:
                applied_action[0] += random.uniform(-0.3, 0.3) # Sadece simülasyona gidecek eylemi boz

            rgb_sensor = env.engine.get_sensor("rgb")
            rgb_img = rgb_sensor.perceive(env.agent) 
            
            # --- ŞERİT VE DERİNLİK HESAPLAMALARI (Aynı Kalacak) ---
            buyuk_maske = get_lane_mask_visual(rgb_img, threshold_value=180)
            kucuk_maske = cv2.resize(buyuk_maske, (84, 84), interpolation=cv2.INTER_AREA)
            lane_norm = (kucuk_maske / 255.0).astype(np.float32)
            lane_map = np.expand_dims(lane_norm, axis=0) 
            
            depth_map = depth_estimator.predict_single(rgb_img, return_tensor=False) 
            
            combined_obs = np.concatenate([depth_map, lane_map], axis=0) 
            
            # --- 3. KAYIT AŞAMASI (KRİTİK DÜZELTME) ---
            combined_observations.append(combined_obs)
            
            # Listeye GÜRÜLTÜLÜ eylemi değil, uzmanın o an olması gereken KUSURSUZ eylemini kaydediyoruz!
            actions.append(expert_action.copy()) 
            
            # Arabayı yoldan çıkarmak için simülasyona GÜRÜLTÜLÜ eylemi gönderiyoruz
            obs, reward, terminated, truncated, info = env.step(applied_action)
            
            done = terminated or truncated
            total_steps += 1
            
        # --- Bölüm sonu Kaydetme ---
        # Artık gereksiz RGB ham verisini kaydetmiyoruz (Diskten GB'larca tasarruf)
        np.savez_compressed(
            os.path.join(save_dir, f"episode_{ep}.npz"),
            depth=np.array(combined_observations), 
            action=np.array(actions),
        )
        print(f"Bölüm {ep+1}/{num_episodes} kaydedildi. (Adım: {len(actions)})")

    env.close()

# ==========================================
# 3. MODEL MİMARİSİ VE DATASET
# ==========================================
class DrivingPolicyNet(nn.Module):
    def __init__(self, action_dim=2):
        super().__init__()
        
        # --- 1. DİREKSİYON BEYNİ (Sadece Şeritleri görecek) ---
        # Giriş kanalı 1 yapıldı (Çünkü sadece lane_mask girecek)
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
            nn.Linear(10, 1), # SADECE 1 ÇIKTI (Direksiyon: action_dim 0)
        )

        # --- 2. GAZ/FREN BEYNİ (Sadece Derinliği görecek) ---
        # Giriş kanalı 1 yapıldı (Çünkü sadece depth_map girecek)
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
            nn.Linear(10, 1), # SADECE 1 ÇIKTI (Gaz/Fren: action_dim 1)
        )
        
        self.flatten = nn.Flatten()

    def forward(self, x):
        # x'in boyutu: (Batch, 2, 84, 84)
        
        # Tensörü kanallarına ayırıyoruz (Beyinlere paylaştırıyoruz)
        depth_input = x[:, 0:1, :, :] # 0. Kanal: Derinlik
        lane_input  = x[:, 1:2, :, :] # 1. Kanal: Şerit
        
        # Direksiyon Tahmini (Sadece Şeritleri Kullanarak)
        s_feat = self.steer_conv(lane_input)
        s_feat = self.flatten(s_feat)
        steer_pred = self.steer_fc(s_feat) # Çıktı: (Batch, 1)
        
        # Gaz/Fren Tahmini (Sadece Derinliği Kullanarak)
        a_feat = self.accel_conv(depth_input)
        a_feat = self.flatten(a_feat)
        accel_pred = self.accel_fc(a_feat) # Çıktı: (Batch, 1)
        
        # İki beynin sonucunu yan yana yapıştır: Çıktı -> (Batch, 2)
        return torch.cat([steer_pred, accel_pred], dim=1)


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
    #criterion = nn.MSELoss()

    def custom_driving_loss(pred, target):
        # 1. Normal karesel hatayı hesapla
        mse = (pred - target) ** 2
        
        # 2. Frenleme anlarını tespit et (Uzmanın eylemi 0'dan küçükse frendir)
        # target[:, 1] gaz/fren sütunudur.
        brake_mask = (target[:, 1] < 0.0).float()
        
        # 3. Ceza Çarpanı (Fren kaçırılırsa hatayı 10 kat daha fazla cezalandır)
        # Normal durumlarda çarpan 1.0, fren anlarında 10.0 olur.
        brake_mult = 2.0 # sweet spot olmayabilir daha denenebilir
        penalty_weight = 1.0 + (brake_mask * brake_mult) 
        
        # 4. Sadece gaz/fren (1. indeks) tahminindeki hatayı bu cezayla çarp
        mse_weighted = mse.clone()
        mse_weighted[:, 1] = mse[:, 1] * penalty_weight
        
        return mse_weighted.mean()
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
            #loss = criterion(pred, batch_actions)
            loss = custom_driving_loss(pred, batch_actions)
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

    # PolicyNet'i FP16 (Yarım Hassasiyet) moduna geçir
    # PolicyNet küçük bir model olduğundan .half() kullanmıyoruz,
    # autocast ile inference zamanında sarmalayacağız
    use_fp16_policy = (device.type == 'cuda')
    if use_fp16_policy:
        print("[INFO] PolicyNet FP16 autocast modu etkinleştirildi!")

    config = {
        "use_render": True,
        "image_observation": True,
        "sensors": {"rgb": (RGBCamera, 400, 400)},
        "vehicle_config": {"image_source": "rgb"},
        "show_interface": False,
        "image_on_cuda": False,
        "preload_models": True,
    }
    env = MetaDriveEnv(config)

    depth_estimator = DepthEstimationModel() # Kendi gri modelimizi çağırıyoruz

    # ── Online metrik toplayıcılar ──
    success_flags       = []
    route_completions   = []
    episode_rewards     = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        done           = False
        ep_reward      = 0.0

        # --- YENİ EKLENEN: HAFIZAYI SIFIRLA ---
        # Her bölüm başında modeli yeniden yükleyerek önceki bölümün hafızasını (cache) siliyoruz.
        depth_estimator = DepthEstimationModel() 
        # --------------------------------------

        with torch.no_grad():
            import time
            step_count = 0
            last_time = time.time()
            anlik_fps = 0.0
            
            while not done:
                step_count += 1
                
                rgb_sensor = env.engine.get_sensor("rgb")
                rgb_img = rgb_sensor.perceive(env.agent)
                
                # --- VDA ile Anlık Derinlik Üretimi ---
                # return_tensor=True ile VDA-PolicyNet arasına giren Numpy darboğazını eziyoruz! 
                # 1. Derinlik Haritası (Mevcut)
                depth_tensor = depth_estimator.predict_single(rgb_img, return_tensor=True)
                
                # 2. Şerit Maskesi Çıkarma ve Tensöre Çevirme
                buyuk_maske = get_lane_mask_visual(rgb_img, threshold_value=180)
                kucuk_maske = cv2.resize(buyuk_maske, (84, 84), interpolation=cv2.INTER_AREA)
                lane_norm = (kucuk_maske / 255.0).astype(np.float32)
                
                # Numpy dizisini PyTorch tensörüne çevirip GPU'ya atıyoruz
                lane_tensor = torch.tensor(lane_norm, device=device).unsqueeze(0).unsqueeze(0) # (1, 1, 84, 84)
                
                # 3. İkisini Birleştir: (1, 2, 84, 84) boyutunda girdi elde ediyoruz
                combined_tensor = torch.cat([depth_tensor, lane_tensor], dim=1)
                
                # Ekranda Göstermek İçin Yapay Zekaya Giren Derinlik(Depth) Haritasını Çizdiriyoruz
                # Performansı artırmak için GPU->CPU çekimini ve ekran renderlamayı sadece 4 adımda bir (seyrek) yapıyoruz
                FPS_DIVIDER = 1
                if step_count % FPS_DIVIDER == 0:
                    # DERINLIK HARITASI
                    depth_vis = (depth_tensor[0, 0].cpu().numpy() * 255.0).astype(np.uint8)
                    depth_vis_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
                    vis_image_resized = cv2.resize(depth_vis_colored, (400, 400), interpolation=cv2.INTER_NEAREST)
                    cv2.imshow("Test Asamasi - AI (Depth Map)", vis_image_resized)

                    # 2. ŞERİT MASKESİ (YENİ EKLENEN KISIM)
                    # Orijinal RGB kameradan şeritleri süz ve ekranda göster
                    lane_mask = get_lane_mask_visual(rgb_img)
                    lane_vis_resized = cv2.resize(lane_mask, (400, 400), interpolation=cv2.INTER_NEAREST)
                    cv2.imshow("AI Gozu - Seritler (Lane)", lane_vis_resized)
                    cv2.waitKey(1)

                # depth_tensor ZATEN GPU'da hazır! FP16 autocast ile Tensor Core'dan geçiriyoruz
                if use_fp16_policy:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        pred_action = policy_model(combined_tensor).float().cpu().numpy()[0]
                else:
                    pred_action = policy_model(combined_tensor).cpu().numpy()[0]

                # FPS Hesaplama (İlk karelerdeki saniyelik Tensor JIT/Compile derlemesini filtreleyen moving average)
                current_time = time.time()
                elapsed = current_time - last_time
                last_time = current_time
                
                if step_count > 1 and elapsed > 0: # 1. karenin devasa derleme süresini yoksay
                    current_fps = 1.0 / elapsed
                    if anlik_fps == 0.0:
                        anlik_fps = current_fps
                    else:
                        anlik_fps = 0.85 * anlik_fps + 0.15 * current_fps
                
                print(f"Direksiyon: {pred_action[0]:+0.3f} | Gaz/Fren: {pred_action[1]:+0.3f} | AI Beyin FPS: {anlik_fps:.1f} | SİMÜLASYON FPS: {(anlik_fps * 3):.1f}    ", end="\r")
                
                # Action Repeat (Frame Skip): Aynı eylemi 3 frame boyunca simüle ederiz
                # Yapay Zeka beyni (DinoV2) 3 frame süresince yeni tahmin yapmakla yorulmaz, hız fırlar!
                action_repeat = 3
                for _ in range(action_repeat):
                    obs, reward, terminated, truncated, info = env.step(pred_action)
                    ep_reward += reward
                    done = terminated or truncated
                    if done:
                        break                

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
    cv2.destroyAllWindows() 
    
    cv2.destroyAllWindows() # <--- DÖRDÜNCÜ EKLEME: Test bitince pencereyi güvenle kapatır

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