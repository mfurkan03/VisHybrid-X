import sys
# xformers kütüphanesini sistemden gizleyerek CPU'da çökmesini engelliyoruz
sys.modules['xformers'] = None

import cv2
import numpy as np
import torch
from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2

config = {
    "use_render": True,          
    "show_interface": True,      
    "start_seed": 0,             
    "num_scenarios": 1,          
    "manual_control": True,      
    
    "image_observation": True,
    "sensors": {"rgb": (RGBCamera, 200, 200)},
    "vehicle_config": {"image_source": "rgb"},
    "image_on_cuda": False,
    
    "stack_size": 1 
}

def laneMask(img, threshold=150):
    bottom_half = img.copy()
    r, c = bottom_half.shape[:2]
    bottom_half[0:r//2, :] = 0
    bottom_half = cv2.cvtColor(bottom_half, cv2.COLOR_RGB2GRAY)
    
    ret, lane_mask = cv2.threshold(bottom_half, threshold, 255, cv2.THRESH_BINARY)
    return lane_mask

def getDepth(img):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not hasattr(getDepth, "model"):
        model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
        model = DepthAnythingV2(**model_configs['vits'])
        model.load_state_dict(torch.load("Depth_Anything_V2/depth_anything_v2_vits.pth", map_location=DEVICE, weights_only=False))
        
        if DEVICE == 'cuda':
            model = model.to(DEVICE).half().eval()
            getDepth.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE).half()
            getDepth.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE).half()
        else:
            model = model.to(DEVICE).eval()
            getDepth.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE).float()
            getDepth.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE).float()
            
        getDepth.model = model
    else:
        model = getDepth.model

    resized_img = cv2.resize(img, (84, 84))
    img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    
    if DEVICE == 'cuda':
        img_tensor = img_tensor.half()
    else:
        img_tensor = img_tensor.float()
        
    img_tensor = img_tensor / 255.0
    img_tensor = (img_tensor - getDepth.mean) / getDepth.std

    with torch.inference_mode():
        depth = model(img_tensor)

    depth_numpy = depth.squeeze().cpu().numpy().astype(np.float32)
    return depth_numpy

def main():
    env = MetaDriveEnv(config)
    try:
        obs, info = env.reset()
        
        print("\n" + "="*60)
        print("🎮 3D Ekranına TIKLA ve aracı sür! Kameralar yan tarafta akacak.")
        print("="*60 + "\n")
        
        while True:
            obs, reward, terminated, truncated, info = env.step([0.0, 0.0])

            raw_image = obs["image"] 
            
            if raw_image.ndim == 4:
                raw_image = raw_image[..., -1]
            elif raw_image.ndim == 3 and raw_image.shape[-1] > 3:
                raw_image = raw_image[..., -3:]
            
            img_uint8 = (raw_image * 255).astype(np.uint8)
            img_contiguous = np.ascontiguousarray(img_uint8)

            half_img = laneMask(img_contiguous, 180) 
            
            depth_map = getDepth(img_contiguous)  
            
            depth_normalized = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX)
            depth_uint8 = depth_normalized.astype(np.uint8)
            
            depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
            img_bgr = cv2.cvtColor(img_contiguous, cv2.COLOR_RGB2BGR)
            
            # Görüntüleri ekranda daha büyük göstermek için sadece gösterim amaçlı büyütüyoruz
            disp_size = (400, 400)
            cv2.imshow("RL Ajaninin Gozunden (RGB)", cv2.resize(img_bgr, disp_size))
            cv2.imshow("Derinlik Haritasi (Depth)", cv2.resize(depth_colored, disp_size))
            cv2.imshow("Serit Maskesi (Lane)", cv2.resize(half_img, disp_size)) 
            cv2.waitKey(1)
            
            if terminated or truncated:
                obs, info = env.reset()
                
    except Exception as e:
        print(f"Hata oluştu: {e}")
        if 'raw_image' in locals():
            print(f"Hata anındaki dizinin boyutu: {raw_image.shape}")
    finally:
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()