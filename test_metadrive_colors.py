import cv2
import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera

def main():
    print("Initializing MetaDrive Environment...")
    config = {
        "use_render": False,
        "image_observation": True,
        "sensors": {"rgb": (RGBCamera, 400, 400)},
        "vehicle_config": dict(image_source="rgb"),
    }
    env = MetaDriveEnv(config)
    env.reset()
    
    # Step a few times to get a good view
    for _ in range(10):
        env.step([0, 1])
        
    print("Fetching image from RGBCamera...")
    raw_img = env.engine.get_sensor("rgb").perceive(
        to_float=False, new_parent_node=env.agent.origin
    )
    if hasattr(raw_img, "get"):
        raw_img = raw_img.get()
    raw_img = np.array(raw_img, dtype=np.uint8)

    print("\n--- VISUAL TEST ---")
    print("OpenCV's cv2.imshow ALWAYS expects images in BGR format.")
    print("If an image looks correct (blue sky), it means the array passed to it is BGR.")
    print("If an image looks incorrect (orange/red sky), the array passed to it is RGB.\n")
    
    print("Press any key on the image windows to close and exit.")

    # 1. Raw Output (What MetaDrive gives us natively)
    cv2.imshow("1. Raw MetaDrive Output (Passed directly to imshow)", raw_img)
    
    # 2. What your code was doing previously
    wrong_conversion = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
    cv2.imshow("2. Previous Code (cv2.cvtColor(img, COLOR_RGB2BGR))", wrong_conversion)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    env.close()

if __name__ == "__main__":
    main()
