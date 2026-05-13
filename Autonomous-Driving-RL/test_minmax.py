import numpy as np
import cv2
arr = np.random.rand(10, 10).astype(np.float16)
try:
    cv2.normalize(arr, None, 0, 255, norm_type=cv2.NORM_MINMAX)
except Exception as e:
    print("Error:", e)
