Autonomous Driving Project - Dual-Brain V2 & Asymmetric Loss
This project is built on the MetaDrive simulator to perform autonomous driving tasks using imitation learning. With recent updates, the project has evolved from a simple imitator; it has reached a level where it can perform driving and Autonomous Emergency Braking (AEB) using purely AI, without relying on any rule-based hacks, utilizing Sensor Fusion, Dual-Stream Architecture (V2), and an Asymmetric Loss Function.

What's Changed & What's New?
Dual-Brain V2: Independent Full Vision (Dual-Stream Architecture)
In the first Dual-Brain attempt, the steering branch only looked at the lane, while the throttle/brake branch only looked at depth. This caused a "Blind Steering" issue: when a car suddenly cut in front, the acceleration brain applied the brakes, but the steering brain couldn't see the obstacle, so it didn't know how to change lanes.

Solution (V2): The DrivingPolicyNet architecture was updated. The weights and neurons of the Steering Branch and Acceleration Branch remain completely separated, but both brains are now fed the 2-Channel full input (Lane + Depth). As a result, the steering brain can now see obstacles and independently learn to change lanes to avoid collisions.

Autonomous Emergency Braking via Pure AI: Asymmetric Loss Function (Brake Penalty)
In Behavioral Cloning (Imitation Learning), over 90% of the dataset consists of accelerating (positive) actions. When using a standard MSE Loss, the AI treated the "braking" action as an insignificant detail to keep its overall error rate low, which resulted in crashing straight into obstacles.

Solution: Instead of forcing the system with hardcoded if/else blocks, a mathematical intelligence was integrated into the Loss Function. Thanks to the custom_driving_loss, if the AI makes a mistake on an empty road, it receives a standard 1x penalty. However, if it misses a required braking action, it faces a penalty multiplier 3 times (x3) larger. Thanks to this "Asymmetric Penalty" system, the AI has learned to brake on its own initiative when it detects obstacles, without any rule-based intervention.

Installation Setup
To run this project, you need to set up the main environment and integrate the Depth Anything V2 repository.

1. Create and activate a new Conda environment

Bash
conda create -n driving-new python=3.11 -y
conda activate driving-new
2. Install PyTorch (Adjust the CUDA version to match your system, e.g., cu118 or cu121 or else)

Bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu1xx
3. Install the main project requirements

Bash
pip install -r requirements.txt
4. Clone and Install Video-Depth-Anything

Bash
git clone https://github.com/DepthAnything/Depth-Anything-V2
cd Depth-Anything-V2
pip install -r requirements.txt
cd ..
Important Parameters and Arguments
The project is now modularized into specialized scripts. Here are the key arguments you can use:

Data Collection (generate_expert_dataset.py)

--episodes: How many episodes of expert data to collect (automatically splits into train/val/test).

--num_cameras: Number of cameras distributed around the vehicle (default: 1).

--image_on_cuda: Highly recommended. Processes the lane masks and camera data directly on the GPU for a massive speedup.

Depth Processing (train_dpt.py)

--mode train: Fine-tunes the Depth-Anything-V2 model on your dataset.

--mode precompute: Runs the DPT model over your dataset and caches the tensors to disk. This skips live inference during policy training.

Policy Training & Testing (train_test_policy.py)

--pred_dir: The directory containing your precomputed DPT predictions. Passing this drastically speeds up training.

--test_mode: Choose between offline (evaluates on the test dataset split), simulation (runs the live MetaDrive environment), or all.

How to Run the Project
The pipeline consists of 4 distinct steps: Data Collection, Depth Fine-tuning, Precomputing (for speed), and Policy Training/Testing.

1. Data Collection
Generate the expert dataset. The script automatically separates the data into train, val, and test folders. We recommend using the --image_on_cuda flag if you have a dedicated GPU.

Bash
python src/generate_expert_dataset.py --save_dir dataset --episodes 100 --image_on_cuda
2. Fine-tune the Depth Model (Optional but Recommended)
Train the Depth-Anything-V2 model specifically on your MetaDrive environment data so it better understands the simulator's depth geometry.

Bash
python src/train_dpt.py --mode train --epochs 5 --data_dir dataset --model_path models/dpt_finetuned.pth
3. Precompute Depth Predictions (Speed Optimization)
To avoid running the heavy Depth-Anything-V2 model on every single frame during policy training, precompute and cache the depth and lane mask tensors.

Bash
python src/train_dpt.py --mode precompute \
    --model_path models/dpt_finetuned.pth \
    --data_dir dataset \
    --out_dir data/processed/dpt_pred
4. Train the Driving Policy
Train the Dual-Brain Steering and Throttle/Brake networks using the Asymmetric Loss Function. Point it to the precomputed directory so it trains at maximum speed.

Bash
python src/train_test_policy.py --mode train \
    --epochs 30 \
    --pred_dir data/processed/dpt_pred \
    --model_path models/policy_model.pth
5. Autonomous Testing
Test the fully trained policy. Using --test_mode all will first evaluate the model against the offline test dataset (checking MSE and Direction Accuracy) and then launch the live MetaDrive simulation so you can watch the AI drive.

Bash
python src/train_test_policy.py --mode test \
    --model_path models/policy_model.pth \
    --dpt_path models/dpt_finetuned.pth \
    --data_dir dataset \
    --pred_dir data/processed/dpt_pred \
    --test_mode all (use --test_mode simulation for only online testing)