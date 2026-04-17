# Autonomous Driving Project - Dual-Brain V2 & Asymmetric Loss
This project is built on the MetaDrive simulator to perform autonomous driving tasks using imitation learning. With recent updates, the project has evolved from a simple imitator; it has reached a level where it can perform driving and Autonomous Emergency Braking (AEB) using purely AI, without relying on any rule-based hacks, utilizing Sensor Fusion, Dual-Stream Architecture (V2), and an Asymmetric Loss Function.

What's Changed & What's New?
1. Dual-Brain V2: Independent Full Vision (Dual-Stream Architecture)
In the first Dual-Brain attempt, the steering branch only looked at the lane, while the throttle/brake branch only looked at depth. This caused a "Blind Steering" issue: when a car suddenly cut in front, the acceleration brain applied the brakes, but the steering brain couldn't see the obstacle, so it didn't know how to change lanes.

Solution (V2): The DrivingPolicyNet architecture was updated. The weights and neurons of the Steering Branch and Acceleration Branch remain completely separated, but both brains are now fed the 2-Channel full input (Lane + Depth). As a result, the steering brain can now see obstacles and independently learn to change lanes to avoid collisions.

2. Autonomous Emergency Braking via Pure AI: Asymmetric Loss Function (Brake Penalty)
In Behavioral Cloning (Imitation Learning), over 90% of the dataset consists of accelerating (positive) actions. When using a standard MSE Loss, the AI treated the "braking" action as an insignificant detail to keep its overall error rate low, which resulted in crashing straight into obstacles.

Solution: Instead of forcing the system with hardcoded if/else blocks, a mathematical intelligence was integrated into the Loss Function. Thanks to the custom_driving_loss, if the AI makes a mistake on an empty road, it receives a standard 1x penalty. However, if it misses a required braking action, it faces a penalty multiplier 3 times (x3) larger. Thanks to this "Asymmetric Penalty" system, the AI has learned to brake on its own initiative when it detects obstacles, without any rule-based intervention.

Installation Setup
To run this project, you need to set up the main environment and integrate the Depth Anything V2 repository.


# 1. Create and activate a new Conda environment
```bash
conda create -n driving-new python=3.11 -y
conda activate driving-new
```
# 2. Install PyTorch (Adjust the CUDA version to match your system, e.g., cu118 or cu121 or else)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu1xx
```
# 3. Install the main project requirements
```bash
pip install -r requirements.txt
```
# 4. Clone and Install Video-Depth-Anything
```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2
cd Depth-Anything-V2
pip install -r requirements.txt
cd ..
```
Important Parameters and Variables
Parameters you can tweak inside src/single_script.py:
```bash
--episodes (Data Collection): Determines how many episodes of expert data the algorithm will collect.

--epochs (Training): Determines how many times the dataset will be iterated over during training. (20-30 epochs is the sweet spot for the dual-brain asymmetric loss).
```
brake_mult = 2.0: The braking sensitivity within the training function. Increasing this value makes the vehicle brake more cautiously (paranoid mode); decreasing it allows the vehicle to get closer to obstacles before reacting.

FPS_DIVIDER=1: The refresh rate of the AI vision windows during the testing phase. (e.g., if set to 4, the visual feeds update less frequently, but the simulation FPS skyrockets).

How to Run the Project
The project is modular and consists of 3 steps: collect, train, and test.

Note: Since the network architecture (Dual-Brain V2) and Loss Function have changed, old weight files are no longer valid. If you already have pre-collected 2-channel fusion data, you only need to retrain the model using the --mode train command.

1- Data Collection (Data Collect)
The expert vehicle instantly converts images from the 400x400 cameras into a Lane Mask and a Depth Map, adding "noise" to record recovery maneuvers to the disk.

```bash
python src/generate_expert_dataset.py --save_dir "data/raw" --episodes 10
```

2- Training the Network with the New Architecture (Train)
The Steering and Throttle/Brake networks are trained using the Asymmetric Loss Function using the data saved in the dataset folder.

```bash
python src/single_script.py --mode train --epochs 20
```

3- Autonomous Testing (Test)
The model is tested in real-time within the simulation. The AI's Depth and Lane vision processing can be monitored live on the screen.

```bash
python src/single_script.py --mode test
```
