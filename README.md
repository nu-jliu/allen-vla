# Allen's Awesome Vision-Language-Action (VLA) Models

Repository to run experiments with various VLA (Vision-Language-Action) models on SoArm-101.

## Project Goal

This project aims to explore and benchmark different VLA models for robotic manipulation tasks using the SoArm-101 robot arm. The workflow consists of two main phases:

1. **Teleoperation & Data Collection Pipeline**: Setup LeRobot with SoArm for teleoperating the robot and collecting high-quality demonstration datasets
2. **Training & Deployment Pipeline**: Implement, train, and deploy various VLA models (π0, π0.5, ACT, etc.) and evaluate their performance

## Installation

### Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and resolver

### 1. Install uv

If you don't have uv installed, install it using:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or on macOS/Linux using brew:

```bash
brew install uv
```

### 2. Clone the Repository

```bash
git clone https://github.com/jliu6718/allen-vla.git
cd allen-vla
```

### 3. Install Dependencies

```bash
uv sync
```

This will create a virtual environment and install all dependencies specified in `pyproject.toml`:
- huggingface-hub (>=0.35.3)
- lerobot[feetech] (>=0.4.2)
- opencv-python (>=4.12.0.88)
- pynput (>=1.8.1)
- uuid (>=1.30)

### 4. Activate the Virtual Environment

```bash
source .venv/bin/activate
```

Or if you're using uv to run commands directly:

```bash
uv run python <script>.py
```

## Project Structure

```
allen-vla/
├── assets/
│   └── data_collection_demo.gif
├── calibration/
│   ├── __init__.py
│   └── calibrate.py              # Calibration script for SO101 follower arm
├── data_collection/
│   ├── __init__.py
│   └── collect.py                # Main data collection script
├── policy/
│   ├── __init__.py
│   └── act/
│       ├── __init__.py
│       ├── train_act.py          # Training script for ACT policy
│       ├── inference_act.py      # Local inference (policy + robot on same machine)
│       ├── inference_server.py   # TCP server for remote inference (GPU machine)
│       └── inference_client.py   # Robot client (connects to inference server)
├── scripts/
│   ├── deploy_remote.bash        # Deploy project files to remote machine
│   ├── download_data.bash        # Download collected data from remote machine
│   └── download_model.bash       # Download trained models from remote machine
├── teleop/
│   ├── __init__.py
│   └── teleop.py                 # Teleoperation script for leader-follower control
├── udev/
│   ├── 99-so101.rules            # Udev rules for consistent device naming
│   └── README.md
├── __init__.py
├── main.py
├── robot_utils.py                # Shared utilities for robot initialization
├── utils.py                      # Common utilities including colored logging
├── pyproject.toml
└── README.md
```

All scripts use a common logging infrastructure with color-coded output for better visibility during operations.

## Hardware Setup

### Configure Udev Rules for SO101 Robot Arms

To access the robot arms without sudo and with consistent device names:

#### 1. Install Udev Rules

```bash
sudo cp udev/99-so101.rules /etc/udev/rules.d/
```

#### 2. Add Your User to the dialout Group

```bash
sudo usermod -a -G dialout $USER
```

**Important**: Log out and log back in for the group change to take effect.

#### 3. Reload Udev Rules

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

#### 4. Test Hardware Connection

Disconnect and reconnect your SO101 arms. You should now be able to access them without `sudo`.

For detailed hardware setup instructions including creating symbolic links for consistent device naming, see [udev/README.md](udev/README.md).

## Deployment

### Deploy to Remote Device

To deploy this project to a remote device (e.g., Jetson):

```bash
./scripts/deploy_remote.bash <username> <hostname>
```

**Example:**

```bash
./scripts/deploy_remote.bash allen jetson
```

This script uses rsync to sync all project files (excluding virtual environments, cache files, data, and models) to the remote machine at `/home/<username>/.ws/vla_ws/`.

### Download Data from Remote

To download collected datasets from a remote machine:

```bash
./scripts/download_data.bash <username> <hostname>
```

**Example:**

```bash
./scripts/download_data.bash allen jetson
```

This syncs the `data/` directory from the remote machine to your local project.

### Download Models from Remote

To download trained models from a remote machine:

```bash
./scripts/download_model.bash <username> <hostname>
```

**Example:**

```bash
./scripts/download_model.bash allen jetson
```

This syncs the `model/` directory from the remote machine to your local project.

## Usage

### Calibration

Before using the robot, you may need to calibrate the follower arm to set the zero positions for each joint:

```bash
python calibration/calibrate.py --port /dev/ttyACM0 --id my_follower
```

Or using uv:

```bash
uv run python calibration/calibrate.py --port /dev/ttyACM0 --id my_follower
```

The script accepts the following command-line arguments:

- `--port`: Serial port for the follower arm (required)
- `--id`: Robot ID for the follower arm (required)

The calibration process will:
1. Connect to the follower arm
2. Run the calibration routine to set joint zero positions
3. Disconnect when complete

### Teleoperation

Run the teleoperation script to control the follower arm using the leader arm:

```bash
python teleop/teleop.py
```

Or using uv:

```bash
uv run python teleop/teleop.py
```

The script accepts the following command-line arguments:

- `--leader-port`: Serial port for the leader arm (default: `/dev/ttyACM0`)
- `--leader-id`: ID for the leader arm (default: `my_leader`)
- `--follower-port`: Serial port for the follower arm (default: `/dev/ttyACM1`)
- `--follower-id`: ID for the follower arm (default: `my_follower`)
- `--frequency`: Control frequency in Hz (default: `20.0`)

**Example with custom ports:**

```bash
python teleop/teleop.py --leader-port /dev/ttyACM2 --follower-port /dev/ttyACM3
```

**Example with custom IDs:**

```bash
python teleop/teleop.py --leader-id leader_arm --follower-id follower_arm
```

The teleoperation interface will:
1. Connect to both leader and follower arms
2. Disable torque on the leader arm (so you can move it by hand)
3. Mirror the leader arm's movements on the follower arm in real-time
4. Print observation data to the console
5. Run until interrupted with `Ctrl+C`

### Data Collection

![Data Collection Demo](assets/data_collection_demo.gif)

Collect demonstration datasets for training VLA models:

```bash
python data_collection/collect.py
```

Or using uv:

```bash
uv run python data_collection/collect.py
```

The script accepts the following command-line arguments:

- `--leader-port`: Serial port for the leader arm (default: `/dev/ttyACM0`)
- `--leader-id`: ID for the leader arm (default: `my_leader`)
- `--follower-port`: Serial port for the follower arm (default: `/dev/ttyACM1`)
- `--follower-id`: ID for the follower arm (default: `my_follower`)
- `--repo-id`: HuggingFace repository ID in format `username/repo-name` (default: `jliu6718/lerobot-so101`)
- `--hz`: Control loop frequency in Hz (default: `30`)
- `--push`: Push dataset to Hugging Face Hub after collection (flag)

**Example:**

```bash
python data_collection/collect.py --repo-id your_username/my_dataset
```

**Example with push to Hugging Face Hub:**

```bash
python data_collection/collect.py --repo-id your_username/my_dataset --push
```

The data collection interface will:
1. Connect to both leader and follower arms
2. Create a LeRobot dataset with unique UUID
3. Press `ENTER` to start/stop recording episodes
4. Capture observations and actions at 30 Hz during recording
5. Save episodes to the dataset
6. Press `Ctrl+C` to finalize and upload the dataset to Hugging Face Hub

### Training ACT Policy

Train an ACT (Action Chunking Transformer) policy on your collected datasets:

```bash
python policy/act/train_act.py --repo-id your_username/your_dataset --output-dir ./outputs/act_run1
```

Or using uv:

```bash
uv run python policy/act/train_act.py --repo-id your_username/your_dataset --output-dir ./outputs/act_run1
```

#### Required Arguments

- `--repo-id`: HuggingFace dataset repo ID (e.g., `jliu6718/lerobot-so101-abc123`)
- `--output-dir`: Directory to save checkpoints and logs

#### Training Hyperparameters

- `--batch-size`: Training batch size (default: `8`)
- `--steps`: Total training steps (default: `100,000`)
- `--num-workers`: Number of dataloader workers (default: `4`)
- `--seed`: Random seed for reproducibility (default: `1000`)

#### ACT-Specific Hyperparameters

- `--chunk-size`: Action prediction chunk size (default: `100`)
- `--n-action-steps`: Number of action steps to execute per query (default: `100`)
- `--lr`: Learning rate (default: `1e-5`)
- `--kl-weight`: KL divergence loss weight for VAE (default: `10.0`)
- `--dropout`: Dropout rate in transformer (default: `0.1`)

#### Logging and Checkpointing

- `--log-freq`: Log training metrics every N steps (default: `250`)
- `--save-freq`: Save checkpoint every N steps (default: `5000`)
- `--progress-bar` / `--no-progress-bar`: Show/hide tqdm progress bar (default: enabled)

#### Weights & Biases Integration

- `--wandb-enable`: Enable Weights & Biases logging
- `--wandb-project`: Wandb project name (default: `soarm-act-training`)
- `--wandb-entity`: Wandb entity/team name (optional)
- `--wandb-notes`: Wandb run notes/description (optional)

#### Advanced Options

- `--resume`: Resume training from checkpoint in output-dir
- `--push`: Push checkpoints to HuggingFace Hub
- `--dataset-root`: Local dataset cache directory (optional)

**Example with custom hyperparameters:**

```bash
python policy/act/train_act.py \
  --repo-id jliu6718/lerobot-so101-abc123 \
  --output-dir ./outputs/act_experiment1 \
  --batch-size 16 \
  --steps 50000 \
  --lr 5e-5 \
  --wandb-enable \
  --wandb-project my-act-training
```

**Example resuming from checkpoint:**

```bash
python policy/act/train_act.py \
  --repo-id jliu6718/lerobot-so101-abc123 \
  --output-dir ./outputs/act_experiment1 \
  --resume
```

The training script will:
1. Load the dataset from HuggingFace Hub
2. Initialize the ACT model with specified hyperparameters
3. Train using LeRobot's full training infrastructure with Accelerate
4. Save checkpoints at specified intervals
5. Log metrics to console and optionally to Weights & Biases
6. Support distributed training out of the box

### ACT Policy Inference

Run inference with a trained ACT policy on the SO101 robot:

```bash
python policy/act/inference_act.py \
  --checkpoint ./outputs/act_training/pretrained_model \
  --robot-port /dev/ttyACM0 \
  --camera-index 0 \
  --repo-id your_username/eval_results
```

Or using uv:

```bash
uv run python policy/act/inference_act.py \
  --checkpoint ./outputs/act_training/pretrained_model \
  --robot-port /dev/ttyACM0 \
  --camera-index 0 \
  --repo-id your_username/eval_results
```

#### Required Arguments

- `--checkpoint`: Path to trained policy checkpoint or HuggingFace repo ID
- `--robot-port`: Robot port (e.g., `/dev/ttyACM0`)
- `--camera-index`: Camera index or path (e.g., `0` for `/dev/video0`)
- `--repo-id`: Dataset repo ID for saving evaluation results

#### Robot Configuration

- `--robot-type`: Robot type (default: `so101_follower`)
- `--robot-id`: Robot ID (default: `eval_robot`)

#### Camera Configuration

- `--camera-name`: Camera name in config (default: `front`)
- `--camera-width`: Camera width (default: `640`)
- `--camera-height`: Camera height (default: `480`)
- `--camera-fps`: Camera FPS (default: `30`)

#### Evaluation Parameters

- `--num-episodes`: Number of episodes to evaluate (default: `10`)
- `--task-description`: Task description for this evaluation run
- `--fps`: Control frequency in Hz (default: `30`)
- `--episode-time`: Maximum time per episode in seconds (default: `60`)
- `--reset-time`: Time for resetting between episodes in seconds (default: `60`)

#### Data Saving Options

- `--root`: Root directory to save dataset locally
- `--push-to-hub`: Push evaluation dataset to HuggingFace Hub
- `--video`: Encode videos in the dataset (default: enabled)

#### Display Options

- `--display-data` / `--no-display`: Show/hide camera feed during evaluation
- `--play-sounds`: Enable vocal synthesis for events

**Example with HuggingFace Hub model:**

```bash
python policy/act/inference_act.py \
  --checkpoint username/act_policy \
  --robot-port /dev/ttyACM0 \
  --camera-index 0 \
  --num-episodes 5 \
  --repo-id username/eval_results \
  --push-to-hub
```

The inference script will:
1. Load the trained policy from checkpoint or HuggingFace Hub
2. Connect to the robot and camera
3. Run autonomous policy control for the specified number of episodes
4. Save evaluation results as a dataset for analysis
5. Optionally push the evaluation dataset to HuggingFace Hub

### Client-Server Inference (Remote GPU)

For setups where the robot runs on a low-power device (e.g., Jetson) and inference runs on a remote GPU server, use the client-server architecture:

#### Start the Inference Server (GPU Machine)

```bash
python policy/act/inference_server.py \
  --checkpoint ./outputs/act_training/pretrained_model \
  --port 8000 \
  --device cuda
```

**Server Arguments:**

- `--checkpoint`: Path to trained policy checkpoint or HuggingFace repo ID (required)
- `--host`: Host address to bind to (default: `0.0.0.0`)
- `--port`: Port to listen on (default: `8000`)
- `--device`: Device to run inference on (default: `cuda`)
- `--task`: Task description for inference

#### Start the Robot Client (Robot Machine)

```bash
python policy/act/inference_client.py \
  --server-host <gpu_server_ip> \
  --server-port 8000 \
  --robot-port /dev/ttyACM0 \
  --camera-index 0 \
  --num-episodes 10
```

**Client Arguments:**

- `--server-host`: Inference server host address (required)
- `--server-port`: Inference server port (default: `8000`)
- `--robot-port`: Robot serial port (required)
- `--camera-index`: Camera index or path (required)
- `--camera-name`: Camera name in config (default: `front`)
- `--camera-width`: Camera width (default: `640`)
- `--camera-height`: Camera height (default: `480`)
- `--camera-fps`: Camera FPS (default: `30`)
- `--num-episodes`: Number of episodes to run (default: `10`)
- `--episode-time`: Duration of each episode in seconds (default: `60`)
- `--reset-time`: Time for reset between episodes in seconds (default: `60`)
- `--fps`: Control frequency in Hz (default: `30`)

**Example with Jetson and Remote GPU:**

On the GPU server (e.g., `192.168.1.100`):
```bash
python policy/act/inference_server.py \
  --checkpoint username/act_policy \
  --port 8000 \
  --device cuda
```

On the Jetson (robot machine):
```bash
python policy/act/inference_client.py \
  --server-host 192.168.1.100 \
  --server-port 8000 \
  --robot-port /dev/ttyACM0 \
  --camera-index 0 \
  --num-episodes 5 \
  --fps 30
```

The client-server architecture:
1. **Server**: Loads the policy model and handles inference requests over TCP
2. **Client**: Connects to the robot, gathers observations, sends them to the server, and actuates the robot with returned actions
3. **Communication**: Uses pickle serialization over TCP with length-prefixed messages
4. **Multi-client**: Server supports multiple concurrent robot connections via threading
5. **Episode management**: Client sends reset signals to server between episodes to clear policy state

## Todo List

### Phase 1: Teleoperation & Data Collection ✓
- [x] Setup LeRobot environment and dependencies
- [x] Configure SoArm-101 hardware integration with LeRobot
- [x] Implement teleoperation interface
- [x] Setup camera(s) and sensor pipeline
- [x] Develop data collection scripts
- [x] Define data format and storage structure (using LeRobot dataset format)
- [x] Test teleoperation and record sample demonstrations
- [x] Create dataset management utilities (keyboard-controlled recording, HF Hub integration)

### Phase 2: Training & Deployment Pipeline
- [x] Setup training infrastructure (GPU environment, configs)
- [x] Implement ACT (Action Chunking Transformer) training pipeline
- [x] Implement ACT inference/evaluation pipeline
- [ ] Implement π0 model training pipeline
- [ ] Implement π0.5 model training pipeline
- [ ] Add additional VLA models as needed
- [ ] Create evaluation metrics and benchmarking scripts
- [ ] Test deployed models on real hardware
- [ ] Compare model performance and document results

### Documentation & Experimentation
- [ ] Document hardware setup and calibration procedures
- [ ] Create training guides for each model
- [ ] Log experimental results and hyperparameters
- [ ] Build visualization tools for trajectories and predictions

## License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2025 Allen Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Citation

If you use this codebase in your research, please cite:

```bibtex
@software{liu2025allen_vla,
  author = {Liu, Allen},
  title = {Allen's Awesome Vision-Language-Action (VLA) Models},
  year = {2025},
  url = {https://github.com/jliu6718/allen-vla},
  note = {Repository for VLA model experiments on SoArm-101}
}
```

This project builds upon [LeRobot](https://github.com/huggingface/lerobot) by Hugging Face:

```bibtex
@misc{cadene2024lerobot,
  author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Wolf, Thomas},
  title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/lerobot}}
}
```
