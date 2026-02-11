# Diffusion Policy

This document covers training, inference, and deployment of the Diffusion policy for the SoArm-101 robot.

## Training

Train a Diffusion policy on your collected datasets:

```bash
python policy/diffusion/train.py --repo-id your_username/your_dataset --output-dir ./outputs/diffusion_run1
```

Or using uv:

```bash
uv run python policy/diffusion/train.py --repo-id your_username/your_dataset --output-dir ./outputs/diffusion_run1
```

> **Example**: See [`example/diffusion/train.bash`](../example/diffusion/train.bash) for a complete example.

### Required Arguments

- `--output-dir`: Directory to save checkpoints and logs

### Dataset Source (mutually exclusive, one required)

- `--repo-id`: HuggingFace Hub dataset repo ID (e.g., `username/diffusion-so101-pick_place`)
- `--local-dir`: Path to local dataset directory
- `--revision`: Dataset revision/branch to use (default: `main`)

### Training Hyperparameters

- `--batch-size`: Training batch size (default: `8`)
- `--steps`: Total training steps (default: `100,000`)
- `--num-workers`: Number of dataloader workers (default: `4`)
- `--seed`: Random seed for reproducibility (default: `42`)

### Diffusion-Specific Hyperparameters

- `--horizon`: Diffusion model action prediction horizon (default: `16`)
- `--n-action-steps`: Number of action steps to execute per query (default: `8`)
- `--n-obs-steps`: Number of observation steps for context (default: `2`)
- `--num-train-timesteps`: Number of diffusion training timesteps (default: `100`)
- `--num-inference-steps`: Number of diffusion inference steps (default: same as train timesteps)
- `--noise-scheduler`: Noise scheduler type, `DDPM` or `DDIM` (default: `DDPM`)
- `--lr`: Learning rate (default: `1e-4`)
- `--vision-backbone`: Vision backbone for image encoding (default: `resnet18`)
- `--crop-shape`: Image crop shape for preprocessing (default: `84 84`)

### Logging and Checkpointing

- `--log-freq`: Log training metrics every N steps (default: `250`)
- `--save-freq`: Save checkpoint every N steps (default: `5000`)
- `--progress-bar` / `--no-progress-bar`: Show/hide tqdm progress bar (default: enabled)

### Model Push Configuration

Required when using `--local-dir` with `--push`:

- `--username`: HuggingFace username
- `--policy-type`: Policy type (e.g., `diffusion`)
- `--robot-type`: Robot type (e.g., `so101`)
- `--task`: Task name for the model repo

### Advanced Options

- `--resume`: Resume training from checkpoint in output-dir
- `--push`: Push checkpoints to HuggingFace Hub
- `--force-redownload`: Force re-download dataset from HuggingFace Hub (ignores cache)

**Example with HuggingFace Hub dataset:**

```bash
python policy/diffusion/train.py \
  --repo-id username/diffusion-so101-pick_place \
  --output-dir ./outputs/diffusion_experiment1 \
  --batch-size 16 \
  --steps 50000 \
  --lr 5e-4 \
  --horizon 32 \
  --n-action-steps 16
```

**Example with local dataset:**

```bash
python policy/diffusion/train.py \
  --local-dir ./data/diffusion-so101-pick_place \
  --output-dir ./outputs/diffusion_experiment1 \
  --push \
  --username my_username \
  --policy-type diffusion \
  --robot-type so101 \
  --task pick_place
```

**Example resuming from checkpoint:**

```bash
python policy/diffusion/train.py \
  --repo-id username/diffusion-so101-pick_place \
  --output-dir ./outputs/diffusion_experiment1 \
  --resume
```

The training script will:
1. Load the dataset from HuggingFace Hub or local directory
2. Initialize the Diffusion model with specified hyperparameters
3. Train using LeRobot's full training infrastructure with Accelerate
4. Save checkpoints at specified intervals
5. Log metrics to console
6. Support distributed training out of the box

## Inference

Run inference with a trained Diffusion policy on the SO101 robot:

```bash
python policy/diffusion/inference.py \
  --checkpoint ./outputs/diffusion_training/pretrained_model \
  --robot-port /dev/ttyACM0 \
  --username my_username \
  --policy-type diffusion \
  --robot-type so101 \
  --task pick_place
```

Or using uv:

```bash
uv run python policy/diffusion/inference.py \
  --checkpoint ./outputs/diffusion_training/pretrained_model \
  --robot-port /dev/ttyACM0 \
  --username my_username \
  --policy-type diffusion \
  --robot-type so101 \
  --task pick_place
```

> **Example**: See [`example/diffusion/inference.bash`](../example/diffusion/inference.bash) for a complete example.

The evaluation repo ID is automatically constructed as `{username}/eval_{policy-type}-{robot-type}-{task}`.

### Required Arguments

- `--checkpoint`: Path to trained policy checkpoint or HuggingFace repo ID
- `--robot-port`: Robot port (e.g., `/dev/ttyACM0`)
- `--username`: HuggingFace username
- `--policy-type`: Policy type (e.g., `diffusion`)
- `--robot-type`: Robot type (e.g., `so101`)
- `--task`: Task name for the evaluation dataset (e.g., `pick_place`)

### Robot Configuration

- `--robot-id`: Robot ID (default: `eval_robot`)

### Camera Configuration

- `--camera-config`: Path to camera config TOML file (default: `config/camera.toml`)

### Evaluation Parameters

- `--episode`: Number of episodes to evaluate (default: `1`)
- `--task-description`: Task description for this evaluation run (default: `Policy evaluation`)
- `--fps`: Control frequency in Hz (default: `30`)
- `--episode-time`: Maximum time per episode in seconds (default: `60`)
- `--reset-time`: Time for resetting between episodes in seconds (default: `60`)

### Data Saving Options

- `--root`: Root directory to save dataset locally (default: `~/.cache/lerobot`)
- `--push-to-hub`: Push evaluation dataset to HuggingFace Hub
- `--video`: Encode videos in the dataset (default: enabled)

### Display Options

- `--display-data` / `--no-display`: Show/hide camera feed during evaluation (default: disabled for headless mode)
- `--play-sounds`: Enable vocal synthesis for events

**Example with HuggingFace Hub model:**

```bash
python policy/diffusion/inference.py \
  --checkpoint username/diffusion-so101-pick_place \
  --robot-port /dev/ttyACM0 \
  --camera-config config/camera.toml \
  --episode 5 \
  --username username \
  --policy-type diffusion \
  --robot-type so101 \
  --task pick_place \
  --push-to-hub
```

The inference script will:
1. Load the trained policy from checkpoint or HuggingFace Hub
2. Connect to the robot and cameras (configured via TOML)
3. Run autonomous policy control for the specified number of episodes
4. Save evaluation results as a dataset for analysis
5. Optionally push the evaluation dataset to HuggingFace Hub

## Client-Server Inference (Remote GPU)

For setups where the robot runs on a low-power device (e.g., Jetson) and inference runs on a remote GPU server, use the client-server architecture:

> **Examples**: See [`example/diffusion/inference_server.bash`](../example/diffusion/inference_server.bash) and [`example/diffusion/inference_client.bash`](../example/diffusion/inference_client.bash) for complete examples.

### Start the Inference Server (GPU Machine)

```bash
python policy/diffusion/inference_server.py \
  --checkpoint ./outputs/diffusion_training/pretrained_model \
  --port 8000 \
  --device cuda
```

**Server Arguments:**

- `--checkpoint`: Path to trained policy checkpoint or HuggingFace repo ID (required)
- `--host`: Host address to bind to (default: `0.0.0.0`)
- `--port`: Port to listen on (default: `8000`)
- `--device`: Device to run inference on (default: `cuda`)
- `--task`: Task description for inference (default: `Policy evaluation`)

### Start the Robot Client (Robot Machine)

```bash
python policy/diffusion/inference_client.py \
  --server-host <gpu_server_ip> \
  --robot-port /dev/ttyACM0 \
  --episode 10
```

**Client Arguments:**

- `--server-host`: Inference server host address (required)
- `--server-port`: Inference server port (default: `8000`)
- `--robot-port`: Robot serial port (required)
- `--robot-id`: Robot ID (default: `inference_robot`)
- `--camera-config`: Path to camera config TOML file (default: `config/camera.toml`)
- `--episode`: Number of episodes to run (default: `1`)
- `--episode-time`: Duration of each episode in seconds (default: `60`)
- `--reset-time`: Time for reset between episodes in seconds (default: `60`)
- `--fps`: Control frequency in Hz (default: `30`)

### Example with Jetson and Remote GPU

On the GPU server (e.g., `192.168.1.100`):
```bash
python policy/diffusion/inference_server.py \
  --checkpoint username/diffusion-so101-pick_place \
  --port 8000 \
  --device cuda
```

On the Jetson (robot machine):
```bash
python policy/diffusion/inference_client.py \
  --server-host 192.168.1.100 \
  --server-port 8000 \
  --robot-port /dev/ttyACM0 \
  --camera-config config/camera.toml \
  --episode 5 \
  --fps 30
```

### Architecture Overview

The client-server architecture:
1. **Server**: Loads the policy model and handles inference requests over TCP
2. **Client**: Connects to the robot, gathers observations, sends them to the server, and actuates the robot with returned actions
3. **Communication**: Uses pickle serialization over TCP with length-prefixed messages
4. **Multi-client**: Server supports multiple concurrent robot connections via threading
5. **Episode management**: Client sends reset signals to server between episodes to clear policy state
6. **Safe disconnect**: Client records initial joint positions on connect and returns to them on disconnect
