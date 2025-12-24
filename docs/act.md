# ACT (Action Chunking Transformer) Policy

This document covers training, inference, and deployment of the ACT policy for the SoArm-101 robot.

## Training

Train an ACT policy on your collected datasets:

```bash
python policy/act/train.py --repo-id your_username/your_dataset --output-dir ./outputs/act_run1
```

Or using uv:

```bash
uv run python policy/act/train.py --repo-id your_username/your_dataset --output-dir ./outputs/act_run1
```

> **Example**: See [`example/train_act.bash`](../example/train_act.bash) for a complete example.

### Required Arguments

- `--repo-id`: HuggingFace dataset repo ID (e.g., `jliu6718/lerobot-so101-abc123`)
- `--output-dir`: Directory to save checkpoints and logs

### Training Hyperparameters

- `--batch-size`: Training batch size (default: `8`)
- `--steps`: Total training steps (default: `100,000`)
- `--num-workers`: Number of dataloader workers (default: `4`)
- `--seed`: Random seed for reproducibility (default: `1000`)

### ACT-Specific Hyperparameters

- `--chunk-size`: Action prediction chunk size (default: `100`)
- `--n-action-steps`: Number of action steps to execute per query (default: `100`)
- `--lr`: Learning rate (default: `1e-5`)
- `--kl-weight`: KL divergence loss weight for VAE (default: `10.0`)
- `--dropout`: Dropout rate in transformer (default: `0.1`)

### Logging and Checkpointing

- `--log-freq`: Log training metrics every N steps (default: `250`)
- `--save-freq`: Save checkpoint every N steps (default: `5000`)
- `--progress-bar` / `--no-progress-bar`: Show/hide tqdm progress bar (default: enabled)

### Weights & Biases Integration

- `--wandb-enable`: Enable Weights & Biases logging
- `--wandb-project`: Wandb project name (default: `soarm-act-training`)
- `--wandb-entity`: Wandb entity/team name (optional)
- `--wandb-notes`: Wandb run notes/description (optional)

### Advanced Options

- `--resume`: Resume training from checkpoint in output-dir
- `--push`: Push checkpoints to HuggingFace Hub
- `--dataset-root`: Local dataset cache directory (optional)

**Example with custom hyperparameters:**

```bash
python policy/act/train.py \
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
python policy/act/train.py \
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

## Inference

![Inference Demo](../assets/inference_demo.gif)

Run inference with a trained ACT policy on the SO101 robot:

```bash
python policy/act/inference.py \
  --checkpoint ./outputs/act_training/pretrained_model \
  --robot-port /dev/ttyACM0 \
  --camera-index 0 \
  --repo-id your_username/eval_results
```

Or using uv:

```bash
uv run python policy/act/inference.py \
  --checkpoint ./outputs/act_training/pretrained_model \
  --robot-port /dev/ttyACM0 \
  --camera-index 0 \
  --repo-id your_username/eval_results
```

> **Example**: See [`example/inference_act.bash`](../example/inference_act.bash) for a complete example.

### Required Arguments

- `--checkpoint`: Path to trained policy checkpoint or HuggingFace repo ID
- `--robot-port`: Robot port (e.g., `/dev/ttyACM0`)
- `--camera-index`: Camera index or path (e.g., `0` for `/dev/video0`)
- `--repo-id`: Dataset repo ID for saving evaluation results

### Robot Configuration

- `--robot-type`: Robot type (default: `so101_follower`)
- `--robot-id`: Robot ID (default: `eval_robot`)

### Camera Configuration

- `--camera-name`: Camera name in config (default: `front`)
- `--camera-width`: Camera width (default: `640`)
- `--camera-height`: Camera height (default: `480`)
- `--camera-fps`: Camera FPS (default: `30`)

### Evaluation Parameters

- `--num-episodes`: Number of episodes to evaluate (default: `10`)
- `--task-description`: Task description for this evaluation run
- `--fps`: Control frequency in Hz (default: `30`)
- `--episode-time`: Maximum time per episode in seconds (default: `60`)
- `--reset-time`: Time for resetting between episodes in seconds (default: `60`)

### Data Saving Options

- `--root`: Root directory to save dataset locally
- `--push-to-hub`: Push evaluation dataset to HuggingFace Hub
- `--video`: Encode videos in the dataset (default: enabled)

### Display Options

- `--display-data` / `--no-display`: Show/hide camera feed during evaluation
- `--play-sounds`: Enable vocal synthesis for events

**Example with HuggingFace Hub model:**

```bash
python policy/act/inference.py \
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

## Client-Server Inference (Remote GPU)

For setups where the robot runs on a low-power device (e.g., Jetson) and inference runs on a remote GPU server, use the client-server architecture:

> **Examples**: See [`example/inference_act_server.bash`](../example/inference_act_server.bash) and [`example/inference_act_client.bash`](../example/inference_act_client.bash) for complete examples.

### Start the Inference Server (GPU Machine)

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

### Start the Robot Client (Robot Machine)

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

### Example with Jetson and Remote GPU

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

### Architecture Overview

The client-server architecture:
1. **Server**: Loads the policy model and handles inference requests over TCP
2. **Client**: Connects to the robot, gathers observations, sends them to the server, and actuates the robot with returned actions
3. **Communication**: Uses pickle serialization over TCP with length-prefixed messages
4. **Multi-client**: Server supports multiple concurrent robot connections via threading
5. **Episode management**: Client sends reset signals to server between episodes to clear policy state
