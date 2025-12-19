# Allen's Awesome Vision-Language-Action (VLA) Models

Repository to run experiments with various VLA (Vision-Language-Action) models on SoArm-101.

## Project Goal

This project aims to explore and benchmark different VLA models for robotic manipulation tasks using the SoArm-101 robot arm. The workflow consists of two main phases:

1. **Teleoperation & Data Collection Pipeline**: Setup LeRobot with SoArm for teleoperating the robot and collecting high-quality demonstration datasets
2. **Training & Deployment Pipeline**: Implement, train, and deploy various VLA models (π0, π0.5, ACT, etc.) and evaluate their performance

## Usage

### Teleoperation

Run the teleoperation script to control the follower arm using the leader arm:

```bash
python teleop/teleop.py
```

The script accepts the following command-line arguments:

- `--leader-port`: Serial port for the leader arm (default: `/dev/ttyACM0`)
- `--leader-id`: ID for the leader arm (default: `my_leader`)
- `--follower-port`: Serial port for the follower arm (default: `/dev/ttyACM1`)
- `--follower-id`: ID for the follower arm (default: `my_follower`)

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

## Todo List

### Phase 1: Teleoperation & Data Collection
- [x] Setup LeRobot environment and dependencies
- [x] Configure SoArm-101 hardware integration with LeRobot
- [x] Implement teleoperation interface
- [ ] Setup camera(s) and sensor pipeline
- [x] Develop data collection scripts
- [ ] Define data format and storage structure
- [x] Test teleoperation and record sample demonstrations
- [ ] Create dataset management utilities

### Phase 2: Training & Deployment Pipeline
- [ ] Setup training infrastructure (GPU environment, configs)
- [ ] Implement π0 model training pipeline
- [ ] Implement π0.5 model training pipeline
- [ ] Implement ACT (Action Chunking Transformer) pipeline
- [ ] Add additional VLA models as needed
- [ ] Create evaluation metrics and benchmarking scripts
- [ ] Implement model deployment interface for SoArm
- [ ] Test deployed models on real hardware
- [ ] Compare model performance and document results

### Documentation & Experimentation
- [ ] Document hardware setup and calibration procedures
- [ ] Create training guides for each model
- [ ] Log experimental results and hyperparameters
- [ ] Build visualization tools for trajectories and predictions
