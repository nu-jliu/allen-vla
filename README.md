# allen-vla

Repository to run experiments with various VLA (Vision-Language-Action) models on SoArm-101.

## Project Goal

This project aims to explore and benchmark different VLA models for robotic manipulation tasks using the SoArm-101 robot arm. The workflow consists of two main phases:

1. **Teleoperation & Data Collection Pipeline**: Setup LeRobot with SoArm for teleoperating the robot and collecting high-quality demonstration datasets
2. **Training & Deployment Pipeline**: Implement, train, and deploy various VLA models (π0, π0.5, ACT, etc.) and evaluate their performance

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
