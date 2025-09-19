# 🧠 BCI EEG Application - Azure Deployment

Brain-Computer Interface EEG Signal-Based Controller for wheelchair control, emotion detection, and robotic arm control.

## 🚀 Quick Deploy

### Prerequisites
- Azure CLI installed (`az login`)
- Docker Desktop running
- kubectl installed

### Option 1: Container Instances (Fastest - 5 minutes)
```bash
chmod +x final-deploy-aci.sh
./final-deploy-aci.sh
```

### Option 2: Kubernetes with GPU (Production - 20 minutes)
```bash
chmod +x deploy-with-gpu.sh
./deploy-with-gpu.sh
```

## 🎯 Features
- 🧠 EEG Signal Classification (Wheelchair Control)
- 😊 Emotion Detection from EEG
- 🤖 Robotic Arm Control
- 📊 Real-time Visualization
- 📁 Batch Processing

## 📁 Essential Files
- `eeg_app.py` - Main application
- `requirements.txt` - Dependencies
- `Dockerfile` - Container config
- `*.joblib` - ML models (5 files)
- `sample_*.csv` - Sample data
- `final-deploy-aci.sh` - Container deployment
- `deploy-with-gpu.sh` - Kubernetes deployment
- `k8s/` - Kubernetes manifests

## 🔧 Troubleshooting
- **Azure CLI not found**: `winget install Microsoft.AzureCLI`
- **Docker not running**: Start Docker Desktop
- **External IP pending**: Wait 5-10 minutes

**After deployment, you'll get a public URL to access your BCI EEG application!** 🌍