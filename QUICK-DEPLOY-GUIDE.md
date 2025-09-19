# 🚀 Quick Deploy Guide: BCI EEG App on GPU-Enabled AKS

## Problem Statement
Deploy BCI-EEG app on GPU-enabled Azure Kubernetes Service (AKS) for public access.

## Solution Overview
- **Platform**: Azure Kubernetes Service (AKS)
- **GPU Support**: NVIDIA Tesla V100/K80 nodes
- **Public Access**: LoadBalancer + Ingress Controller
- **Auto-scaling**: HPA with GPU resource management
- **SSL**: Let's Encrypt certificates

## 📋 Prerequisites Checklist
- [ ] Azure CLI installed (`az --version`)
- [ ] kubectl installed (`kubectl version --client`)
- [ ] Docker installed (`docker --version`)
- [ ] Azure subscription with GPU quota
- [ ] Logged into Azure (`az login`)

## 🎯 One-Command Deployment

### Step 1: Setup AKS Cluster (15-20 minutes)
```bash
# Make executable and run
chmod +x scripts/setup-aks-gpu.sh
./scripts/setup-aks-gpu.sh
```

### Step 2: Build & Push Image (5 minutes)
```bash
# Update ACR name in the script first, then run
chmod +x scripts/build-and-push.sh
./scripts/build-and-push.sh
```

### Step 3: Deploy Application (3 minutes)
```bash
# Update cluster details in the script first, then run
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```

### Step 4: Monitor Deployment
```bash
chmod +x scripts/monitor.sh
./scripts/monitor.sh
```

## 🔧 Configuration Updates Required

After running `setup-aks-gpu.sh`, update these files with your actual values:

### 1. Update `scripts/build-and-push.sh`:
```bash
ACR_NAME="your-actual-acr-name"  # From cluster-info.txt
RESOURCE_GROUP="bci-eeg-rg"
```

### 2. Update `scripts/deploy.sh`:
```bash
CLUSTER_NAME="bci-eeg-aks"
RESOURCE_GROUP="bci-eeg-rg"
ACR_NAME="your-actual-acr-name"
```

## 🌐 Public Access Options

### Option 1: LoadBalancer (Immediate Access)
```bash
# Get external IP after deployment
kubectl get service bci-eeg-service -n bci-eeg
# Access via: http://<EXTERNAL-IP>
```

### Option 2: Custom Domain with SSL
1. Update `k8s/ingress.yaml` with your domain
2. Point your domain's DNS to ingress IP
3. Apply ingress: `kubectl apply -f k8s/ingress.yaml`
4. Access via: https://your-domain.com

## 📊 Verification Commands

```bash
# Check GPU nodes
kubectl get nodes -l accelerator=nvidia

# Check pods are running
kubectl get pods -n bci-eeg

# Check GPU allocation
kubectl describe nodes | grep -A 5 "nvidia.com/gpu"

# Get public IP
kubectl get service bci-eeg-service -n bci-eeg
```

## 🎛️ GPU Configuration Details

- **GPU VM Size**: Standard_NC6s_v3 (1x NVIDIA Tesla V100)
- **Auto-scaling**: 1-3 GPU nodes
- **Resource Request**: 1 GPU per pod
- **Taints**: GPU nodes dedicated for GPU workloads

## 💰 Cost Optimization

```bash
# Scale down GPU nodes when not needed
az aks nodepool scale --resource-group bci-eeg-rg --cluster-name bci-eeg-aks --name gpunodes --node-count 0

# Scale up when needed
az aks nodepool scale --resource-group bci-eeg-rg --cluster-name bci-eeg-aks --name gpunodes --node-count 1
```

## 🚨 Troubleshooting

### GPU Not Available
```bash
kubectl get pods -n kube-system | grep nvidia
kubectl describe nodes -l accelerator=nvidia
```

### Pod Not Starting
```bash
kubectl describe pod <pod-name> -n bci-eeg
kubectl logs <pod-name> -n bci-eeg
```

### Service Not Accessible
```bash
kubectl get endpoints -n bci-eeg
kubectl get ingress -n bci-eeg
```

## 🧹 Cleanup

```bash
# Delete application
kubectl delete namespace bci-eeg

# Delete entire cluster (if needed)
az group delete --name bci-eeg-rg
```

## 📈 Expected Results

After successful deployment:
- ✅ BCI EEG app running on GPU-enabled AKS
- ✅ Publicly accessible via LoadBalancer IP
- ✅ Auto-scaling based on CPU/Memory usage
- ✅ SSL-enabled (if using ingress)
- ✅ Monitoring dashboard available

## 🎯 Success Metrics

- **Deployment Time**: ~25 minutes total
- **Public Access**: Immediate via LoadBalancer
- **GPU Utilization**: Visible in monitoring
- **Auto-scaling**: 2-10 pods based on load
- **Uptime**: 99.9% with multi-replica setup