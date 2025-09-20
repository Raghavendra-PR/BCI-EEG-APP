#!/bin/bash

# Deploy BCI EEG with GPU Support

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}🚀 Deploying BCI EEG with GPU Support${NC}"
echo -e "${BLUE}====================================${NC}"

# Configuration
RESOURCE_GROUP="bci-eeg-rg"
CLUSTER_NAME="bci-eeg-gpu-aks"
ACR_NAME="bcieegacr1758269708"

echo -e "\n${YELLOW}Step 1: Getting AKS credentials...${NC}"
az aks get-credentials --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME --overwrite-existing

echo -e "\n${YELLOW}Step 2: Checking GPU availability...${NC}"

# Check if jq is available, if not use alternative method
if command -v jq &> /dev/null; then
    GPU_AVAILABLE=$(kubectl get nodes -o json | jq -r '.items[].status.allocatable."nvidia.com/gpu"' | grep -v null | wc -l)
else
    # Alternative method without jq
    GPU_AVAILABLE=$(kubectl get nodes -o custom-columns="GPU:.status.allocatable.nvidia\.com/gpu" --no-headers | grep -v "<none>" | wc -l)
fi

if [ $GPU_AVAILABLE -gt 0 ]; then
    echo -e "${GREEN}✅ GPU nodes available, deploying with GPU support${NC}"
    
    # Create GPU-enabled deployment
    cat > k8s/deployment-gpu.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bci-eeg-app
  namespace: bci-eeg
  labels:
    app: bci-eeg-app
spec:
  replicas: 1
  selector:
    matchLabels:
      app: bci-eeg-app
  template:
    metadata:
      labels:
        app: bci-eeg-app
    spec:
      containers:
      - name: bci-eeg-app
        image: $ACR_NAME.azurecr.io/bci-eeg:latest
        ports:
        - containerPort: 8501
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
            nvidia.com/gpu: 1
          limits:
            memory: "4Gi"
            cpu: "2000m"
            nvidia.com/gpu: 1
        env:
        - name: PYTHONUNBUFFERED
          value: "1"
        - name: STREAMLIT_SERVER_ADDRESS
          value: "0.0.0.0"
        - name: STREAMLIT_SERVER_PORT
          value: "8501"
      tolerations:
      - key: "sku"
        operator: "Equal"
        value: "gpu"
        effect: "NoSchedule"
      - key: "nvidia.com/gpu"
        operator: "Exists"
        effect: "NoSchedule"
EOF
    
    DEPLOYMENT_FILE="k8s/deployment-gpu.yaml"
    echo -e "${GREEN}🎯 Using GPU-enabled deployment${NC}"
    
else
    echo -e "${YELLOW}⚠️  No GPU nodes available, using CPU-only deployment${NC}"
    
    # Create CPU-only deployment
    cat > k8s/deployment-cpu.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bci-eeg-app
  namespace: bci-eeg
  labels:
    app: bci-eeg-app
spec:
  replicas: 2
  selector:
    matchLabels:
      app: bci-eeg-app
  template:
    metadata:
      labels:
        app: bci-eeg-app
    spec:
      containers:
      - name: bci-eeg-app
        image: $ACR_NAME.azurecr.io/bci-eeg:latest
        ports:
        - containerPort: 8501
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        env:
        - name: PYTHONUNBUFFERED
          value: "1"
        - name: STREAMLIT_SERVER_ADDRESS
          value: "0.0.0.0"
        - name: STREAMLIT_SERVER_PORT
          value: "8501"
EOF
    
    DEPLOYMENT_FILE="k8s/deployment-cpu.yaml"
    echo -e "${BLUE}🎯 Using CPU-only deployment${NC}"
fi

echo -e "\n${YELLOW}Step 3: Deploying application...${NC}"

# Clean up existing deployment
kubectl delete namespace bci-eeg --ignore-not-found=true
sleep 10

# Deploy application
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f $DEPLOYMENT_FILE
kubectl apply -f k8s/service.yaml

echo -e "\n${YELLOW}Step 4: Waiting for deployment...${NC}"
kubectl wait --for=condition=available --timeout=300s deployment/bci-eeg-app -n bci-eeg

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}🎉 Deployment successful!${NC}"
    
    # Get external IP
    EXTERNAL_IP=""
    COUNTER=0
    while [ -z "$EXTERNAL_IP" ] && [ $COUNTER -lt 15 ]; do
        EXTERNAL_IP=$(kubectl get service bci-eeg-service -n bci-eeg -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
        if [ -z "$EXTERNAL_IP" ] || [ "$EXTERNAL_IP" = "null" ]; then
            echo -e "${YELLOW}⏳ Waiting for external IP... (${COUNTER}/15)${NC}"
            sleep 20
            COUNTER=$((COUNTER + 1))
        fi
    done
    
    echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}🚀 BCI EEG App Deployed Successfully!${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    if [ ! -z "$EXTERNAL_IP" ] && [ "$EXTERNAL_IP" != "null" ]; then
        echo -e "${GREEN}🌐 Public URL: http://$EXTERNAL_IP${NC}"
    fi
    
    if [ $GPU_AVAILABLE -gt 0 ]; then
        echo -e "${GREEN}🎯 GPU-Accelerated: YES (1 GPU per pod)${NC}"
    else
        echo -e "${BLUE}🎯 GPU-Accelerated: NO (CPU-only)${NC}"
    fi
    
    echo -e "\n${BLUE}📊 Current Status:${NC}"
    kubectl get pods -n bci-eeg -o wide
    kubectl get services -n bci-eeg
    
else
    echo -e "${RED}❌ Deployment failed${NC}"
    kubectl get pods -n bci-eeg
fi