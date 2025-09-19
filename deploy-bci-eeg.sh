#!/bin/bash

# BCI EEG Application - Complete Deployment Script for GPU-Enabled AKS
# This script automates the entire deployment process

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
RESOURCE_GROUP="bci-eeg-rg"
CLUSTER_NAME="bci-eeg-aks"
LOCATION="eastus"
ACR_NAME="bcieegacr$(date +%s)"
NAMESPACE="bci-eeg"

echo -e "${PURPLE}🧠 BCI EEG Application - GPU-Enabled AKS Deployment${NC}"
echo -e "${BLUE}====================================================${NC}"
echo -e "${GREEN}This script will deploy your BCI EEG app to Azure Kubernetes Service with GPU support${NC}"
echo -e "${BLUE}====================================================${NC}"

# Function to check prerequisites
check_prerequisites() {
    echo -e "\n${YELLOW}🔍 Checking Prerequisites...${NC}"
    
    # Check Azure CLI
    if ! command -v az &> /dev/null; then
        echo -e "${RED}❌ Azure CLI not found. Please install it first.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Azure CLI found${NC}"
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        echo -e "${RED}❌ kubectl not found. Please install it first.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ kubectl found${NC}"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker not found. Please install it first.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Docker found${NC}"
    
    # Check Azure login
    if ! az account show &> /dev/null; then
        echo -e "${YELLOW}⚠️  Not logged in to Azure. Please run 'az login' first.${NC}"
        read -p "Do you want to login now? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            az login
        else
            exit 1
        fi
    fi
    echo -e "${GREEN}✅ Azure CLI authenticated${NC}"
}

# Function to setup AKS cluster
setup_aks_cluster() {
    echo -e "\n${YELLOW}🚀 Setting up AKS Cluster with GPU Support...${NC}"
    
    # Create resource group
    echo -e "${BLUE}Creating resource group: $RESOURCE_GROUP${NC}"
    az group create --name $RESOURCE_GROUP --location $LOCATION
    
    # Create ACR
    echo -e "${BLUE}Creating Azure Container Registry: $ACR_NAME${NC}"
    az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Standard --location $LOCATION
    
    # Create AKS cluster
    echo -e "${BLUE}Creating AKS cluster: $CLUSTER_NAME (this may take 10-15 minutes)${NC}"
    az aks create \
        --resource-group $RESOURCE_GROUP \
        --name $CLUSTER_NAME \
        --location $LOCATION \
        --node-count 2 \
        --node-vm-size Standard_D2s_v3 \
        --generate-ssh-keys \
        --attach-acr $ACR_NAME \
        --enable-managed-identity \
        --enable-addons monitoring \
        --network-plugin azure \
        --network-policy azure \
        --load-balancer-sku standard \
        --vm-set-type VirtualMachineScaleSets \
        --kubernetes-version "1.28.0" \
        --node-osdisk-size 100 \
        --max-pods 110
    
    # Add GPU node pool
    echo -e "${BLUE}Adding GPU node pool...${NC}"
    az aks nodepool add \
        --resource-group $RESOURCE_GROUP \
        --cluster-name $CLUSTER_NAME \
        --name gpunodes \
        --node-count 1 \
        --node-vm-size Standard_NC6s_v3 \
        --node-taints sku=gpu:NoSchedule \
        --aks-custom-headers UseGPUDedicatedVHD=true \
        --enable-cluster-autoscaler \
        --min-count 1 \
        --max-count 3
    
    # Get credentials
    echo -e "${BLUE}Getting AKS credentials...${NC}"
    az aks get-credentials --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME --overwrite-existing
    
    # Install NVIDIA device plugin
    echo -e "${BLUE}Installing NVIDIA device plugin...${NC}"
    kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml
    
    # Install NGINX Ingress Controller
    echo -e "${BLUE}Installing NGINX Ingress Controller...${NC}"
    kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/cloud/deploy.yaml
    
    # Wait for ingress controller
    kubectl wait --namespace ingress-nginx \
        --for=condition=ready pod \
        --selector=app.kubernetes.io/component=controller \
        --timeout=300s
}

# Function to build and push image
build_and_push_image() {
    echo -e "\n${YELLOW}🐳 Building and Pushing Docker Image...${NC}"
    
    # Login to ACR
    az acr login --name $ACR_NAME
    
    # Build image
    echo -e "${BLUE}Building Docker image...${NC}"
    docker build -t $ACR_NAME.azurecr.io/bci-eeg:latest .
    
    # Push image
    echo -e "${BLUE}Pushing image to ACR...${NC}"
    docker push $ACR_NAME.azurecr.io/bci-eeg:latest
}

# Function to deploy application
deploy_application() {
    echo -e "\n${YELLOW}☸️  Deploying Application to Kubernetes...${NC}"
    
    # Update deployment with correct ACR name
    sed -i "s/your-acr-name/$ACR_NAME/g" k8s/deployment.yaml
    
    # Apply manifests
    echo -e "${BLUE}Applying Kubernetes manifests...${NC}"
    kubectl apply -f k8s/namespace.yaml
    kubectl apply -f k8s/configmap.yaml
    kubectl apply -f k8s/pvc.yaml
    kubectl apply -f k8s/deployment.yaml
    kubectl apply -f k8s/service.yaml
    kubectl apply -f k8s/hpa.yaml
    
    # Wait for deployment
    echo -e "${BLUE}Waiting for deployment to be ready...${NC}"
    kubectl wait --for=condition=available --timeout=300s deployment/bci-eeg-app -n $NAMESPACE
}

# Function to get access information
get_access_info() {
    echo -e "\n${YELLOW}🌐 Getting Access Information...${NC}"
    
    # Wait for external IP
    echo -e "${BLUE}Waiting for external IP assignment...${NC}"
    EXTERNAL_IP=""
    COUNTER=0
    while [ -z "$EXTERNAL_IP" ] && [ $COUNTER -lt 30 ]; do
        EXTERNAL_IP=$(kubectl get service bci-eeg-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
        if [ -z "$EXTERNAL_IP" ] || [ "$EXTERNAL_IP" = "null" ]; then
            echo -e "${YELLOW}⏳ Waiting... (${COUNTER}/30)${NC}"
            sleep 10
            COUNTER=$((COUNTER + 1))
        fi
    done
    
    if [ ! -z "$EXTERNAL_IP" ] && [ "$EXTERNAL_IP" != "null" ]; then
        echo -e "\n${GREEN}🎉 Deployment Successful!${NC}"
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${GREEN}🌐 Public URL: http://$EXTERNAL_IP${NC}"
        echo -e "${GREEN}📱 Access from anywhere: http://$EXTERNAL_IP${NC}"
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    else
        echo -e "${YELLOW}⚠️  External IP not assigned yet. Check later with:${NC}"
        echo -e "kubectl get service bci-eeg-service -n $NAMESPACE"
    fi
}

# Function to save deployment info
save_deployment_info() {
    cat > deployment-info.txt <<EOF
BCI EEG Application Deployment Information
==========================================

Resource Group: $RESOURCE_GROUP
Cluster Name: $CLUSTER_NAME
ACR Name: $ACR_NAME
Location: $LOCATION
Namespace: $NAMESPACE

Access Commands:
- Get URL: ./scripts/get-access-url.sh
- Monitor: ./scripts/monitor.sh
- Validate: ./scripts/validate-deployment.sh

Management Commands:
- Scale: kubectl scale deployment bci-eeg-app --replicas=5 -n $NAMESPACE
- Logs: kubectl logs -f deployment/bci-eeg-app -n $NAMESPACE
- Status: kubectl get pods -n $NAMESPACE

Cleanup:
- Delete app: kubectl delete namespace $NAMESPACE
- Delete cluster: az group delete --name $RESOURCE_GROUP
EOF
    
    echo -e "\n${GREEN}📄 Deployment information saved to deployment-info.txt${NC}"
}

# Main execution
main() {
    check_prerequisites
    
    echo -e "\n${BLUE}📋 Deployment Configuration:${NC}"
    echo -e "Resource Group: $RESOURCE_GROUP"
    echo -e "Cluster Name: $CLUSTER_NAME"
    echo -e "ACR Name: $ACR_NAME"
    echo -e "Location: $LOCATION"
    
    read -p "Do you want to proceed with the deployment? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Deployment cancelled.${NC}"
        exit 0
    fi
    
    setup_aks_cluster
    build_and_push_image
    deploy_application
    get_access_info
    save_deployment_info
    
    echo -e "\n${GREEN}🎯 Deployment completed successfully!${NC}"
    echo -e "${BLUE}Next steps:${NC}"
    echo -e "1. Run ./scripts/validate-deployment.sh to verify everything is working"
    echo -e "2. Run ./scripts/get-access-url.sh to get the public URL"
    echo -e "3. Run ./scripts/monitor.sh to monitor the application"
}

# Run main function
main "$@"