#!/bin/bash

# Simple BCI EEG Deployment Script for GPU-Enabled AKS
# This script deploys the essential components only

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}🧠 BCI EEG - Simple Kubernetes Deployment${NC}"
echo -e "${BLUE}===========================================${NC}"

# Configuration (UPDATE THESE)
ACR_NAME="your-acr-name"  # Replace with your ACR name
RESOURCE_GROUP="your-resource-group"  # Replace with your resource group
CLUSTER_NAME="your-cluster-name"  # Replace with your cluster name

echo -e "\n${YELLOW}📋 Current Configuration:${NC}"
echo -e "ACR Name: $ACR_NAME"
echo -e "Resource Group: $RESOURCE_GROUP"
echo -e "Cluster Name: $CLUSTER_NAME"

if [ "$ACR_NAME" = "your-acr-name" ]; then
    echo -e "\n${RED}❌ Please update the configuration in this script first!${NC}"
    echo -e "${YELLOW}Edit simple-deploy.sh and update:${NC}"
    echo -e "- ACR_NAME"
    echo -e "- RESOURCE_GROUP"
    echo -e "- CLUSTER_NAME"
    exit 1
fi

# Check prerequisites
echo -e "\n${YELLOW}🔍 Checking Prerequisites...${NC}"

if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}❌ kubectl not found${NC}"
    exit 1
fi
echo -e "${GREEN}✅ kubectl found${NC}"

if ! command -v az &> /dev/null; then
    echo -e "${RED}❌ Azure CLI not found${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Azure CLI found${NC}"

# Get AKS credentials
echo -e "\n${YELLOW}🔑 Getting AKS credentials...${NC}"
az aks get-credentials --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME --overwrite-existing

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to get AKS credentials${NC}"
    exit 1
fi

# Update deployment with correct ACR name
echo -e "\n${YELLOW}🔧 Updating deployment configuration...${NC}"
sed -i "s/your-acr-name/$ACR_NAME/g" k8s/deployment.yaml

# Deploy essential components
echo -e "\n${YELLOW}☸️  Deploying to Kubernetes...${NC}"

echo -e "${BLUE}Creating namespace...${NC}"
kubectl apply -f k8s/namespace.yaml

echo -e "${BLUE}Applying configuration...${NC}"
kubectl apply -f k8s/configmap.yaml

echo -e "${BLUE}Deploying application...${NC}"
kubectl apply -f k8s/deployment.yaml

echo -e "${BLUE}Creating service...${NC}"
kubectl apply -f k8s/service.yaml

echo -e "${BLUE}Setting up auto-scaling...${NC}"
kubectl apply -f k8s/hpa.yaml

# Wait for deployment
echo -e "\n${YELLOW}⏳ Waiting for deployment to be ready...${NC}"
kubectl wait --for=condition=available --timeout=300s deployment/bci-eeg-app -n bci-eeg

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ Deployment successful!${NC}"
    
    # Get service information
    echo -e "\n${YELLOW}🌐 Getting service information...${NC}"
    kubectl get services -n bci-eeg
    
    # Wait for external IP
    echo -e "\n${YELLOW}⏳ Waiting for external IP (this may take a few minutes)...${NC}"
    EXTERNAL_IP=""
    COUNTER=0
    
    while [ -z "$EXTERNAL_IP" ] && [ $COUNTER -lt 20 ]; do
        EXTERNAL_IP=$(kubectl get service bci-eeg-service -n bci-eeg -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
        if [ -z "$EXTERNAL_IP" ] || [ "$EXTERNAL_IP" = "null" ]; then
            echo -e "${YELLOW}⏳ Waiting... (${COUNTER}/20)${NC}"
            sleep 15
            COUNTER=$((COUNTER + 1))
        fi
    done
    
    if [ ! -z "$EXTERNAL_IP" ] && [ "$EXTERNAL_IP" != "null" ]; then
        echo -e "\n${GREEN}🎉 SUCCESS! Your BCI EEG app is now publicly accessible!${NC}"
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${GREEN}🌐 Public URL: http://$EXTERNAL_IP${NC}"
        echo -e "${GREEN}📱 Access from anywhere: http://$EXTERNAL_IP${NC}"
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        
        # Test connectivity
        echo -e "\n${YELLOW}🔗 Testing connectivity...${NC}"
        HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 10 http://$EXTERNAL_IP/ 2>/dev/null)
        if [ "$HTTP_STATUS" = "200" ]; then
            echo -e "${GREEN}✅ Application is responding correctly!${NC}"
        else
            echo -e "${YELLOW}⚠️  Application is starting up (HTTP $HTTP_STATUS)${NC}"
        fi
        
    else
        echo -e "\n${YELLOW}⚠️  External IP not assigned yet.${NC}"
        echo -e "${BLUE}Check status with: kubectl get service bci-eeg-service -n bci-eeg${NC}"
    fi
    
    # Show pod status
    echo -e "\n${YELLOW}📊 Pod Status:${NC}"
    kubectl get pods -n bci-eeg
    
    # Show useful commands
    echo -e "\n${BLUE}🔧 Useful Commands:${NC}"
    echo -e "• Check status: kubectl get pods -n bci-eeg"
    echo -e "• View logs: kubectl logs -f deployment/bci-eeg-app -n bci-eeg"
    echo -e "• Scale up: kubectl scale deployment bci-eeg-app --replicas=5 -n bci-eeg"
    echo -e "• Get URL: kubectl get service bci-eeg-service -n bci-eeg"
    
else
    echo -e "\n${RED}❌ Deployment failed or timed out${NC}"
    echo -e "${YELLOW}Checking pod status...${NC}"
    kubectl get pods -n bci-eeg
    kubectl describe pods -n bci-eeg
fi