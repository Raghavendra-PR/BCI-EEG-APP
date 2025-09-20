#!/bin/bash

# Final Deploy to Azure Container Instances - All Issues Fixed

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${YELLOW}🚀 Final Deploy BCI EEG to Azure Container Instances${NC}"
echo -e "${BLUE}====================================================${NC}"

# Configuration
RESOURCE_GROUP="bci-eeg-rg"
ACR_NAME="bcieegacr1758269708"
CONTAINER_NAME="bci-eeg-app"
LOCATION="eastus"

echo -e "\n${YELLOW}Step 1: Cleaning up any existing container...${NC}"
az container delete --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --yes 2>/dev/null || echo "No existing container to delete"

echo -e "\n${YELLOW}Step 2: Getting ACR credentials...${NC}"
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query username --output tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query passwords[0].value --output tsv)

echo -e "${GREEN}✅ ACR Username: $ACR_USERNAME${NC}"

echo -e "\n${YELLOW}Step 3: Creating unique DNS name...${NC}"
DNS_NAME="bci-eeg-app-$(date +%s)"
echo -e "${GREEN}✅ DNS Name: $DNS_NAME${NC}"

echo -e "\n${YELLOW}Step 4: Deploying to Azure Container Instances...${NC}"
az container create \
    --resource-group $RESOURCE_GROUP \
    --name $CONTAINER_NAME \
    --image $ACR_NAME.azurecr.io/bci-eeg:latest \
    --registry-login-server $ACR_NAME.azurecr.io \
    --registry-username $ACR_USERNAME \
    --registry-password $ACR_PASSWORD \
    --ports 8501 \
    --dns-name-label $DNS_NAME \
    --location $LOCATION \
    --os-type Linux \
    --cpu 2 \
    --memory 4 \
    --environment-variables STREAMLIT_SERVER_ADDRESS=0.0.0.0 STREAMLIT_SERVER_PORT=8501

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}🎉 Deployment successful!${NC}"
    
    echo -e "\n${YELLOW}⏳ Waiting for container to start...${NC}"
    sleep 30
    
    # Get the public URL
    FQDN=$(az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query ipAddress.fqdn --output tsv)
    IP=$(az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query ipAddress.ip --output tsv)
    STATE=$(az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query containers[0].instanceView.currentState.state --output tsv)
    
    echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}🌐 Your BCI EEG Application is Live!${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}🔗 Public URL: http://$FQDN:8501${NC}"
    echo -e "${GREEN}🌍 IP Address: $IP${NC}"
    echo -e "${GREEN}📊 Status: $STATE${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    echo -e "\n${BLUE}🎯 Your BCI EEG Application Features:${NC}"
    echo -e "• 🧠 EEG Signal Classification (Wheelchair Control)"
    echo -e "• 😊 Emotion Detection from EEG"
    echo -e "• 🤖 Robotic Arm Control"
    echo -e "• 📊 Real-time Visualization"
    echo -e "• 📁 Batch Processing"
    echo -e "• 📈 Interactive Dashboards"
    
    echo -e "\n${BLUE}🔧 Management Commands:${NC}"
    echo -e "• View logs: az container logs --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME"
    echo -e "• Restart: az container restart --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME"
    echo -e "• Delete: az container delete --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME"
    
    echo -e "\n${YELLOW}🔗 Testing connectivity...${NC}"
    if command -v curl &> /dev/null; then
        HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 10 http://$FQDN:8501/ 2>/dev/null)
        if [ "$HTTP_STATUS" = "200" ]; then
            echo -e "${GREEN}✅ Application is responding correctly!${NC}"
        else
            echo -e "${YELLOW}⏳ Application is starting up (HTTP $HTTP_STATUS)${NC}"
            echo -e "${BLUE}💡 Try accessing the URL in a few minutes${NC}"
        fi
    else
        echo -e "${BLUE}💡 Open http://$FQDN:8501 in your browser${NC}"
    fi
    
else
    echo -e "${RED}❌ Deployment failed${NC}"
    echo -e "${YELLOW}Let's check the error details...${NC}"
    
    # Show any existing container info
    az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME 2>/dev/null || echo "No container information available"
fi