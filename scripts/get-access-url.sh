#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

NAMESPACE="bci-eeg"

echo -e "${GREEN}🌐 BCI EEG Application Access Information${NC}"
echo -e "${BLUE}========================================${NC}"

# Check if service exists
if ! kubectl get service bci-eeg-service -n $NAMESPACE &>/dev/null; then
    echo -e "${RED}❌ Service 'bci-eeg-service' not found in namespace '$NAMESPACE'${NC}"
    echo -e "${YELLOW}💡 Make sure the application is deployed first.${NC}"
    exit 1
fi

echo -e "\n${YELLOW}📊 Service Status:${NC}"
kubectl get service bci-eeg-service -n $NAMESPACE

# Get external IP
echo -e "\n${YELLOW}🔍 Checking for external IP...${NC}"
EXTERNAL_IP=""
COUNTER=0
MAX_ATTEMPTS=30

while [ -z "$EXTERNAL_IP" ] && [ $COUNTER -lt $MAX_ATTEMPTS ]; do
    EXTERNAL_IP=$(kubectl get service bci-eeg-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
    
    if [ -z "$EXTERNAL_IP" ] || [ "$EXTERNAL_IP" = "null" ]; then
        echo -e "${YELLOW}⏳ Waiting for external IP assignment... (${COUNTER}/${MAX_ATTEMPTS})${NC}"
        sleep 10
        COUNTER=$((COUNTER + 1))
    else
        break
    fi
done

if [ ! -z "$EXTERNAL_IP" ] && [ "$EXTERNAL_IP" != "null" ]; then
    echo -e "\n${GREEN}✅ External IP assigned: $EXTERNAL_IP${NC}"
    
    # Test connectivity
    echo -e "\n${YELLOW}🔗 Testing connectivity...${NC}"
    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 10 http://$EXTERNAL_IP/ 2>/dev/null)
    
    if [ "$HTTP_STATUS" = "200" ]; then
        echo -e "${GREEN}✅ Application is accessible!${NC}"
        
        echo -e "\n${BLUE}🎯 Access Information:${NC}"
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${GREEN}🌐 Public URL: http://$EXTERNAL_IP${NC}"
        echo -e "${GREEN}📱 Mobile Access: http://$EXTERNAL_IP${NC}"
        echo -e "${GREEN}🖥️  Desktop Access: http://$EXTERNAL_IP${NC}"
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        
        echo -e "\n${BLUE}📋 Application Features Available:${NC}"
        echo -e "   • 🧠 EEG Signal Classification (Wheelchair Control)"
        echo -e "   • 😊 Emotion Detection from EEG"
        echo -e "   • 🤖 Robotic Arm Control"
        echo -e "   • 📊 Real-time Visualization"
        echo -e "   • 📁 Batch Processing"
        echo -e "   • 📈 Interactive Dashboards"
        
    elif [ "$HTTP_STATUS" = "000" ]; then
        echo -e "${YELLOW}⏳ Application is starting up... (Connection timeout)${NC}"
        echo -e "${BLUE}💡 Try accessing http://$EXTERNAL_IP in a few minutes${NC}"
    else
        echo -e "${YELLOW}⚠️  Application responded with HTTP $HTTP_STATUS${NC}"
        echo -e "${BLUE}💡 URL: http://$EXTERNAL_IP${NC}"
    fi
    
else
    echo -e "\n${RED}❌ External IP not assigned after ${MAX_ATTEMPTS} attempts${NC}"
    echo -e "${YELLOW}💡 This might take longer. Check Azure portal or try:${NC}"
    echo -e "   kubectl get service bci-eeg-service -n $NAMESPACE --watch"
fi

# Check ingress if available
echo -e "\n${YELLOW}🔍 Checking for Ingress configuration...${NC}"
if kubectl get ingress bci-eeg-ingress -n $NAMESPACE &>/dev/null; then
    echo -e "${GREEN}✅ Ingress controller found${NC}"
    kubectl get ingress bci-eeg-ingress -n $NAMESPACE
    
    INGRESS_IP=$(kubectl get ingress bci-eeg-ingress -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
    if [ ! -z "$INGRESS_IP" ] && [ "$INGRESS_IP" != "null" ]; then
        echo -e "${GREEN}🌐 Ingress URL: https://your-domain.com${NC}"
    fi
else
    echo -e "${BLUE}ℹ️  No ingress configured (using LoadBalancer only)${NC}"
fi

# Show pod status
echo -e "\n${YELLOW}📊 Pod Status:${NC}"
kubectl get pods -n $NAMESPACE -l app=bci-eeg-app

# Show resource usage if available
echo -e "\n${YELLOW}💻 Resource Usage:${NC}"
kubectl top pods -n $NAMESPACE --containers 2>/dev/null || echo -e "${BLUE}ℹ️  Metrics not available yet${NC}"

echo -e "\n${BLUE}🔧 Management Commands:${NC}"
echo -e "   • Monitor: ./scripts/monitor.sh"
echo -e "   • Validate: ./scripts/validate-deployment.sh"
echo -e "   • Scale up: kubectl scale deployment bci-eeg-app --replicas=5 -n $NAMESPACE"
echo -e "   • View logs: kubectl logs -f deployment/bci-eeg-app -n $NAMESPACE"

if [ ! -z "$EXTERNAL_IP" ] && [ "$EXTERNAL_IP" != "null" ]; then
    echo -e "\n${GREEN}🎉 Your BCI EEG application is publicly accessible!${NC}"
    echo -e "${GREEN}🔗 Share this URL: http://$EXTERNAL_IP${NC}"
fi