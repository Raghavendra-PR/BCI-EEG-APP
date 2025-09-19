#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

NAMESPACE="bci-eeg"

echo -e "${GREEN}🔍 BCI EEG Deployment Validation${NC}"
echo -e "${BLUE}================================${NC}"

# Function to check status
check_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2${NC}"
        return 0
    else
        echo -e "${RED}❌ $2${NC}"
        return 1
    fi
}

# Function to check with timeout
check_with_timeout() {
    timeout 30s $1 &>/dev/null
    return $?
}

echo -e "\n${YELLOW}1. Checking Kubernetes Connection...${NC}"
kubectl cluster-info &>/dev/null
check_status $? "Kubernetes cluster connection"

echo -e "\n${YELLOW}2. Checking Namespace...${NC}"
kubectl get namespace $NAMESPACE &>/dev/null
check_status $? "Namespace '$NAMESPACE' exists"

echo -e "\n${YELLOW}3. Checking GPU Nodes...${NC}"
GPU_NODES=$(kubectl get nodes -l accelerator=nvidia --no-headers 2>/dev/null | wc -l)
if [ $GPU_NODES -gt 0 ]; then
    echo -e "${GREEN}✅ Found $GPU_NODES GPU node(s)${NC}"
    kubectl get nodes -l accelerator=nvidia
else
    echo -e "${RED}❌ No GPU nodes found${NC}"
fi

echo -e "\n${YELLOW}4. Checking NVIDIA Device Plugin...${NC}"
kubectl get pods -n kube-system | grep nvidia-device-plugin &>/dev/null
check_status $? "NVIDIA device plugin is running"

echo -e "\n${YELLOW}5. Checking Application Pods...${NC}"
READY_PODS=$(kubectl get pods -n $NAMESPACE -l app=bci-eeg-app --no-headers 2>/dev/null | grep "Running" | wc -l)
TOTAL_PODS=$(kubectl get pods -n $NAMESPACE -l app=bci-eeg-app --no-headers 2>/dev/null | wc -l)

if [ $READY_PODS -gt 0 ] && [ $READY_PODS -eq $TOTAL_PODS ]; then
    echo -e "${GREEN}✅ $READY_PODS/$TOTAL_PODS pods are running${NC}"
    kubectl get pods -n $NAMESPACE -l app=bci-eeg-app
else
    echo -e "${RED}❌ Only $READY_PODS/$TOTAL_PODS pods are running${NC}"
    kubectl get pods -n $NAMESPACE -l app=bci-eeg-app
fi

echo -e "\n${YELLOW}6. Checking Services...${NC}"
kubectl get service bci-eeg-service -n $NAMESPACE &>/dev/null
check_status $? "LoadBalancer service exists"

# Get external IP
EXTERNAL_IP=$(kubectl get service bci-eeg-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
if [ ! -z "$EXTERNAL_IP" ] && [ "$EXTERNAL_IP" != "null" ]; then
    echo -e "${GREEN}✅ External IP assigned: $EXTERNAL_IP${NC}"
else
    echo -e "${YELLOW}⏳ External IP pending...${NC}"
fi

echo -e "\n${YELLOW}7. Checking HPA...${NC}"
kubectl get hpa bci-eeg-hpa -n $NAMESPACE &>/dev/null
check_status $? "Horizontal Pod Autoscaler configured"

echo -e "\n${YELLOW}8. Checking GPU Resource Allocation...${NC}"
GPU_ALLOCATED=$(kubectl describe nodes -l accelerator=nvidia 2>/dev/null | grep "nvidia.com/gpu" | grep -v "0" | wc -l)
if [ $GPU_ALLOCATED -gt 0 ]; then
    echo -e "${GREEN}✅ GPU resources are allocated${NC}"
else
    echo -e "${YELLOW}⚠️  No GPU resources currently allocated${NC}"
fi

echo -e "\n${YELLOW}9. Testing Application Connectivity...${NC}"
if [ ! -z "$EXTERNAL_IP" ] && [ "$EXTERNAL_IP" != "null" ]; then
    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 10 http://$EXTERNAL_IP/ 2>/dev/null)
    if [ "$HTTP_STATUS" = "200" ]; then
        echo -e "${GREEN}✅ Application is accessible at http://$EXTERNAL_IP${NC}"
    else
        echo -e "${YELLOW}⏳ Application not yet accessible (HTTP $HTTP_STATUS)${NC}"
    fi
else
    echo -e "${YELLOW}⏳ Waiting for external IP assignment${NC}"
fi

echo -e "\n${YELLOW}10. Checking Ingress (if configured)...${NC}"
kubectl get ingress bci-eeg-ingress -n $NAMESPACE &>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Ingress controller configured${NC}"
    kubectl get ingress bci-eeg-ingress -n $NAMESPACE
else
    echo -e "${BLUE}ℹ️  Ingress not configured (using LoadBalancer only)${NC}"
fi

echo -e "\n${BLUE}================================${NC}"
echo -e "${GREEN}🎯 Deployment Summary${NC}"
echo -e "${BLUE}================================${NC}"

if [ ! -z "$EXTERNAL_IP" ] && [ "$EXTERNAL_IP" != "null" ]; then
    echo -e "${GREEN}🌐 Public Access URL: http://$EXTERNAL_IP${NC}"
else
    echo -e "${YELLOW}🌐 Public Access: Pending IP assignment${NC}"
fi

echo -e "${BLUE}📊 Resource Information:${NC}"
echo -e "   • GPU Nodes: $GPU_NODES"
echo -e "   • Running Pods: $READY_PODS/$TOTAL_PODS"
echo -e "   • Namespace: $NAMESPACE"

echo -e "\n${BLUE}🔧 Useful Commands:${NC}"
echo -e "   • Monitor: ./scripts/monitor.sh"
echo -e "   • Logs: kubectl logs -f deployment/bci-eeg-app -n $NAMESPACE"
echo -e "   • Scale: kubectl scale deployment bci-eeg-app --replicas=3 -n $NAMESPACE"

if [ $READY_PODS -gt 0 ] && [ ! -z "$EXTERNAL_IP" ]; then
    echo -e "\n${GREEN}🎉 Deployment validation completed successfully!${NC}"
    exit 0
else
    echo -e "\n${YELLOW}⚠️  Deployment validation completed with warnings.${NC}"
    echo -e "   Run this script again in a few minutes to check progress."
    exit 1
fi