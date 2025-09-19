#!/bin/bash

# Check if all required files exist for BCI EEG deployment

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔍 BCI EEG Deployment Files Check${NC}"
echo -e "${BLUE}=================================${NC}"

# Required files
REQUIRED_FILES=(
    "k8s/namespace.yaml"
    "k8s/deployment.yaml"
    "k8s/service.yaml"
    "k8s/configmap.yaml"
    "k8s/hpa.yaml"
    "eeg_app.py"
    "requirements.txt"
    "Dockerfile"
    "best_eeg_model.joblib"
    "best_emotion_model.joblib"
    "eeg_scaler.joblib"
    "emotion_scaler.joblib"
    "emotion_feature_selector.joblib"
)

# Optional files
OPTIONAL_FILES=(
    "k8s/ingress.yaml"
    "k8s/pvc.yaml"
    "simple-deploy.sh"
    "deploy-bci-eeg.sh"
)

echo -e "\n${YELLOW}📋 Checking Required Files:${NC}"
MISSING_REQUIRED=0

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅ $file${NC}"
    else
        echo -e "${RED}❌ $file (MISSING)${NC}"
        MISSING_REQUIRED=$((MISSING_REQUIRED + 1))
    fi
done

echo -e "\n${YELLOW}📋 Checking Optional Files:${NC}"
for file in "${OPTIONAL_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅ $file${NC}"
    else
        echo -e "${YELLOW}⚠️  $file (optional)${NC}"
    fi
done

echo -e "\n${BLUE}📊 Summary:${NC}"
if [ $MISSING_REQUIRED -eq 0 ]; then
    echo -e "${GREEN}🎉 All required files are present!${NC}"
    echo -e "${GREEN}✅ Ready for deployment${NC}"
    
    echo -e "\n${BLUE}🚀 Next Steps:${NC}"
    echo -e "1. Update configuration in simple-deploy.sh:"
    echo -e "   - ACR_NAME"
    echo -e "   - RESOURCE_GROUP"
    echo -e "   - CLUSTER_NAME"
    echo -e ""
    echo -e "2. Run deployment:"
    echo -e "   chmod +x simple-deploy.sh"
    echo -e "   ./simple-deploy.sh"
    
else
    echo -e "${RED}❌ $MISSING_REQUIRED required files are missing${NC}"
    echo -e "${YELLOW}Please ensure all required files are present before deployment${NC}"
fi

echo -e "\n${BLUE}📁 Current Directory Structure:${NC}"
echo -e "$(tree -L 2 2>/dev/null || find . -type d -name .git -prune -o -type f -print | head -20)"