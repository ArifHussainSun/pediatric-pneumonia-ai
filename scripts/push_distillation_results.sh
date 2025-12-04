#!/bin/bash
# Push knowledge distillation (ResNet50 teacher) results to GitHub
# Run this on DGX: bash scripts/push_distillation_results.sh

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}=== Push Distillation Results to GitHub ===${NC}\n"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Step 1: Check current branch
echo -e "${BLUE}Step 1: Checking git status...${NC}"
git status

echo -e "\n${BLUE}Step 2: Adding distillation model files...${NC}"

# Force add model files (override .gitignore)
if [ -d "outputs/mobilenet_v5_distilled_resnet50" ]; then
    echo "Adding model files from outputs/mobilenet_v5_distilled_resnet50/"
    git add -f outputs/mobilenet_v5_distilled_resnet50/best_model.pth 2>/dev/null || echo "  best_model.pth not found"
    git add -f outputs/mobilenet_v5_distilled_resnet50/final_model.pth 2>/dev/null || echo "  final_model.pth not found"

    # Add any text files (logs, stats, etc.)
    git add outputs/mobilenet_v5_distilled_resnet50/*.txt 2>/dev/null || echo "  No .txt files found"
    git add outputs/mobilenet_v5_distilled_resnet50/*.json 2>/dev/null || echo "  No .json files found"

    echo -e "${GREEN}✓${NC} Distillation model files added"
else
    echo -e "${YELLOW}Warning: outputs/mobilenet_v5_distilled_resnet50/ not found${NC}"
fi

# Step 3: Add validation report
echo -e "\n${BLUE}Step 3: Adding validation report...${NC}"

if [ -d "validation/reports/mobilenet_v5_distilled_resnet50_test_results" ]; then
    git add validation/reports/mobilenet_v5_distilled_resnet50_test_results/
    echo -e "${GREEN}✓${NC} Validation report added"
else
    echo -e "${YELLOW}Warning: validation report not found${NC}"
fi

# Step 4: Show what will be committed
echo -e "\n${BLUE}Step 4: Review changes${NC}"
git status

# Step 5: Create commit
echo -e "\n${BLUE}Step 5: Creating commit...${NC}"

git commit -m "$(cat <<'EOF'
feat(distillation): add MobileNet v5 distilled with ResNet50 teacher

Trained MobileNetV1 v5 using knowledge distillation with ResNet50
teacher to further improve v4 model performance.

Training Configuration:
- Student: MobileNetV1 v4 (CLAHE-augmented)
- Teacher: ResNet50 (ImageNet pretrained)
- Distillation: Temperature 3.0, alpha 0.5
- Training: 29 epochs, early stopping at epoch 22
- Best validation accuracy: 95.77%

Test Results (1000 images):
- Overall Accuracy: 95.70%
- Sensitivity: 99.80% (+0.60% vs v4)
- Specificity: 91.60%
- False Negatives: 1 (75% reduction from v4's 4 FN)
- False Positives: 42

Key Achievement:
Reduced false negatives from 4 to 1, achieving 99.80% sensitivity.
Only 1 missed pneumonia case out of 500 test samples.

Comparison to V4:
- FN reduction: 4 → 1 (-75%)
- Sensitivity improvement: 99.20% → 99.80%
- Trade-off: 7 more false positives (35 → 42)

False Negative Case:
- person54_bacteria_258.jpeg (also missed by v4)

Successfully identified 3 cases v4 missed:
- person485_bacteria_2049.jpeg
- person1481_virus_2567.jpeg
- person1676_virus_2892.jpeg

Model files: outputs/mobilenet_v5_distilled_resnet50/
Test report: validation/reports/mobilenet_v5_distilled_resnet50_test_results/
EOF
)" || echo -e "${YELLOW}No changes to commit or commit failed${NC}"

# Step 6: Push to GitHub
echo -e "\n${BLUE}Step 6: Pushing to GitHub...${NC}"
read -p "Push to main branch? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git push origin main
    echo -e "\n${GREEN}=== Successfully pushed distillation results to GitHub! ===${NC}"
    echo -e "${GREEN}Model: outputs/mobilenet_v5_distilled_resnet50/best_model.pth${NC}"
    echo -e "${GREEN}Results: 95.70% accuracy, 99.80% sensitivity, 1 FN${NC}"
else
    echo -e "${YELLOW}Aborted. Changes committed locally but not pushed.${NC}"
fi
