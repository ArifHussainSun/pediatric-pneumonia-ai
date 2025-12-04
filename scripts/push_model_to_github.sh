#!/bin/bash
# Push trained model and results to GitHub

set -e  # Exit on error

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Push Model to GitHub ===${NC}\n"

# Configuration
BRANCH="${1:-RunningModels4GPU}"
MODEL_DIR="${2:-outputs/clahe_augmented_finetune_v4}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_ROOT"

# Check if we're on the right branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "$BRANCH" ]; then
    echo -e "${YELLOW}Warning: Currently on branch '$CURRENT_BRANCH', expected '$BRANCH'${NC}"
    read -p "Switch to $BRANCH? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git checkout "$BRANCH"
    else
        echo "Aborted."
        exit 1
    fi
fi

# Check git status
echo -e "${BLUE}Current git status:${NC}"
git status

# Add files
echo -e "\n${BLUE}Adding model files...${NC}"
git add "$MODEL_DIR/" 2>/dev/null || echo "Model directory already tracked"
git add validation/reports/ 2>/dev/null || echo "Validation reports already tracked"
git add scripts/retrain_with_clahe_augmentation.py 2>/dev/null || echo "Training script already tracked"
git add scripts/validate_model.sh 2>/dev/null || echo "Validation script already tracked"
git add scripts/push_model_to_github.sh 2>/dev/null || echo "Push script already tracked"

# Show what will be committed
echo -e "\n${BLUE}Files to be committed:${NC}"
git status --short

# Confirm before committing
echo -e "\n${YELLOW}Ready to commit and push.${NC}"
read -p "Continue? (y/n) " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Create commit
echo -e "\n${BLUE}Creating commit...${NC}"

git commit -m "$(cat <<'EOF'
feat(model): add fine-tuned v4 model with 81% reduction in false negatives

Add CLAHE-augmented MobileNetV1 model that dramatically improves sensitivity
on blurry/low-quality chest X-rays. Model achieves 99.20% sensitivity compared
to baseline's 95.80%, reducing missed pneumonia cases from 21 to 4.

Training details:
- Fine-tuned from baseline with 40% CLAHE augmentation
- Batch size: 128, Learning rate: 0.0002
- Best validation accuracy: 97.01% at epoch 11
- Early stopping at epoch 18

Test results (1000 images):
- Accuracy: 96.10%
- Sensitivity: 99.20% (+3.4% vs baseline)
- Specificity: 93.00% (-6.4% vs baseline)
- False negatives: 4 (81% reduction from 21)
- False positives: 35 (trade-off for high sensitivity)

Model path: outputs/clahe_augmented_finetune_v4/best_model.pth
Validation report: outputs/clahe_augmented_finetune_v4/VALIDATION_SUMMARY.md

Includes:
- Trained model weights and checkpoints
- Validation summary and statistics
- Test results and reports
- Training and validation scripts
EOF
)" || echo -e "${YELLOW}No changes to commit${NC}"

# Push to GitHub
echo -e "\n${BLUE}Pushing to GitHub...${NC}"
git push origin "$BRANCH"

echo -e "\n${GREEN}=== Successfully pushed to GitHub! ===${NC}"
echo -e "Branch: $BRANCH"
echo -e "Model: $MODEL_DIR"
