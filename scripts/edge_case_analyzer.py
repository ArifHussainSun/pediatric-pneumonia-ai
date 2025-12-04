#!/usr/bin/env python3
"""
Edge Case Detection and Failure Analysis for MobileNet V1

This script identifies problematic cases where the model:
1. Has low confidence but correct predictions
2. Has high confidence but incorrect predictions
3. Shows inconsistent behavior with similar images
4. Fails to improve with preprocessing
"""

import os
import sys
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from scripts.batch_validation import BatchValidator, TestResult
from src.preprocessing import assess_image_quality

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EdgeCase:
    """Represents an edge case or failure mode."""
    case_type: str
    image_path: str
    ground_truth: str
    prediction: str
    confidence: float
    description: str
    severity: str  # 'low', 'medium', 'high'
    image_quality: Optional[Dict] = None
    preprocessing_helped: bool = False

class EdgeCaseAnalyzer:
    """Analyzes model failures and edge cases."""

    def __init__(self):
        self.edge_cases: List[EdgeCase] = []
        self.test_results: List[TestResult] = []

    def analyze_results(self, results: List[TestResult]) -> List[EdgeCase]:
        """Analyze test results to identify edge cases."""
        self.test_results = results
        self.edge_cases = []

        logger.info(f"🔍 Analyzing {len(results)} test results for edge cases...")

        # Different types of edge case detection
        self._detect_low_confidence_correct()
        self._detect_high_confidence_incorrect()
        self._detect_preprocessing_failures()
        self._detect_overconfident_wrong()
        self._detect_borderline_cases()

        logger.info(f"Found {len(self.edge_cases)} edge cases")
        return self.edge_cases

    def _detect_low_confidence_correct(self):
        """Find cases where model is correct but has low confidence."""
        for result in self.test_results:
            if result.correct_processed and result.confidence_processed < 0.6:
                edge_case = EdgeCase(
                    case_type="low_confidence_correct",
                    image_path=result.image_path,
                    ground_truth=result.ground_truth,
                    prediction=result.prediction_processed,
                    confidence=result.confidence_processed,
                    description=f"Correct prediction but low confidence ({result.confidence_processed:.3f})",
                    severity="medium" if result.confidence_processed < 0.5 else "low",
                    preprocessing_helped=result.improvement > 0
                )
                self.edge_cases.append(edge_case)

    def _detect_high_confidence_incorrect(self):
        """Find cases where model is wrong but has high confidence."""
        for result in self.test_results:
            if not result.correct_processed and result.confidence_processed > 0.7:
                edge_case = EdgeCase(
                    case_type="high_confidence_incorrect",
                    image_path=result.image_path,
                    ground_truth=result.ground_truth,
                    prediction=result.prediction_processed,
                    confidence=result.confidence_processed,
                    description=f"Incorrect prediction with high confidence ({result.confidence_processed:.3f})",
                    severity="high",
                    preprocessing_helped=result.improvement > 0
                )
                self.edge_cases.append(edge_case)

    def _detect_preprocessing_failures(self):
        """Find cases where preprocessing made things worse."""
        for result in self.test_results:
            if result.improvement < -0.1:  # Significant confidence drop
                edge_case = EdgeCase(
                    case_type="preprocessing_failure",
                    image_path=result.image_path,
                    ground_truth=result.ground_truth,
                    prediction=result.prediction_processed,
                    confidence=result.confidence_processed,
                    description=f"Preprocessing reduced confidence by {-result.improvement:.3f}",
                    severity="medium",
                    preprocessing_helped=False
                )
                self.edge_cases.append(edge_case)

    def _detect_overconfident_wrong(self):
        """Find cases with overconfidence (>0.9) that are wrong."""
        for result in self.test_results:
            if not result.correct_processed and result.confidence_processed > 0.9:
                edge_case = EdgeCase(
                    case_type="overconfident_wrong",
                    image_path=result.image_path,
                    ground_truth=result.ground_truth,
                    prediction=result.prediction_processed,
                    confidence=result.confidence_processed,
                    description=f"Overconfident wrong prediction ({result.confidence_processed:.3f})",
                    severity="high",
                    preprocessing_helped=result.improvement > 0
                )
                self.edge_cases.append(edge_case)

    def _detect_borderline_cases(self):
        """Find borderline cases around 0.5 confidence."""
        for result in self.test_results:
            if 0.45 <= result.confidence_processed <= 0.55:
                edge_case = EdgeCase(
                    case_type="borderline_case",
                    image_path=result.image_path,
                    ground_truth=result.ground_truth,
                    prediction=result.prediction_processed,
                    confidence=result.confidence_processed,
                    description=f"Borderline confidence ({result.confidence_processed:.3f})",
                    severity="low",
                    preprocessing_helped=result.improvement > 0
                )
                self.edge_cases.append(edge_case)

    async def analyze_image_characteristics(self, max_cases: int = 20):
        """Analyze image characteristics of edge cases."""
        logger.info("🖼️  Analyzing image characteristics of edge cases...")

        # Limit analysis to most severe cases
        high_severity_cases = [ec for ec in self.edge_cases if ec.severity == "high"]
        cases_to_analyze = high_severity_cases[:max_cases]

        for edge_case in cases_to_analyze:
            try:
                # Load and analyze image
                image_path = Path(edge_case.image_path)
                if not image_path.exists():
                    continue

                # Get image quality assessment
                with Image.open(image_path) as img:
                    quality_assessment = assess_image_quality(img)
                    edge_case.image_quality = quality_assessment

                logger.info(f"  📊 {image_path.name}: {edge_case.case_type} - "
                           f"Quality: {quality_assessment['overall_quality']}")

            except Exception as e:
                logger.warning(f"  ⚠️  Failed to analyze {edge_case.image_path}: {e}")

    def generate_failure_analysis(self, output_dir: Path = Path("edge_case_analysis")):
        """Generate comprehensive failure analysis report."""
        output_dir.mkdir(exist_ok=True)

        if not self.edge_cases:
            logger.warning("No edge cases to analyze")
            return

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save edge cases data
        edge_cases_data = [asdict(ec) for ec in self.edge_cases]
        df = pd.DataFrame(edge_cases_data)

        results_file = output_dir / f"edge_cases_{timestamp}.csv"
        df.to_csv(results_file, index=False)
        logger.info(f"💾 Saved edge cases to {results_file}")

        # Generate analysis plots
        self._generate_edge_case_plots(output_dir, timestamp)

        # Generate detailed report
        self._generate_failure_report(output_dir, timestamp)

        # Generate recommendations
        self._generate_improvement_recommendations(output_dir, timestamp)

    def _generate_edge_case_plots(self, output_dir: Path, timestamp: str):
        """Generate visualization plots for edge cases."""
        if not self.edge_cases:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Edge case distribution by type
        case_types = [ec.case_type for ec in self.edge_cases]
        type_counts = pd.Series(case_types).value_counts()

        axes[0,0].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
        axes[0,0].set_title('Distribution of Edge Case Types')

        # 2. Severity distribution
        severities = [ec.severity for ec in self.edge_cases]
        severity_counts = pd.Series(severities).value_counts()

        colors = {'low': 'green', 'medium': 'orange', 'high': 'red'}
        severity_colors = [colors.get(sev, 'gray') for sev in severity_counts.index]

        axes[0,1].bar(severity_counts.index, severity_counts.values, color=severity_colors)
        axes[0,1].set_title('Edge Cases by Severity')
        axes[0,1].set_ylabel('Count')

        # 3. Confidence distribution for edge cases
        confidences = [ec.confidence for ec in self.edge_cases]
        axes[0,2].hist(confidences, bins=20, alpha=0.7, edgecolor='black')
        axes[0,2].set_title('Confidence Distribution in Edge Cases')
        axes[0,2].set_xlabel('Confidence')
        axes[0,2].set_ylabel('Count')

        # 4. Edge cases by ground truth class
        ground_truths = [ec.ground_truth for ec in self.edge_cases]
        gt_counts = pd.Series(ground_truths).value_counts()

        axes[1,0].bar(gt_counts.index, gt_counts.values, color=['lightblue', 'lightcoral'])
        axes[1,0].set_title('Edge Cases by True Class')
        axes[1,0].set_ylabel('Count')

        # 5. Preprocessing help rate by case type
        df_cases = pd.DataFrame([asdict(ec) for ec in self.edge_cases])
        help_rate = df_cases.groupby('case_type')['preprocessing_helped'].mean()

        axes[1,1].bar(range(len(help_rate)), help_rate.values)
        axes[1,1].set_xticks(range(len(help_rate)))
        axes[1,1].set_xticklabels(help_rate.index, rotation=45, ha='right')
        axes[1,1].set_title('Preprocessing Help Rate by Case Type')
        axes[1,1].set_ylabel('Fraction Helped')

        # 6. Confidence vs Case Type
        case_type_conf = df_cases.boxplot(column='confidence', by='case_type', ax=axes[1,2])
        axes[1,2].set_title('Confidence Distribution by Case Type')
        axes[1,2].set_xlabel('Case Type')
        axes[1,2].set_ylabel('Confidence')

        plt.tight_layout()

        plot_file = output_dir / f"edge_case_analysis_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"📊 Saved edge case plots to {plot_file}")

    def _generate_failure_report(self, output_dir: Path, timestamp: str):
        """Generate detailed failure analysis report."""
        from datetime import datetime

        report_file = output_dir / f"failure_analysis_report_{timestamp}.txt"

        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MobileNet V1 Failure Analysis Report\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Edge Cases: {len(self.edge_cases)}\n")
            f.write(f"Total Test Results: {len(self.test_results)}\n")
            f.write(f"Edge Case Rate: {len(self.edge_cases)/len(self.test_results):.1%}\n\n")

            # Summary by case type
            case_type_counts = {}
            for ec in self.edge_cases:
                case_type_counts[ec.case_type] = case_type_counts.get(ec.case_type, 0) + 1

            f.write("EDGE CASE BREAKDOWN:\n")
            f.write("-" * 40 + "\n")
            for case_type, count in sorted(case_type_counts.items()):
                percentage = count / len(self.edge_cases) * 100
                f.write(f"{case_type.replace('_', ' ').title()}: {count} ({percentage:.1f}%)\n")
            f.write("\n")

            # Severity analysis
            severity_counts = {'high': 0, 'medium': 0, 'low': 0}
            for ec in self.edge_cases:
                severity_counts[ec.severity] += 1

            f.write("SEVERITY BREAKDOWN:\n")
            f.write("-" * 40 + "\n")
            f.write(f"High Severity:   {severity_counts['high']} cases\n")
            f.write(f"Medium Severity: {severity_counts['medium']} cases\n")
            f.write(f"Low Severity:    {severity_counts['low']} cases\n\n")

            # Most problematic cases
            high_severity_cases = [ec for ec in self.edge_cases if ec.severity == "high"]
            if high_severity_cases:
                f.write("MOST PROBLEMATIC CASES:\n")
                f.write("-" * 40 + "\n")
                for i, ec in enumerate(high_severity_cases[:10], 1):
                    f.write(f"{i}. {Path(ec.image_path).name}\n")
                    f.write(f"   Type: {ec.case_type}\n")
                    f.write(f"   Ground Truth: {ec.ground_truth} -> Predicted: {ec.prediction}\n")
                    f.write(f"   Confidence: {ec.confidence:.3f}\n")
                    f.write(f"   Description: {ec.description}\n\n")

            # Preprocessing effectiveness
            helped_count = sum(1 for ec in self.edge_cases if ec.preprocessing_helped)
            f.write("PREPROCESSING ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Cases where preprocessing helped: {helped_count}/{len(self.edge_cases)} "
                   f"({helped_count/len(self.edge_cases):.1%})\n\n")

            # Image quality analysis (if available)
            quality_cases = [ec for ec in self.edge_cases if ec.image_quality]
            if quality_cases:
                f.write("IMAGE QUALITY ANALYSIS:\n")
                f.write("-" * 40 + "\n")

                quality_distribution = {}
                for ec in quality_cases:
                    quality = ec.image_quality['overall_quality']
                    quality_distribution[quality] = quality_distribution.get(quality, 0) + 1

                for quality, count in sorted(quality_distribution.items()):
                    f.write(f"{quality.capitalize()}: {count} cases\n")
                f.write("\n")

        logger.info(f"📋 Saved failure analysis report to {report_file}")

    def _generate_improvement_recommendations(self, output_dir: Path, timestamp: str):
        """Generate specific recommendations for improvement."""
        recommendations_file = output_dir / f"improvement_recommendations_{timestamp}.txt"

        with open(recommendations_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Model Improvement Recommendations\n")
            f.write("=" * 80 + "\n\n")

            # Analyze patterns in edge cases
            high_conf_wrong = [ec for ec in self.edge_cases if ec.case_type == "high_confidence_incorrect"]
            low_conf_correct = [ec for ec in self.edge_cases if ec.case_type == "low_confidence_correct"]
            preprocessing_failures = [ec for ec in self.edge_cases if ec.case_type == "preprocessing_failure"]

            f.write("IMMEDIATE ACTIONS:\n")
            f.write("-" * 40 + "\n")

            if len(high_conf_wrong) > 5:
                f.write("🚨 HIGH PRIORITY: Model Overconfidence Issue\n")
                f.write(f"   Found {len(high_conf_wrong)} cases of high-confidence wrong predictions\n")
                f.write("   Recommendations:\n")
                f.write("   - Consider temperature scaling or confidence calibration\n")
                f.write("   - Review training data for similar challenging cases\n")
                f.write("   - Add uncertainty estimation to the model\n\n")

            if len(low_conf_correct) > 10:
                f.write("⚠️  MEDIUM PRIORITY: Low Confidence on Correct Predictions\n")
                f.write(f"   Found {len(low_conf_correct)} cases of low-confidence correct predictions\n")
                f.write("   Recommendations:\n")
                f.write("   - Improve preprocessing for these image types\n")
                f.write("   - Consider data augmentation for similar cases\n")
                f.write("   - Review model architecture for feature extraction\n\n")

            if len(preprocessing_failures) > 5:
                f.write("🔧 PREPROCESSING ISSUES:\n")
                f.write(f"   Found {len(preprocessing_failures)} cases where preprocessing hurt performance\n")
                f.write("   Recommendations:\n")
                f.write("   - Fine-tune CLAHE parameters for these specific cases\n")
                f.write("   - Consider adaptive preprocessing based on image quality\n")
                f.write("   - Add preprocessing quality checks\n\n")

            f.write("LONG-TERM IMPROVEMENTS:\n")
            f.write("-" * 40 + "\n")
            f.write("1. Data Collection:\n")
            f.write("   - Collect more examples similar to high-severity edge cases\n")
            f.write("   - Focus on underrepresented image quality types\n\n")

            f.write("2. Model Architecture:\n")
            f.write("   - Consider attention mechanisms for better feature focus\n")
            f.write("   - Evaluate ensemble methods for improved reliability\n\n")

            f.write("3. Training Improvements:\n")
            f.write("   - Use hard negative mining for difficult cases\n")
            f.write("   - Implement focal loss for handling class imbalance\n")
            f.write("   - Add regularization for overconfidence\n\n")

            f.write("4. Preprocessing Enhancements:\n")
            f.write("   - Implement adaptive CLAHE based on image characteristics\n")
            f.write("   - Add more sophisticated quality assessment\n")
            f.write("   - Consider multi-scale preprocessing approaches\n\n")

        logger.info(f"💡 Saved improvement recommendations to {recommendations_file}")

async def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze edge cases and failures in model predictions")
    parser.add_argument("dataset_path", help="Path to dataset for analysis")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--max-samples", type=int, default=100, help="Max samples per class")
    parser.add_argument("--output-dir", default="edge_case_analysis", help="Output directory")

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        return

    # First run batch validation to get test results
    logger.info("Running batch validation to collect test results...")
    async with BatchValidator(args.api_url) as validator:
        await validator.validate_dataset(dataset_path, args.max_samples)
        test_results = validator.results

    if not test_results:
        logger.error("No test results available for analysis")
        return

    # Analyze edge cases
    analyzer = EdgeCaseAnalyzer()
    edge_cases = analyzer.analyze_results(test_results)

    # Analyze image characteristics
    await analyzer.analyze_image_characteristics()

    # Generate comprehensive analysis
    analyzer.generate_failure_analysis(Path(args.output_dir))

    logger.info(f"🎉 Edge case analysis complete!")
    logger.info(f"Found {len(edge_cases)} edge cases out of {len(test_results)} total tests")

if __name__ == "__main__":
    asyncio.run(main())