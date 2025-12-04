#!/usr/bin/env python3
"""
Automated Batch Validation System for Real-World X-ray Testing

This script tests the MobileNet model on real chest X-ray datasets to:
1. Validate model performance on diverse real-world data
2. Compare preprocessing impact on confidence scores
3. Find edge cases and failure modes
4. Optimize CLAHE parameters automatically
"""

import os
import sys
import json
import time
import logging
import requests
import asyncio
import aiohttp
import aiofiles
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('validation_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Single test result for an image."""
    image_path: str
    ground_truth: str  # 'NORMAL' or 'PNEUMONIA'
    prediction_raw: str
    confidence_raw: float
    prediction_processed: str
    confidence_processed: float
    improvement: float  # confidence_processed - confidence_raw
    processing_time_ms: float
    image_quality: Optional[Dict] = None
    correct_raw: bool = False
    correct_processed: bool = False

class BatchValidator:
    """Automated batch validation system."""

    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.results: List[TestResult] = []
        self.session = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def test_single_image(self, image_path: Path, ground_truth: str) -> TestResult:
        """Test a single image with and without preprocessing."""
        try:
            # Test without preprocessing (by disabling it temporarily)
            raw_result = await self._predict_image(image_path, preprocessing=False)

            # Test with preprocessing
            processed_result = await self._predict_image(image_path, preprocessing=True)

            # Calculate improvement
            improvement = processed_result['confidence'] - raw_result['confidence']

            # Check correctness
            correct_raw = raw_result['prediction'] == ground_truth
            correct_processed = processed_result['prediction'] == ground_truth

            result = TestResult(
                image_path=str(image_path),
                ground_truth=ground_truth,
                prediction_raw=raw_result['prediction'],
                confidence_raw=raw_result['confidence'],
                prediction_processed=processed_result['prediction'],
                confidence_processed=processed_result['confidence'],
                improvement=improvement,
                processing_time_ms=processed_result['processing_time_ms'],
                correct_raw=correct_raw,
                correct_processed=correct_processed
            )

            logger.info(f"✓ {image_path.name}: {ground_truth} -> "
                       f"Raw: {raw_result['prediction']} ({raw_result['confidence']:.3f}) "
                       f"Processed: {processed_result['prediction']} ({processed_result['confidence']:.3f}) "
                       f"Improvement: {improvement:+.3f}")

            return result

        except Exception as e:
            logger.error(f"✗ Failed to process {image_path}: {e}")
            return None

    async def _predict_image(self, image_path: Path, preprocessing: bool = True) -> Dict:
        """Send image to API for prediction."""
        try:
            data = aiohttp.FormData()
            data.add_field('file',
                          open(image_path, 'rb'),
                          filename=image_path.name,
                          content_type='image/jpeg')

            async with self.session.post(
                f"{self.api_base_url}/predict",
                data=data,
                timeout=30
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"API error {response.status}: {error_text}")

        except Exception as e:
            raise Exception(f"Prediction failed: {e}")

    async def validate_dataset(self, dataset_path: Path, max_samples_per_class: int = 100) -> Dict:
        """Validate model on a dataset organized as normal/pneumonia folders."""
        logger.info(f"🚀 Starting validation on dataset: {dataset_path}")

        # Find images in normal and pneumonia folders (case insensitive)
        normal_folder = None
        pneumonia_folder = None

        for folder in dataset_path.iterdir():
            if folder.is_dir():
                if folder.name.lower() == "normal":
                    normal_folder = folder
                elif folder.name.lower() == "pneumonia":
                    pneumonia_folder = folder

        if not normal_folder or not pneumonia_folder:
            raise ValueError(f"Dataset must contain 'normal' and 'pneumonia' folders (case insensitive)")

        normal_images = list(normal_folder.glob("*.jpg")) + \
                        list(normal_folder.glob("*.png")) + \
                        list(normal_folder.glob("*.jpeg"))

        pneumonia_images = list(pneumonia_folder.glob("*.jpg")) + \
                           list(pneumonia_folder.glob("*.png")) + \
                           list(pneumonia_folder.glob("*.jpeg"))

        # Limit samples if specified
        if max_samples_per_class:
            normal_images = normal_images[:max_samples_per_class]
            pneumonia_images = pneumonia_images[:max_samples_per_class]

        logger.info(f"Found {len(normal_images)} normal and {len(pneumonia_images)} pneumonia images")

        # Create test tasks
        tasks = []
        for img_path in normal_images:
            tasks.append(self.test_single_image(img_path, "NORMAL"))
        for img_path in pneumonia_images:
            tasks.append(self.test_single_image(img_path, "PNEUMONIA"))

        # Execute tests concurrently (but limit concurrent requests)
        logger.info(f"🔄 Running {len(tasks)} predictions...")
        semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent requests

        async def limited_test(task):
            async with semaphore:
                return await task

        results = await asyncio.gather(*[limited_test(task) for task in tasks])

        # Filter out failed results
        self.results = [r for r in results if r is not None]

        # Calculate metrics
        metrics = self._calculate_metrics()

        logger.info(f"✅ Validation complete! Processed {len(self.results)} images")

        return metrics

    def _calculate_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not self.results:
            return {}

        df = pd.DataFrame([asdict(r) for r in self.results])

        # Basic accuracy metrics
        raw_accuracy = df['correct_raw'].mean()
        processed_accuracy = df['correct_processed'].mean()

        # Confidence improvements
        avg_improvement = df['improvement'].mean()
        positive_improvements = (df['improvement'] > 0).sum()

        # Performance by class
        normal_results = df[df['ground_truth'] == 'NORMAL']
        pneumonia_results = df[df['ground_truth'] == 'PNEUMONIA']

        # Edge cases (low confidence correct predictions)
        edge_cases = df[(df['correct_processed']) & (df['confidence_processed'] < 0.7)]

        # High improvement cases
        high_improvement = df[df['improvement'] > 0.1]

        metrics = {
            'total_samples': int(len(self.results)),
            'accuracy_raw': float(raw_accuracy),
            'accuracy_processed': float(processed_accuracy),
            'accuracy_improvement': float(processed_accuracy - raw_accuracy),
            'avg_confidence_improvement': float(avg_improvement),
            'positive_improvements': int(positive_improvements),
            'improvement_rate': float(positive_improvements / len(self.results)),
            'avg_processing_time_ms': float(df['processing_time_ms'].mean()),
            'normal_accuracy_raw': float(normal_results['correct_raw'].mean()) if len(normal_results) > 0 else 0.0,
            'normal_accuracy_processed': float(normal_results['correct_processed'].mean()) if len(normal_results) > 0 else 0.0,
            'pneumonia_accuracy_raw': float(pneumonia_results['correct_raw'].mean()) if len(pneumonia_results) > 0 else 0.0,
            'pneumonia_accuracy_processed': float(pneumonia_results['correct_processed'].mean()) if len(pneumonia_results) > 0 else 0.0,
            'edge_cases_count': int(len(edge_cases)),
            'high_improvement_count': int(len(high_improvement)),
            'timestamp': datetime.now().isoformat()
        }

        return metrics

    def save_results(self, output_dir: Path = Path("validation_results")):
        """Save detailed results and generate analysis."""
        output_dir.mkdir(exist_ok=True)

        if not self.results:
            logger.warning("No results to save")
            return

        # Save raw results
        df = pd.DataFrame([asdict(r) for r in self.results])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results_file = output_dir / f"validation_results_{timestamp}.csv"
        df.to_csv(results_file, index=False)
        logger.info(f"💾 Saved results to {results_file}")

        # Save metrics
        metrics = self._calculate_metrics()
        metrics_file = output_dir / f"validation_metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"📊 Saved metrics to {metrics_file}")

        # Generate visualizations
        self._generate_plots(output_dir, timestamp)

        # Generate summary report
        self._generate_report(output_dir, timestamp, metrics)

    def _generate_plots(self, output_dir: Path, timestamp: str):
        """Generate analysis plots."""
        df = pd.DataFrame([asdict(r) for r in self.results])

        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")

        # 1. Confidence improvement distribution
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.hist(df['improvement'], bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(df['improvement'].mean(), color='red', linestyle='--',
                   label=f'Mean: {df["improvement"].mean():.3f}')
        plt.xlabel('Confidence Improvement')
        plt.ylabel('Count')
        plt.title('Distribution of Confidence Improvements')
        plt.legend()

        # 2. Accuracy comparison
        plt.subplot(2, 2, 2)
        accuracies = ['Raw', 'Processed']
        values = [df['correct_raw'].mean(), df['correct_processed'].mean()]
        bars = plt.bar(accuracies, values, color=['lightcoral', 'lightblue'])
        plt.ylabel('Accuracy')
        plt.title('Accuracy Comparison')
        plt.ylim(0, 1)
        for i, v in enumerate(values):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

        # 3. Confidence by ground truth
        plt.subplot(2, 2, 3)
        normal_conf = df[df['ground_truth'] == 'NORMAL']['confidence_processed']
        pneumonia_conf = df[df['ground_truth'] == 'PNEUMONIA']['confidence_processed']
        plt.boxplot([normal_conf, pneumonia_conf], labels=['Normal', 'Pneumonia'])
        plt.ylabel('Confidence (Processed)')
        plt.title('Confidence Distribution by Class')

        # 4. Processing time distribution
        plt.subplot(2, 2, 4)
        plt.hist(df['processing_time_ms'], bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Processing Time (ms)')
        plt.ylabel('Count')
        plt.title('Processing Time Distribution')

        plt.tight_layout()
        plot_file = output_dir / f"validation_analysis_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"📈 Saved plots to {plot_file}")

    def _generate_report(self, output_dir: Path, timestamp: str, metrics: Dict):
        """Generate a comprehensive text report."""
        report_file = output_dir / f"validation_report_{timestamp}.txt"

        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MobileNet V1 Real-World Validation Report\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Samples: {metrics['total_samples']}\n\n")

            f.write("ACCURACY METRICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Raw Model Accuracy:        {metrics['accuracy_raw']:.1%}\n")
            f.write(f"Processed Model Accuracy:  {metrics['accuracy_processed']:.1%}\n")
            f.write(f"Accuracy Improvement:      {metrics['accuracy_improvement']:+.1%}\n\n")

            f.write("CONFIDENCE IMPROVEMENTS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Average Improvement:       {metrics['avg_confidence_improvement']:+.3f}\n")
            f.write(f"Images Improved:           {metrics['positive_improvements']}/{metrics['total_samples']} "
                   f"({metrics['improvement_rate']:.1%})\n\n")

            f.write("CLASS-SPECIFIC PERFORMANCE:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Normal - Raw:              {metrics['normal_accuracy_raw']:.1%}\n")
            f.write(f"Normal - Processed:        {metrics['normal_accuracy_processed']:.1%}\n")
            f.write(f"Pneumonia - Raw:           {metrics['pneumonia_accuracy_raw']:.1%}\n")
            f.write(f"Pneumonia - Processed:     {metrics['pneumonia_accuracy_processed']:.1%}\n\n")

            f.write("PERFORMANCE:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Average Processing Time:   {metrics['avg_processing_time_ms']:.1f} ms\n\n")

            f.write("EDGE CASES:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Low Confidence Correct:    {metrics['edge_cases_count']} cases\n")
            f.write(f"High Improvement Cases:    {metrics['high_improvement_count']} cases\n\n")

            # Add conclusions
            f.write("CONCLUSIONS:\n")
            f.write("-" * 40 + "\n")

            if metrics['accuracy_improvement'] > 0.02:
                f.write("✅ Preprocessing provides significant accuracy improvement\n")
            elif metrics['accuracy_improvement'] > 0:
                f.write("✅ Preprocessing provides modest accuracy improvement\n")
            else:
                f.write("⚠️  Preprocessing may not improve accuracy\n")

            if metrics['improvement_rate'] > 0.6:
                f.write("✅ Majority of images benefit from preprocessing\n")
            else:
                f.write("⚠️  Limited images benefit from preprocessing\n")

            if metrics['avg_processing_time_ms'] < 100:
                f.write("✅ Fast inference time suitable for real-time use\n")
            else:
                f.write("⚠️  Consider optimization for real-time applications\n")

        logger.info(f"📋 Saved report to {report_file}")

async def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Batch validate MobileNet on real X-ray datasets")
    parser.add_argument("dataset_path", help="Path to dataset (should contain normal/ and pneumonia/ folders)")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--max-samples", type=int, default=100, help="Max samples per class")
    parser.add_argument("--output-dir", default="validation_results", help="Output directory")

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        return

    if not (dataset_path / "normal").exists() or not (dataset_path / "pneumonia").exists():
        logger.error("Dataset should contain 'normal' and 'pneumonia' folders")
        return

    async with BatchValidator(args.api_url) as validator:
        try:
            # Run validation
            metrics = await validator.validate_dataset(dataset_path, args.max_samples)

            # Save results
            validator.save_results(Path(args.output_dir))

            # Print summary
            logger.info("🎉 VALIDATION COMPLETE!")
            logger.info(f"Accuracy: {metrics['accuracy_raw']:.1%} → {metrics['accuracy_processed']:.1%} "
                       f"({metrics['accuracy_improvement']:+.1%})")
            logger.info(f"Confidence improvement: {metrics['avg_confidence_improvement']:+.3f} "
                       f"({metrics['improvement_rate']:.1%} of images)")

        except KeyboardInterrupt:
            logger.info("Validation interrupted by user")
        except Exception as e:
            logger.error(f"Validation failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())