#!/usr/bin/env python3
"""
Automated CLAHE Parameter Optimization for MobileNet V1

This script automatically finds the optimal CLAHE parameters for your specific model
by testing different combinations on a validation dataset and measuring:
1. Accuracy improvement
2. Confidence score improvements
3. Processing speed trade-offs
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from scripts.batch_validation import BatchValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CLAHEConfig:
    """CLAHE configuration parameters."""
    clip_limit: float
    tile_grid_size: Tuple[int, int]

    def __str__(self):
        return f"clip_{self.clip_limit}_grid_{self.tile_grid_size[0]}x{self.tile_grid_size[1]}"

@dataclass
class OptimizationResult:
    """Result of optimization for specific CLAHE parameters."""
    config: CLAHEConfig
    accuracy_improvement: float
    avg_confidence_improvement: float
    improvement_rate: float
    avg_processing_time: float
    total_samples: int

class CLAHEOptimizer:
    """Automated CLAHE parameter optimizer."""

    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.results: List[OptimizationResult] = []

    def generate_parameter_grid(self) -> List[CLAHEConfig]:
        """Generate grid of CLAHE parameters to test."""
        # Based on medical imaging literature and our preprocessing module
        clip_limits = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        tile_grids = [(6, 6), (8, 8), (10, 10), (12, 12)]

        configs = []
        for clip, grid in product(clip_limits, tile_grids):
            configs.append(CLAHEConfig(clip_limit=clip, tile_grid_size=grid))

        logger.info(f"Generated {len(configs)} parameter combinations to test")
        return configs

    async def optimize_parameters(self,
                                 dataset_path: Path,
                                 max_samples_per_class: int = 50,
                                 output_dir: Path = Path("clahe_optimization")) -> CLAHEConfig:
        """
        Find optimal CLAHE parameters for the given dataset.

        Args:
            dataset_path: Path to validation dataset (normal/pneumonia folders)
            max_samples_per_class: Samples to use for optimization
            output_dir: Directory to save results

        Returns:
            Best CLAHE configuration
        """
        output_dir.mkdir(exist_ok=True)

        # Generate parameter combinations
        configs = self.generate_parameter_grid()

        logger.info(f"🚀 Starting CLAHE optimization with {len(configs)} configurations")
        logger.info(f"Dataset: {dataset_path}")
        logger.info(f"Samples per class: {max_samples_per_class}")

        # Test each configuration
        for i, config in enumerate(configs, 1):
            logger.info(f"[{i}/{len(configs)}] Testing {config}")

            try:
                # Temporarily update preprocessing config for this test
                await self._update_preprocessing_config(config)

                # Run validation with this configuration
                async with BatchValidator(self.api_base_url) as validator:
                    metrics = await validator.validate_dataset(
                        dataset_path, max_samples_per_class
                    )

                    # Store result
                    result = OptimizationResult(
                        config=config,
                        accuracy_improvement=metrics.get('accuracy_improvement', 0),
                        avg_confidence_improvement=metrics.get('avg_confidence_improvement', 0),
                        improvement_rate=metrics.get('improvement_rate', 0),
                        avg_processing_time=metrics.get('avg_processing_time_ms', 0),
                        total_samples=metrics.get('total_samples', 0)
                    )

                    self.results.append(result)

                    logger.info(f"  ✓ Accuracy improvement: {result.accuracy_improvement:+.1%}")
                    logger.info(f"  ✓ Confidence improvement: {result.avg_confidence_improvement:+.3f}")
                    logger.info(f"  ✓ Processing time: {result.avg_processing_time:.1f}ms")

            except Exception as e:
                logger.error(f"  ✗ Failed to test {config}: {e}")
                continue

        # Find best configuration
        best_config = self._find_best_configuration()

        # Save results
        self._save_optimization_results(output_dir)

        # Generate analysis
        self._generate_optimization_analysis(output_dir)

        logger.info(f"🎉 Optimization complete! Best config: {best_config}")

        return best_config

    async def _update_preprocessing_config(self, config: CLAHEConfig):
        """Update the preprocessing configuration for testing."""
        # This would update the config file or send API request to change parameters
        # For now, we'll assume the API has an endpoint to update CLAHE params
        # In practice, you might need to modify the preprocessing config file

        config_data = {
            "clahe_enhancement": {
                "clip_limit": config.clip_limit,
                "tile_grid_size": list(config.tile_grid_size)
            }
        }

        # Save to preprocessing config
        config_file = Path("config/preprocessing_config.json")
        if config_file.exists():
            with open(config_file, 'r') as f:
                full_config = json.load(f)

            full_config["preprocessing"]["clahe_enhancement"]["clip_limit"] = config.clip_limit
            full_config["preprocessing"]["clahe_enhancement"]["tile_grid_size"] = list(config.tile_grid_size)

            with open(config_file, 'w') as f:
                json.dump(full_config, f, indent=2)

    def _find_best_configuration(self) -> CLAHEConfig:
        """Find the best CLAHE configuration based on multiple criteria."""
        if not self.results:
            raise ValueError("No optimization results available")

        # Create scoring function that balances accuracy and confidence improvements
        # while penalizing excessive processing time
        def score_config(result: OptimizationResult) -> float:
            # Normalize metrics
            accuracy_score = result.accuracy_improvement * 100  # -10 to +10 range
            confidence_score = result.avg_confidence_improvement * 100  # -10 to +10 range
            improvement_rate_score = result.improvement_rate * 10  # 0 to 10 range

            # Penalize slow processing (anything over 200ms gets penalty)
            time_penalty = max(0, (result.avg_processing_time - 200) / 100)

            # Weighted combination
            total_score = (
                0.4 * accuracy_score +           # 40% weight on accuracy
                0.3 * confidence_score +         # 30% weight on confidence
                0.2 * improvement_rate_score +   # 20% weight on improvement rate
                0.1 * (-time_penalty)            # 10% penalty for slow processing
            )

            return total_score

        # Score all configurations
        scored_results = [(score_config(r), r) for r in self.results]
        scored_results.sort(key=lambda x: x[0], reverse=True)

        # Log top 5 configurations
        logger.info("🏆 Top 5 CLAHE configurations:")
        for i, (score, result) in enumerate(scored_results[:5], 1):
            logger.info(f"  {i}. {result.config} (score: {score:.2f})")
            logger.info(f"     Accuracy: {result.accuracy_improvement:+.1%}, "
                       f"Confidence: {result.avg_confidence_improvement:+.3f}, "
                       f"Time: {result.avg_processing_time:.1f}ms")

        return scored_results[0][1].config

    def _save_optimization_results(self, output_dir: Path):
        """Save detailed optimization results."""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save results as CSV
        results_data = []
        for result in self.results:
            results_data.append({
                'clip_limit': result.config.clip_limit,
                'tile_grid_x': result.config.tile_grid_size[0],
                'tile_grid_y': result.config.tile_grid_size[1],
                'accuracy_improvement': result.accuracy_improvement,
                'avg_confidence_improvement': result.avg_confidence_improvement,
                'improvement_rate': result.improvement_rate,
                'avg_processing_time': result.avg_processing_time,
                'total_samples': result.total_samples
            })

        df = pd.DataFrame(results_data)
        results_file = output_dir / f"clahe_optimization_results_{timestamp}.csv"
        df.to_csv(results_file, index=False)

        logger.info(f"💾 Saved optimization results to {results_file}")

    def _generate_optimization_analysis(self, output_dir: Path):
        """Generate comprehensive analysis plots."""
        from datetime import datetime

        if not self.results:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Prepare data for plotting
        clip_limits = [r.config.clip_limit for r in self.results]
        grid_sizes = [r.config.tile_grid_size[0] for r in self.results]  # Assuming square grids
        accuracy_improvements = [r.accuracy_improvement for r in self.results]
        confidence_improvements = [r.avg_confidence_improvement for r in self.results]
        processing_times = [r.avg_processing_time for r in self.results]

        # Create subplot figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Accuracy improvement heatmap
        df_acc = pd.DataFrame({
            'clip_limit': clip_limits,
            'grid_size': grid_sizes,
            'accuracy_improvement': accuracy_improvements
        })
        pivot_acc = df_acc.pivot(index='grid_size', columns='clip_limit', values='accuracy_improvement')
        sns.heatmap(pivot_acc, annot=True, fmt='.3f', cmap='RdYlGn', center=0, ax=axes[0,0])
        axes[0,0].set_title('Accuracy Improvement by CLAHE Parameters')

        # 2. Confidence improvement heatmap
        df_conf = pd.DataFrame({
            'clip_limit': clip_limits,
            'grid_size': grid_sizes,
            'confidence_improvement': confidence_improvements
        })
        pivot_conf = df_conf.pivot(index='grid_size', columns='clip_limit', values='confidence_improvement')
        sns.heatmap(pivot_conf, annot=True, fmt='.3f', cmap='RdYlGn', center=0, ax=axes[0,1])
        axes[0,1].set_title('Confidence Improvement by CLAHE Parameters')

        # 3. Processing time heatmap
        df_time = pd.DataFrame({
            'clip_limit': clip_limits,
            'grid_size': grid_sizes,
            'processing_time': processing_times
        })
        pivot_time = df_time.pivot(index='grid_size', columns='clip_limit', values='processing_time')
        sns.heatmap(pivot_time, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[0,2])
        axes[0,2].set_title('Processing Time (ms) by CLAHE Parameters')

        # 4. Clip limit effect
        clip_groups = pd.DataFrame({'clip_limit': clip_limits, 'accuracy': accuracy_improvements}).groupby('clip_limit').mean()
        axes[1,0].plot(clip_groups.index, clip_groups['accuracy'], 'o-')
        axes[1,0].set_xlabel('Clip Limit')
        axes[1,0].set_ylabel('Average Accuracy Improvement')
        axes[1,0].set_title('Effect of Clip Limit on Accuracy')
        axes[1,0].grid(True, alpha=0.3)

        # 5. Grid size effect
        grid_groups = pd.DataFrame({'grid_size': grid_sizes, 'accuracy': accuracy_improvements}).groupby('grid_size').mean()
        axes[1,1].plot(grid_groups.index, grid_groups['accuracy'], 's-')
        axes[1,1].set_xlabel('Tile Grid Size')
        axes[1,1].set_ylabel('Average Accuracy Improvement')
        axes[1,1].set_title('Effect of Grid Size on Accuracy')
        axes[1,1].grid(True, alpha=0.3)

        # 6. Pareto plot: accuracy vs processing time
        axes[1,2].scatter(processing_times, accuracy_improvements, c=confidence_improvements,
                         cmap='viridis', s=50, alpha=0.7)
        axes[1,2].set_xlabel('Processing Time (ms)')
        axes[1,2].set_ylabel('Accuracy Improvement')
        axes[1,2].set_title('Accuracy vs Processing Time\n(Color = Confidence Improvement)')

        # Add colorbar for the scatter plot
        scatter = axes[1,2].collections[0]
        plt.colorbar(scatter, ax=axes[1,2], label='Confidence Improvement')

        plt.tight_layout()

        # Save plot
        plot_file = output_dir / f"clahe_optimization_analysis_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"📊 Saved optimization analysis to {plot_file}")

        # Generate summary report
        self._generate_optimization_report(output_dir, timestamp)

    def _generate_optimization_report(self, output_dir: Path, timestamp: str):
        """Generate optimization summary report."""
        from datetime import datetime

        best_config = self._find_best_configuration()
        best_result = next(r for r in self.results if r.config == best_config)

        report_file = output_dir / f"clahe_optimization_report_{timestamp}.txt"

        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CLAHE Parameter Optimization Report\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configurations tested: {len(self.results)}\n\n")

            f.write("OPTIMAL CONFIGURATION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Clip Limit:        {best_config.clip_limit}\n")
            f.write(f"Tile Grid Size:    {best_config.tile_grid_size}\n\n")

            f.write("PERFORMANCE WITH OPTIMAL CONFIG:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Accuracy Improvement:     {best_result.accuracy_improvement:+.1%}\n")
            f.write(f"Confidence Improvement:   {best_result.avg_confidence_improvement:+.3f}\n")
            f.write(f"Improvement Rate:         {best_result.improvement_rate:.1%}\n")
            f.write(f"Processing Time:          {best_result.avg_processing_time:.1f} ms\n\n")

            f.write("PARAMETER ANALYSIS:\n")
            f.write("-" * 40 + "\n")

            # Analyze clip limit effects
            clip_effects = {}
            for result in self.results:
                clip = result.config.clip_limit
                if clip not in clip_effects:
                    clip_effects[clip] = []
                clip_effects[clip].append(result.accuracy_improvement)

            best_clip = max(clip_effects.keys(), key=lambda x: np.mean(clip_effects[x]))
            f.write(f"Best Clip Limit:    {best_clip} (avg improvement: {np.mean(clip_effects[best_clip]):+.1%})\n")

            # Analyze grid size effects
            grid_effects = {}
            for result in self.results:
                grid = result.config.tile_grid_size[0]  # Assuming square
                if grid not in grid_effects:
                    grid_effects[grid] = []
                grid_effects[grid].append(result.accuracy_improvement)

            best_grid = max(grid_effects.keys(), key=lambda x: np.mean(grid_effects[x]))
            f.write(f"Best Grid Size:     {best_grid}x{best_grid} (avg improvement: {np.mean(grid_effects[best_grid]):+.1%})\n\n")

            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"✅ Use clip_limit = {best_config.clip_limit}\n")
            f.write(f"✅ Use tile_grid_size = {best_config.tile_grid_size}\n")

            if best_result.accuracy_improvement > 0.02:
                f.write("✅ Significant accuracy improvement expected\n")
            elif best_result.accuracy_improvement > 0:
                f.write("✅ Modest accuracy improvement expected\n")
            else:
                f.write("⚠️  Limited accuracy improvement with current dataset\n")

            if best_result.avg_processing_time < 100:
                f.write("✅ Fast processing suitable for real-time use\n")
            else:
                f.write("⚠️  Consider speed vs quality trade-offs for real-time use\n")

        logger.info(f"📋 Saved optimization report to {report_file}")

async def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Optimize CLAHE parameters for MobileNet")
    parser.add_argument("dataset_path", help="Path to validation dataset")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--max-samples", type=int, default=50, help="Max samples per class for optimization")
    parser.add_argument("--output-dir", default="clahe_optimization", help="Output directory")

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        return

    optimizer = CLAHEOptimizer(args.api_url)

    try:
        best_config = await optimizer.optimize_parameters(
            dataset_path=dataset_path,
            max_samples_per_class=args.max_samples,
            output_dir=Path(args.output_dir)
        )

        logger.info(f"🎉 OPTIMIZATION COMPLETE!")
        logger.info(f"Best CLAHE configuration: {best_config}")

    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
    except Exception as e:
        logger.error(f"Optimization failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())