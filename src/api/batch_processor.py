"""
Batch Prediction Support for REST API

This module provides efficient batch processing capabilities for the
pneumonia detection REST API. Handles concurrent processing of multiple
images with proper resource management and progress tracking.

Features:
- Concurrent batch processing
- Progress tracking and status updates
- Resource management and throttling
- Error handling and partial results
- Queue management for large batches
- Background task processing

Designed for high-throughput clinical environments where multiple
chest X-rays need to be processed simultaneously.
"""

import os
import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from fastapi import BackgroundTasks
import torch
from PIL import Image

# Setup logging
logger = logging.getLogger(__name__)

class BatchStatus(Enum):
    """Batch processing status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"

@dataclass
class BatchItem:
    """Individual item in a batch processing request."""
    item_id: str
    image: Union[Image.Image, bytes, str]
    patient_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time_ms: Optional[float] = None

@dataclass
class BatchRequest:
    """Complete batch processing request."""
    batch_id: str
    items: List[BatchItem]
    model_type: str
    confidence_threshold: float
    processing_options: Dict[str, Any]
    created_at: float
    status: BatchStatus = BatchStatus.PENDING
    progress: float = 0.0
    results: Optional[List[Dict[str, Any]]] = None
    error_summary: Optional[Dict[str, Any]] = None

class BatchProcessor:
    """
    High-performance batch processor for pneumonia detection.

    Manages concurrent processing of multiple images with proper
    resource allocation and progress tracking for clinical workflows.
    """

    def __init__(self,
                 max_concurrent: int = 5,
                 max_batch_size: int = 50,
                 timeout_seconds: int = 300):
        """
        Initialize batch processor.

        Args:
            max_concurrent: Maximum concurrent processing tasks
            max_batch_size: Maximum number of items per batch
            timeout_seconds: Maximum processing timeout per batch
        """
        self.max_concurrent = max_concurrent
        self.max_batch_size = max_batch_size
        self.timeout_seconds = timeout_seconds

        # Batch tracking
        self.active_batches: Dict[str, BatchRequest] = {}
        self.completed_batches: Dict[str, BatchRequest] = {}
        self.processing_semaphore = asyncio.Semaphore(max_concurrent)

        # Performance metrics
        self.total_batches_processed = 0
        self.total_items_processed = 0
        self.total_processing_time = 0.0

        logger.info(f"BatchProcessor initialized: max_concurrent={max_concurrent}, max_batch_size={max_batch_size}")

    async def submit_batch(self,
                          images: List[Union[Image.Image, bytes, str]],
                          patient_ids: Optional[List[str]] = None,
                          model_type: str = "mobilenet",
                          confidence_threshold: float = 0.5,
                          processing_options: Optional[Dict[str, Any]] = None,
                          callback: Optional[Callable] = None) -> str:
        """
        Submit a batch for processing.

        Args:
            images: List of images to process
            patient_ids: Optional list of patient identifiers
            model_type: Model type for processing
            confidence_threshold: Confidence threshold for predictions
            processing_options: Additional processing options
            callback: Optional callback function for completion

        Returns:
            Batch ID for tracking progress
        """
        # Validate batch size
        if len(images) > self.max_batch_size:
            raise ValueError(f"Batch size {len(images)} exceeds maximum {self.max_batch_size}")

        if patient_ids and len(patient_ids) != len(images):
            raise ValueError("Number of patient IDs must match number of images")

        # Create batch request
        batch_id = str(uuid.uuid4())

        # Create batch items
        items = []
        for i, image in enumerate(images):
            item = BatchItem(
                item_id=f"{batch_id}-{i}",
                image=image,
                patient_id=patient_ids[i] if patient_ids else None,
                metadata={"batch_index": i}
            )
            items.append(item)

        batch_request = BatchRequest(
            batch_id=batch_id,
            items=items,
            model_type=model_type,
            confidence_threshold=confidence_threshold,
            processing_options=processing_options or {},
            created_at=time.time()
        )

        # Store batch
        self.active_batches[batch_id] = batch_request

        # Start processing (non-blocking)
        asyncio.create_task(self._process_batch(batch_request, callback))

        logger.info(f"Batch submitted: {batch_id} with {len(images)} items")
        return batch_id

    async def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current status of a batch.

        Args:
            batch_id: Batch identifier

        Returns:
            Batch status information or None if not found
        """
        # Check active batches
        if batch_id in self.active_batches:
            batch = self.active_batches[batch_id]
        elif batch_id in self.completed_batches:
            batch = self.completed_batches[batch_id]
        else:
            return None

        # Calculate progress
        completed_items = sum(1 for item in batch.items if item.status in ["completed", "failed"])
        progress = completed_items / len(batch.items) if batch.items else 0.0

        return {
            "batch_id": batch_id,
            "status": batch.status.value,
            "progress": progress,
            "total_items": len(batch.items),
            "completed_items": completed_items,
            "failed_items": sum(1 for item in batch.items if item.status == "failed"),
            "created_at": batch.created_at,
            "processing_time_seconds": time.time() - batch.created_at if batch.status == BatchStatus.PROCESSING else None,
            "model_type": batch.model_type,
            "has_results": batch.results is not None
        }

    async def get_batch_results(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """
        Get results of a completed batch.

        Args:
            batch_id: Batch identifier

        Returns:
            Batch results or None if not found/completed
        """
        batch = self.completed_batches.get(batch_id) or self.active_batches.get(batch_id)

        if not batch or batch.status not in [BatchStatus.COMPLETED, BatchStatus.PARTIAL]:
            return None

        # Compile results
        results = []
        for item in batch.items:
            if item.result:
                results.append({
                    "item_id": item.item_id,
                    "patient_id": item.patient_id,
                    "status": item.status,
                    "result": item.result,
                    "processing_time_ms": item.processing_time_ms
                })
            elif item.error:
                results.append({
                    "item_id": item.item_id,
                    "patient_id": item.patient_id,
                    "status": item.status,
                    "error": item.error
                })

        return {
            "batch_id": batch_id,
            "status": batch.status.value,
            "total_items": len(batch.items),
            "successful_items": sum(1 for item in batch.items if item.status == "completed"),
            "failed_items": sum(1 for item in batch.items if item.status == "failed"),
            "results": results,
            "batch_metadata": {
                "model_type": batch.model_type,
                "confidence_threshold": batch.confidence_threshold,
                "processing_options": batch.processing_options,
                "created_at": batch.created_at,
                "total_processing_time": time.time() - batch.created_at
            }
        }

    async def cancel_batch(self, batch_id: str) -> bool:
        """
        Cancel a pending or processing batch.

        Args:
            batch_id: Batch identifier

        Returns:
            True if cancelled successfully, False otherwise
        """
        if batch_id not in self.active_batches:
            return False

        batch = self.active_batches[batch_id]

        if batch.status == BatchStatus.PENDING:
            batch.status = BatchStatus.FAILED
            batch.error_summary = {"error": "Batch cancelled by user"}
            self._move_to_completed(batch_id)
            logger.info(f"Batch cancelled: {batch_id}")
            return True

        return False

    async def _process_batch(self, batch_request: BatchRequest, callback: Optional[Callable] = None):
        """Process a batch request with concurrent item processing."""
        batch_id = batch_request.batch_id
        start_time = time.time()

        try:
            batch_request.status = BatchStatus.PROCESSING
            logger.info(f"Starting batch processing: {batch_id}")

            # Process items concurrently
            tasks = []
            for item in batch_request.items:
                task = asyncio.create_task(self._process_single_item(item, batch_request))
                tasks.append(task)

            # Wait for all tasks with timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self.timeout_seconds
                )
            except asyncio.TimeoutError:
                logger.error(f"Batch processing timeout: {batch_id}")
                batch_request.status = BatchStatus.FAILED
                batch_request.error_summary = {"error": "Processing timeout"}

            # Determine final status
            successful_items = sum(1 for item in batch_request.items if item.status == "completed")
            failed_items = sum(1 for item in batch_request.items if item.status == "failed")

            if successful_items == len(batch_request.items):
                batch_request.status = BatchStatus.COMPLETED
            elif successful_items > 0:
                batch_request.status = BatchStatus.PARTIAL
                batch_request.error_summary = {
                    "partial_success": True,
                    "successful_items": successful_items,
                    "failed_items": failed_items
                }
            else:
                batch_request.status = BatchStatus.FAILED
                batch_request.error_summary = {"error": "All items failed"}

            # Update metrics
            processing_time = time.time() - start_time
            self.total_batches_processed += 1
            self.total_items_processed += len(batch_request.items)
            self.total_processing_time += processing_time

            logger.info(f"Batch processing completed: {batch_id} ({successful_items}/{len(batch_request.items)} successful)")

            # Move to completed
            self._move_to_completed(batch_id)

            # Call callback if provided
            if callback:
                try:
                    await callback(batch_request)
                except Exception as e:
                    logger.error(f"Batch callback failed: {e}")

        except Exception as e:
            logger.error(f"Batch processing failed: {batch_id} - {e}")
            batch_request.status = BatchStatus.FAILED
            batch_request.error_summary = {"error": str(e)}
            self._move_to_completed(batch_id)

    async def _process_single_item(self, item: BatchItem, batch_request: BatchRequest):
        """Process a single item within a batch."""
        async with self.processing_semaphore:
            item_start_time = time.perf_counter()

            try:
                # Mock prediction processing (replace with actual inference)
                await asyncio.sleep(0.1)  # Simulate processing time

                # Mock prediction result
                mock_confidence = 0.85
                mock_prediction = "PNEUMONIA" if mock_confidence > batch_request.confidence_threshold else "NORMAL"

                item.result = {
                    "prediction": mock_prediction,
                    "confidence": mock_confidence,
                    "probabilities": {
                        "NORMAL": 1 - mock_confidence,
                        "PNEUMONIA": mock_confidence
                    },
                    "model_type": batch_request.model_type
                }
                item.status = "completed"

                processing_time = (time.perf_counter() - item_start_time) * 1000
                item.processing_time_ms = processing_time

                logger.debug(f"Item processed: {item.item_id} - {mock_prediction} ({mock_confidence:.3f})")

            except Exception as e:
                item.status = "failed"
                item.error = str(e)
                logger.error(f"Item processing failed: {item.item_id} - {e}")

    def _move_to_completed(self, batch_id: str):
        """Move batch from active to completed storage."""
        if batch_id in self.active_batches:
            batch = self.active_batches.pop(batch_id)
            self.completed_batches[batch_id] = batch

            # Clean up old completed batches (keep last 100)
            if len(self.completed_batches) > 100:
                oldest_batches = sorted(
                    self.completed_batches.items(),
                    key=lambda x: x[1].created_at
                )
                for old_batch_id, _ in oldest_batches[:-100]:
                    del self.completed_batches[old_batch_id]

    def get_processor_stats(self) -> Dict[str, Any]:
        """Get processor performance statistics."""
        avg_processing_time = (
            self.total_processing_time / self.total_batches_processed
            if self.total_batches_processed > 0 else 0.0
        )

        return {
            "total_batches_processed": self.total_batches_processed,
            "total_items_processed": self.total_items_processed,
            "avg_batch_processing_time_seconds": avg_processing_time,
            "active_batches": len(self.active_batches),
            "completed_batches_cached": len(self.completed_batches),
            "max_concurrent": self.max_concurrent,
            "max_batch_size": self.max_batch_size,
            "current_queue_size": sum(len(batch.items) for batch in self.active_batches.values())
        }

    async def cleanup_old_batches(self, max_age_hours: int = 24):
        """Clean up old batch records."""
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)

        # Clean up completed batches
        to_remove = [
            batch_id for batch_id, batch in self.completed_batches.items()
            if batch.created_at < cutoff_time
        ]

        for batch_id in to_remove:
            del self.completed_batches[batch_id]

        logger.info(f"Cleaned up {len(to_remove)} old batch records")


# Global batch processor instance
_batch_processor: Optional[BatchProcessor] = None

def get_batch_processor() -> BatchProcessor:
    """Get or create global batch processor instance."""
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = BatchProcessor()
    return _batch_processor

def init_batch_processor(max_concurrent: int = 5, max_batch_size: int = 50) -> BatchProcessor:
    """Initialize global batch processor with custom settings."""
    global _batch_processor
    _batch_processor = BatchProcessor(max_concurrent=max_concurrent, max_batch_size=max_batch_size)
    return _batch_processor


if __name__ == "__main__":
    # Test batch processing
    print("Testing batch processor...")

    async def test_batch_processing():
        try:
            processor = BatchProcessor(max_concurrent=3, max_batch_size=10)

            # Create test images
            test_images = [Image.new('RGB', (224, 224), color='gray') for _ in range(5)]
            patient_ids = [f"patient_{i}" for i in range(5)]

            # Submit batch
            batch_id = await processor.submit_batch(
                images=test_images,
                patient_ids=patient_ids,
                model_type="mobilenet"
            )

            print(f"Batch submitted: {batch_id}")

            # Wait for completion
            while True:
                status = await processor.get_batch_status(batch_id)
                print(f"Progress: {status['progress']:.1%}")

                if status['status'] in ['completed', 'failed', 'partial']:
                    break

                await asyncio.sleep(0.5)

            # Get results
            results = await processor.get_batch_results(batch_id)
            print(f"Batch completed: {results['successful_items']}/{results['total_items']} successful")

            print("Batch processing test successful!")

        except Exception as e:
            print(f"Test failed: {e}")

    # Run test
    asyncio.run(test_batch_processing())
    print("Batch processing system ready!")