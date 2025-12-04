"""
Monitoring and Logging for REST API

Provides comprehensive monitoring, metrics collection, and structured
logging for the pneumonia detection API. Tracks performance, errors,
and clinical metrics for production deployment.
"""

import logging
import time
import functools
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading

import psutil


@dataclass
class RequestMetrics:
    """Metrics for individual API requests."""
    timestamp: float
    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    model_type: Optional[str] = None
    batch_size: Optional[int] = None
    patient_id: Optional[str] = None


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    active_requests: int
    queue_size: int


class MetricsCollector:
    """
    Collects and aggregates API metrics for monitoring.

    Tracks request performance, system resources, and
    clinical metrics with configurable retention.
    """

    def __init__(self, max_history: int = 10000):
        """Initialize metrics collector."""
        self.max_history = max_history

        # Request metrics
        self.request_history: deque = deque(maxlen=max_history)
        self.request_counts = defaultdict(int)
        self.error_counts = defaultdict(int)

        # Performance metrics
        self.response_times: deque = deque(maxlen=1000)
        self.active_requests = 0
        self.total_requests = 0

        # Clinical metrics
        self.prediction_counts = defaultdict(int)
        self.model_usage = defaultdict(int)

        # System metrics
        self.system_history: deque = deque(maxlen=100)

        # Thread safety
        self._lock = threading.Lock()

    def record_request(self, metrics: RequestMetrics):
        """Record metrics for an API request."""
        with self._lock:
            self.request_history.append(metrics)
            self.request_counts[metrics.endpoint] += 1
            self.response_times.append(metrics.response_time_ms)
            self.total_requests += 1

            if metrics.status_code >= 400:
                self.error_counts[metrics.endpoint] += 1

            if metrics.model_type:
                self.model_usage[metrics.model_type] += 1

    def record_prediction(self, prediction: str, confidence: float, model_type: str):
        """Record prediction metrics."""
        with self._lock:
            self.prediction_counts[prediction] += 1

    def record_system_metrics(self):
        """Record current system metrics."""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            metrics = SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 ** 2),
                disk_usage_percent=disk.percent,
                active_requests=self.active_requests,
                queue_size=0  # Would be populated from actual queue
            )

            with self._lock:
                self.system_history.append(metrics)

        except Exception as e:
            logger.error(f"Failed to record system metrics: {e}")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        with self._lock:
            # Calculate response time stats
            if self.response_times:
                avg_response_time = sum(self.response_times) / len(self.response_times)
                sorted_times = sorted(self.response_times)
                p95_response_time = sorted_times[int(len(sorted_times) * 0.95)]
            else:
                avg_response_time = 0
                p95_response_time = 0

            # Error rate
            total_errors = sum(self.error_counts.values())
            error_rate = total_errors / self.total_requests if self.total_requests > 0 else 0

            # Recent system metrics
            recent_system = self.system_history[-1] if self.system_history else None

            return {
                "request_metrics": {
                    "total_requests": self.total_requests,
                    "active_requests": self.active_requests,
                    "avg_response_time_ms": avg_response_time,
                    "p95_response_time_ms": p95_response_time,
                    "error_rate": error_rate,
                    "requests_by_endpoint": dict(self.request_counts),
                    "errors_by_endpoint": dict(self.error_counts)
                },
                "clinical_metrics": {
                    "predictions_by_type": dict(self.prediction_counts),
                    "model_usage": dict(self.model_usage)
                },
                "system_metrics": {
                    "cpu_percent": recent_system.cpu_percent if recent_system else 0,
                    "memory_percent": recent_system.memory_percent if recent_system else 0,
                    "memory_used_mb": recent_system.memory_used_mb if recent_system else 0,
                    "disk_usage_percent": recent_system.disk_usage_percent if recent_system else 0
                } if recent_system else {}
            }


class APILogger:
    """
    Structured logging for API operations.

    Provides consistent logging format for requests, errors,
    and clinical operations with proper correlation.
    """

    def __init__(self, name: str = "pneumonia_api"):
        """Initialize API logger."""
        self.logger = logging.getLogger(name)
        self._setup_logging()

    def _setup_logging(self):
        """Setup structured logging format."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def log_request(self, request_data: Dict[str, Any]):
        """Log incoming API request."""
        self.logger.info(
            f"Request: {request_data.get('method')} {request_data.get('endpoint')} "
            f"- Patient: {request_data.get('patient_id', 'N/A')} "
            f"- Model: {request_data.get('model_type', 'N/A')}"
        )

    def log_prediction(self, prediction_data: Dict[str, Any]):
        """Log prediction result."""
        self.logger.info(
            f"Prediction: {prediction_data.get('prediction')} "
            f"- Confidence: {prediction_data.get('confidence', 0):.3f} "
            f"- Patient: {prediction_data.get('patient_id', 'N/A')} "
            f"- Time: {prediction_data.get('processing_time_ms', 0):.1f}ms"
        )

    def log_error(self, error_data: Dict[str, Any]):
        """Log API error."""
        self.logger.error(
            f"Error: {error_data.get('error_type')} "
            f"- Endpoint: {error_data.get('endpoint')} "
            f"- Patient: {error_data.get('patient_id', 'N/A')} "
            f"- Details: {error_data.get('details', 'N/A')}"
        )

    def log_clinical_event(self, event_data: Dict[str, Any]):
        """Log clinical-specific events."""
        self.logger.info(
            f"Clinical: {event_data.get('event_type')} "
            f"- Patient: {event_data.get('patient_id', 'N/A')} "
            f"- Details: {event_data.get('details', 'N/A')}"
        )


class APIMonitor:
    """
    Comprehensive API monitoring system.

    Combines metrics collection and logging for complete
    observability of the pneumonia detection API.
    """

    def __init__(self):
        """Initialize API monitor."""
        self.metrics = MetricsCollector()
        self.logger = APILogger()
        self._monitoring_active = True

    def middleware(self, request_handler: Callable) -> Callable:
        """Monitoring middleware decorator."""
        @functools.wraps(request_handler)
        async def wrapper(*args, **kwargs):
            start_time = time.perf_counter()

            # Extract request info
            request_info = self._extract_request_info(*args, **kwargs)

            # Log request
            self.logger.log_request(request_info)

            # Track active requests
            self.metrics.active_requests += 1

            try:
                # Execute request
                result = await request_handler(*args, **kwargs)

                # Calculate metrics
                processing_time = (time.perf_counter() - start_time) * 1000

                # Record metrics
                metrics = RequestMetrics(
                    timestamp=time.time(),
                    endpoint=request_info.get('endpoint', 'unknown'),
                    method=request_info.get('method', 'unknown'),
                    status_code=200,  # Assume success if no exception
                    response_time_ms=processing_time,
                    model_type=request_info.get('model_type'),
                    batch_size=request_info.get('batch_size'),
                    patient_id=request_info.get('patient_id')
                )

                self.metrics.record_request(metrics)

                # Log prediction if applicable
                if hasattr(result, 'prediction'):
                    prediction_data = {
                        'prediction': result.prediction,
                        'confidence': result.confidence,
                        'patient_id': request_info.get('patient_id'),
                        'processing_time_ms': processing_time
                    }
                    self.logger.log_prediction(prediction_data)
                    self.metrics.record_prediction(
                        result.prediction, result.confidence,
                        request_info.get('model_type', 'unknown')
                    )

                return result

            except Exception as e:
                # Record error metrics
                processing_time = (time.perf_counter() - start_time) * 1000

                error_metrics = RequestMetrics(
                    timestamp=time.time(),
                    endpoint=request_info.get('endpoint', 'unknown'),
                    method=request_info.get('method', 'unknown'),
                    status_code=500,
                    response_time_ms=processing_time,
                    model_type=request_info.get('model_type'),
                    patient_id=request_info.get('patient_id')
                )

                self.metrics.record_request(error_metrics)

                # Log error
                error_data = {
                    'error_type': type(e).__name__,
                    'endpoint': request_info.get('endpoint'),
                    'patient_id': request_info.get('patient_id'),
                    'details': str(e)
                }
                self.logger.log_error(error_data)

                raise

            finally:
                self.metrics.active_requests -= 1

        return wrapper

    def _extract_request_info(self, *args, **kwargs) -> Dict[str, Any]:
        """Extract relevant information from request."""
        # This would be customized based on actual request structure
        return {
            'endpoint': kwargs.get('endpoint', 'unknown'),
            'method': kwargs.get('method', 'POST'),
            'patient_id': kwargs.get('patient_id'),
            'model_type': kwargs.get('model_type'),
            'batch_size': kwargs.get('batch_size')
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        # Record current system metrics
        self.metrics.record_system_metrics()

        # Get metrics summary
        metrics = self.metrics.get_metrics_summary()

        # Determine health status
        cpu_ok = metrics['system_metrics'].get('cpu_percent', 0) < 80
        memory_ok = metrics['system_metrics'].get('memory_percent', 0) < 80
        error_rate_ok = metrics['request_metrics'].get('error_rate', 0) < 0.1

        overall_status = "healthy" if all([cpu_ok, memory_ok, error_rate_ok]) else "degraded"

        return {
            "status": overall_status,
            "timestamp": time.time(),
            "metrics": metrics,
            "checks": {
                "cpu_usage_ok": cpu_ok,
                "memory_usage_ok": memory_ok,
                "error_rate_ok": error_rate_ok
            }
        }

    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        metrics = self.metrics.get_metrics_summary()

        prometheus_lines = []

        # Request metrics
        prometheus_lines.append(f"api_requests_total {metrics['request_metrics']['total_requests']}")
        prometheus_lines.append(f"api_requests_active {metrics['request_metrics']['active_requests']}")
        prometheus_lines.append(f"api_response_time_avg {metrics['request_metrics']['avg_response_time_ms']}")
        prometheus_lines.append(f"api_error_rate {metrics['request_metrics']['error_rate']}")

        # System metrics
        if metrics['system_metrics']:
            prometheus_lines.append(f"system_cpu_percent {metrics['system_metrics']['cpu_percent']}")
            prometheus_lines.append(f"system_memory_percent {metrics['system_metrics']['memory_percent']}")

        # Clinical metrics
        for prediction_type, count in metrics['clinical_metrics']['predictions_by_type'].items():
            prometheus_lines.append(f"predictions_total{{type=\"{prediction_type}\"}} {count}")

        return "\n".join(prometheus_lines)


# Global monitor instance
_api_monitor: Optional[APIMonitor] = None

def get_api_monitor() -> APIMonitor:
    """Get or create global API monitor instance."""
    global _api_monitor
    if _api_monitor is None:
        _api_monitor = APIMonitor()
    return _api_monitor


# Setup module-level logger
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Test monitoring system
    monitor = APIMonitor()

    # Simulate some metrics
    test_metrics = RequestMetrics(
        timestamp=time.time(),
        endpoint="/predict",
        method="POST",
        status_code=200,
        response_time_ms=150.5,
        model_type="mobilenet",
        patient_id="test_001"
    )

    monitor.metrics.record_request(test_metrics)
    monitor.metrics.record_prediction("PNEUMONIA", 0.85, "mobilenet")

    # Get health status
    health = monitor.get_health_status()
    print(f"Health status: {health['status']}")

    print("Monitoring system ready!")