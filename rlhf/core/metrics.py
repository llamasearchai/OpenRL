from typing import Optional, Dict, Any

from rlhf.core.logging import get_logger

logger = get_logger("Metrics")

# Initialize metrics based on available backend
try:
    from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server
    PROMETHEUS_AVAILABLE = True
    logger.info("Prometheus metrics available")
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus metrics not available. Install with 'pip install prometheus-client'")
    # Create dummy metric classes for type checking
    class DummyMetric:
        def __init__(self, *args, **kwargs):
            pass
        
        def labels(self, **kwargs):
            return self
        
        def inc(self, amount=1):
            pass
        
        def dec(self, amount=1):
            pass
        
        def set(self, value):
            pass
        
        def observe(self, value):
            pass
    
    Counter = Gauge = Histogram = Summary = DummyMetric


class MetricsRegistry:
    """Registry for system metrics"""
    
    def __init__(self):
        """Initialize metrics registry"""
        self.metrics = {}
        self.initialized = False
    
    def initialize(self, expose_http: bool = True, port: int = 8000) -> None:
        """Initialize metrics
        
        Args:
            expose_http: Whether to expose metrics HTTP endpoint
            port: Port for metrics HTTP endpoint
        """
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus metrics not available. Metrics will be no-ops.")
            return
        
        if self.initialized:
            logger.warning("Metrics already initialized")
            return
        
        # Start HTTP server if requested
        if expose_http:
            start_http_server(port)
            logger.info(f"Prometheus metrics server started on port {port}")
        
        # Create metrics
        self._create_metrics()
        
        self.initialized = True
        logger.info("Metrics initialized")
    
    def _create_metrics(self) -> None:
        """Create metrics"""
        # Training metrics
        self.metrics["training_steps"] = Counter(
            "rlhf_training_steps_total",
            "Total number of training steps",
            ["algorithm", "model"]
        )
        
        self.metrics["training_loss"] = Gauge(
            "rlhf_training_loss",
            "Current training loss",
            ["algorithm", "model"]
        )
        
        self.metrics["learning_rate"] = Gauge(
            "rlhf_learning_rate",
            "Current learning rate",
            ["algorithm", "model"]
        )
        
        self.metrics["gpu_memory_usage"] = Gauge(
            "rlhf_gpu_memory_usage_bytes",
            "GPU memory usage in bytes",
            ["device"]
        )
        
        # Reward metrics
        self.metrics["reward_value"] = Histogram(
            "rlhf_reward_value",
            "Distribution of reward values",
            ["model"],
            buckets=(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        )
        
        # Inference metrics
        self.metrics["inference_latency"] = Histogram(
            "rlhf_inference_latency_seconds",
            "Inference latency in seconds",
            ["model"],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
        )
        
        self.metrics["generation_tokens_per_second"] = Gauge(
            "rlhf_generation_tokens_per_second",
            "Generation speed in tokens per second",
            ["model"]
        )
        
        # API metrics
        self.metrics["api_requests"] = Counter(
            "rlhf_api_requests_total",
            "Total number of API requests",
            ["endpoint", "method", "status"]
        )
        
        self.metrics["api_latency"] = Histogram(
            "rlhf_api_latency_seconds",
            "API request latency in seconds",
            ["endpoint"],
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
        )
    
    def get_metric(self, name: str) -> Any:
        """Get metric by name
        
        Args:
            name: Metric name
            
        Returns:
            Metric object
        """
        if not self.initialized:
            logger.warning("Metrics not initialized")
            if not PROMETHEUS_AVAILABLE:
                return DummyMetric()
        
        if name not in self.metrics:
            logger.warning(f"Metric '{name}' not found")
            if not PROMETHEUS_AVAILABLE:
                return DummyMetric()
            raise KeyError(f"Metric '{name}' not found")
        
        return self.metrics[name]


# Global metrics registry
metrics_registry = MetricsRegistry()