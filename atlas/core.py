import json
import threading
import time
import logging
from enum import Enum
from typing import Callable, Optional, Any
from contextlib import contextmanager
from queue import Queue
from websockets.sync.client import connect

from atlas.gpu import _get_num_gpus, _query_gpu

logger = logging.getLogger(__name__)


class Metric(Enum):
    MFU = "mfu"
    THROUGHPUT = "throughput"
    STEP_TIME = "step_time"


class Atlas:
    def __init__(self, uri: str, gpu_interval: Optional[float] = None):
        self.uri = uri
        self.gpu_interval = gpu_interval
        self.ws = None
        self.message_queue: Queue = Queue()
        self.sender_thread: Optional[threading.Thread] = None
        self.gpu_thread: Optional[threading.Thread] = None
        self.num_gpus = 0
        self.static_info_sent = False
        self.metrics: dict[Metric, tuple[Callable, dict]] = {}
        self.metric_cache: dict[Metric, Any] = {}
        self._stop_event = threading.Event()

    def __enter__(self):
        self._connect()
        self._start_sender_thread()
        if self.gpu_interval:
            self._start_gpu_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_gpu_monitoring()
        self._stop_sender_thread()  # Flush queue before disconnecting
        self._disconnect()

    def _connect(self):
        try:
            self.ws = connect(self.uri)
            logger.info(f"Connected to {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise

    def _disconnect(self):
        if self.ws:
            self.ws.close()
            logger.info("Disconnected")

    def _start_sender_thread(self):
        """Start background thread that sends queued messages"""
        self._stop_event.clear()
        self.sender_thread = threading.Thread(
            target=self._sender_loop,
            daemon=False,
        )
        self.sender_thread.start()

    def _stop_sender_thread(self):
        """Stop sender thread and flush remaining messages"""
        self._stop_event.set()
        if self.sender_thread:
            self.sender_thread.join(timeout=10)

    def _sender_loop(self):
        """Background thread that drains message queue"""
        while not self._stop_event.is_set():
            try:
                msg = self.message_queue.get(timeout=0.1)
                if self.ws:
                    self.ws.send(json.dumps(msg))
            except Exception as e:
                logger.error(f"Send error: {e}")

    def metric(
        self,
        metric: Metric,
        fn: Callable[[], float],
        cache: bool = False,
        **kwargs,
    ) -> None:
        """Register a computed metric"""
        self.metrics[metric] = (fn, {"cache": cache, **kwargs})

    def _evaluate_metrics(self) -> dict[str, float]:
        """Evaluate all registered metrics"""
        result = {}
        for metric, (fn, config) in self.metrics.items():
            try:
                if config.get("cache") and metric in self.metric_cache:
                    result[metric.value] = self.metric_cache[metric]
                else:
                    value = fn()
                    result[metric.value] = value
                    if config.get("cache"):
                        self.metric_cache[metric] = value
            except Exception as e:
                logger.error(f"Failed to evaluate {metric.value}: {e}")
        return result

    def static(
        self,
        smi_ver: Optional[str] = None,
        cuda_ver: Optional[str] = None,
        driver_ver: Optional[str] = None,
    ) -> None:
        """Send static GPU info"""
        msg = {
            "type": "static_metrics",
            "metrics": {
                "smi_ver": smi_ver,
                "cuda_ver": cuda_ver,
                "driver_ver": driver_ver,
            },
        }
        self._queue_message(msg)

    def training(self, step: int, **metrics) -> None:
        """Log training metrics (auto-includes registered computed metrics)"""
        computed = self._evaluate_metrics()
        all_metrics = {**metrics, **computed}

        msg = {
            "type": "global_update",
            "step": step,
            "metrics": all_metrics,
        }
        self._queue_message(msg)

    def layer(self, layer_id: int, step: int, **metrics) -> None:
        """Log layer metrics"""
        msg = {
            "type": "layer_update",
            "layer_id": layer_id,
            "step": step,
            "metrics": metrics,
        }
        self._queue_message(msg)

    def _queue_message(self, msg: dict) -> None:
        """Add message to send queue (thread-safe)"""
        self.message_queue.put(msg)

    def _start_gpu_monitoring(self) -> None:
        """Start background GPU monitoring thread"""
        self._stop_event.clear()
        self.gpu_thread = threading.Thread(
            target=self._gpu_monitor_loop,
            daemon=False,
        )
        self.gpu_thread.start()

    def _stop_gpu_monitoring(self) -> None:
        """Stop GPU monitoring thread"""
        if self.gpu_thread:
            self.gpu_thread.join(timeout=5)

    def _gpu_monitor_loop(self) -> None:
        """Background thread that polls GPU metrics"""
        self.num_gpus = _get_num_gpus()

        if self.num_gpus == 0:
            logger.warning("No GPUs detected")
            return

        while not self._stop_event.is_set():
            try:
                for gpu_id in range(self.num_gpus):
                    metrics, static_info = _query_gpu(gpu_id)

                    # Send static info once on first query
                    if not self.static_info_sent and static_info:
                        self.static(
                            smi_ver=static_info.smi_version,
                            cuda_ver=static_info.cuda_version,
                            driver_ver=static_info.driver_version,
                        )
                        self.static_info_sent = True

                    # Send GPU metrics
                    if metrics:
                        msg = {
                            "type": "gpu_update",
                            "id": gpu_id,
                            "timestamp": time.time(),
                            "metrics": {
                                "temp": metrics.temp,
                                "perf": metrics.perf,
                                "pwr_draw": metrics.power_draw,
                                "max_pwr": metrics.max_power,
                                "mem_usg": metrics.memory_usage,
                                "max_mem": metrics.max_memory,
                                "gpu_util": metrics.gpu_util,
                            },
                        }
                        self._queue_message(msg)

                time.sleep(self.gpu_interval or 2.0)
            except Exception as e:
                logger.error(f"GPU monitor error: {e}")
                time.sleep(self.gpu_interval or 2.0)


@contextmanager
def session(uri: str, gpu_interval: Optional[float] = None):
    """Context manager for Atlas sessions"""
    tracker = Atlas(uri, gpu_interval)
    with tracker:
        yield tracker
