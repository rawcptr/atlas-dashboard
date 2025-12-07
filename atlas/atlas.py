import time
import logging
import threading
from typing import Callable, Optional, Any, Collection
from contextlib import contextmanager

from .gpu import _get_num_gpus, _query_gpu
from .metrics import AutoMetric, Metric, _ema
from .transport import AtlasTransport

logger = logging.getLogger(__name__)


class Atlas:
    def __init__(
        self,
        uri: str,
        gpu_interval: Optional[float] = None,
        *,
        batch_size: Optional[int] = None,
        max_flops: Optional[float] = None,
        flops_per_sample: Optional[float] = None,
        flops_fn: Optional[Callable[[], float]] = None,
        auto_metrics: Optional[Collection[AutoMetric]] = None,
        ema_alpha: float = 0.2,
    ) -> None:
        self.uri = uri
        self.gpu_interval = gpu_interval
        # session management
        self._stop_event = threading.Event()
        self._transport = AtlasTransport(uri, self._stop_event)
        # Auto-metric config/state
        self.batch_size = batch_size
        self.max_flops = max_flops
        self._flops_per_sample = flops_per_sample
        self._flops_fn = flops_fn
        self._auto_enabled = self._init_auto_metrics(auto_metrics)
        self._ema_alpha = ema_alpha
        # Auto-metric delta state
        self._last_step: Optional[int] = None
        self._last_time: Optional[float] = None
        self._ema_throughput: Optional[float] = None
        self._ema_step_time: Optional[float] = None
        # Manual computed metrics registry
        self._manual_metrics: dict[Metric, tuple[Callable[[], float], dict[str, Any]]] = {}
        self._metric_cache: dict[Metric, Any] = {}
        # GPU monitoring
        self._static_sent = False
        self.gpu_thread: Optional[threading.Thread] = None

    def _init_auto_metrics(self, auto_metrics: Optional[Collection[AutoMetric]]) -> set[AutoMetric]:
        """Initializes the set of enabled auto-metrics."""
        if auto_metrics is not None:
            return set(auto_metrics)
        return {AutoMetric.THROUGHPUT, AutoMetric.STEP_TIME, AutoMetric.MFU}

    # Public API

    def compute(
        self,
        metric: Metric,
        fn: Callable[[], float],
        *,
        cache: bool = False,
        **meta: Any,
    ) -> None:
        """Register a manual computed metric (evaluated during training())."""
        self._manual_metrics[metric] = (fn, {"cache": cache, **meta})

    metric = compute  # alias

    def static(
        self,
        smi_ver: Optional[str] = None,
        cuda_ver: Optional[str] = None,
        driver_ver: Optional[str] = None,
    ) -> None:
        """
        Send static system/GPU information.
        This can be called manually, or will be called by the GPU monitor
        thread once it retrieves static info.
        """
        self._transport.queue_message(
            {
                "type": "static_metrics",
                "metrics": {
                    "smi_ver": smi_ver,
                    "cuda_ver": cuda_ver,
                    "driver_ver": driver_ver,
                },
            }
        )

    def training(
        self,
        *,
        step: int,
        samples: Optional[int] = None,
        **metrics: float,
    ) -> None:
        """
        Log training metrics for a specific step.
        Auto-metrics (throughput, step_time, MFU) are computed based on
        deltas from the previous call and automatically included.
        """
        auto_metrics = self._calculate_auto_metrics(step=step, samples=samples)
        manual_metrics = self._evaluate_manual_metrics()
        all_metrics = {**metrics, **auto_metrics, **manual_metrics}
        self._transport.queue_message(
            {
                "type": "global_update",
                "step": step,
                "metrics": all_metrics,
            }
        )

    def layer(self, *, layer_id: int, step: int, **metrics: float) -> None:
        """Log per-layer metrics."""
        self._transport.queue_message(
            {
                "type": "layer_update",
                "layer_id": layer_id,
                "step": step,
                "metrics": metrics,
            }
        )

    # Context management

    def __enter__(self) -> "Atlas":
        self._transport.connect()
        self._transport.start_sender_thread()
        if self.gpu_interval is not None:
            self._start_gpu_monitoring()
        return self

    def __exit__(self, exc_t, exc, tb) -> None:
        self._stop_gpu_monitoring()
        self._transport.stop_sender_thread()
        self._transport.disconnect()

    # Internals: auto metrics

    def _calculate_auto_metrics(self, *, step: int, samples: Optional[int]) -> dict[str, float]:
        """
        Orchestrates the calculation of auto-metrics.
        """
        now = time.perf_counter()
        if not self._can_compute_deltas(step, now):
            return {}
        ds = step - self._last_step  # type: ignore
        dt = max(0.0, now - self._last_time)  # type: ignore
        if not self._is_valid_delta(ds, dt):
            self._update_last_state(step, now)
            return {}
        out = {}
        eff_samples = self._get_effective_samples(ds, samples)
        self._compute_throughput(out, eff_samples, dt)
        self._compute_step_time(out, ds, dt)
        self._compute_mfu(out, eff_samples)
        self._update_last_state(step, now)
        return out

    def _can_compute_deltas(self, step: int, now: float) -> bool:
        """Checks if there's enough history to compute deltas."""
        if self._last_step is None or self._last_time is None:
            self._update_last_state(step, now)
            return False
        return True

    def _is_valid_delta(self, ds: int, dt: float) -> bool:
        """Checks if the computed deltas are valid for metric calculation."""
        return ds > 0 and dt > 1e-6

    def _update_last_state(self, step: int, now: float) -> None:
        """Updates the last step and time for delta calculations."""
        self._last_step = step
        self._last_time = now

    def _get_effective_samples(self, ds: int, samples: Optional[int]) -> Optional[int]:
        """Determines the effective number of samples for the current delta window."""
        if samples is not None:
            return samples
        if self.batch_size is not None:
            return self.batch_size * ds
        return None

    def _compute_throughput(
        self, out: dict[str, float], eff_samples: Optional[int], dt: float) -> None:
        """Computes and updates the EMA throughput."""
        if AutoMetric.THROUGHPUT in self._auto_enabled and eff_samples is not None:
            inst_tput = float(eff_samples) / dt
            self._ema_throughput = _ema(self._ema_throughput, inst_tput, self._ema_alpha)
            out[AutoMetric.THROUGHPUT.value] = self._ema_throughput

    def _compute_step_time(self, out: dict[str, float], ds: int, dt: float) -> None:
        """Computes and updates the EMA step time."""
        if AutoMetric.STEP_TIME in self._auto_enabled:
            inst_step_time = dt / float(ds)
            self._ema_step_time = _ema(self._ema_step_time, inst_step_time, self._ema_alpha)
            out[AutoMetric.STEP_TIME.value] = self._ema_step_time

    def _compute_mfu(self, out: dict[str, float], eff_samples: Optional[int]) -> None:
        """Computes MFU if conditions are met."""
        if (
            AutoMetric.MFU in self._auto_enabled
            and eff_samples is not None
            and self.max_flops is not None
            and self._ema_throughput is not None
        ):
            fps = self._get_flops_per_sample()
            if fps is not None:
                achieved_flops = self._ema_throughput * fps
                mfu = 100.0 * achieved_flops / self.max_flops if self.max_flops > 0 else 0.0
                out[AutoMetric.MFU.value] = mfu

    def _get_flops_per_sample(self) -> Optional[float]:
        """
        Retrieves or computes FLOPs per sample.
        Caches the result if computed via flops_fn.
        """
        if self._flops_per_sample is not None:
            return self._flops_per_sample
        if self._flops_fn is not None:
            try:
                self._flops_per_sample = float(self._flops_fn())
                self._flops_fn = None
                return self._flops_per_sample
            except Exception as e:
                logger.error("flops_fn failed: %s", e)
                self._flops_fn = None
                return None
        return None

    def _evaluate_manual_metrics(self) -> dict[str, float]:
        """Evaluates all registered manual computed metrics."""
        out: dict[str, float] = {}
        for metric, (fn, cfg) in self._manual_metrics.items():
            self._evaluate_single_manual_metric(out, metric, fn, cfg)
        return out

    def _evaluate_single_manual_metric(
        self, out: dict[str, float], metric: Metric, fn: Callable[[], float], cfg: dict[str, Any]
    ) -> None:
        """Evaluates a single manual metric, with caching if configured."""
        try:
            if cfg.get("cache") and metric in self._metric_cache:
                out[metric.value] = self._metric_cache[metric]
            else:
                val = float(fn())
                out[metric.value] = val
                if cfg.get("cache"):
                    self._metric_cache[metric] = val
        except Exception as e:
            logger.error("Manual metric %s failed: %s", metric.value, e)

    # Internals: GPU thread

    def _start_gpu_monitoring(self) -> None:
        """Starts the GPU monitoring thread."""
        self._stop_event.clear()
        self.gpu_thread = threading.Thread(target=self._gpu_monitor_loop, daemon=True) # Changed to daemon=True
        self.gpu_thread.start()

    def _stop_gpu_monitoring(self) -> None:
        """Stops the GPU monitoring thread and waits for it to finish."""
        if self.gpu_thread:
            self._stop_event.set()
            self.gpu_thread.join(timeout=5)
        self.gpu_thread = None
        logger.debug("GPU monitoring thread stopped.")

    def _gpu_monitor_loop(self) -> None:
        """Background thread that polls GPU metrics."""
        num_gpus = _get_num_gpus()
        if num_gpus == 0:
            logger.warning("No NVIDIA GPUs detected. Skipping GPU monitoring.")
            return
        while not self._stop_event.is_set():
            try:
                self._poll_gpus(num_gpus)
                self._stop_event.wait(timeout=self.gpu_interval or 2.0)
            except Exception as e:
                logger.error(f"GPU monitor loop error: {e}")
                self._stop_event.wait(timeout=self.gpu_interval or 2.0)
        logger.debug("GPU monitoring loop finished.")

    def _poll_gpus(self, num_gpus: int) -> None:
        """Polls metrics for all detected GPUs."""
        for gpu_id in range(num_gpus):
            metrics, static_info = _query_gpu(gpu_id)
            self._handle_gpu_static_info(static_info)
            self._handle_gpu_dynamic_metrics(metrics)

    def _handle_gpu_static_info(self, static_info: Any) -> None:
        """Sends static GPU info if it hasn't been sent yet."""
        if not self._static_sent and static_info:
            self.static(
                smi_ver=static_info.smi_version,
                cuda_ver=static_info.cuda_version,
                driver_ver=static_info.driver_version,
            )
            self._static_sent = True

    def _handle_gpu_dynamic_metrics(self, metrics: Any) -> None:
        """Queues dynamic GPU metrics if available."""
        if metrics:
            msg = {
                "type": "gpu_update",
                "id": metrics.gpu_id,
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
            self._transport.queue_message(msg)


@contextmanager
def session(
    uri: str,
    gpu_interval: Optional[float] = None,
    *,
    batch_size: Optional[int] = None,
    max_flops: Optional[float] = None,
    flops_per_sample: Optional[float] = None,
    flops_fn: Optional[Callable[[], float]] = None,
    auto_metrics: Optional[Collection[AutoMetric]] = None,
    ema_alpha: float = 0.2,
):
    tracker = Atlas(
        uri,
        gpu_interval,
        batch_size=batch_size,
        max_flops=max_flops,
        flops_per_sample=flops_per_sample,
        flops_fn=flops_fn,
        auto_metrics=auto_metrics,
        ema_alpha=ema_alpha,
    )
    with tracker:
        yield tracker