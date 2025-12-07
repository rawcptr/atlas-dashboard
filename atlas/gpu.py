import subprocess
import re
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class GPUMetrics:
    gpu_id: int
    gpu_util: Optional[int]
    memory_usage: Optional[int]
    max_memory: Optional[int]
    temp: Optional[int]
    power_draw: Optional[float]
    max_power: Optional[float]
    perf: Optional[str]


@dataclass
class GPUStaticInfo:
    smi_version: Optional[str]
    cuda_version: Optional[str]
    driver_version: Optional[str]


def _get_num_gpus() -> int:
    """Detect number of GPUs available"""
    try:
        result = subprocess.run(
            "nvidia-smi --query-gpu=index --format=csv,noheader",
            shell=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            logger.warning("nvidia-smi failed, assuming 0 GPUs")
            return 0

        lines = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
        num = len(lines)
        logger.info(f"Detected {num} GPU(s)")
        return num
    except Exception as e:
        logger.error(f"Failed to detect GPUs: {e}")
        return 0


def _query_gpu(gpu_id: int) -> tuple[Optional[GPUMetrics], Optional[GPUStaticInfo]]:
    """Query GPU metrics and optional static info"""
    try:
        result = subprocess.run(
            f"nvidia-smi --id={gpu_id} "
            "--query-gpu=utilization.gpu,memory.used,memory.total,"
            "temperature.gpu,power.draw,power.limit,pstate "
            "--format=csv,noheader",
            shell=True,
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            raise RuntimeError(f"nvidia-smi failed: {result.stderr}")

        parts = [x.strip() for x in result.stdout.strip().split(",")]

        metrics = GPUMetrics(
            gpu_id=gpu_id,
            gpu_util=int(parts[0].split()[0]) if parts[0] else None,
            memory_usage=int(parts[1].split()[0]) if parts[1] else None,
            max_memory=int(parts[2].split()[0]) if parts[2] else None,
            temp=int(parts[3]) if parts[3] else None,
            power_draw=float(parts[4].split()[0]) if parts[4] else None,
            max_power=float(parts[5].split()[0]) if parts[5] else None,
            perf=parts[6] if parts[6] else None,
        )

        static_info = None
        if gpu_id == 0:
            static_info = _get_static_info()

        return metrics, static_info

    except Exception as e:
        logger.error(f"GPU {gpu_id} query error: {e}")
        return None, None


def _get_static_info() -> GPUStaticInfo:
    """Query static GPU/driver info"""
    try:
        smi_result = subprocess.run(
            "nvidia-smi",
            shell=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        smi_output = smi_result.stdout

        smi_match = re.search(r"NVIDIA-SMI\s+([\d.]+)", smi_output)
        smi_version = smi_match.group(1) if smi_match else None

        cuda_match = re.search(r"CUDA Version:\s+([\d.]+)", smi_output)
        cuda_version = cuda_match.group(1) if cuda_match else None

        driver_result = subprocess.run(
            "nvidia-smi --query-gpu=driver_version --format=csv,noheader",
            shell=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        driver_version = driver_result.stdout.strip() or None

        return GPUStaticInfo(
            smi_version=smi_version,
            cuda_version=cuda_version,
            driver_version=driver_version,
        )
    except Exception as e:
        logger.error(f"Failed to get static GPU info: {e}")
        return GPUStaticInfo(smi_version=None, cuda_version=None, driver_version=None)
