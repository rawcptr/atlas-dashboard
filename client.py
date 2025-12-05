import asyncio
import json
from websockets.asyncio.client import connect

_client = None


class AtlasClient:
    def __init__(self, url: str):
        self.url = url
        self.ws = None
        self.queue = asyncio.Queue()
        self.task = None
        self.closed = False

    async def connect(self):
        self.ws = await connect(self.url)
        self.task = asyncio.create_task(self._sender())

    async def _send(self, msg: dict):
        if self.closed:
            return
        await self.queue.put(msg)

    async def _sender(self):
        while not self.closed:
            msg = await self.queue.get()
            try:
                if self.ws is not None:
                    await self.ws.send(json.dumps(msg))
            except Exception:
                await asyncio.sleep(1.0)
                self.ws = await connect(self.url)

    async def close(self):
        self.closed = True
        if self.ws is not None:
            await self.ws.close()


class AtlasSession:
    def __init__(self, url: str):
        self.url = url
        self.client = AtlasClient(url)

    async def __aenter__(self):
        await self.client.connect()
        global _client
        _client = self.client
        return self.client

    async def __aexit__(self, exc_type, exc, tb):
        await self.client.close()
        return False


async def init(url: str):
    global _client
    _client = AtlasClient(url)
    await _client.connect()
    return _client


def _c():
    if _client is None:
        raise RuntimeError("atlas_can.init() must be called first.")
    return _client


async def update_dynamic(
    *,
    step: int,
    loss=None,
    lr=None,
    tokens_per_second=None,
    samples_per_second=None,
    **extra,
):
    data = {
        "step": step,
        "loss": loss,
        "lr": lr,
        "tokens_per_second": tokens_per_second,
        "samples_per_second": samples_per_second,
        **extra,
    }
    data = {k: v for k, v in data.items() if v is not None}

    await _c()._send(
        {
            "type": "dynamic_metrics",
            "data": data,
        }
    )


async def update_gpu(
    *,
    gpu_id=0,
    util=None,
    mem_allocated=None,
    mem_total=None,
    temperature=None,
    power_watts=None,
    **extra,
):
    data = {
        "gpu_id": gpu_id,
        "util": util,
        "mem_allocated": mem_allocated,
        "mem_total": mem_total,
        "temperature": temperature,
        "power_watts": power_watts,
        **extra,
    }
    data = {k: v for k, v in data.items() if v is not None}

    await _c()._send(
        {
            "type": "gpu_metrics",
            "data": data,
        }
    )


async def update_layer(
    *,
    layer_id: int,
    step: int,
    mean=None,
    stddev=None,
    absmax=None,
    attn_entropy=None,
    grad_norm=None,
):
    data = {
        "mean": mean,
        "stddev": stddev,
        "absmax": absmax,
        "attn_entropy": attn_entropy,
        "grad_norm": grad_norm,
    }
    data = {k: v for k, v in data.items() if v is not None}

    if not data:
        return

    await _c()._send(
        {
            "type": "layer_metrics",
            "layer_id": layer_id,
            "step": step,
            "data": data,
        }
    )


async def close():
    await _c().close()
