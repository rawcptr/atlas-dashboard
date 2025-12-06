import asyncio
import json
from websockets.asyncio.client import connect

_client = None


class AtlasClient:
    def __init__(self, url: str):
        self.url = url
        self.ws = None
        self.queue = asyncio.Queue()
        self.running = True
        self.sender_task = None

    async def connect(self):
        print(f"[DEBUG] Connecting to {self.url}...")
        self.ws = await connect(self.url)
        print("[DEBUG] Connected!")
        self.sender_task = asyncio.create_task(self._sender())

        if self.sender_task.done():
            try:
                self.sender_task.result()
            except Exception as e:
                print(f"[ERROR] Task exception: {e}")

    async def _send(self, msg: dict):
        await self.queue.put(msg)
        await asyncio.sleep(0)

    async def _sender(self):
        while self.running:
            msg = await self.queue.get()
            try:
                if self.ws is not None:
                    await self.ws.send(json.dumps(msg))
            except Exception as e1:
                print(f"[ERROR] Send failed: {e1}")
                await asyncio.sleep(0.5)
                try:
                    self.ws = await connect(self.url)
                    print("[DEBUG] Reconnected successfully")
                except Exception as e2:
                    print(f"[ERROR] Reconnect failed: {e2}")
                    continue
                await self.ws.send(json.dumps(msg))

    async def close(self):
        self.running = False
        if self.sender_task:
            self.sender_task.cancel()
        if self.ws:
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
    try:
        await _c()._send(
            {
                "type": "dynamic_metrics",
                "data": data,
            }
        )
    except Exception as e:
        print(f"[ERROR] update_dynamic failed: {e}")
        import traceback

        traceback.print_exc()


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
