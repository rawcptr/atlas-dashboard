from .client import AtlasClient, init, update_layer, update_dynamic, update_gpu, close, AtlasSession


def session(url: str):
    return AtlasSession(url)


__all__ = [
    "AtlasClient",
    "init",
    "update_layer",
    "update_dynamic",
    "update_gpu",
    "close",
]
