from .config import SBFlow_Config

__all__ = ["Model", "SBFlow_Config"]


def __getattr__(name):
    if name == "Model":
        from .Model import Model

        return Model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
