"""Decoder registry for stage 1 models."""

from typing import Callable, Dict, Optional, Type, Union

DECODER_REGISTRY: Dict[str, Type] = {}
__all__ = ["DECODER_REGISTRY", "register_decoder"]


def _add_to_registry(name: str, cls: Type) -> Type:
    if name in DECODER_REGISTRY and DECODER_REGISTRY[name] is not cls:
        raise ValueError(f"Decoder '{name}' is already registered.")
    DECODER_REGISTRY[name] = cls
    return cls


def register_decoder(cls: Optional[Type] = None, *, name: Optional[str] = None) -> Union[Callable[[Type], Type], Type]:
    """Register a decoder class in ``DECODER_REGISTRY``.

    Usage: ``@register_decoder(name="GeneralDecoder_qformer")`` on the class,
    or ``register_decoder(MyClass, name="MyDecoder")`` after definition.
    """
    def decorator(inner_cls: Type) -> Type:
        decoder_name = name or inner_cls.__name__
        return _add_to_registry(decoder_name, inner_cls)

    if cls is None:
        return decorator
    return decorator(cls)


# Import modules that perform registration on import.
# Only decoders used by released configs are included and imported.
from .decoder import GeneralDecoder
from .vit_decoder import ViTDecoder

for _name, _cls in [
    ("GeneralDecoder", GeneralDecoder),
    ("ViTDecoder", ViTDecoder),
]:
    if _name not in DECODER_REGISTRY:
        _add_to_registry(_name, _cls)
