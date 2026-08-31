from typing import Callable, Dict, Type
from src.core.base_dictionary import BaseDictionary

DICTIONARY_REGISTRY: Dict[str, Type[BaseDictionary]] = {}


def register_dictionary(name: str) -> Callable[[Type[BaseDictionary]], Type[BaseDictionary]]:
    """
    Decorator to register a dictionary architecture class.
    """
    def decorator(cls: Type[BaseDictionary]) -> Type[BaseDictionary]:
        DICTIONARY_REGISTRY[name.lower()] = cls
        DICTIONARY_REGISTRY[cls.__name__.lower()] = cls
        return cls
    return decorator


def _ensure_registered() -> None:
    """Ensure all subpackages are imported and registered."""
    try:
        import src.architectures.sae  # noqa: F401
        import src.architectures.transcoders  # noqa: F401
        import src.architectures.crosscoders  # noqa: F401
    except ImportError:
        pass


def get_dictionary_cls(name: str) -> Type[BaseDictionary]:
    _ensure_registered()
    name_clean = name.lower().replace("-", "_")
    if name_clean in DICTIONARY_REGISTRY:
        return DICTIONARY_REGISTRY[name_clean]
    
    # Also check if class name matches directly
    for k, cls in DICTIONARY_REGISTRY.items():
        if cls.__name__.lower() == name_clean or k == name_clean:
            return cls

    available = list(DICTIONARY_REGISTRY.keys())
    raise KeyError(f"Architecture '{name}' not found in registry. Available architectures: {available}")


def build_dictionary(architecture: str, d_in: int, d_sae: int, **kwargs) -> BaseDictionary:
    """
    Factory function to instantiate any registered dictionary model.
    """
    cls = get_dictionary_cls(architecture)
    return cls(d_in=d_in, d_sae=d_sae, **kwargs)
