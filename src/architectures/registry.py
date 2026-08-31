from typing import Callable, Dict, Type, Union
from src.core.base_dictionary import BaseDictionary

DICTIONARY_REGISTRY: Dict[str, Type[BaseDictionary]] = {}


def register_dictionary(name: str) -> Callable[[Type[BaseDictionary]], Type[BaseDictionary]]:
    """
    Decorator to register a dictionary architecture class.
    """
    def decorator(cls: Type[BaseDictionary]) -> Type[BaseDictionary]:
        DICTIONARY_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_dictionary_cls(name: str) -> Type[BaseDictionary]:
    name_clean = name.lower().replace("-", "_")
    if name_clean not in DICTIONARY_REGISTRY:
        available = list(DICTIONARY_REGISTRY.keys())
        raise KeyError(f"Architecture '{name}' not found in registry. Available architectures: {available}")
    return DICTIONARY_REGISTRY[name_clean]


def build_dictionary(architecture: str, d_in: int, d_sae: int, **kwargs) -> BaseDictionary:
    """
    Factory function to instantiate any registered dictionary model.
    """
    cls = get_dictionary_cls(architecture)
    return cls(d_in=d_in, d_sae=d_sae, **kwargs)
