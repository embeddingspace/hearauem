import logging
from typing import Callable

from .base import AuemBaseModel

logger = logging.getLogger(__name__)

model_registry = {}

def register_model(wrapped_class: Callable) -> Callable:
    global model_registry

    name = wrapped_class.__name__
    if name in model_registry:
        logger.warning('Executor %s already exists. Will replace it', name)
    model_registry[name] = wrapped_class

    return wrapped_class


def create_model(name: str, **kwargs) -> AuemBaseModel:
    """Factory command to create a model from name."""
    global model_registry
    if name not in model_registry:
        logger.warning(f"Model {name} does not exist in the registry\n"
                       "Registry Models:\n"
                       f"{model_registry.keys()}")
        return None

    model_cls = model_registry[name]
    return model_cls(**kwargs)
