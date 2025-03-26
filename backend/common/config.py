from pydantic import BaseModel
import os
import yaml
from typing import TypeVar, Type, Generic

# Create a generic type variable for BaseModel subclasses
T = TypeVar('T', bound=BaseModel)


def load_environment_variables(model: BaseModel) -> None:
    """Recursively load environment variables into the model."""
    for field_name, field in model.__annotations__.items():
        env_var = field_name.upper()
        if env_var in os.environ:
            setattr(model, field_name, os.environ[env_var])

        # If the field is a nested model, recursively process it
        value = getattr(model, field_name, None)
        if isinstance(value, BaseModel):
            load_environment_variables(value)


def new_configuration(model_class: Type[T], config_path: str = None) -> T:
    """
    Create a new configuration object from YAML and environment variables.

    Args:
        model_class: The Pydantic model class to instantiate
        config_path: Path to the YAML config file (optional)

    Returns:
        An instance of the provided model_class with values from YAML and env vars
    """
    # Get config path from environment or use provided path or default
    if config_path is None:
        config_path = os.environ.get("CONFIG_PATH", f"classifier/config.yml")

    # Load from YAML file
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    # Create config object from YAML
    config = model_class.parse_obj(config_data)

    # Override with environment variables
    load_environment_variables(config)

    return config
