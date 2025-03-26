from pydantic import BaseModel
import os
import yaml


class Config(BaseModel):
    witness_dir_path: str
    inputs_dir_path: str
    proofs_dir_path: str

    circuit_path: str
    proving_key_path: str
    verification_key_path: str
    srs_path: str


def load_environment_variables(model: BaseModel) -> None:
    """Recursively load environment variables into the model."""
    for field_name, field in model.__fields__.items():
        env_var = field_name.upper()
        if env_var in os.environ:
            setattr(model, field_name, os.environ[env_var])

        # If the field is a nested model, recursively process it
        value = getattr(model, field_name)
        if isinstance(value, BaseModel):
            load_environment_variables(value)


def new_configuration() -> Config:
    """Create a new configuration object from YAML and environment variables."""
    # 1. Load from YAML file
    config_path = os.environ.get("CONFIG_PATH", "classifier/config.yml")
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    # 2. Create config object from YAML
    config = Config.parse_obj(config_data)

    # 3. Override with environment variables
    load_environment_variables(config)

    return config
