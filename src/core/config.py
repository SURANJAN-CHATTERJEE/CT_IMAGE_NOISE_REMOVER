import os
import yaml
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ConfigManager:
    """Central configuration system with full YAML support."""
    _cache: Dict[str, Any] = {}

    @classmethod
    def load(cls, config_name: str, file_path: str) -> None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        try:
            with open(file_path, "r") as f:
                data = yaml.safe_load(f)
                cls._cache[config_name] = data or {}
                logger.info(f"Loaded config: {config_name} from {file_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load config {file_path}: {e}")

    @classmethod
    def reload(cls, config_name: str, file_path: str) -> None:
        cls.load(config_name, file_path)

    @classmethod
    def save(cls, config_name: str, file_path: str) -> None:
        if config_name not in cls._cache:
            raise ValueError(f"Config {config_name} not found in cache.")
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                yaml.safe_dump(cls._cache[config_name], f, default_flow_style=False)
                logger.info(f"Saved config: {config_name} to {file_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save config {file_path}: {e}")

    @classmethod
    def exists(cls, config_name: str) -> bool:
        return config_name in cls._cache

    @classmethod
    def get(cls, config_name: str, key: str = None, default: Any = None) -> Any:
        config = cls._cache.get(config_name, {})
        if key:
            return config.get(key, default)
        return config

    @classmethod
    def set(cls, config_name: str, key: str, value: Any) -> None:
        if config_name not in cls._cache:
            cls._cache[config_name] = {}
        cls._cache[config_name][key] = value

    @classmethod
    def validate(cls, config_name: str, required_keys: list) -> bool:
        if not cls.exists(config_name):
            return False
        config = cls._cache[config_name]
        return all(k in config for k in required_keys)
