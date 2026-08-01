import os
import yaml
import logging
from typing import Any, Dict, List
from src.interfaces.expert import BaseExpert

logger = logging.getLogger(__name__)

class ExpertRegistry:
    """Centralized registry for managing denoising experts."""
    _experts: Dict[str, BaseExpert] = {}
    _manifests: Dict[str, dict] = {}

    @classmethod
    def register(cls, expert_name: str, expert_instance: BaseExpert, overwrite: bool = False) -> None:
        cls.validate(expert_instance)
        if expert_name in cls._experts and not overwrite:
            logger.error(f"Duplicate registration prevented for {expert_name}.")
            return
        cls._experts[expert_name] = expert_instance
        logger.info(f"Registered expert: {expert_name}")

    @classmethod
    def get(cls, expert_name: str) -> BaseExpert:
        if not cls.exists(expert_name):
            raise ValueError(f"Expert {expert_name} not found in registry.")
        return cls._experts[expert_name]

    @classmethod
    def exists(cls, expert_name: str) -> bool:
        return expert_name in cls._experts

    @classmethod
    def unregister(cls, expert_name: str) -> None:
        if expert_name in cls._experts:
            del cls._experts[expert_name]
            logger.info(f"Unregistered expert: {expert_name}")

    @classmethod
    def clear(cls) -> None:
        cls._experts.clear()
        cls._manifests.clear()
        logger.info("Expert registry cleared.")

    @classmethod
    def validate(cls, expert_instance: Any) -> None:
        if not isinstance(expert_instance, BaseExpert):
            raise TypeError("Registered object must inherit from BaseExpert.")

    @classmethod
    def list_experts(cls) -> List[str]:
        return list(cls._experts.keys())

    @classmethod
    def discover_manifests(cls, plugins_dir: str) -> None:
        """Auto-discover plugin manifests (expert.yaml)."""
        if not os.path.exists(plugins_dir):
            return
        for root, _, files in os.walk(plugins_dir):
            if "expert.yaml" in files:
                manifest_path = os.path.join(root, "expert.yaml")
                try:
                    with open(manifest_path, "r") as f:
                        manifest = yaml.safe_load(f)
                        if "ExpertName" in manifest:
                            cls._manifests[manifest["ExpertName"]] = manifest
                            logger.info(f"Discovered manifest for {manifest['ExpertName']}")
                except Exception as e:
                    logger.error(f"Failed to load manifest {manifest_path}: {e}")
