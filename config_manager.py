"""
CONVIO - Configuration Manager
==============================

Loads, validates, and exposes application settings from ``config.yaml``.

This module deliberately has no GUI dependencies so it can be imported from
analysis code, tests, or a future headless/CLI entry point without pulling
PyQt into the dependency graph.
"""

import os
import sys
import logging
from typing import Any, Dict, List

import yaml


class ConfigManager:
    """
    Manages application configuration from ``config.yaml`` (YAML-first, single source).

    Responsibilities:
    - Locate and parse the YAML file (works both in source and PyInstaller builds).
    - Validate that required sections and keys are present.
    - Configure the root logger based on the ``logging`` section.
    - Create on-disk directories referenced by the ``paths`` section.
    - Provide dotted-path read access via :py:meth:`get`.
    """

    # Minimum set of sections/keys that must be present in config.yaml.
    # Anything the application is free to live without has a default and is
    # read lazily through ``.get()`` instead of being listed here.
    _REQUIRED_SECTIONS = ["paths", "graph_loader", "logging", "error_handling", "reproducibility"]
    _REQUIRED_PATH_KEYS = ["data_dir", "export_dir", "log_dir"]
    _REQUIRED_GRAPH_LOADER_KEYS = [
        "min_direct_node_distance_mm",
        "allow_projection_on_edge",
        "skip_self_loops",
    ]

    def __init__(self, config_path: str = None):
        # Resolve the base directory so exports/logs land next to the executable
        # when frozen, and next to the source file in development mode.
        if getattr(sys, "frozen", False):
            self.base_dir = os.path.dirname(sys.executable)
        else:
            self.base_dir = os.path.dirname(os.path.abspath(__file__))

        self.config_path = config_path or os.path.join(self.base_dir, "config.yaml")

        self.config: Dict[str, Any] = self._load_config_yaml_only()
        self._setup_logging()
        self._create_directories()

    # ------------------------------------------------------------------
    # Loading & validation
    # ------------------------------------------------------------------
    def _load_config_yaml_only(self) -> Dict[str, Any]:
        """Load configuration strictly from ``config.yaml``."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Config file not found at: {self.config_path}. "
                "Please add config.yaml next to main.py."
            )

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                raise ValueError("Top-level configuration must be a mapping (YAML object).")

            self._require_keys(data, self._REQUIRED_SECTIONS)
            self._require_keys(data["paths"], self._REQUIRED_PATH_KEYS)
            self._require_keys(data["graph_loader"], self._REQUIRED_GRAPH_LOADER_KEYS)
            return data
        except Exception as e:
            raise RuntimeError(f"Failed to load/validate configuration: {e}") from e

    @staticmethod
    def _require_keys(node: Dict[str, Any], keys: List[str]) -> None:
        missing = [k for k in keys if k not in node]
        if missing:
            raise ValueError(f"Missing required configuration keys: {missing}")

    # ------------------------------------------------------------------
    # Side-effects performed at startup
    # ------------------------------------------------------------------
    def _setup_logging(self) -> None:
        """Configure the root logger from the ``logging`` section."""
        log_config = self.config.get("logging", {})
        log_level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
        fmt = log_config.get("fmt", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        datefmt = log_config.get("datefmt", "%H:%M:%S")

        logging.basicConfig(level=log_level, format=fmt, datefmt=datefmt)

        if log_config.get("enable_log_to_file", True):
            log_dir = self.config["paths"]["log_dir"]
            os.makedirs(log_dir, exist_ok=True)
            log_file = self.config["paths"].get("log_file", os.path.join(log_dir, "optimizer.log"))

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
            logging.getLogger().addHandler(file_handler)

    def _create_directories(self) -> None:
        """Create directories referenced by the ``paths`` section."""
        paths = self.config["paths"]

        # Anchor relative paths to ``self.base_dir`` rather than the CWD so
        # behaviour is identical in source and frozen (PyInstaller) runs.
        for key in self._REQUIRED_PATH_KEYS:
            default = {"data_dir": "./data", "export_dir": "./export", "log_dir": "./logs"}[key]
            paths[key] = os.path.join(self.base_dir, paths.get(key, default))
            os.makedirs(paths[key], exist_ok=True)

    # ------------------------------------------------------------------
    # Public accessor
    # ------------------------------------------------------------------
    def get(self, path: str, default: Any = None) -> Any:
        """
        Retrieve a configuration value using dotted notation.

        Example:
            ``config.get("gui.window_size", [1500, 900])``
        """
        value: Any = self.config
        for key in path.split("."):
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
