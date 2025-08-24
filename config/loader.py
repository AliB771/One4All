# -*- coding: utf-8 -*-
import os
import yaml

class ConfigLoader:
    def __init__(self, config_dir: str):
        """
        config_dir: path to the directory containing yaml files
        """
        self.config_dir = config_dir
        self._configs = self._load_all_configs()

    def _load_all_configs(self):
        """
        Load all YAML files inside the config directory
        Returns a dict with filenames (without extension) as keys
        """
        configs = {}
        for file in os.listdir(self.config_dir):
            if file.endswith((".yml", ".yaml")):
                key = os.path.splitext(file)[0]  # filename without extension
                path = os.path.join(self.config_dir, file)
                with open(path, "r", encoding="utf-8") as f:
                    configs[key] = yaml.safe_load(f) or {}
        return configs

    def get(self, name: str):
        """Get config data by name (filename without extension)"""
        return self._configs.get(name, {})

    def all(self):
        """Return all configs as a dict"""
        return self._configs
