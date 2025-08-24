# -*- coding: utf-8 -*-
import os
from .loader import ConfigLoader
# Initialize loader with current directory
_config_dir = os.path.dirname(__file__)
_loader = ConfigLoader(_config_dir)

# Expose helpers at package level
get_config = _loader.get
all_configs = _loader.all
