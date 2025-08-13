#!/usr/bin/env python3
"""
Router training script
"""

import yaml
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from router.trainer import RouterTrainer
from utils.config_loader import load_config

def main():
    parser = argparse.ArgumentParser(description="Train Router Model")
    parser.add_argument("--config", default="config/router_config.yaml", 
                       help="Path to router config file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize trainer
    trainer = RouterTrainer(config)
    
    # Start training
    trainer.train()
    
    print("Router training completed!")

if __name__ == "__main__":
    main()
