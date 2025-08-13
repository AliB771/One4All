#!/usr/bin/env python3
"""
Pipeline execution script
"""

import sys
from pathlib import Path

# Add src to path  
sys.path.append(str(Path(__file__).parent.parent / "src"))

from pipeline.combined_pipeline import CombinedPipeline
from utils.config_loader import load_config

def main():
    # Load configurations
    router_config = load_config("config/router_config.yaml")
    expert_config = load_config("config/expert_config.yaml")
    pipeline_config = load_config("config/pipeline_config.yaml")
    
    # Initialize pipeline
    pipeline = CombinedPipeline(
        router_config=router_config,
        expert_config=expert_config,
        pipeline_config=pipeline_config
    )
    
    # Example input
    sample_input = "This is a sample input for routing"
    
    # Run pipeline
    result = pipeline.predict(sample_input)
    
    print(f"Pipeline result: {result}")

if __name__ == "__main__":
    main()
