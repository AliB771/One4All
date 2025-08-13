#!/usr/bin/env python3
"""
Script to create ML Router-Expert project structure
"""

import os
import sys
from pathlib import Path

def create_directory_structure():
    """Create the complete directory structure for ML Router-Expert project"""
    
    project_name = "ml-router-expert-project"
    
    # Define the directory structure
    directories = [
        "",  # Root directory
        "config",
        "data/raw",
        "data/processed", 
        "data/training",
        "data/validation",
        "models/router/checkpoints",
        "models/router/trained",
        "models/router/artifacts",
        "models/experts/expert_1/lora_weights",
        "models/experts/expert_1/checkpoints",
        "models/experts/expert_2",
        "src/router",
        "src/experts", 
        "src/pipeline",
        "src/core",
        "src/utils",
        "scripts",
        "tests/test_router",
        "tests/test_experts",
        "tests/test_pipeline", 
        "tests/test_utils",
        "notebooks",
        "api/routes",
        "api/schemas",
        "deployment/docker",
        "deployment/kubernetes",
        "deployment/cloud",
        "docs"
    ]
    
    # Create directories
    print(f"Creating project structure for: {project_name}")
    base_path = Path(project_name)
    
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {dir_path}")
    
    return base_path

def create_init_files(base_path):
    """Create __init__.py files for Python packages"""
    
    init_files = [
        "config/__init__.py",
        "src/__init__.py",
        "src/router/__init__.py", 
        "src/experts/__init__.py",
        "src/pipeline/__init__.py",
        "src/core/__init__.py",
        "src/utils/__init__.py",
        "tests/__init__.py",
        "api/__init__.py"
    ]
    
    print("\nCreating __init__.py files...")
    for init_file in init_files:
        file_path = base_path / init_file
        file_path.write_text("# -*- coding: utf-8 -*-\n")
        print(f"✓ Created: {file_path}")

def create_config_files(base_path):
    """Create configuration files"""
    
    # Router config
    router_config = """# Router Configuration
model:
  name: "router_model"
  architecture: "transformer"
  hidden_size: 768
  num_experts: 8
  top_k: 2

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  save_steps: 1000
  
data:
  train_path: "data/training/router_data.json"
  val_path: "data/validation/router_data.json"
  
paths:
  model_save_path: "models/router/trained/"
  checkpoint_path: "models/router/checkpoints/"
"""

    # Expert config
    expert_config = """# Expert Configuration
experts:
  expert_1:
    name: "expert_domain_1"
    lora_weights_path: "models/experts/expert_1/lora_weights/"
    specialization: "domain_1"
    active: true
    
  expert_2:
    name: "expert_domain_2" 
    lora_weights_path: "models/experts/expert_2/lora_weights/"
    specialization: "domain_2"
    active: true

lora:
  rank: 16
  alpha: 32
  dropout: 0.1
  
base_model:
  name: "base_transformer_model"
  path: "models/base_model/"
"""

    # Pipeline config
    pipeline_config = """# Pipeline Configuration
pipeline:
  name: "router_expert_pipeline"
  max_batch_size: 16
  timeout: 30
  
routing:
  strategy: "top_k"
  k: 2
  threshold: 0.5
  
expert_management:
  lazy_loading: true
  cache_size: 2
  unload_after_idle: 300  # seconds
  
monitoring:
  log_level: "INFO"
  metrics_enabled: true
  performance_tracking: true
"""

    configs = [
        ("config/router_config.yaml", router_config),
        ("config/expert_config.yaml", expert_config), 
        ("config/pipeline_config.yaml", pipeline_config)
    ]
    
    print("\nCreating configuration files...")
    for config_file, content in configs:
        file_path = base_path / config_file
        file_path.write_text(content)
        print(f"✓ Created: {file_path}")

def create_core_files(base_path):
    """Create core Python files with basic structure"""
    
    # Requirements file
    requirements = """torch>=1.9.0
transformers>=4.20.0
datasets>=2.0.0
accelerate>=0.20.0
peft>=0.4.0
yaml>=6.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
fastapi>=0.68.0
uvicorn>=0.15.0
pydantic>=1.8.0
pytest>=6.0.0
black>=22.0.0
flake8>=4.0.0
"""

    # README file
    readme = """# ML Router-Expert Project

A machine learning project implementing a router-based expert system with LoRA weights.

## Project Structure

- `src/router/`: Router model implementation
- `src/experts/`: Expert system with LoRA weights
- `src/pipeline/`: Pipeline integration
- `config/`: Configuration files
- `scripts/`: Training and execution scripts

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Train router:
```bash
python scripts/train_router.py
```

2. Configure experts:
```bash
python scripts/train_expert.py
```

3. Run pipeline:
```bash
python scripts/run_pipeline.py
```

## Configuration

Edit YAML files in `config/` directory to customize:
- Router settings
- Expert configurations  
- Pipeline parameters
"""

    # .gitignore
    gitignore = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Models and data
models/*/checkpoints/
models/*/trained/
data/raw/*
data/processed/*
*.pkl
*.pth
*.bin

# Jupyter Notebook
.ipynb_checkpoints

# Environment
.env
.venv
env/
venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""

    # Setup.py
    setup_py = """from setuptools import setup, find_packages

setup(
    name="ml-router-expert",
    version="0.1.0",
    description="ML Router-Expert System with LoRA",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.20.0", 
        "peft>=0.4.0",
        "pyyaml>=6.0",
    ],
)
"""

    files = [
        ("requirements.txt", requirements),
        ("README.md", readme),
        (".gitignore", gitignore),
        ("setup.py", setup_py)
    ]
    
    print("\nCreating core project files...")
    for file_name, content in files:
        file_path = base_path / file_name
        file_path.write_text(content)
        print(f"✓ Created: {file_path}")

def create_sample_scripts(base_path):
    """Create sample script files"""
    
    # Train router script
    train_router = """#!/usr/bin/env python3
\"\"\"
Router training script
\"\"\"

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
"""

    # Run pipeline script
    run_pipeline = """#!/usr/bin/env python3
\"\"\"
Pipeline execution script
\"\"\"

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
"""

    scripts = [
        ("scripts/train_router.py", train_router),
        ("scripts/run_pipeline.py", run_pipeline)
    ]
    
    print("\nCreating sample scripts...")
    for script_file, content in scripts:
        file_path = base_path / script_file
        file_path.write_text(content)
        file_path.chmod(0o755)  # Make executable
        print(f"✓ Created: {file_path}")

def main():
    """Main function to create the complete project structure"""
    try:
        # Create directory structure
        base_path = create_directory_structure()
        
        # Create Python package files
        create_init_files(base_path)
        
        # Create configuration files
        create_config_files(base_path)
        
        # Create core project files
        create_core_files(base_path)
        
        # Create sample scripts
        create_sample_scripts(base_path)
        
        print(f"\n🎉 Project structure created successfully!")
        print(f"📁 Project location: {base_path.absolute()}")
        print(f"\nNext steps:")
        print(f"1. cd {base_path}")
        print(f"2. python -m venv venv")
        print(f"3. source venv/bin/activate  # Linux/Mac")
        print(f"4. pip install -r requirements.txt")
        print(f"5. Start developing your router and expert models!")
        
    except Exception as e:
        print(f"❌ Error creating project structure: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()