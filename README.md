# ML Router-Expert Project

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
