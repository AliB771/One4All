# -*- coding: utf-8 -*-

import os
from config import get_config
from src.router.model import ModelModule
from src.router.trainer import TrainingModule
from src.data.dataset import DatasetModule

class TrainingPipeline:
    """
    TrainingPipeline handles the complete workflow of loading datasets, 
    building or loading a model, and training it.
    """
    def __init__(self, config_name="router_config", output_dir="./trained_model"):
        self.config = get_config(config_name)
        self.dataset_config = self.config.get("Dataset")
        self.model_config = self.config.get("Model")
        self.tokenizer_name = self.dataset_config.get("Tokenizer").get("TokenizerName")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, load_existing_model=None):
        """
        Execute the training pipeline.
        Args:
            load_existing_model (str, optional): Path to previously trained model weights.
        """

        # 1. Load and tokenize datasets
        dataset_module = DatasetModule(self.config)
        train_ds, valid_ds, test_ds = dataset_module.load_and_prepare()
        print(f"✅ Datasets loaded: train={len(train_ds)}, valid={len(valid_ds)}, test={len(test_ds)}")

        # 2. Build or load model
        model_module = ModelModule(self.model_config)
        model = model_module.build_model(load_from=load_existing_model)
        print(f"✅ Model ready: {model.__class__.__name__}")

        # 3. Train the model
        trainer_module = TrainingModule(
            model=model,
            train_ds=train_ds,
            valid_ds=valid_ds,
            tokenizer=dataset_module.tokenizer,
            output_dir=self.output_dir
        )
        trainer_module.train()
        print(f"✅ Training pipeline completed. Model saved at {self.output_dir}")


# Example usage:
if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()  # or pipeline.run(load_existing_model="./trained_model")
