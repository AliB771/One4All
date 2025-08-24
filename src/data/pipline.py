import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List
from datasets import Dataset, concatenate_datasets, load_dataset
from src.data.processing import SQLiteDatasetLoader
from config import all_configs, get_config
from src.utils.logging import setup_logger


logger = setup_logger(
    name=__name__,
    log_file="logs/pipeline.log",  
    level=logging.INFO
)


class ConcurrentDataPipeline:
    """
    Pipeline that processes multiple classes and databases concurrently,
    merges all processed datasets into one final dataset.
    """
    def __init__(self, output_dir: str = None, max_workers: int = 3):
        if output_dir == None:
            self.config = get_config("router_config").get('DataProcessing')
            self.output_dir = self.config.get("OutputDir")
        os.makedirs(self.output_dir, exist_ok=True)
        self.max_workers = max_workers
        self.parquet_files: List[str] = []

    def _process_single_class(self, class_name) -> str:

        logger.info(f"Starting processing class '{class_name}' ")
        
        db_path = get_config("router_config").get('DataProcessing').get("BasePath")
        db_path = os.path.join(db_path, "raw")

        config = get_config("router_config").get('DataProcessing').get(class_name)
        
        loader = SQLiteDatasetLoader(
            db_path = db_path,
            config = config
        )

        dataset = loader.load_all_encoded_dataset()
        loader.close()

        output_path = os.path.join(self.output_dir, f"{config.get('Name')}.parquet")
        dataset.to_parquet(output_path)
        logger.info(f"Finished processing '{config.get('Name')}', saved to: {output_path}")
        return output_path

    def run_pipeline(self) -> Dataset:
        """Run concurrent processing and merge all datasets."""
        logger.info("🚀 Starting concurrent data pipeline...")

        # 1. Process all classes concurrently
        futures = []
        class_list = self.config.get("ClassList")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for i, class_name in enumerate(class_list):
                futures.append(
                    executor.submit(self._process_single_class, class_name)
                )

            # Collect results
            for future in as_completed(futures):
                parquet_file = future.result()
                if parquet_file:
                    self.parquet_files.append(parquet_file)

        if not self.parquet_files:
            logger.warning("No datasets were processed. Exiting pipeline.")
            return Dataset.from_dict({})

        logger.info("Loading and combining parquet datasets...")
        datasets = [load_dataset("parquet", data_files=f, split="train") for f in self.parquet_files]
        combined = concatenate_datasets(datasets)
        combined = combined.shuffle()

        logger.info("Splitting dataset into train/valid/test...")
        splits = combined.train_test_split(test_size = self.config.get("TestSize"))   
        test_valid_split = splits["test"].train_test_split(test_size = self.config.get("ValidationSize"))  

        final_dataset = {
            "training": splits["train"],
            "validation": test_valid_split["train"],
            "test": test_valid_split["test"]
        }
        BasePath = get_config("router_config").get('DataProcessing').get("BasePath")
        for split_name, dataset in final_dataset.items():
            save_path = os.path.join(BasePath, split_name, "router")
            os.makedirs(save_path, exist_ok=True)
            split_path = os.path.join(save_path, f"{split_name}_dataset.parquet")
            dataset.to_parquet(split_path)
            logger.info(f"{split_name.capitalize()} dataset saved at: {split_path}")

        logger.info("Concurrent data pipeline completed successfully.")

        return combined


if __name__ == "__main__":
    pipeline = ConcurrentDataPipeline(max_workers=3)
    final_dataset = pipeline.run_pipeline()
    print(final_dataset)
