# -*- coding: utf-8 -*-

from config import all_configs, get_config
from datasets import load_dataset
from transformers import AutoTokenizer


class DatasetModule:
    """
    DatasetModule handles loading and tokenizing datasets when train/validation/test
    are already split into separate files.
    """
    def __init__(self, config: Dict[str, Any]):
        # Load the dataset configuration from router_config
        self.config = config.get("Dataset")
        
        # Extract tokenizer-specific configuration
        self.Tokenizer_config = self.config.get("Tokenizer")
        
        # Path to the dataset splits (train/validation/test)
        self.dataset_path = self.config.get("DatasetPath")
        
        # Initialize the HuggingFace tokenizer using the configured tokenizer name
        self.tokenizer = AutoTokenizer.from_pretrained(self.Tokenizer_config.get("TokenizerName"))

    def load_and_prepare(self):
        """
        Load each split (train/validation/test), tokenize them, and return datasets.
        Returns:
            train_ds, valid_ds, test_ds
        """
        # Load the train split from the parquet file
        train_ds = load_dataset("parquet", data_files=self.dataset_paths["Train"], split="train")
        
        # Load the validation split from the parquet file
        valid_ds = load_dataset("parquet", data_files=self.dataset_paths["Validation"], split="train")
        
        # Load the test split from the parquet file
        test_ds = load_dataset("parquet", data_files=self.dataset_paths["Test"], split="train")

        # Tokenize the train split
        train_ds = train_ds.map(
            self.tokenize_fn,
            batched=self.tokenizer.get("Batched"),  # Whether to tokenize in batch mode
            remove_columns=self.tokenizer.get("RemoveColumns"),  # Columns to remove after tokenization
            num_proc=self.tokenizer.get("NumProc")  # Number of processes for parallel tokenization
        )

        # Tokenize the validation split
        valid_ds = valid_ds.map(
            self.tokenize_fn,
            batched=self.tokenizer.get("Batched"),
            remove_columns=self.tokenizer.get("RemoveColumns"),
            num_proc=self.tokenizer.get("NumProc")
        )

        # Tokenize the test split
        test_ds = test_ds.map(
            self.tokenize_fn,
            batched=self.tokenizer.get("Batched"),
            remove_columns=self.tokenizer.get("RemoveColumns"),
            num_proc=self.tokenizer.get("NumProc")
        )

        # Return the tokenized datasets
        return train_ds, valid_ds, test_ds

    def tokenize_fn(self, examples):
        """
        Tokenize the 'chunk' column of the dataset using the HuggingFace tokenizer.
        Args:
            examples (dict): A batch of examples from the dataset
        Returns:
            dict: Tokenized input ready for model consumption
        """
        return self.tokenizer(
            examples["chunk"],
            truncation=True,  # Truncate sequences to max_length
            padding=self.tokenizer.get("Padding"),  # Use padding strategy from config
            max_length=self.tokenizer.get("MaxLength")  # Maximum token length
        )
