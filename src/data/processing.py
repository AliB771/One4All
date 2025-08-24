import sqlite3
import pandas as pd
from typing import List, Optional, Dict, Any, Generator
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import json
from src.utils.logging import setup_logger
from config import all_configs, get_config
import pprint
import logging
from tqdm import tqdm

logger = setup_logger(
    name=__name__,
    log_file="logs/processing.log",  # optional
    level=logging.INFO
)



class SQLiteDatasetLoader:
    """
    Loader for SQLite datasets, with support for chunking, label encoding,
    JSONL export, and HuggingFace Dataset conversion.
    """

    def __init__(
        self,
        db_path: str,
        config : Dict
    ):
        self.db_path = os.path.join(db_path, config.get("Path"))
        self.table_name = config.get("TableName")
        self.text_column = config.get("TextColumn")
        self.label_column = config.get("LabelColumn")
        self.title_column = config.get("TitleColumn")
        self.chunk_column = config.get("ChunkColumn")
        self.max_chunk_word = 1000
        self.chunk_repeat_title = config.get("ChunkRepeatTitle")
        self.batch_size = config.get("BatchSize")
        self.min_chunk_words = config.get("MinChunkWords")
        self.label = config.get("Label")
        self.AllowedCategories = config.get("AllowedCategories")
        self.use_chunks = config.get("UseChunks")
        self.Shuffle = config.get("Shuffle")
        self.Name = config.get("Name")

        self.conn = sqlite3.connect(self.db_path)
        self.label_encoder = LabelEncoder()
        logger.info(f"Connected to database: {self.db_path}")


    def get_category_counts(self) -> dict[str, int]:
        query = f"""
            SELECT {self.label_column}, COUNT(*) as count
            FROM {self.table_name}
            GROUP BY {self.label_column}
        """
        df = pd.read_sql(query, self.conn)
        counts = dict(zip(df[self.label_column], df["count"]))
        logger.info(f"Category counts: {counts}")
        return counts

    def iter_rows(
        self,
        allowed_categories: Optional[List[str]] = None,
        max_batch: Optional[int] = None
    ) -> Generator[pd.DataFrame, None, None]:
        """Yield batches of rows from SQLite table."""
        offset = 0
        batch_count = 0
        if allowed_categories == None:
            allowed_categories = self.AllowedCategories

        while True:
            query = f"SELECT * FROM {self.table_name} "
            params: List[Any] = []

            if allowed_categories:
                placeholders = ",".join(["?"] * len(allowed_categories))
                query += f"WHERE {self.label_column} IN ({placeholders}) "
                params = allowed_categories

            query += f"LIMIT {self.batch_size} OFFSET {offset}"
            df = pd.read_sql(query, self.conn, params=params)

            if df.empty:
                break

            yield df
            offset += self.batch_size
            batch_count += 1

            if max_batch and batch_count >= max_batch:
                break


    def _explode_chunks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Explode chunk column into individual rows and attach title if needed."""
        try:
            df[self.chunk_column] = df[self.chunk_column].apply(eval)
        except Exception as e:
            logger.warning(f"Failed to eval chunks: {e}")
            df[self.chunk_column] = df[self.chunk_column].apply(lambda x: [])

        df = df.explode(self.chunk_column)
        df.rename(columns={self.chunk_column: "chunk"}, inplace=True)

        if self.chunk_repeat_title and self.title_column in df.columns:
            df["chunk"] = df.apply(lambda row: f"{row[self.title_column]}\n\n{row['chunk']}", axis=1)

        df = df[df["chunk"].apply(lambda x: len(str(x).split()) >= self.min_chunk_words)]
        keep_cols = [self.label_column, self.title_column, "chunk"]
        return df[keep_cols]

    def _chunk_full_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Split full text into chunks based on max_chunk_word."""
        def split_text(text: str, title: str) -> List[str]:
            words = text.split()
            chunks = []
            for i in range(0, len(words), self.max_chunk_word):
                chunk = " ".join(words[i:i + self.max_chunk_word])
                if self.chunk_repeat_title:
                    chunk = f"{title}\n\n{chunk}"
                if len(chunk.split()) >= self.min_chunk_words:
                    chunks.append(chunk)
            return chunks

        df[self.chunk_column] = df.apply(lambda row: split_text(row[self.text_column], row[self.title_column]), axis=1)
        df = df.explode(self.chunk_column)
        df.rename(columns={self.chunk_column: "chunk"}, inplace=True)
        keep_cols = [self.label_column, self.title_column, "chunk"]
        return df[keep_cols]

    # --- Public processing methods ---
    def load_all_encoded_dataset(
        self,
        allowed_categories: Optional[List[str]] = None,
        max_batch: Optional[int] = None,
        use_chunks: bool = False,
        label: Optional[int] = None
    ) -> Dataset:
        """Load all data as HuggingFace Dataset, optionally with chunking and label override."""
        full_df = pd.DataFrame()
        if allowed_categories == None:
            allowed_categories = self.AllowedCategories
        
        if label == None:
            label = self.label

        for batch_df in tqdm(self.iter_rows(allowed_categories=allowed_categories, max_batch=max_batch)):
            if use_chunks and self.chunk_column:
                batch_df = self._explode_chunks(batch_df)
            else:
                batch_df = self._chunk_full_text(batch_df)
            full_df = pd.concat([full_df, batch_df], ignore_index=True)

        label_list = self.label_encoder.fit_transform(full_df[self.label_column])
        if label is not None:
            label_list = [label] * len(label_list)

        full_df["label"] = label_list
        dataset = Dataset.from_pandas(full_df[[self.label_column, self.title_column, "chunk", "label"]])
        logger.info(f"Loaded dataset with {len(full_df)} samples.")
        return dataset

    def export_jsonl(
        self,
        output_path: str,
        allowed_categories: Optional[List[str]] = None,
        max_batch: Optional[int] = None
    ):
        """Export dataset to JSONL format."""
        with open(output_path, "w", encoding="utf-8") as f_out:
            for batch_df in self.iter_rows(allowed_categories=allowed_categories, max_batch=max_batch):
                for _, row in batch_df.iterrows():
                    title = row[self.title_column]
                    text = row[self.text_column]

                    sections = None
                    if "sections" in row and row["sections"]:
                        try:
                            sections = json.loads(row["sections"])
                        except Exception as e:
                            logger.warning(f"Error parsing sections JSON: {e}")

                    if sections and isinstance(sections, list) and len(sections) > 0:
                        for section in sections:
                            question = section.get("question", "").strip()
                            answer = section.get("answer", "").strip()
                            if not question or not answer:
                                continue
                            input_text = f"{title}\n\n{question}"
                            record = {"input": input_text, "output": answer}
                            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    else:
                        f_out.write(json.dumps({"input": title, "output": text}, ensure_ascii=False) + "\n")
        logger.info(f"Exported data to {output_path}")

    def save_label_encoder(self, path: str = "label_encoder.pkl"):
        """Save label encoder to disk."""
        if path == "label_encoder.pkl":
            path = os.path.basename(self.db_path) + "_label_encoder.pkl"
        joblib.dump(self.label_encoder, path)
        logger.info(f"Saved label encoder to {path}")

    def close(self):
        """Close SQLite connection."""
        if self.conn:
            self.conn.close()
            logger.info(f"Closed database connection: {self.db_path}")



