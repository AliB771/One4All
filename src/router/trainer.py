import os
import numpy as np
from litgpt import Trainer, TrainingArguments
from config.settings import BATCH_SIZE, GRAD_ACCUM, EPOCHS, LR, FP16

class TrainingModule:
    """
    Only imported when training is required.
    """
    def __init__(self, model, train_ds, valid_ds, tokenizer, output_dir="./trained_model"):
        self.model = model
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def compute_metrics(self, pred):
        from sklearn.metrics import precision_recall_fscore_support
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
        return {"precision": precision, "recall": recall, "f1": f1}

    def train(self):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            num_train_epochs=EPOCHS,
            learning_rate=LR,
            logging_steps=50,
            logging_dir=os.path.join(self.output_dir,"logs"),
            evaluation_strategy="no",
            save_strategy="epoch",
            fp16=FP16
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_ds,
            eval_dataset=self.valid_ds,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )

        trainer.train()
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print("✅ Model trained and saved successfully!")
