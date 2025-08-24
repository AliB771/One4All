import torch
from typing import Dict
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType, PeftModel

class ModelModule:
    """
    ModelModule builds a new model or loads trained weights from a directory.
    """
    def __init__(self, config: Dict[str, Any]):
        self.base_model = config.get("BASEMODEL")
        self.num_labels = config.get("NumLabels", 4)
        self.device_map = config.get("DeviceMap", "auto")

    def build_model(self, load_from=None):
        """
        If load_from is provided, load trained weights.
        Otherwise, build a new model with LoRA.
        """
        if load_from is not None:
            model = AutoModelForSequenceClassification.from_pretrained(self.base_model, num_labels=self.num_labels)
            model = PeftModel.from_pretrained(model, load_from)
            model.eval()
            return model

        # Quantization config (4-bit)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        # Load base model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels=self.num_labels,
            quantization_config=bnb_config,
            device_map=self.device_map
        )

        model = prepare_model_for_kbit_training(model)

        # LoRA configuration
        lora_config = LoraConfig(
            r=8, lora_alpha=32, lora_dropout=0.1, bias="none",
            task_type=TaskType.SEQ_CLS, target_modules=["query","value"]
        )

        model = get_peft_model(model, lora_config)
        return model
