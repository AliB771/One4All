import torch

class Router:
    """
    Multi-model Router for inference.
    """
    def __init__(self):
        self.models = {}

    def register_model(self, name, model, tokenizer):
        """
        Register multiple models by name.
        """
        self.models[name] = {"model": model, "tokenizer": tokenizer}

    def predict(self, name, text):
        """
        Send input text to the chosen registered model.
        """
        if name not in self.models:
            raise ValueError(f"Model '{name}' not registered.")

        model = self.models[name]["model"]
        tokenizer = self.models[name]["tokenizer"]

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = outputs.logits.argmax(dim=-1).item()
        return prediction
