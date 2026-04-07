"""
Baseline 1: Vanilla RoBERTa
----------------------------
Standard RobertaForSequenceClassification with 6 output classes.
pooler_output → Linear(768 → 6)
"""

import torch
import torch.nn as nn
from transformers import RobertaForSequenceClassification, RobertaConfig

NUM_CLASSES = 6
MODEL_NAME = "roberta-base"


class VanillaRoberta(nn.Module):
    """
    Vanilla RoBERTa classifier.
    Uses HuggingFace RobertaForSequenceClassification directly.
    """

    def __init__(self, num_labels: int = NUM_CLASSES, model_name: str = MODEL_NAME):
        super().__init__()
        self.roberta = RobertaForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sentiment_score: torch.Tensor = None,  # Unused, kept for API compat
        labels: torch.Tensor = None,
    ):
        """
        Forward pass.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            sentiment_score: ignored (API compatibility)
            labels: (batch,) optional, for loss computation

        Returns:
            dict with 'logits' and optionally 'loss'
        """
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return {
            "logits": outputs.logits,
            "loss": outputs.loss if labels is not None else None,
            "gate_values": None,  # No gate in vanilla model
        }


if __name__ == "__main__":
    # Quick sanity check
    model = VanillaRoberta()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    dummy_ids = torch.ones(2, 128, dtype=torch.long)
    dummy_mask = torch.ones(2, 128, dtype=torch.long)
    dummy_labels = torch.tensor([0, 3])
    
    out = model(dummy_ids, dummy_mask, labels=dummy_labels)
    print(f"Logits shape: {out['logits'].shape}")
    print(f"Loss: {out['loss'].item():.4f}")
    print("✓ VanillaRoberta forward pass OK")
