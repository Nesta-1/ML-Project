"""
Baseline 2: Concatenation Fusion
----------------------------------
RoBERTa CLS output (768) + sentiment_features (4) → concat (772) → Linear(772 → 6)
"""

import torch
import torch.nn as nn
from transformers import RobertaModel

NUM_CLASSES = 6
MODEL_NAME = "roberta-base"
HIDDEN_DIM = 768
SENTIMENT_DIM = 4  # compound, pos, neg, neu


class ConcatFusionRoberta(nn.Module):
    """
    RoBERTa + Sentiment Concatenation model.
    
    Architecture:
        CLS output (768) + sentiment_features (4) = 772
        → Linear(772 → num_labels)
    """

    def __init__(self, num_labels: int = NUM_CLASSES, model_name: str = MODEL_NAME):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(HIDDEN_DIM + SENTIMENT_DIM, num_labels)  # 772 → 6
        self.loss_fn = nn.CrossEntropyLoss()

        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sentiment_score: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        """
        Forward pass.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            sentiment_score: (batch, 4) sentiment features [compound, pos, neg, neu]
            labels: (batch,) optional

        Returns:
            dict with 'logits', 'loss', 'gate_values'
        """
        # Use CLS token (pretrained) instead of pooler_output (random for RoBERTa)
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch, 768)

        # Ensure sentiment_score has correct shape
        if sentiment_score is None:
            sentiment_score = torch.zeros(cls_output.size(0), SENTIMENT_DIM, device=cls_output.device)
        elif sentiment_score.dim() == 1:
            # Backward compat: single scalar → expand to 4-dim
            sentiment_score = sentiment_score.unsqueeze(1).expand(-1, SENTIMENT_DIM)
        if sentiment_score.size(-1) == 1:
            sentiment_score = sentiment_score.expand(-1, SENTIMENT_DIM)

        # Concatenate: (batch, 772)
        fused = torch.cat([cls_output, sentiment_score], dim=1)
        fused = self.dropout(fused)
        logits = self.classifier(fused)  # (batch, 6)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {
            "logits": logits,
            "loss": loss,
            "gate_values": None,  # No gate in concat model
        }


if __name__ == "__main__":
    # Quick sanity check
    model = ConcatFusionRoberta()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    dummy_ids = torch.ones(2, 128, dtype=torch.long)
    dummy_mask = torch.ones(2, 128, dtype=torch.long)
    dummy_sent = torch.tensor([[0.3, 0.1, 0.6, 0.3], [0.8, 0.4, 0.1, 0.5]])
    dummy_labels = torch.tensor([0, 3])

    out = model(dummy_ids, dummy_mask, sentiment_score=dummy_sent, labels=dummy_labels)
    print(f"Logits shape: {out['logits'].shape}")
    print(f"Loss: {out['loss'].item():.4f}")
    print("✓ ConcatFusionRoberta forward pass OK")
