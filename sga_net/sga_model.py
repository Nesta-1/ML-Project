"""
SGA-RoBERTa: Sentiment-Gated Attention Network
-------------------------------------------------
Main model. Uses a SOFT-GATING mechanism conditioned on sentiment features
to modulate RoBERTa feature relevance via residual connection.

Architecture:
    RoBERTa → CLS token output (B, 768)
    
    Gate Network:
        sentiment_features (B, 4)  [compound, pos, neg, neu]
        → Linear(4 → 128) → ReLU → LayerNorm
        → Linear(128 → 768) → Sigmoid
        → gate_vector (B, 768)
    
    Soft gating with residual:
        gated = cls_output + α * (gate_vector * cls_output)
        (α is a learnable scalar, initialized to 0.1)
    
    Classifier:
        Dropout(0.1) → Linear(768 → 6)
"""

import torch
import torch.nn as nn
from transformers import RobertaModel

NUM_CLASSES = 6
MODEL_NAME = "roberta-base"
HIDDEN_DIM = 768
GATE_HIDDEN = 128
SENTIMENT_DIM = 4  # compound, pos, neg, neu


class SentimentGateNetwork(nn.Module):
    """
    Gate Network: transforms 4-dim sentiment features into a 768-dim gate vector.
    
    Linear(4 → 128) → ReLU → LayerNorm → Linear(128 → 768) → Sigmoid
    """

    def __init__(self, input_dim: int = SENTIMENT_DIM, gate_hidden: int = GATE_HIDDEN, output_dim: int = HIDDEN_DIM):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, gate_hidden),
            nn.ReLU(),
            nn.LayerNorm(gate_hidden),
            nn.Linear(gate_hidden, output_dim),
            nn.Sigmoid(),
        )
        # Initialize weights
        for module in self.gate:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, sentiment_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sentiment_features: (batch, 4) — [compound, pos, neg, neu]
        
        Returns:
            gate_vector: (batch, 768) values in [0, 1] via Sigmoid
        """
        return self.gate(sentiment_features)


class SentimentGatedRoberta(nn.Module):
    """
    SGA-RoBERTa: Sentiment-Gated Attention Network.
    
    Uses SOFT GATING with residual connection:
        output = cls_output + α * (gate_vector ⊙ cls_output)
    
    This preserves the original representation while allowing sentiment
    to selectively modulate features. α is learned during training.
    """

    def __init__(self, num_labels: int = NUM_CLASSES, model_name: str = MODEL_NAME):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.gate_network = SentimentGateNetwork(
            input_dim=SENTIMENT_DIM, gate_hidden=GATE_HIDDEN, output_dim=HIDDEN_DIM
        )
        # Learnable gate strength — starts small (0.1) so model initially
        # relies on RoBERTa features and gradually learns gating
        self.gate_alpha = nn.Parameter(torch.tensor(0.1))
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(HIDDEN_DIM, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

        # Initialize classifier
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
        Forward pass with sentiment-gated feature modulation.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            sentiment_score: (batch, 4) sentiment features [compound, pos, neg, neu]
            labels: (batch,) optional

        Returns:
            dict with 'logits', 'loss', 'gate_values', 'gate_alpha'
        """
        # RoBERTa encoding — use CLS token (pretrained) instead of pooler (random)
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch, 768)

        # Prepare sentiment input
        if sentiment_score is None:
            sentiment_score = torch.full(
                (cls_output.size(0), SENTIMENT_DIM), 0.5, device=cls_output.device
            )
        elif sentiment_score.dim() == 1:
            # Backward compat: single scalar → expand to 4-dim
            sentiment_score = sentiment_score.unsqueeze(1).expand(-1, SENTIMENT_DIM)
        
        # Ensure 4-dim input
        if sentiment_score.size(-1) == 1:
            sentiment_score = sentiment_score.expand(-1, SENTIMENT_DIM)

        # Gate computation
        gate_vector = self.gate_network(sentiment_score)  # (batch, 768)

        # SOFT GATING with residual connection
        # output = cls + α * (gate ⊙ cls)
        # This preserves original features and adds modulated features
        gated_embedding = cls_output + self.gate_alpha * (gate_vector * cls_output)

        # Classification
        gated_embedding = self.dropout(gated_embedding)
        logits = self.classifier(gated_embedding)  # (batch, 6)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {
            "logits": logits,
            "loss": loss,
            "gate_values": gate_vector,  # For visualization
            "gate_alpha": self.gate_alpha.item(),  # Current gate strength
        }


if __name__ == "__main__":
    # Quick sanity check
    model = SentimentGatedRoberta()
    total = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total:,}")
    print(f"Gate alpha (init): {model.gate_alpha.item():.4f}")

    dummy_ids = torch.ones(2, 128, dtype=torch.long)
    dummy_mask = torch.ones(2, 128, dtype=torch.long)
    dummy_sent = torch.tensor([[0.3, 0.1, 0.6, 0.3], [0.8, 0.4, 0.1, 0.5]])
    dummy_labels = torch.tensor([0, 3])

    out = model(dummy_ids, dummy_mask, sentiment_score=dummy_sent, labels=dummy_labels)
    print(f"Logits shape: {out['logits'].shape}")
    print(f"Loss: {out['loss'].item():.4f}")
    print(f"Gate values shape: {out['gate_values'].shape}")
    print(f"Gate alpha: {out['gate_alpha']:.4f}")
    print("✓ SentimentGatedRoberta forward pass OK")
