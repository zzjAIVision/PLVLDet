"""
PowerBERT: Domain-Adapted Text Encoder for Power Line Inspection

This module implements the PowerBERT text encoder, which is based on BERT-base-Chinese
and adapted for power line domain terminology through continued pre-training with MLM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from transformers import BertModel, BertTokenizer, BertConfig


class DomainAdaptationLayer(nn.Module):
    """
    Domain adaptation layer for projecting BERT outputs to visual embedding space.
    
    Applies domain-specific transformation followed by dimensional projection
    to align text embeddings with visual features.
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 768,
        output_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Domain-specific linear transformation
        self.domain_linear = nn.Linear(input_dim, hidden_dim)
        self.domain_norm = nn.LayerNorm(hidden_dim)
        self.domain_act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Projection to visual embedding dimension
        self.projection = nn.Linear(hidden_dim, output_dim)
        self.proj_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through domain adaptation layer.
        
        Args:
            x: Input tensor of shape (batch_size, hidden_dim)
            
        Returns:
            Projected tensor of shape (batch_size, output_dim)
        """
        # Domain adaptation
        x = self.domain_linear(x)
        x = self.domain_norm(x)
        x = self.domain_act(x)
        x = self.dropout(x)
        
        # Projection
        x = self.projection(x)
        x = self.proj_norm(x)
        
        return x


class PowerBERT(nn.Module):
    """
    PowerBERT: Domain-adapted text encoder for power line inspection.
    
    Based on BERT-base-Chinese, adapted for power line domain terminology
    through continued pre-training with masked language modeling (MLM).
    
    The encoder produces text embeddings aligned with the visual feature space,
    enabling vision-language fusion for object detection.
    
    Attributes:
        bert: Pre-trained BERT model
        tokenizer: BERT tokenizer with extended vocabulary
        domain_adapter: Domain adaptation and projection layers
    """
    
    def __init__(
        self,
        pretrained_model: str = "bert-base-chinese",
        hidden_size: int = 768,
        output_dim: int = 512,
        dropout: float = 0.1,
        freeze: bool = True,
        max_length: int = 64
    ):
        """
        Initialize PowerBERT.
        
        Args:
            pretrained_model: Name or path of pre-trained BERT model
            hidden_size: Hidden dimension of BERT
            output_dim: Output dimension for visual alignment
            dropout: Dropout rate for domain adapter
            freeze: Whether to freeze BERT parameters
            max_length: Maximum sequence length
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.max_length = max_length
        self.freeze_bert = freeze
        
        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        
        # Domain adaptation layer
        self.domain_adapter = DomainAdaptationLayer(
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            output_dim=output_dim,
            dropout=dropout
        )
        
        # Freeze BERT if specified
        if freeze:
            self._freeze_bert()
            
        # Cache for pre-computed text embeddings
        self._embedding_cache: Dict[str, torch.Tensor] = {}
        
    def _freeze_bert(self):
        """Freeze BERT parameters to stabilize cross-modal alignment."""
        for param in self.bert.parameters():
            param.requires_grad = False
            
    def unfreeze_bert(self):
        """Unfreeze BERT parameters for fine-tuning."""
        for param in self.bert.parameters():
            param.requires_grad = True
        self.freeze_bert = False
        
    def extend_vocabulary(self, additional_tokens: List[str]):
        """
        Extend tokenizer vocabulary with domain-specific tokens.
        
        Args:
            additional_tokens: List of power line domain terms to add
        """
        num_added = self.tokenizer.add_tokens(additional_tokens)
        if num_added > 0:
            self.bert.resize_token_embeddings(len(self.tokenizer))
            print(f"Added {num_added} tokens to vocabulary")
            
    def encode_text(
        self,
        texts: List[str],
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Encode a list of text prompts into embeddings.
        
        Args:
            texts: List of text descriptions
            device: Target device for output tensor
            
        Returns:
            Text embeddings of shape (num_texts, output_dim)
        """
        if device is None:
            device = next(self.parameters()).device
            
        # Tokenize with standard BERT format: [CLS] text [SEP]
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        token_type_ids = encodings.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
            
        # Forward through BERT
        with torch.set_grad_enabled(not self.freeze_bert):
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
        # Extract [CLS] token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)
        
        # Apply domain adaptation
        text_embedding = self.domain_adapter(cls_embedding)  # (batch, output_dim)
        
        return text_embedding
    
    def encode_categories(
        self,
        category_texts: Dict[str, List[str]],
        device: Optional[torch.device] = None,
        use_cache: bool = True
    ) -> torch.Tensor:
        """
        Encode all category text prompts into an embedding matrix.
        
        For each category, randomly selects one text prompt from available
        synonyms during training, or uses the first prompt during inference.
        
        Args:
            category_texts: Dictionary mapping category names to text prompts
            device: Target device for output tensor
            use_cache: Whether to use cached embeddings (for inference)
            
        Returns:
            Text embedding matrix of shape (num_categories, output_dim)
        """
        if device is None:
            device = next(self.parameters()).device
            
        category_names = list(category_texts.keys())
        
        # Check cache
        if use_cache and not self.training:
            cache_key = "_".join(category_names)
            if cache_key in self._embedding_cache:
                return self._embedding_cache[cache_key].to(device)
        
        # Select text prompts
        if self.training:
            # Randomly sample one prompt per category for training
            import random
            texts = [random.choice(prompts) for prompts in category_texts.values()]
        else:
            # Use first prompt for inference
            texts = [prompts[0] for prompts in category_texts.values()]
            
        # Encode texts
        embeddings = self.encode_text(texts, device)
        
        # Cache for inference
        if use_cache and not self.training:
            self._embedding_cache[cache_key] = embeddings.detach().cpu()
            
        return embeddings
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        
    def forward(
        self,
        texts: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through PowerBERT.
        
        Can accept either raw text strings or pre-tokenized inputs.
        
        Args:
            texts: List of text strings (alternative to tokenized inputs)
            input_ids: Tokenized input IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            
        Returns:
            Text embeddings of shape (batch_size, output_dim)
        """
        if texts is not None:
            return self.encode_text(texts)
            
        # Use pre-tokenized inputs
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
            
        with torch.set_grad_enabled(not self.freeze_bert):
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        text_embedding = self.domain_adapter(cls_embedding)
        
        return text_embedding


class PowerBERTForMLM(nn.Module):
    """
    PowerBERT with Masked Language Modeling head for domain adaptation pre-training.
    
    This module is used for continued pre-training of BERT on power line domain
    corpus using the MLM objective.
    """
    
    def __init__(
        self,
        pretrained_model: str = "bert-base-chinese",
        hidden_size: int = 768,
        vocab_size: int = 21128
    ):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.vocab_size = vocab_size
        
        # MLM prediction head
        self.mlm_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, vocab_size)
        )
        
        # Tie weights with input embeddings
        self.mlm_head[-1].weight = self.bert.embeddings.word_embeddings.weight
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for MLM pre-training.
        
        Args:
            input_ids: Input token IDs with some tokens masked
            attention_mask: Attention mask
            labels: Original token IDs for computing loss (masked positions only)
            
        Returns:
            Tuple of (logits, loss) where loss is None if labels not provided
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs.last_hidden_state
        logits = self.mlm_head(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))
            
        return logits, loss


def build_powerbert(cfg: dict, pretrained_path: Optional[str] = None) -> PowerBERT:
    """
    Build PowerBERT model from configuration.
    
    Args:
        cfg: Model configuration dictionary
        pretrained_path: Path to pre-trained PowerBERT weights
        
    Returns:
        Initialized PowerBERT model
    """
    model = PowerBERT(
        pretrained_model=cfg.get("pretrained", "bert-base-chinese"),
        hidden_size=cfg.get("hidden_size", 768),
        output_dim=cfg.get("output_dim", 512),
        dropout=cfg.get("dropout", 0.1),
        freeze=cfg.get("freeze", True),
        max_length=cfg.get("max_length", 64)
    )
    
    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded PowerBERT weights from {pretrained_path}")
        
    return model
