"""
LoRA-adapted VideoMAE backbone for UCF-Crime fine-tuning.
Uses HuggingFace PEFT to inject low-rank adapters into Q and V projections.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import VideoMAEModel, VideoMAEImageProcessor
from peft import LoraConfig, get_peft_model

# Same backbone constants as stage1
BACKBONE_NAME = "MCG-NJU/videomae-base"
CLIP_LEN = 16
IMG_SIZE = 224
HIDDEN_DIM = 768
NUM_TOKENS_NO_CLS = 1568


class LoRAVideoMAE(nn.Module):
    """
    VideoMAE-Base with LoRA adapters on Q, V projections.
    Classification head: Linear(768, num_classes).
    """

    def __init__(
        self,
        num_classes: int = 14,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[list] = None,
        use_gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.backbone = VideoMAEModel.from_pretrained(BACKBONE_NAME)
        self.processor = VideoMAEImageProcessor.from_pretrained(BACKBONE_NAME)

        if use_gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        if target_modules is None:
            target_modules = ["query", "value"]

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        self.backbone = get_peft_model(self.backbone, lora_config)
        self.classifier = nn.Linear(HIDDEN_DIM, num_classes)
        self.backbone.print_trainable_parameters()

    def _normalize(self, clips_bcthw: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(
            self.processor.image_mean, dtype=torch.float32, device=clips_bcthw.device
        ).view(1, 3, 1, 1, 1)
        std = torch.tensor(
            self.processor.image_std, dtype=torch.float32, device=clips_bcthw.device
        ).view(1, 3, 1, 1, 1)
        return (clips_bcthw - mean) / std

    def extract_features(self, clips_bcthw: torch.Tensor) -> torch.Tensor:
        """Extract 768-dim global features (no classifier head)."""
        pixel_values = self._normalize(clips_bcthw)
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4).contiguous()  # -> (B, T, C, H, W)

        tokens = self.backbone(pixel_values=pixel_values).last_hidden_state
        if tokens.shape[1] == NUM_TOKENS_NO_CLS + 1:
            tokens = tokens[:, 1:, :]

        return tokens.mean(dim=1)

    def forward(self, clips_bcthw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, features)."""
        features = self.extract_features(clips_bcthw)
        return self.classifier(features), features

    def save_lora_checkpoint(self, save_dir: str, epoch: int, extra_meta: dict = None) -> str:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "lora_state_dict": {
                k: v.cpu() for k, v in self.backbone.state_dict().items()
                if "lora_" in k or "classifier" in k
            },
            "classifier_state_dict": self.classifier.state_dict(),
            "meta": {
                "epoch": epoch,
                "lora_rank": self.backbone.peft_config["default"].r,
                "lora_alpha": self.backbone.peft_config["default"].lora_alpha,
                "target_modules": list(self.backbone.peft_config["default"].target_modules),
                "num_classes": self.num_classes,
                "backbone": BACKBONE_NAME,
                **(extra_meta or {}),
            },
        }

        out_path = str(save_path / f"lora_ckpt_epoch{epoch}.pt")
        torch.save(checkpoint, out_path)
        return out_path

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
        use_gradient_checkpointing: bool = False,
    ) -> "LoRAVideoMAE":
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        meta = ckpt["meta"]

        model = cls(
            num_classes=meta["num_classes"],
            lora_rank=meta["lora_rank"],
            lora_alpha=meta["lora_alpha"],
            target_modules=meta["target_modules"],
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        # Load LoRA weights
        model.backbone.load_state_dict(ckpt["lora_state_dict"], strict=False)
        # Load classifier
        model.classifier.load_state_dict(ckpt["classifier_state_dict"])

        model.to(device)
        model.eval()
        return model