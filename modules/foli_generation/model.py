
from __future__ import annotations

from typing import Optional

import torch

from .config import FoliGenConfig


class AudioLDM2Model:

    def __init__(self, config: Optional[FoliGenConfig] = None):
        self.config = config or FoliGenConfig()
        self.pipe = None
        self.device = None
        self.sampling_rate = self.config.sampling_rate or 16000

    def load(self):
        from diffusers import AudioLDM2Pipeline
        import logging
        
        logger = logging.getLogger("audiobook.foli_generation")

        # Determine device
        if self.config.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = self.config.device

        # Choose dtype
        dtype = torch.float16 if (self.device == "cuda" and self.config.use_fp16) else torch.float32

        logger.info(f"Загрузка модели AudioLDM2: {self.config.repo_id}")
        logger.info(f"Это может занять несколько минут при первом запуске (загрузка ~2-3 ГБ)...")
        logger.info(f"Устройство: {self.device}, Тип данных: {dtype}")

        # Load pipeline
        self.pipe = AudioLDM2Pipeline.from_pretrained(
            self.config.repo_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

        logger.info("Модель загружена, перемещение на устройство...")

        # Move to device
        self.pipe = self.pipe.to(self.device)
        
        # Compatibility: some checkpoints may provide GPT2Model instead of GPT2LMHeadModel
        try:
            from transformers import GPT2LMHeadModel, AutoModelForCausalLM
            language_model = getattr(self.pipe, "language_model", None)
            if language_model is not None and language_model.__class__.__name__ == "GPT2Model":
                lm_name = getattr(language_model, "name_or_path", "gpt2")
                try:
                    upgraded_lm = GPT2LMHeadModel.from_pretrained(lm_name)
                except Exception:
                    upgraded_lm = AutoModelForCausalLM.from_pretrained(lm_name)
                self.pipe.language_model = upgraded_lm.to(self.device)
        except Exception as e:
            logger.warning(f"Не удалось обновить language_model до GPT2LMHeadModel: {e}")
        
        logger.info("Модель AudioLDM2 готова к использованию")

        # AudioLDM2 outputs audios at 16 kHz typically
        try:
            # Some versions expose feature extractor with sampling rate
            self.sampling_rate = int(getattr(self.pipe, 'feature_extractor').sampling_rate)
        except Exception:
            self.sampling_rate = self.sampling_rate or 16000

    def is_loaded(self) -> bool:
        return self.pipe is not None


