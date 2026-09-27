# Copyright (c) 2026 ysnrfd
"""Post-install verification: core imports intact + siglip processor reachable."""

import soundfile as sf
import torch
import transformers

print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("soundfile", sf.__version__)

from transformers import AutoProcessor

proc = AutoProcessor.from_pretrained("hf-internal-testing/tiny-random-SiglipModel")
print("siglip processor:", type(proc).__name__)
print("has tokenizer:", hasattr(proc, "tokenizer"), "| has image_processor:", hasattr(proc, "image_processor"))
