from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


class InferenceRunner:
    """Shared config, model loading, prompt generation, and persistence helpers."""

    def __init__(self, config: dict[str, Any], *, config_path: str | Path | None = None) -> None:
        self.config = config
        self.config_path = Path(config_path).resolve() if config_path else None
        self.config_dir = self.config_path.parent if self.config_path else Path.cwd()
        self.tokenizer = None
        self.model = None
        self.device = self._resolve_device(self.config.get("model", {}).get("device", "auto"))

    @classmethod
    def from_config(cls, config_path: str | Path) -> "InferenceRunner":
        path = Path(config_path)
        return cls(load_config(path), config_path=path)

    def load_model(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        model_config = self.config.get("model", {})
        model_name = model_config["model_name"]
        trust_remote_code = bool(model_config.get("trust_remote_code", True))
        local_files_only = bool(model_config.get("local_files_only", False))
        if local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        model_ref = self._resolve_model_ref(model_name, local_files_only)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_ref,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        dtype = self._resolve_dtype(model_config.get("torch_dtype", "float16"))
        if dtype is not None:
            model_kwargs["dtype"] = dtype

        self.model = AutoModelForCausalLM.from_pretrained(model_ref, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()

    def generate_text(self, prompt: str, **generation_overrides: Any) -> str:
        self.load_model()
        assert self.model is not None
        assert self.tokenizer is not None

        generation_config = copy.deepcopy(self.config.get("generation", {}))
        generation_config.update(generation_overrides)
        max_input_tokens = generation_config.pop("max_input_tokens", None)
        max_new_tokens = int(generation_config.get("max_new_tokens", 768))
        if not generation_config.get("do_sample", False):
            generation_config.pop("temperature", None)
            generation_config.pop("top_p", None)
            generation_config.pop("top_k", None)

        max_input_tokens = self._effective_input_length(max_input_tokens, max_new_tokens)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=max_input_tokens is not None,
            max_length=max_input_tokens,
        )
        encoded = encoded.to(self.device)
        prompt_length = encoded["input_ids"].shape[-1]

        if "pad_token_id" not in generation_config and self.tokenizer.pad_token_id is not None:
            generation_config["pad_token_id"] = self.tokenizer.pad_token_id
        if "eos_token_id" not in generation_config and self.tokenizer.eos_token_id is not None:
            generation_config["eos_token_id"] = self.tokenizer.eos_token_id

        with torch.inference_mode():
            output_ids = self.model.generate(**encoded, **generation_config)

        generated_ids = output_ids[0][prompt_length:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path

        candidates = [
            Path.cwd() / path,
            self.config_dir / path,
            self.config_dir.parent / path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return candidates[0].resolve()

    def output_dir(self) -> Path:
        output_dir = self.resolve_path(self.config.get("runtime", {}).get("output_dir", "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _effective_input_length(
        self,
        requested_max_input_tokens: int | None,
        max_new_tokens: int,
    ) -> int | None:
        if self.tokenizer is None:
            return requested_max_input_tokens

        model_max_length = getattr(self.tokenizer, "model_max_length", None)
        if not isinstance(model_max_length, int) or model_max_length > 100_000:
            return requested_max_input_tokens

        available = max(model_max_length - max_new_tokens, 1)
        if requested_max_input_tokens is None:
            return available
        return min(int(requested_max_input_tokens), available)

    def _resolve_device(self, requested_device: str) -> str:
        if requested_device != "auto":
            return requested_device
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _resolve_dtype(self, dtype_name: str | None) -> torch.dtype | None:
        if dtype_name is None or dtype_name == "auto":
            return None

        if self.device == "cpu" and dtype_name in {"float16", "fp16", "bfloat16", "bf16"}:
            return torch.float32

        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if dtype_name not in mapping:
            raise ValueError(f"Unsupported torch_dtype: {dtype_name}")
        return mapping[dtype_name]

    def _resolve_model_ref(self, model_name: str, local_files_only: bool) -> str:
        if not local_files_only or Path(model_name).exists():
            return model_name
        return snapshot_download(repo_id=model_name, local_files_only=True)


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as handle:
        if suffix == ".json":
            return json.load(handle)
        if suffix in {".yaml", ".yml"}:
            import yaml

            loaded = yaml.safe_load(handle)
            return loaded or {}
    raise ValueError(f"Unsupported config format: {path}")
