"""
Shared activation extractor for residual stream activations.

Lazily loads a causal LM and extracts activations at specified layers/positions
using forward hooks. Used by tasks to pre-extract activations that methods consume.
"""

from typing import Dict, Optional

import numpy as np
import torch


def compute_token_boundaries(tokenizer, full_prompt: str, raw_response: str) -> Dict[str, int]:
    """Compute token boundary indices for prompt/thinking/response segments.

    Args:
        tokenizer: HuggingFace tokenizer instance.
        full_prompt: The full prompt text (everything before the model's response).
        raw_response: The model's raw response text.

    Returns:
        Dict with keys:
            last_input: last token index of the prompt
            last_thinking: last token index before </think> (-1 if no thinking block)
            last_response: last token index of full text
    """
    full_text = full_prompt + raw_response

    prompt_tokens = tokenizer.encode(full_prompt, add_special_tokens=False)
    all_tokens = tokenizer.encode(full_text, add_special_tokens=False)

    last_input = len(prompt_tokens) - 1
    last_response = len(all_tokens) - 1

    if "</think>" in raw_response:
        think_prefix = full_prompt + raw_response.split("</think>")[0]
        think_tokens = tokenizer.encode(think_prefix, add_special_tokens=False)
        last_thinking = len(think_tokens) - 1
    else:
        last_thinking = -1

    return {
        "last_input": last_input,
        "last_thinking": last_thinking,
        "last_response": last_response,
    }

class ActivationExtractor:
    """
    Extracts residual stream activations from a causal LM at specific layers/positions.
    Lazy-loads the model on first use.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        torch_dtype=None,
        load_in_4bit: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype or torch.bfloat16
        self._load_in_4bit = load_in_4bit
        self._model = None
        self._tokenizer = None

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    def _load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading {self.model_name}...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        load_kwargs = {
            "torch_dtype": self.torch_dtype,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if self._load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.torch_dtype,
                bnb_4bit_quant_type="nf4",
            )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **load_kwargs
        )
        self._model.eval()
        print(f"Model loaded: {len(self._model.model.layers)} layers")

    def extract_activation(
        self, text: str, layer: int, token_position: int, max_length: int = 4096,
    ) -> np.ndarray:
        """Extract activation at a specific layer and token position. Returns [hidden_dim]."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        ).to(self.model.device)

        seq_len = inputs["input_ids"].shape[1]
        token_position = max(0, min(token_position, seq_len - 1))

        activations = {}

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                activations["resid"] = output[0].detach()
            else:
                activations["resid"] = output.detach()

        handle = self.model.model.layers[layer].register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                self.model(**inputs)
        finally:
            handle.remove()

        return activations["resid"][0, token_position, :].cpu().float().numpy()

    def extract_full_sequence(
        self, text: str, layer: int, max_length: int = 4096,
    ) -> np.ndarray:
        """Extract full-sequence activations at a layer. Returns [seq_len, hidden_dim]."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        ).to(self.model.device)

        activations = {}

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                activations["resid"] = output[0].detach()
            else:
                activations["resid"] = output.detach()

        handle = self.model.model.layers[layer].register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                self.model(**inputs)
        finally:
            handle.remove()

        return activations["resid"][0].cpu().float().numpy()  # [seq_len, hidden_dim]

    def extract_rollout_activations(
        self,
        prompts_by_sentence: Dict[int, str],
        layer: int,
        max_length: int = 4096,
    ) -> Optional[tuple]:
        """Extract prompt-only activations for all sentences in a rollout with one forward pass.

        Because prompts within a rollout are nested prefixes (sentence 0's prompt
        is a prefix of sentence 1's, etc.), a single forward pass on the longest
        prompt gives correct activations for ALL sentences — the caller just
        slices ``full_activations[:token_counts[i], :]`` for sentence ``i``.

        Args:
            prompts_by_sentence: ``{sentence_idx: full_prompt_text}`` — must be
                nested prefixes (each shorter prompt is a prefix of longer ones).
            layer: Model layer to extract from.
            max_length: Tokenizer truncation limit.

        Returns:
            ``(full_activations [max_tokens, hidden_dim], token_counts {sentence_idx: n_tokens})``
            or ``None`` if prompts_by_sentence is empty.
        """
        if not prompts_by_sentence:
            return None

        # Find the sentence with the longest prompt
        longest_idx = max(prompts_by_sentence, key=lambda k: len(prompts_by_sentence[k]))
        longest_prompt = prompts_by_sentence[longest_idx]

        # Tokenize each prompt to get its token count.
        # Use self.tokenizer() (not .encode(add_special_tokens=False)) to match
        # how extract_full_sequence tokenizes the longest prompt for the forward pass.
        token_counts: Dict[int, int] = {}
        for sent_idx, prompt_text in prompts_by_sentence.items():
            ids = self.tokenizer(
                prompt_text, truncation=True, max_length=max_length,
            )["input_ids"]
            token_counts[sent_idx] = len(ids)

        # Run ONE forward pass on the longest prompt
        full_activations = self.extract_full_sequence(longest_prompt, layer, max_length)

        # Clamp token counts to actual sequence length (handles truncation)
        for sent_idx in token_counts:
            token_counts[sent_idx] = min(token_counts[sent_idx], full_activations.shape[0])

        return full_activations, token_counts

    def find_token_positions(self, prompt: str, thinking: str, answer: str) -> Dict[str, int]:
        """Find token indices for last_input, last_thinking, last_response."""
        full_response = f"<think>{thinking}</think>{answer}"
        full_text = prompt + full_response

        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        all_tokens = self.tokenizer.encode(full_text, add_special_tokens=False)
        think_end_tokens = self.tokenizer.encode(prompt + f"<think>{thinking}", add_special_tokens=False)

        return {
            "last_input": len(prompt_tokens) - 1,
            "last_thinking": len(think_end_tokens) - 1,
            "last_response": len(all_tokens) - 1,
            "full_text": full_text,
        }
