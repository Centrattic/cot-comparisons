"""
Interpret SAE features by finding their top-activating tokens/contexts in FineWeb.

For each feature of interest, collects the top-K tokens that maximally activate it,
along with surrounding context. This helps understand what concept each feature encodes.
"""

import json
import sys
import os
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
import heapq

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from huggingface_hub import hf_hub_download

HIGH_NORM_MULTIPLIER = 10.0


class BatchTopKSAE(nn.Module):
    def __init__(self, activation_dim: int, dict_size: int, k: int):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.register_buffer("k", torch.tensor(k, dtype=torch.int))
        self.register_buffer("threshold", torch.tensor(-1.0, dtype=torch.float32))
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.encoder = nn.Linear(activation_dim, dict_size)
        self.b_dec = nn.Parameter(torch.zeros(activation_dim))

    def encode(self, x: torch.Tensor, use_threshold: bool = True) -> torch.Tensor:
        post_relu_feat_acts = nn.functional.relu(self.encoder(x - self.b_dec))
        if use_threshold:
            encoded_acts = post_relu_feat_acts * (post_relu_feat_acts > self.threshold)
        else:
            flattened_acts = post_relu_feat_acts.flatten()
            post_topk = flattened_acts.topk(self.k.item() * x.size(0), sorted=False, dim=-1)
            encoded_acts = (
                torch.zeros_like(post_relu_feat_acts.flatten())
                .scatter_(-1, post_topk.indices, post_topk.values)
                .reshape(post_relu_feat_acts.shape)
            )
        return encoded_acts

    @classmethod
    def from_pretrained(cls, repo_id="adamkarvonen/qwen3-32b-saes", layer=32, trainer=0, device="cuda"):
        subdir = f"saes_Qwen_Qwen3-32B_batch_top_k/resid_post_layer_{layer}/trainer_{trainer}"
        config_path = hf_hub_download(repo_id=repo_id, filename=f"{subdir}/config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        tc = config["trainer"]
        sae = cls(activation_dim=tc["activation_dim"], dict_size=tc["dict_size"], k=tc["k"])
        weights_path = hf_hub_download(repo_id=repo_id, filename=f"{subdir}/ae.pt")
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        sae.load_state_dict(state_dict)
        sae = sae.to(device)
        sae.eval()
        print(f"Loaded SAE: layer={layer}, dict_size={tc['dict_size']}, k={tc['k']}, threshold={sae.threshold.item():.4f}")
        return sae


# ── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-32B"
SAE_LAYER = 32
TOP_K_EXAMPLES = 20          # top activating examples per feature
NUM_DOCUMENTS = 500           # number of FineWeb documents to process
MAX_SEQ_LEN = 1024            # tokens per chunk
CONTEXT_WINDOW = 30           # tokens of context around max-activating token
BATCH_SIZE = 4                # sequences per forward pass

# Features to study - union of top features across all 3 variants
# Format: {feature_idx: description_of_which_variant}
FEATURES_OF_INTEREST = {
    # suggest_wrong top features (strongest probe, r=0.705)
    706: "suggest_wrong negative r=-0.683",
    4479: "suggest_wrong positive r=+0.661",
    13772: "suggest_wrong negative r=-0.624",
    11896: "suggest_wrong negative r=-0.599",
    1930: "suggest_wrong positive r=+0.590",
    16233: "suggest_wrong negative r=-0.576",
    15376: "suggest_wrong negative r=-0.574",
    9523: "suggest_wrong positive r=+0.571",
    2817: "suggest_wrong positive r=+0.549",
    3938: "suggest_wrong negative r=-0.548",
    # first_person top features (strongest probe, r=0.828)
    4346: "first_person positive r=+0.553",
    9778: "first_person negative r=-0.538",
    63: "first_person positive r=+0.521",
    12160: "first_person positive r=+0.516",
    4658: "first_person positive r=+0.509",
    9996: "first_person positive r=+0.506",
    570: "first_person positive r=+0.500",
    12907: "first_person negative r=-0.494",
    12346: "first_person positive r=+0.494",
    2813: "first_person positive r=+0.491",
    # suggest_right top features (r=0.742)
    3266: "suggest_right positive r=+0.563",
    3306: "suggest_right negative r=-0.546",
    2612: "suggest_right positive r=+0.536",
    9045: "suggest_right positive r=+0.535",
    13363: "suggest_right negative r=-0.535",
    7264: "suggest_right positive r=+0.525",
    15588: "suggest_right positive r=+0.525",
    7911: "suggest_right positive r=+0.517",
    6265: "suggest_right negative r=-0.513",
    3138: "suggest_right negative r=-0.510",
}

FEATURE_INDICES = sorted(FEATURES_OF_INTEREST.keys())


@dataclass
class ActivationExample:
    """One top-activating example for a feature."""
    activation: float
    token: str
    context_before: str
    context_after: str
    doc_idx: int


def main():
    device = "cuda:0"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    print("Loading model (across 2 GPUs)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Figure out which device layer 32 ended up on
    layer_device = next(model.model.layers[SAE_LAYER].parameters()).device
    print(f"Layer {SAE_LAYER} is on {layer_device}")

    print("Loading SAE...")
    sae = BatchTopKSAE.from_pretrained(layer=SAE_LAYER, device=str(layer_device))

    print(f"Loading FineWeb (streaming {NUM_DOCUMENTS} docs)...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    # Min-heaps per feature: store (activation, example) tuples
    # Using negative activation for max-heap behavior with heapq
    feature_heaps: dict[int, list] = {f: [] for f in FEATURE_INDICES}

    # Hook to capture residual stream at target layer
    activations_store = {}

    def hook_fn(module, input, output):
        # output is a tuple; first element is the hidden states
        if isinstance(output, tuple):
            activations_store["resid"] = output[0].detach()
        else:
            activations_store["resid"] = output.detach()

    hook = model.model.layers[SAE_LAYER].register_forward_hook(hook_fn)

    doc_count = 0
    total_tokens = 0

    try:
        for doc in tqdm(ds, total=NUM_DOCUMENTS, desc="Processing FineWeb"):
            if doc_count >= NUM_DOCUMENTS:
                break

            text = doc.get("text", "")
            if len(text.strip()) < 100:
                continue

            # Tokenize
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) < 20:
                continue

            # Process in chunks
            for chunk_start in range(0, len(tokens), MAX_SEQ_LEN):
                chunk_tokens = tokens[chunk_start:chunk_start + MAX_SEQ_LEN]
                if len(chunk_tokens) < 10:
                    continue

                input_ids = torch.tensor([chunk_tokens], device=model.device)

                with torch.no_grad():
                    model(input_ids, use_cache=False)

                resid = activations_store["resid"]  # [1, seq_len, hidden_dim]
                resid = resid.squeeze(0)  # [seq_len, hidden_dim]

                # High-norm filtering
                norms = resid.norm(dim=-1)  # [seq_len]
                median_norm = norms.median()
                mask = norms <= (median_norm * HIGH_NORM_MULTIPLIER)

                # Get SAE features for ALL tokens (we need position info)
                resid_f32 = resid.float()
                with torch.no_grad():
                    all_features = sae.encode(resid_f32)  # [seq_len, dict_size]

                # Zero out high-norm positions
                all_features[~mask] = 0.0

                # Extract only features of interest
                foi_tensor = all_features[:, FEATURE_INDICES]  # [seq_len, n_features]
                foi_np = foi_tensor.cpu().numpy()

                # For each feature, find if any tokens beat current top-k
                decoded_tokens = [tokenizer.decode([t]) for t in chunk_tokens]

                for feat_local_idx, feat_global_idx in enumerate(FEATURE_INDICES):
                    feat_acts = foi_np[:, feat_local_idx]
                    # Find top activations in this chunk
                    top_positions = np.argsort(feat_acts)[-5:]  # top 5 per chunk

                    for pos in top_positions:
                        act_val = float(feat_acts[pos])
                        if act_val <= 0:
                            continue

                        heap = feature_heaps[feat_global_idx]

                        if len(heap) < TOP_K_EXAMPLES or act_val > heap[0][0]:
                            # Build context
                            ctx_start = max(0, pos - CONTEXT_WINDOW)
                            ctx_end = min(len(decoded_tokens), pos + CONTEXT_WINDOW + 1)

                            ctx_before = "".join(decoded_tokens[ctx_start:pos])
                            tok = decoded_tokens[pos]
                            ctx_after = "".join(decoded_tokens[pos+1:ctx_end])

                            example = ActivationExample(
                                activation=act_val,
                                token=tok,
                                context_before=ctx_before,
                                context_after=ctx_after,
                                doc_idx=doc_count,
                            )

                            if len(heap) < TOP_K_EXAMPLES:
                                heapq.heappush(heap, (act_val, id(example), example))
                            else:
                                heapq.heapreplace(heap, (act_val, id(example), example))

                total_tokens += len(chunk_tokens)

            doc_count += 1

            if doc_count % 50 == 0:
                print(f"  Processed {doc_count} docs, {total_tokens:,} tokens")

    finally:
        hook.remove()

    # ── Output results ──────────────────────────────────────────────────────
    print(f"\nDone! Processed {doc_count} docs, {total_tokens:,} tokens total.\n")

    output = {}
    for feat_idx in FEATURE_INDICES:
        heap = feature_heaps[feat_idx]
        # Sort by activation descending
        examples = sorted(heap, key=lambda x: -x[0])

        variant_info = FEATURES_OF_INTEREST[feat_idx]
        print(f"\n{'='*80}")
        print(f"Feature {feat_idx} ({variant_info})")
        print(f"{'='*80}")

        feat_examples = []
        for rank, (act_val, _, ex) in enumerate(examples[:TOP_K_EXAMPLES]):
            print(f"\n  #{rank+1} activation={act_val:.3f} (doc {ex.doc_idx})")
            # Show context with the activating token highlighted
            context = f"...{ex.context_before}>>>{ex.token}<<<{ex.context_after}..."
            print(f"  {context[:200]}")
            feat_examples.append({
                "rank": rank + 1,
                "activation": act_val,
                "token": ex.token,
                "context_before": ex.context_before[-100:],
                "context_after": ex.context_after[:100],
                "doc_idx": ex.doc_idx,
            })

        output[str(feat_idx)] = {
            "variant_info": variant_info,
            "examples": feat_examples,
        }

    # Save to JSON
    output_path = Path(__file__).parent / "data" / "scruples-qwen3-32b" / "feature_interpretations.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved interpretations to {output_path}")


if __name__ == "__main__":
    main()
