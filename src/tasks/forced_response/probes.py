"""
White-box probes for the Forced Response task.

Uses activations from the last token of partial CoT prefixes to predict
answer distributions.

Supports two question types:
- multiple_choice (bgel): 4-class classifier (A, B, C, D)
- binary_judge (blackmail): 2-class classifier (YES=blackmail, NO=no_blackmail)

Model: Qwen/Qwen2.5-32B-Instruct (or similar)
Activations: Residual stream at mid layer, last token of partial CoT prefix
"""

import contextlib
import io
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .data_loader import GPQAQuestion, BinaryJudgeQuestion, Question
from .prompts import get_cumulative_cot_segments
from .task import ForcedResponseTask


# Model configuration
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"
DEFAULT_PROBE_LAYER = 32  # Mid layer for 64-layer model

# Label definitions for different question types
MULTIPLE_CHOICE_LABELS = ["A", "B", "C", "D"]
BINARY_JUDGE_LABELS = ["YES", "NO"]  # YES = bad outcome (blackmail), NO = no blackmail


@dataclass
class ActivationSample:
    """A single activation sample for training/inference."""
    sentence_idx: int
    activation: np.ndarray  # [hidden_dim] - last token activation
    answer_counts: Dict[str, int]  # {"A": 5, "B": 10, ...} or {"YES": 2, "NO": 8}
    answer_distribution: Dict[str, float]  # Normalized to sum to 1
    partial_cot_length: int
    question_id: str = ""
    question_type: str = "multiple_choice"


@dataclass
class ProbeResult:
    """Result of probe prediction for a single prefix."""
    sentence_idx: int
    predicted_distribution: Dict[str, float]  # {"A": 0.1, ...} or {"YES": 0.3, "NO": 0.7}
    actual_distribution: Optional[Dict[str, float]] = None
    kl_divergence: Optional[float] = None


@dataclass
class ProbeTrainingResult:
    """Results from training the probe."""
    accuracy: float  # Top-1 accuracy on validation
    mean_kl_divergence: float  # Average KL(actual || predicted)
    per_class_accuracy: Dict[str, float]
    n_train: int
    n_val: int
    question_type: str = "multiple_choice"


class ActivationExtractor:
    """
    Extracts residual stream activations from Qwen at a specified layer.

    Extracts the activation at the last token position, which corresponds
    to the end of the partial CoT prefix.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        layer: int = DEFAULT_PROBE_LAYER,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        load_in_4bit: bool = True,
    ):
        self.model_name = model_name
        self.layer = layer
        self.device = device
        self.torch_dtype = torch_dtype
        self.load_in_4bit = load_in_4bit

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
        """Load Qwen model and tokenizer."""
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

        if self.load_in_4bit:
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

        # Get number of layers for validation
        n_layers = len(self._model.model.layers)
        print(f"Model loaded: {n_layers} layers, using layer {self.layer}")

        if self.layer >= n_layers:
            raise ValueError(f"Layer {self.layer} out of range (model has {n_layers} layers)")

    def extract_last_token_activation(
        self,
        text: str,
        max_length: int = 4096,
    ) -> np.ndarray:
        """
        Extract residual stream activation at the last token.

        Args:
            text: Full text (prompt + partial CoT prefix)
            max_length: Maximum sequence length

        Returns:
            Activation vector [hidden_dim]
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        ).to(self.model.device)

        activations = {}

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                activations["resid"] = output[0].detach()
            else:
                activations["resid"] = output.detach()

        # Hook into the target layer
        layer_module = self._model.model.layers[self.layer]
        handle = layer_module.register_forward_hook(hook_fn)

        with torch.no_grad():
            self._model(**inputs)

        handle.remove()

        # Get last token activation
        resid = activations["resid"]  # [1, seq_len, hidden_dim]
        last_token_act = resid[0, -1, :].cpu().numpy()  # [hidden_dim]

        return last_token_act

    def extract_for_prefix(
        self,
        question: Question,
        partial_cot: str,
    ) -> np.ndarray:
        """
        Extract activation for a partial CoT prefix.

        Builds the full forcing prompt and extracts the last token activation.

        Args:
            question: The question (GPQAQuestion or BinaryJudgeQuestion)
            partial_cot: The partial chain-of-thought prefix

        Returns:
            Activation vector [hidden_dim]
        """
        prompt = build_forcing_prompt(question, partial_cot)
        return self.extract_last_token_activation(prompt)


def build_forcing_prompt(
    question: Question,
    partial_cot: str,
) -> str:
    """
    Build the full prompt used for forcing, including partial CoT.

    This constructs the text that would be sent to a model when forcing it
    to continue from a partial chain of thought.

    Args:
        question: The question (GPQAQuestion or BinaryJudgeQuestion)
        partial_cot: The partial chain of thought prefix

    Returns:
        Full prompt string including question and partial CoT
    """
    if isinstance(question, BinaryJudgeQuestion):
        # Binary judge questions: just the question text
        user_msg = question.question
    else:
        # Multiple choice questions: include choices
        choice_labels = [chr(ord("A") + i) for i in range(len(question.choices))]
        choices_text = "\n".join(
            f"{label}. {choice}" for label, choice in zip(choice_labels, question.choices)
        )
        user_msg = (
            f"Answer the following multiple choice question. Think through the problem "
            f"step by step, then provide your final answer.\n\n"
            f"Question: {question.question}\n\n"
            f"{choices_text}\n\n"
            f"After your reasoning, provide your final answer as just the letter (A, B, C, or D) on a new line."
        )

    # Format as user message followed by assistant's partial thinking
    prompt = f"""{user_msg}

<think>
{partial_cot}"""

    return prompt


class AnswerProbe(nn.Module):
    """
    Classifier for predicting forcing answer distributions.

    Supports both multiple choice (4-class: A, B, C, D) and
    binary judge (2-class: YES, NO) question types.
    """

    def __init__(
        self,
        hidden_dim: int,
        question_type: Literal["multiple_choice", "binary_judge"] = "multiple_choice",
        use_mlp: bool = False,
        mlp_hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.question_type = question_type
        self.use_mlp = use_mlp

        # Set number of classes based on question type
        if question_type == "binary_judge":
            self.n_classes = 2
            self.labels = BINARY_JUDGE_LABELS
        else:
            self.n_classes = 4
            self.labels = MULTIPLE_CHOICE_LABELS

        if use_mlp:
            self.probe = nn.Sequential(
                nn.Linear(hidden_dim, mlp_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, self.n_classes),
            )
        else:
            # Simple linear probe
            self.probe = nn.Linear(hidden_dim, self.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Activations [batch, hidden_dim]

        Returns:
            Logits [batch, n_classes]
        """
        return self.probe(x)

    def predict_distribution(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Predict answer distribution for a single sample.

        Args:
            x: Activation [hidden_dim] or [1, hidden_dim]

        Returns:
            Distribution dict {"A": prob, ...} or {"YES": prob, "NO": prob}
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0)

        return {label: float(probs[i]) for i, label in enumerate(self.labels)}

    def predict_class(self, x: torch.Tensor) -> str:
        """
        Predict the most likely class for a single sample.

        Args:
            x: Activation [hidden_dim] or [1, hidden_dim]

        Returns:
            Predicted class label
        """
        dist = self.predict_distribution(x)
        return max(dist.items(), key=lambda x: x[1])[0]


class ProbeTrainer:
    """
    Trains and evaluates probes for both question types.

    Uses forcing run data to train a classifier that predicts answer
    distributions from activations.
    """

    def __init__(
        self,
        extractor: ActivationExtractor,
        probe: Optional[AnswerProbe] = None,
        device: str = "cuda",
    ):
        self.extractor = extractor
        self.device = device
        self.probe = probe
        self._scaler = None  # StandardScaler for activations
        self.question_type = "multiple_choice"

    def collect_training_data(
        self,
        question_id: str,
        rollout_idx: int = 0,
        forcing_run_dir: Optional[Path] = None,
        verbose: bool = True,
    ) -> List[ActivationSample]:
        """
        Collect activation samples from forcing run data.

        For each sentence in the forcing run, extracts the activation at
        that prefix and pairs it with the actual answer distribution.

        Args:
            question_id: Question ID to collect data for
            rollout_idx: Rollout index
            forcing_run_dir: Path to forcing run directory (auto-detected if None)
            verbose: Print progress

        Returns:
            List of ActivationSample objects
        """
        task = ForcedResponseTask()

        # Find forcing run directory
        if forcing_run_dir is None:
            forcing_dir = task.forcing_dir / question_id / f"rollout_{rollout_idx:03d}"
            if not forcing_dir.exists():
                raise FileNotFoundError(f"No forcing data at {forcing_dir}")
            forcing_run_dir = task.get_latest_run_dir(forcing_dir)

        if forcing_run_dir is None or not forcing_run_dir.exists():
            raise FileNotFoundError(f"No forcing run found for {question_id}")

        # Load summary to get question data
        summary_path = forcing_run_dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"No summary at {summary_path}")

        with open(summary_path) as f:
            summary = json.load(f)

        # Determine question type and create appropriate question object
        question_type = summary.get("question_type", "multiple_choice")
        self.question_type = question_type

        if question_type == "binary_judge":
            # Load from verification summary for judge prompt
            verification_dir = task.verification_dir / question_id
            verification_summary_path = verification_dir / "summary.json"
            with open(verification_summary_path) as f:
                verification_summary = json.load(f)

            question: Question = BinaryJudgeQuestion(
                id=summary["question_id"],
                question=verification_summary["question"],
                judge_prompt=verification_summary["judge_prompt"],
                bad_outcome=verification_summary["bad_outcome"],
                subject=verification_summary.get("subject"),
            )
            valid_labels = BINARY_JUDGE_LABELS
        else:
            question = GPQAQuestion(
                id=summary["question_id"],
                question=summary.get("question", ""),
                choices=summary.get("choices", []),
                correct_answer=summary.get("correct_answer", ""),
                correct_index=ord(summary.get("correct_answer", "A")) - ord("A"),
            )
            valid_labels = MULTIPLE_CHOICE_LABELS

        # Load source CoT
        source_cot = summary.get("source_cot", "")
        if not source_cot:
            # Try to load from verification rollout
            verification_dir = task.verification_dir / question_id
            rollout_path = verification_dir / "rollouts" / f"rollout_{rollout_idx:03d}.json"
            if rollout_path.exists():
                with open(rollout_path) as f:
                    rollout_data = json.load(f)
                source_cot = rollout_data.get("thinking", "") or rollout_data.get("full_response", "")

        if not source_cot:
            raise ValueError(f"No source CoT found for {question_id}")

        cot_segments = get_cumulative_cot_segments(source_cot)

        # Load forcing results per sentence
        samples = []
        sentence_results = summary.get("sentence_results", summary.get("sentence_summaries", []))

        if verbose:
            print(f"Question type: {question_type}")
            print(f"Valid labels: {valid_labels}")
            print(f"Collecting activations for {len(sentence_results)} sentences...")

        for result in tqdm(sentence_results, disable=not verbose, desc="Extracting activations"):
            sentence_idx = result["sentence_idx"]
            answer_counts = result.get("answer_counts", {})

            if not answer_counts:
                continue

            # Get partial CoT for this sentence
            if sentence_idx >= len(cot_segments):
                continue
            partial_cot = cot_segments[sentence_idx]

            # Compute distribution (only over valid labels)
            total = sum(answer_counts.get(label, 0) for label in valid_labels)
            if total == 0:
                continue
            distribution = {label: answer_counts.get(label, 0) / total for label in valid_labels}

            # Extract activation
            activation = self.extractor.extract_for_prefix(question, partial_cot)

            samples.append(ActivationSample(
                sentence_idx=sentence_idx,
                activation=activation,
                answer_counts=answer_counts,
                answer_distribution=distribution,
                partial_cot_length=len(partial_cot),
                question_id=question_id,
                question_type=question_type,
            ))

        if verbose:
            print(f"Collected {len(samples)} samples")

        return samples

    def collect_training_data_from_verification(
        self,
        question_id: str,
        max_rollouts: Optional[int] = None,
        verbose: bool = True,
    ) -> List[ActivationSample]:
        """
        Collect activation samples directly from verification rollouts.

        This is useful when forcing data doesn't exist yet. Each rollout
        provides one sample at the final answer point.

        Args:
            question_id: Question ID to collect data for
            max_rollouts: Maximum number of rollouts to use
            verbose: Print progress

        Returns:
            List of ActivationSample objects
        """
        task = ForcedResponseTask()

        # Load verification summary
        verification_dir = task.verification_dir / question_id
        summary_path = verification_dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"No verification data at {summary_path}")

        with open(summary_path) as f:
            summary = json.load(f)

        # Determine question type
        question_type = summary.get("question_type", "multiple_choice")
        self.question_type = question_type

        if question_type == "binary_judge":
            question: Question = BinaryJudgeQuestion(
                id=summary["question_id"],
                question=summary["question"],
                judge_prompt=summary["judge_prompt"],
                bad_outcome=summary["bad_outcome"],
                subject=summary.get("subject"),
            )
            valid_labels = BINARY_JUDGE_LABELS
        else:
            question = GPQAQuestion(
                id=summary["question_id"],
                question=summary["question"],
                choices=summary["choices"],
                correct_answer=summary["correct_answer"],
                correct_index=ord(summary["correct_answer"]) - ord("A"),
            )
            valid_labels = MULTIPLE_CHOICE_LABELS

        # Get answer counts from summary
        answer_counts = summary.get("answer_counts", {})

        # Load rollouts
        rollouts_dir = verification_dir / "rollouts"
        rollout_files = sorted(rollouts_dir.glob("rollout_*.json"))
        if max_rollouts is not None:
            rollout_files = rollout_files[:max_rollouts]

        if verbose:
            print(f"Question type: {question_type}")
            print(f"Valid labels: {valid_labels}")
            print(f"Answer counts from verification: {answer_counts}")
            print(f"Loading {len(rollout_files)} rollouts...")

        samples = []
        for rollout_path in tqdm(rollout_files, disable=not verbose, desc="Extracting activations"):
            with open(rollout_path) as f:
                rollout_data = json.load(f)

            # Get the answer for this rollout
            answer = rollout_data.get("answer", "").upper()
            if answer not in valid_labels:
                continue

            # Get the full CoT
            source_cot = rollout_data.get("thinking", "") or rollout_data.get("full_response", "")
            if not source_cot:
                continue

            # Extract activation at end of CoT
            activation = self.extractor.extract_for_prefix(question, source_cot)

            # Create one-hot distribution for this sample
            distribution = {label: 1.0 if label == answer else 0.0 for label in valid_labels}

            samples.append(ActivationSample(
                sentence_idx=-1,  # Full CoT, not a prefix
                activation=activation,
                answer_counts={answer: 1},
                answer_distribution=distribution,
                partial_cot_length=len(source_cot),
                question_id=question_id,
                question_type=question_type,
            ))

        if verbose:
            print(f"Collected {len(samples)} samples")
            # Show class distribution
            class_counts = {}
            for s in samples:
                for label, prob in s.answer_distribution.items():
                    if prob > 0.5:
                        class_counts[label] = class_counts.get(label, 0) + 1
            print(f"Class distribution: {class_counts}")

        return samples

    def train(
        self,
        samples: List[ActivationSample],
        val_split: float = 0.2,
        epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        use_mlp: bool = False,
        verbose: bool = True,
    ) -> ProbeTrainingResult:
        """
        Train the probe on collected samples.

        Uses cross-entropy loss with soft labels (the actual answer distribution).

        Args:
            samples: List of ActivationSample objects
            val_split: Fraction of samples for validation
            epochs: Number of training epochs
            lr: Learning rate
            weight_decay: L2 regularization
            use_mlp: Use MLP instead of linear probe
            verbose: Print progress

        Returns:
            ProbeTrainingResult with metrics
        """
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        if not samples:
            raise ValueError("No samples provided for training")

        # Determine question type from samples
        question_type = samples[0].question_type
        self.question_type = question_type

        if question_type == "binary_judge":
            labels = BINARY_JUDGE_LABELS
        else:
            labels = MULTIPLE_CHOICE_LABELS

        # Prepare data
        X = np.stack([s.activation for s in samples])
        # Convert distributions to probability vectors
        y = np.array([
            [s.answer_distribution.get(label, 0.0) for label in labels]
            for s in samples
        ])

        # Split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, random_state=42
        )

        # Standardize
        self._scaler = StandardScaler()
        X_train = self._scaler.fit_transform(X_train)
        X_val = self._scaler.transform(X_val)

        # Convert to tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32, device=self.device)

        # Initialize probe
        hidden_dim = X_train.shape[1]
        self.probe = AnswerProbe(
            hidden_dim=hidden_dim,
            question_type=question_type,
            use_mlp=use_mlp,
        ).to(self.device)

        # Training setup
        optimizer = torch.optim.AdamW(
            self.probe.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Use KL divergence loss (soft labels)
        def soft_cross_entropy(logits, soft_labels):
            log_probs = torch.log_softmax(logits, dim=-1)
            return -(soft_labels * log_probs).sum(dim=-1).mean()

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(epochs):
            # Training
            self.probe.train()
            optimizer.zero_grad()
            logits = self.probe(X_train_t)
            loss = soft_cross_entropy(logits, y_train_t)
            loss.backward()
            optimizer.step()

            # Validation
            self.probe.eval()
            with torch.no_grad():
                val_logits = self.probe(X_val_t)
                val_loss = soft_cross_entropy(val_logits, y_val_t)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.probe.state_dict().items()}

            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{epochs}: train_loss={loss.item():.4f}, val_loss={val_loss.item():.4f}")

        # Load best model
        if best_state is not None:
            self.probe.load_state_dict(best_state)
            self.probe.to(self.device)

        # Evaluate
        self.probe.eval()
        with torch.no_grad():
            val_logits = self.probe(X_val_t)
            val_probs = torch.softmax(val_logits, dim=-1)

            # Top-1 accuracy
            pred_labels = val_probs.argmax(dim=-1)
            true_labels = y_val_t.argmax(dim=-1)
            accuracy = (pred_labels == true_labels).float().mean().item()

            # KL divergence (actual || predicted)
            eps = 1e-8
            kl_div = (y_val_t * (torch.log(y_val_t + eps) - torch.log(val_probs + eps))).sum(dim=-1).mean().item()

            # Per-class accuracy
            per_class_acc = {}
            for i, label in enumerate(labels):
                mask = true_labels == i
                if mask.sum() > 0:
                    per_class_acc[label] = (pred_labels[mask] == i).float().mean().item()
                else:
                    per_class_acc[label] = 0.0

        if verbose:
            print(f"\nTraining complete:")
            print(f"  Question type: {question_type}")
            print(f"  Validation accuracy: {accuracy:.3f}")
            print(f"  Mean KL divergence: {kl_div:.4f}")
            print(f"  Per-class accuracy: {per_class_acc}")

        return ProbeTrainingResult(
            accuracy=accuracy,
            mean_kl_divergence=kl_div,
            per_class_accuracy=per_class_acc,
            n_train=len(X_train),
            n_val=len(X_val),
            question_type=question_type,
        )

    def predict(
        self,
        question: Question,
        partial_cot: str,
    ) -> Dict[str, float]:
        """
        Predict answer distribution for a single prefix.

        Args:
            question: The question (GPQAQuestion or BinaryJudgeQuestion)
            partial_cot: The partial chain-of-thought prefix

        Returns:
            Predicted distribution {"A": prob, ...} or {"YES": prob, "NO": prob}
        """
        if self.probe is None:
            raise RuntimeError("Probe not trained. Call train() first.")
        if self._scaler is None:
            raise RuntimeError("Scaler not fitted. Call train() first.")

        # Extract activation
        activation = self.extractor.extract_for_prefix(question, partial_cot)

        # Standardize
        activation_scaled = self._scaler.transform(activation.reshape(1, -1))

        # Predict
        activation_t = torch.tensor(activation_scaled, dtype=torch.float32, device=self.device)
        return self.probe.predict_distribution(activation_t)

    def predict_all_prefixes(
        self,
        question: Question,
        source_cot: str,
        verbose: bool = True,
    ) -> List[ProbeResult]:
        """
        Predict distributions for all prefix points in a CoT.

        Args:
            question: The question (GPQAQuestion or BinaryJudgeQuestion)
            source_cot: Full chain-of-thought to extract prefixes from
            verbose: Print progress

        Returns:
            List of ProbeResult objects
        """
        cot_segments = get_cumulative_cot_segments(source_cot)

        results = []
        for sentence_idx, partial_cot in enumerate(tqdm(cot_segments, disable=not verbose, desc="Predicting")):
            pred_dist = self.predict(question, partial_cot)
            results.append(ProbeResult(
                sentence_idx=sentence_idx,
                predicted_distribution=pred_dist,
            ))

        return results

    def save(self, path: Path):
        """Save probe and scaler to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save probe weights
        torch.save(self.probe.state_dict(), path / "probe.pt")

        # Save scaler
        np.savez(
            path / "scaler.npz",
            mean=self._scaler.mean_,
            scale=self._scaler.scale_,
        )

        # Save config
        config = {
            "hidden_dim": self.probe.hidden_dim,
            "use_mlp": self.probe.use_mlp,
            "question_type": self.probe.question_type,
            "n_classes": self.probe.n_classes,
            "labels": self.probe.labels,
            "model_name": self.extractor.model_name,
            "layer": self.extractor.layer,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"Saved probe to {path}")

    def load(self, path: Path):
        """Load probe and scaler from disk."""
        from sklearn.preprocessing import StandardScaler

        path = Path(path)

        # Load config
        with open(path / "config.json") as f:
            config = json.load(f)

        # Initialize probe
        self.question_type = config.get("question_type", "multiple_choice")
        self.probe = AnswerProbe(
            hidden_dim=config["hidden_dim"],
            question_type=self.question_type,
            use_mlp=config["use_mlp"],
        ).to(self.device)

        # Load weights
        state_dict = torch.load(path / "probe.pt", map_location=self.device)
        self.probe.load_state_dict(state_dict)
        self.probe.eval()

        # Load scaler
        scaler_data = np.load(path / "scaler.npz")
        self._scaler = StandardScaler()
        self._scaler.mean_ = scaler_data["mean"]
        self._scaler.scale_ = scaler_data["scale"]
        self._scaler.var_ = self._scaler.scale_ ** 2
        self._scaler.n_features_in_ = len(self._scaler.mean_)

        print(f"Loaded probe from {path} (question_type: {self.question_type})")


def run_probe_training(
    question_id: str,
    rollout_idx: int = 0,
    model_name: str = DEFAULT_MODEL_NAME,
    layer: int = DEFAULT_PROBE_LAYER,
    use_mlp: bool = False,
    epochs: int = 100,
    load_in_4bit: bool = True,
    device: str = "cuda",
    verbose: bool = True,
    use_verification_data: bool = False,
    max_rollouts: Optional[int] = None,
) -> Path:
    """
    Train a probe on data from a single question.

    Args:
        question_id: Question ID with forcing data
        rollout_idx: Rollout index (for forcing data)
        model_name: Model to extract activations from
        layer: Layer to extract from
        use_mlp: Use MLP instead of linear probe
        epochs: Training epochs
        load_in_4bit: Load model in 4-bit
        device: Device for training
        verbose: Print progress
        use_verification_data: If True, use verification rollouts instead of forcing data
        max_rollouts: Max rollouts to use (for verification data)

    Returns:
        Path to saved probe
    """
    # Initialize extractor
    extractor = ActivationExtractor(
        model_name=model_name,
        layer=layer,
        device=device,
        load_in_4bit=load_in_4bit,
    )

    # Initialize trainer
    trainer = ProbeTrainer(extractor=extractor, device=device)

    # Collect training data
    if use_verification_data:
        samples = trainer.collect_training_data_from_verification(
            question_id=question_id,
            max_rollouts=max_rollouts,
            verbose=verbose,
        )
    else:
        samples = trainer.collect_training_data(
            question_id=question_id,
            rollout_idx=rollout_idx,
            verbose=verbose,
        )

    if len(samples) < 10:
        raise ValueError(f"Not enough samples ({len(samples)}). Need at least 10.")

    # Train
    result = trainer.train(
        samples=samples,
        epochs=epochs,
        use_mlp=use_mlp,
        verbose=verbose,
    )

    # Save
    task = ForcedResponseTask()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    data_source = "verification" if use_verification_data else f"rollout_{rollout_idx:03d}"
    probe_dir = task.data_dir / "probes" / question_id / data_source / timestamp
    trainer.save(probe_dir)

    # Save training result
    with open(probe_dir / "training_result.json", "w") as f:
        json.dump({
            "accuracy": result.accuracy,
            "mean_kl_divergence": result.mean_kl_divergence,
            "per_class_accuracy": result.per_class_accuracy,
            "n_train": result.n_train,
            "n_val": result.n_val,
            "question_id": question_id,
            "question_type": result.question_type,
            "data_source": data_source,
            "model_name": model_name,
            "layer": layer,
            "use_mlp": use_mlp,
            "epochs": epochs,
        }, f, indent=2)

    return probe_dir


def run_probe_inference(
    question_id: str,
    rollout_idx: int = 0,
    probe_path: Optional[Path] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    layer: int = DEFAULT_PROBE_LAYER,
    load_in_4bit: bool = True,
    device: str = "cuda",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run probe inference on a question.

    Args:
        question_id: Question ID
        rollout_idx: Rollout index
        probe_path: Path to saved probe (auto-detected if None)
        model_name: Model to extract activations from
        layer: Layer to extract from
        load_in_4bit: Load model in 4-bit
        device: Device
        verbose: Print progress

    Returns:
        Dictionary with predictions per sentence
    """
    task = ForcedResponseTask()

    # Find probe
    if probe_path is None:
        probe_base = task.data_dir / "probes" / question_id
        if not probe_base.exists():
            raise FileNotFoundError(f"No probe found at {probe_base}")
        # Try rollout-specific first, then verification
        for subdir in [f"rollout_{rollout_idx:03d}", "verification"]:
            subpath = probe_base / subdir
            if subpath.exists():
                timestamps = sorted(subpath.iterdir())
                if timestamps:
                    probe_path = timestamps[-1]
                    break
        if probe_path is None:
            raise FileNotFoundError(f"No probe timestamps found in {probe_base}")

    # Load verification summary to get question info
    verification_dir = task.verification_dir / question_id
    summary_path = verification_dir / "summary.json"
    with open(summary_path) as f:
        summary = json.load(f)

    # Create appropriate question object
    question_type = summary.get("question_type", "multiple_choice")
    if question_type == "binary_judge":
        question: Question = BinaryJudgeQuestion(
            id=summary["question_id"],
            question=summary["question"],
            judge_prompt=summary["judge_prompt"],
            bad_outcome=summary["bad_outcome"],
            subject=summary.get("subject"),
        )
    else:
        question = GPQAQuestion(
            id=summary["question_id"],
            question=summary["question"],
            choices=summary["choices"],
            correct_answer=summary["correct_answer"],
            correct_index=ord(summary["correct_answer"]) - ord("A"),
        )

    rollout_path = verification_dir / "rollouts" / f"rollout_{rollout_idx:03d}.json"
    with open(rollout_path) as f:
        rollout_data = json.load(f)
    source_cot = rollout_data.get("thinking", "") or rollout_data.get("full_response", "")

    # Initialize extractor and trainer
    extractor = ActivationExtractor(
        model_name=model_name,
        layer=layer,
        device=device,
        load_in_4bit=load_in_4bit,
    )
    trainer = ProbeTrainer(extractor=extractor, device=device)
    trainer.load(probe_path)

    # Run inference
    results = trainer.predict_all_prefixes(question, source_cot, verbose=verbose)

    # Format output
    output = {
        "question_id": question_id,
        "question_type": question_type,
        "num_sentences": len(results),
        "sentence_results": [
            {
                "sentence_idx": r.sentence_idx,
                "predicted_distribution": r.predicted_distribution,
            }
            for r in results
        ],
    }

    if isinstance(question, GPQAQuestion):
        output["correct_answer"] = question.correct_answer
    else:
        output["bad_outcome"] = question.bad_outcome

    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    inference_dir = task.data_dir / "probe_inference" / question_id / f"rollout_{rollout_idx:03d}" / timestamp
    inference_dir.mkdir(parents=True, exist_ok=True)
    with open(inference_dir / "results.json", "w") as f:
        json.dump(output, f, indent=2)

    if verbose:
        print(f"\nSaved inference results to {inference_dir}")

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and run probes for forced response task")
    parser.add_argument("command", choices=["train", "infer"], help="Command to run")
    parser.add_argument("--question-id", "-q", required=True, help="Question ID")
    parser.add_argument("--rollout-idx", "-r", type=int, default=0, help="Rollout index")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Model name")
    parser.add_argument("--layer", type=int, default=DEFAULT_PROBE_LAYER, help="Layer to extract from")
    parser.add_argument("--use-mlp", action="store_true", help="Use MLP instead of linear probe")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--probe-path", type=Path, default=None, help="Path to saved probe")
    parser.add_argument("--load-in-4bit", action="store_true", default=True, help="Load model in 4-bit")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--use-verification-data", action="store_true",
                        help="Use verification rollouts instead of forcing data")
    parser.add_argument("--max-rollouts", type=int, default=None,
                        help="Max rollouts to use (for verification data)")

    args = parser.parse_args()

    if args.command == "train":
        probe_path = run_probe_training(
            question_id=args.question_id,
            rollout_idx=args.rollout_idx,
            model_name=args.model_name,
            layer=args.layer,
            use_mlp=args.use_mlp,
            epochs=args.epochs,
            load_in_4bit=args.load_in_4bit,
            device=args.device,
            verbose=not args.quiet,
            use_verification_data=args.use_verification_data,
            max_rollouts=args.max_rollouts,
        )
        print(f"\nProbe saved to: {probe_path}")

    elif args.command == "infer":
        results = run_probe_inference(
            question_id=args.question_id,
            rollout_idx=args.rollout_idx,
            probe_path=args.probe_path,
            model_name=args.model_name,
            layer=args.layer,
            load_in_4bit=args.load_in_4bit,
            device=args.device,
            verbose=not args.quiet,
        )
        print(f"\nPredictions for {len(results['sentence_results'])} sentences")
