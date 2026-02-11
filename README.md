# cot-comparisons

A research framework for analyzing and evaluating chain-of-thought (CoT) reasoning in large language models. The framework supports:

- **CoT Forcing**: Prefill model thinking with partial CoT and observe answer distributions
- **CoT Compression**: Reduce CoT length while preserving model accuracy
- **Sycophancy Detection**: Detect when models produce biased/user-pleasing responses
- **White-box & Black-box Evaluation**: Linear probes, attention probes, SAE analysis, and LLM monitors

## Architecture

The framework follows a **Task-Method pattern**:

```
TASK (Data Producer)          METHOD (Data Consumer)
────────────────────          ─────────────────────
  run_data()                     train()
  extract_activations()          infer()
  get_data()                     → predictions/results
```

**Tasks** generate rollouts, extract activations, and prepare data. **Methods** consume task data to train probes or run inference (e.g., LLM monitors).

## Setup

```bash
pip install tinker transformers torch numpy pandas scikit-learn python-dotenv openai tqdm
```

Optional for SAE analysis:
```bash
pip install sae-lens
```

Add your API keys to `.env` in the project root:

```
TINKER_API_KEY=<your-key>
OPENROUTER_API_KEY=<your-key>
```

## Running Pipelines

Configuration is done by editing constants at the top of each run script. After configuring, run with:

```bash
python -m src2.runs.<script_name>
```

### 1. Verification — Generate Base Rollouts

Generates N rollouts per question to establish baseline CoT data.

```bash
python -m src2.runs.run_verification
```

**Configuration** (edit `src2/runs/run_verification.py`):
```python
MODEL = "moonshotai/kimi-k2-thinking"   # Model to generate rollouts
NUM_ROLLOUTS = 50                        # Rollouts per question
TEMPERATURE = 0.7                        # Sampling temperature
MAX_WORKERS = 300                        # Parallel API calls
QUESTION_SOURCE = "sample_gpqa"          # "sample_gpqa", "custom", or "gpqa_hf"
```

### 2. Forced Response — CoT Prefill Forcing

Prefills the model's `<think>` block with partial CoT at sentence boundaries via Tinker API.

```bash
python -m src2.runs.run_forcing
```

**Configuration** (edit `src2/runs/run_forcing.py`):
```python
SUBJECT_MODEL = "Qwen/Qwen3-32B"         # Model to force
MONITOR_MODEL = "openai/gpt-5.2"         # LLM monitor for predictions
ACTIVATION_MODEL = "Qwen/Qwen3-32B"      # Model for activation extraction
LAYER = 32                               # Layer for activations

ROLLOUT_IDX = 0                          # Which verification rollout to use
NUM_FORCES = 1                           # Samples per sentence boundary
TEMPERATURE = 0.0                        # Forcing temperature
MAX_SENTENCES = None                     # Limit sentences (None = all)
SENTENCE_STRIDE = 1                      # Force every Nth sentence
```

**Pipeline steps:**
1. Ensures verification rollouts exist for each question
2. Forces model at each sentence boundary
3. Extracts activations at specified layer
4. Runs LLM monitor to predict answers from partial CoT

### 3. CoT Compression — Reduce CoT Length

Compresses CoT regions by N× and evaluates answer distribution change via JS divergence.

```bash
# With command-line arguments:
python -m src2.runs.run_compression <question_id> <num_rollouts>

# Examples:
python -m src2.runs.run_compression starfish 25
python -m src2.runs.run_compression custom_bagel_001 25
python -m src2.runs.run_compression waffle 25
```

**Configuration** (edit `src2/runs/run_compression.py`):
```python
SUBJECT_MODEL = "Qwen/Qwen3-32B"         # Model to evaluate
MONITOR_MODEL = "openai/gpt-5.2"         # LLM for sentence selection

COMPRESSION_FACTOR = 10                  # Target compression ratio (10×)
CHAR_LIMIT_MULTIPLIER = 1.5              # Character limit buffer
COMPRESS_PCT = 0.5                       # Fraction of CoT to compress
REGION = "prefix"                        # "prefix" or "middle"
NUM_RESAMPLES = 50                       # Resamples for distribution estimation
```

**Pipeline steps:**
1. Computes baseline distribution from full CoT (cached after first run)
2. For each rollout: LLM selects which sentences to keep
3. Forces with compressed CoT, collects answer distribution
4. Computes JS divergence and agreement vs baseline

### 4. Scruples Sycophancy — Detect User-Pleasing Bias

Compares model responses between control (neutral) and intervention (user suggests answer) conditions.

```bash
python -m src2.runs.run_scruples
```

**Configuration** (edit `src2/runs/run_scruples.py`):
```python
SUBJECT_MODEL = "Qwen/Qwen3-32B"         # Model to evaluate
MONITOR_MODEL = "openai/gpt-5.2"         # LLM monitor
ACTIVATION_MODEL = "Qwen/Qwen3-32B"      # For activation extraction
LAYER = 32                               # Layer for activations

VARIANTS_TO_RUN = ["suggest_wrong", "suggest_right"]
NUM_SAMPLES = 50                         # Rollouts per arm
MAX_PROMPTS = None                       # Limit prompts (None = all)
MAX_WORKERS = 100                        # Parallel API calls

# Pipeline steps to run (set True/False):
GENERATE_DATA = False                    # Generate control/intervention rollouts
EXTRACT_ACTIVATIONS = False              # Extract model activations
RUN_MONITOR = True                       # Run base sycophancy monitor
RUN_HIGH_CONTEXT_MONITOR = True          # Run few-shot monitor
RUN_DISCRIMINATION = True                # Run paired A/B discrimination
```

**Variants:**
- `suggest_wrong`: User states "I think the author is in the wrong"
- `suggest_right`: User states "I think the author is NOT in the wrong"

### 5. Sycophancy Probe — Train Binary Classifier

Trains an attention probe to predict sycophancy from CoT activations.

```bash
python -m src2.runs.run_sycophancy_probe
```

**Configuration** (edit `src2/runs/run_sycophancy_probe.py`):
```python
SUBJECT_MODEL = "moonshotai/kimi-k2-thinking"
ACTIVATION_MODEL = "Qwen/Qwen3-32B"
LAYER = 32
VARIANTS = ["suggest_wrong", "suggest_right"]

# Training hyperparameters:
NUM_HEADS = 4                            # Attention heads in probe
LR = 1e-4                                # Learning rate
EPOCHS = 40                              # Training epochs
BATCH_SIZE = 8                           # Batch size
TEST_SPLIT = 0.2                         # Test set fraction

EXTRACT_ACTIVATIONS = False              # Set True if activations not yet extracted
```

### 6. Heatmaps — Pairwise Similarity Visualization

Generates pairwise cosine-similarity heatmaps for reasoning rollouts. Supports two similarity methods (layer-44 activations and semantic embeddings) at two granularities (sentence and paragraph).

```bash
# Single rollout, default types (sentence layer44 + sentence semantic)
python -m src2.runs.generate_heatmaps --rollout data/reasoning_evals/rollouts/unlabeled/bagel/rollout_0.json

# Multiple prompts, 3 rollouts each, all four heatmap types
python -m src2.runs.generate_heatmaps --prompt bagel well starfish -n 3 \
  --types sentence_layer44 sentence_semantic paragraph_layer44 paragraph_semantic

# Semantic only (no GPU pod needed)
python -m src2.runs.generate_heatmaps --prompt starfish -n 1 --types sentence_semantic paragraph_semantic

# All rollouts in a directory
python -m src2.runs.generate_heatmaps --rollout-dir data/reasoning_evals/rollouts/unlabeled/bagel

# Plain text file (resolves to data/texts/essay.txt) — outputs to non_cot/ folder
python -m src2.runs.generate_heatmaps --text essay --types sentence_semantic paragraph_semantic

# Mix rollouts and text in one run
python -m src2.runs.generate_heatmaps --prompt bagel -n 1 --text essay --types sentence_semantic
```

**Input modes** (can be combined):
- `--rollout PATH [PATH ...]` — explicit rollout JSON files
- `--rollout-dir PATH [PATH ...]` — directories of rollout JSONs
- `--prompt NAME [NAME ...]` — prompt names from `REASONING_PROMPTS`; auto-generates rollouts via Tinker if fewer than `-n` exist
- `--text NAME [NAME ...]` — text names resolved to `data/texts/NAME.txt`; heatmaps saved to `non_cot/` subfolder

**Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `-n N` | 1 | Rollouts per prompt (with `--prompt`) |
| `--types TYPE [...]` | `sentence_layer44 sentence_semantic` | Heatmap types to generate |
| `--pod ALIAS` | `mats_9` | SSH alias for GPU pod (layer44 only) |
| `--output-dir PATH` | `data/reasoning_evals/heatmaps_2` | Output root |
| `--activations-dir PATH` | `data/reasoning_evals/heatmap_activations` | Activation storage |
| `--no-skip-existing` | off | Regenerate even if outputs exist |

**Heatmap types:**
- `sentence_layer44` / `paragraph_layer44` — Layer-44 mean-pooled activations, mean-centered cosine similarity. Requires GPU pod for extraction.
- `sentence_semantic` / `paragraph_semantic` — Sentence-transformer embeddings (`paraphrase-mpnet-base-v2` / `all-mpnet-base-v2`), mean-centered cosine similarity. Runs locally on CPU.

**Pod setup** (for layer44 types): The pod needs `transformers`, `accelerate`, `nltk`, and the model at `/dev/shm/models/Qwen3-32B`. The script checks for the model and gives instructions if missing.

### 7. Additional Pipelines

```bash
# Generate more scruples data
python -m src2.runs.run_scruples_data

# Run intervention type probe (3-class)
python -m src2.runs.run_intervention_probe

# Run thought anchors resampling
python -m src2.runs.run_thought_anchors

# Run resampling experiments
python -m src2.runs.run_resampling
```

## Methods

### LlmMonitor

Black-box LLM calling via OpenRouter API. Takes a prompt template and runs parallelized inference.

```python
from src2.methods import LlmMonitor
from src2.tasks.scruples.prompts import ScruplesBaseMonitorPrompt

monitor = LlmMonitor(prompt=ScruplesBaseMonitorPrompt(variant), model="openai/gpt-5.2")
monitor.set_task(task)
results = monitor.infer(data)
```

### LinearProbe

Single-token linear/ridge regression probes on model activations.

```python
from src2.methods import LinearProbe

probe = LinearProbe(layer=32, mode="ridge")  # or mode="soft_ce"
probe.set_task(task)
probe.train(probe_data)
predictions = probe.infer(probe_data)
```

### AttentionProbe

Attention-based classification on full sequences or specific positions.

```python
from src2.methods import AttentionProbe

probe = AttentionProbe(layer=32, mode="classification")
probe.set_task(task)
probe.train(probe_data)
predictions = probe.infer(probe_data)
```

### ContrastiveSAE

Sparse autoencoder analysis for feature-level contrastive analysis.

```python
from src2.methods import ContrastiveSAE

sae = ContrastiveSAE(sae_repo="adamkarvonen/qwen3-32b-saes", layer=32)
sae.set_task(task)
features = sae.infer(sae_data)
```

## Directory Structure

```
cot-comparisons/
├── src2/                          # Main source code
│   ├── tasks/                     # Data producers
│   │   ├── base.py               # BaseTask abstract class
│   │   ├── forced_response/       # CoT forcing task
│   │   │   ├── task.py
│   │   │   ├── prompts.py
│   │   │   └── data_loader.py
│   │   ├── compressed_cot/        # CoT compression task
│   │   │   ├── task.py
│   │   │   └── prompts.py
│   │   ├── scruples/              # Sycophancy task
│   │   │   ├── task.py
│   │   │   ├── data_loader.py
│   │   │   └── prompts.py
│   │   └── resampled_response/    # Resampling task
│   │       └── task.py
│   ├── methods/                   # Data consumers
│   │   ├── base.py               # BaseMethod abstract class
│   │   ├── linear_probe.py        # Linear/ridge probes
│   │   ├── attention_probe.py     # Attention classification
│   │   ├── llm_monitor.py         # Black-box LLM monitor
│   │   ├── thought_anchors.py     # Adaptive resampling
│   │   └── contrastive_sae.py     # SAE analysis
│   ├── prompts/                   # Prompt templates
│   │   └── base.py               # BasePrompt abstract class
│   ├── utils/                     # Shared utilities
│   │   ├── questions.py           # Question data types
│   │   ├── activations.py         # Activation extraction
│   │   ├── chat_template.py       # Chat formatting
│   │   ├── output.py              # Timestamped output management
│   │   └── questions.json         # Custom question definitions
│   ├── runs/                      # Executable pipelines
│   │   ├── run_forcing.py
│   │   ├── run_compression.py
│   │   ├── run_scruples.py
│   │   ├── run_thought_anchors.py
│   │   ├── generate_heatmaps.py   # Unified heatmap CLI
│   │   ├── extract_batch_activations.py  # Pod extraction script
│   │   └── ...
│   └── data_slice.py              # Data filtering utility
├── data/                          # Generated data & outputs
│   ├── verification_rollouts/     # Base CoTs for questions
│   ├── forced_response/           # Forcing task outputs
│   ├── compressed_cot/            # Compression task outputs
│   ├── scruples/                  # Sycophancy task outputs
│   ├── texts/                     # Plain text files for --text input
│   │   └── {name}.txt            # e.g. pragmatic.txt, dario.txt
│   └── reasoning_evals/           # Reasoning eval pipeline
│       ├── rollouts/unlabeled/    # Raw rollouts per prompt
│       ├── heatmap_activations/   # .npz + manifest from pod
│       └── heatmaps_2/            # Output heatmaps (PNG + HTML)
│           ├── sentence/
│           │   ├── layer44/{prompt}/
│           │   ├── semantic/{prompt}/
│           │   └── */non_cot/     # --text outputs
│           └── paragraph/
│               ├── layer44/{prompt}/
│               ├── semantic/{prompt}/
│               └── */non_cot/     # --text outputs
├── tests/                         # Unit tests
├── plots/                         # Output visualizations
├── scripts/                       # Helper scripts
└── writeups/                      # Results documents
```

## Data Layout

### Verification Rollouts (shared base data)

```
data/verification_rollouts/{question_id}/
  └── {TIMESTAMP}/
      ├── summary.json              # question metadata + stats
      └── rollouts/
          ├── rollout_000.json       # full response + thinking
          └── ...
```

### Forcing Task

```
data/forced_response/
  ├── forcing/{question_id}/rollout_000/{timestamp}/
  │   ├── sentence_000/
  │   │   ├── force_000.json     # forcing result
  │   │   ├── force_000.npz      # activations [seq_len, hidden]
  │   │   └── summary.json       # answer distribution
  │   └── summary.json           # all sentences summary
  └── llm_monitor_*/{TIMESTAMP}/
      ├── predictions.json
      └── method_config.json
```

### Scruples Task

```
data/scruples/
  ├── results_{variant}.csv          # all runs, one per row
  ├── prompts_{variant}.csv          # anecdote-level aggregation
  └── runs/{timestamp}/{anecdote_id}/
      ├── control_0.json             # control arm run
      ├── control_0.npz              # activations
      ├── intervention_0.json        # intervention arm run
      └── intervention_0.npz
```

### Compression Task

```
data/compressed_cot/
  ├── compressions/{question_id}/rollout_000/{timestamp}/
  │   ├── compression_spec.json
  │   ├── compressed_cot.txt
  │   └── compression_eval.json
  └── llm_monitor_*/{TIMESTAMP}/
      └── predictions.json
```

## Output Management

All results are timestamped (`YYYY-MM-DD_HH-MM-SS/`) with:
- A `latest` symlink pointing to the most recent successful run
- `method_config.json` capturing full configuration
- `git_info.txt` with commit hash and diff

## Data Filtering

Use `DataSlice` to filter which data to process:

```python
from src2.data_slice import DataSlice

DataSlice.all()                    # no filter
DataSlice.from_ids(["id1", "id2"]) # filter by IDs
DataSlice.latest(3)                # most recent N runs
DataSlice.from_paths([p1, p2])     # filter by path
```

## Custom Questions

Questions are defined in `src2/utils/questions.json`:

- **Multiple choice**: Spatial reasoning puzzles (bagel, waffle, starfish)
- **Binary judge**: AI safety scenarios with automated judge prompts

## Testing

```bash
pytest tests/
```

## Research Findings

See `writeups/` for detailed results:
- Sycophancy detection rates across intervention types
- CoT compression vs answer accuracy tradeoffs
- Example CoT switches showing how interventions affect reasoning