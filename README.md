# cot-comparisons

## Forced Response Task

Analyzes chain-of-thought reasoning by seeding the model's thinking with partial CoT prefixes and observing how it completes its reasoning.

### Setup

```bash
pip install tinker transformers python-dotenv openai
```

Add your API keys to `.env` in the project root:

```
TINKER_API_KEY=<your-key>
OPENROUTER_API_KEY=<your-key>
```

### Commands

#### 1. Verification — find high-agreement questions

Run N rollouts to identify questions where the model consistently agrees on an answer:

```bash
python src/tasks/forced_response/run_forced_response.py verify \
  --num-rollouts 50 \
  --threshold 0.8
```

#### 2. Force (Tinker) — true prefill forcing

Prefills the start of the model's `<think>` block with partial CoT via the Tinker sampling API. The model continues its chain of thought from the prefix and produces an answer. Uses `moonshotai/Kimi-K2-Thinking` with raw token-level prompting.

```bash
python src/tasks/forced_response/run_forced_response.py force \
  -q custom_bagel_001 \
  -n 5 \
  -r 0
```

- `-q` / `--question-id`: Which verified question to force
- `-n` / `--num-forces`: Number of samples per sentence (default: 5, all obtained in one Tinker call)
- `-r` / `--rollout-idx`: Which verification rollout's CoT to use as source

Results saved to: `data/forced_response/forcing/{question_id}/rollout_{idx}/`

#### 3. Monitor — CoT in user message

Places partial CoT in the user message and asks the model to predict the answer. This is a "monitor" approach (the model sees the reasoning as context, not as its own thinking). Uses OpenRouter API.

```bash
python src/tasks/forced_response/run_forced_response.py monitor \
  -q custom_bagel_001 \
  -n 5 \
  -w 300 \
  -r 0
```

- `-w` / `--max-workers`: Max concurrent API calls (default: 300)

Results saved to: `data/forced_response/monitor/{question_id}/rollout_{idx}/`

#### 4. Full pipeline — verify then force

```bash
python src/tasks/forced_response/run_forced_response.py full \
  --num-rollouts 50 \
  --num-forces 5
```

#### 5. List results

```bash
python src/tasks/forced_response/run_forced_response.py list
```

### Data Layout

```
data/forced_response/
  verification/{question_id}/
    rollouts/rollout_000.json       # individual rollout results
    summary.json                    # agreement stats
  forcing/{question_id}/
    rollout_000/
      sentence_000/force_000.json   # per-sentence Tinker forcing results
      ...
      summary.json                  # per-rollout summary
  monitor/{question_id}/
    rollout_000/
      sentence_000/force_000.json   # per-sentence monitor results
      ...
      summary.json
  monitor_forcing/                  # legacy data (old forcing approach)
  resampling/{question_id}/         # resample results
```