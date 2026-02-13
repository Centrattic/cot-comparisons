# Scratch: Prompt Injection Testing

Quick-and-dirty tests of indirect prompt injection (IPI) attacks on a
Qwen-32B ReAct agent scaffold.

Based on [InjecAgent (Zhan et al., 2024)](https://arxiv.org/abs/2403.02691).

## Setup
```bash
pip install openai   # used for OpenRouter's OpenAI-compatible endpoint
```

Set `OPENROUTER_API_KEY` env var with your OpenRouter key.

## Usage
```bash
python -m scratch_injection_tests.run_injections
```
