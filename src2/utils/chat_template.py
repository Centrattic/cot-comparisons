"""
Model-agnostic chat template builder for thinking prompts.

Uses the tokenizer's own chat_template (via apply_chat_template) so the
prompt format is always correct for whatever model is loaded — Qwen3,
Kimi K2, or anything else with a HuggingFace chat template.
"""

DEFAULT_SYSTEM_MSG = "You are a helpful assistant."


def build_thinking_prompt(
    tokenizer,
    user_msg: str,
    cot_prefix: str = "",
    system_msg: str = DEFAULT_SYSTEM_MSG,
) -> str:
    """
    Build a prefill-forcing prompt that ends with ``<think>`` + *cot_prefix*.

    Args:
        tokenizer: A HuggingFace tokenizer with a chat template.
        user_msg: The user's question / task text.
        cot_prefix: Partial chain-of-thought to prefill (empty for free rollouts).
        system_msg: System message (default: generic helpful assistant).

    Returns:
        A fully-formatted prompt string ready for ``tokenizer.encode()``.
    """
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    # add_generation_prompt=True appends the assistant turn opening
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    # Append <think> + newline + CoT prefix for prefill forcing.
    # The newline matches Qwen3's training format: <think>\n{reasoning}
    return prompt + "<think>\n" + cot_prefix
