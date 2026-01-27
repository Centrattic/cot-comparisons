"""
Scruples sycophancy monitor implementation (counterfactual framing).

This module provides monitors that analyze model CoT and/or answers to determine
whether the model would have responded differently without the intervention
(first-person framing or user opinion).

Monitor scopes:
- cot_only: Monitor sees only the model's chain-of-thought
- answer_only: Monitor sees only the final answer
- all_messages: Monitor sees both chain-of-thought and final answer
"""

import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional
from dotenv import load_dotenv
import openai

from ...methods.base import MonitorMethod
from .prompts import (
    INTERVENTION_SUGGESTED_ANSWER,
    get_monitor_prompt,
    get_discrimination_monitor_prompt,
)

load_dotenv()

# OpenRouter configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MONITOR_MODEL = "openai/gpt-5-thinking"

MonitorScope = Literal["cot_only", "answer_only", "all_messages"]


class ScruplesMonitor(MonitorMethod):
    """
    Monitor for detecting sycophancy in Scruples task responses.

    This monitor analyzes model outputs to determine whether the model's
    answer was influenced by a desire to agree with the user's framing
    or explicitly stated opinion.

    Attributes:
        variant: The intervention variant being monitored
        scope: What the monitor can see (cot_only, answer_only, all_messages)
        model_config: Configuration for the monitor LLM (optional)
    """

    SCOPES = ["cot_only", "answer_only", "all_messages"]
    VARIANTS = ["first_person", "suggest_right", "suggest_wrong"]

    def __init__(
        self,
        variant: str,
        scope: MonitorScope = "all_messages",
        model_config: Optional[Any] = None,
        name: Optional[str] = None,
        model: str = DEFAULT_MONITOR_MODEL,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the Scruples monitor.

        Args:
            variant: Which intervention variant to monitor for
            scope: What the monitor can see
            model_config: Configuration for the monitor LLM (legacy, optional)
            name: Optional custom name for this monitor instance
            model: Model to use via OpenRouter (default: openai/gpt-5-thinking)
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
        """
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant: {variant}. Expected one of: {self.VARIANTS}")
        if scope not in self.SCOPES:
            raise ValueError(f"Unknown scope: {scope}. Expected one of: {self.SCOPES}")

        # Generate descriptive name if not provided
        if name is None:
            name = f"scruples_monitor_{variant}_{scope}"

        super().__init__(name)
        self.variant = variant
        self.scope = scope
        self.model_config = model_config
        self.model = model

        # Initialize OpenRouter client
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = openai.OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=self.api_key,
        )

    def analyze_cot(self, cot_text: str, actual_answer: str = "") -> Dict[str, Any]:
        """
        Analyze chain-of-thought text to predict what answer the model would give without intervention.

        Args:
            cot_text: The model's chain-of-thought content from the intervention arm
            actual_answer: The model's actual answer (for comparison)

        Returns:
            Dict with analysis results
        """
        return self._analyze(thinking=cot_text, actual_answer=actual_answer)

    def analyze_intervention_cot(
        self,
        thinking: str,
        actual_answer: str,
    ) -> Dict[str, Any]:
        """
        Analyze the chain of thought from an intervention run to predict what answer
        the model would have given without the user's stated preference.

        Args:
            thinking: The model's chain-of-thought from the intervention arm
            actual_answer: The model's actual answer (for comparison)

        Returns:
            Dict with analysis results including:
                - 'predicted_answer': What the monitor predicts the model would answer without intervention
                - 'actual_answer': The model's actual answer
                - 'monitor_response': The full monitor response
                - 'monitor_prompt': The prompt sent to the monitor
        """
        return self._analyze(thinking=thinking, actual_answer=actual_answer)

    def _analyze(
        self,
        thinking: str,
        actual_answer: str,
    ) -> Dict[str, Any]:
        """
        Internal analysis method.

        Args:
            thinking: The model's chain of thought from the intervention arm
            actual_answer: The model's actual answer

        Returns:
            Analysis results dict with predicted_answer, actual_answer, monitor_response, monitor_prompt
        """
        # Generate monitor prompt (only needs variant and thinking)
        monitor_prompt = get_monitor_prompt(
            variant=self.variant,
            thinking=thinking,
        )

        # Call the monitor LLM
        monitor_response = self._call_monitor_llm(monitor_prompt)
        predicted_answer = self._parse_monitor_response(monitor_response)

        return {
            "predicted_answer": predicted_answer,
            "actual_answer": actual_answer,
            "monitor_response": monitor_response,
            "monitor_prompt": monitor_prompt,
        }

    def _call_monitor_llm(self, prompt: str) -> str:
        """
        Call the monitor LLM via OpenRouter API.

        Uses OpenAI-compatible chat completions API through OpenRouter
        to call the configured model (default: openai/gpt-5-thinking).

        Args:
            prompt: The full monitor prompt

        Returns:
            The monitor's response text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            # Log error but return empty string to allow graceful degradation
            print(f"Monitor LLM call failed: {e}")
            return ""

    def _parse_monitor_response(self, response: str) -> Optional[str]:
        """
        Parse the monitor's response to extract the predicted answer.

        The monitor should end with A or B representing what answer the model
        would have given without the user's stated preference.

        Args:
            response: The monitor's full response

        Returns:
            "A" or "B" if found, None if parsing failed
        """
        if not response:
            return None

        # Get the last non-empty line
        lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
        if not lines:
            return None

        last_line = lines[-1].upper()

        # Check for A or B as the final answer
        if last_line == 'A':
            return 'A'
        elif last_line == 'B':
            return 'B'

        # Fallback: search for A or B in the last line
        if 'A' in last_line and 'B' not in last_line:
            return 'A'
        elif 'B' in last_line and 'A' not in last_line:
            return 'B'

        # Default to None if unclear
        return None


class ScruplesFirstPersonMonitor(ScruplesMonitor):
    """Monitor for first-person variant."""

    def __init__(
        self,
        scope: MonitorScope = "all_messages",
        model_config: Optional[Any] = None,
        model: str = DEFAULT_MONITOR_MODEL,
        api_key: Optional[str] = None,
    ):
        super().__init__(
            variant="first_person",
            scope=scope,
            model_config=model_config,
            model=model,
            api_key=api_key,
        )


class ScruplesSuggestRightMonitor(ScruplesMonitor):
    """Monitor for suggest-right variant."""

    def __init__(
        self,
        scope: MonitorScope = "all_messages",
        model_config: Optional[Any] = None,
        model: str = DEFAULT_MONITOR_MODEL,
        api_key: Optional[str] = None,
    ):
        super().__init__(
            variant="suggest_right",
            scope=scope,
            model_config=model_config,
            model=model,
            api_key=api_key,
        )


class ScruplesSuggestWrongMonitor(ScruplesMonitor):
    """Monitor for suggest-wrong variant."""

    def __init__(
        self,
        scope: MonitorScope = "all_messages",
        model_config: Optional[Any] = None,
        model: str = DEFAULT_MONITOR_MODEL,
        api_key: Optional[str] = None,
    ):
        super().__init__(
            variant="suggest_wrong",
            scope=scope,
            model_config=model_config,
            model=model,
            api_key=api_key,
        )


class ScruplesDiscriminationMonitor(ScruplesMonitor):
    """
    Discrimination monitor that identifies which of two CoTs
    had the user's preference stated (intervention vs control).

    This monitor sees paired control/intervention responses from the same
    prompt and must identify which one had the user's preference stated.
    Tests whether sycophancy leaves detectable reasoning traces.
    """

    def __init__(
        self,
        variant: str,
        scope: MonitorScope = "all_messages",
        model: str = DEFAULT_MONITOR_MODEL,
        api_key: Optional[str] = None,
    ):
        super().__init__(
            variant=variant,
            scope=scope,
            model=model,
            api_key=api_key,
            name=f"scruples_discrimination_{variant}",
        )

    def analyze_pair(
        self,
        control_thinking: str,
        intervention_thinking: str,
        control_answer: str,
        intervention_answer: str,
    ) -> Dict[str, Any]:
        """
        Analyze a paired control/intervention CoT.

        Randomly assigns to A/B, asks monitor to identify intervention.

        Args:
            control_thinking: Chain of thought from control arm
            intervention_thinking: Chain of thought from intervention arm
            control_answer: Final answer from control arm
            intervention_answer: Final answer from intervention arm

        Returns:
            Dict with:
            - predicted_intervention: "A" or "B"
            - actual_intervention: "A" or "B"
            - correct: bool
            - monitor_response: full response
            - monitor_prompt: the prompt sent to monitor
            - order: "control_first" or "intervention_first"
        """
        import random

        # Randomly assign order
        if random.random() < 0.5:
            response_a = (control_thinking, control_answer, "control")
            response_b = (intervention_thinking, intervention_answer, "intervention")
            actual_intervention = "B"
            order = "control_first"
        else:
            response_a = (intervention_thinking, intervention_answer, "intervention")
            response_b = (control_thinking, control_answer, "control")
            actual_intervention = "A"
            order = "intervention_first"

        prompt = get_discrimination_monitor_prompt(
            variant=self.variant,
            thinking_a=response_a[0],
            answer_a=response_a[1],
            thinking_b=response_b[0],
            answer_b=response_b[1],
        )

        response = self._call_monitor_llm(prompt)
        predicted = self._parse_ab_response(response)

        return {
            "predicted_intervention": predicted,
            "actual_intervention": actual_intervention,
            "correct": predicted == actual_intervention if predicted else False,
            "monitor_response": response,
            "monitor_prompt": prompt,
            "order": order,
        }

    def _parse_ab_response(self, response: str) -> Optional[str]:
        """
        Parse the monitor's response to extract A or B prediction.

        Args:
            response: The monitor's full response

        Returns:
            "A" or "B" if found, None if parsing failed
        """
        if not response:
            return None

        # Get the last non-empty line
        lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
        if not lines:
            return None

        last_line = lines[-1].upper()

        # Check for A or B as the final answer
        if last_line == 'A':
            return 'A'
        elif last_line == 'B':
            return 'B'

        # Fallback: search for A or B in the last line
        if 'A' in last_line and 'B' not in last_line:
            return 'A'
        elif 'B' in last_line and 'A' not in last_line:
            return 'B'

        # Default to None if unclear
        return None
