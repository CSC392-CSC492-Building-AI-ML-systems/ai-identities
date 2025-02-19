import random
import re
import pandas as pd
from typing import Any, Literal
from dataclasses import dataclass, field
import json
import openai
import ollama

# Define message structure
Message = dict[str, Any]  # Keys: role, content
MessageList = list[Message]

# Regex to extract final answer from response
ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"


class SamplerBase:
    """Base class for defining a model sampling method."""

    def __call__(self, message_list: MessageList) -> str:
        """Override this method in a subclass to define a model's response behavior."""
        raise NotImplementedError

class ModelSampler:
    """A sampler that dynamically selects a model and generates responses."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def __call__(self, message_list: MessageList) -> str:
        """Generates a response using the specified model."""
        if self.model_name.startswith("ollama:chat:"):
            # Call Ollama for local LLMs
            response = ollama.chat(model=self.model_name.split(":")[-1], messages=message_list)
            return response["message"]["content"]

        elif self.model_name.startswith("openai:"):
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model_name.split(":")[-1],
                messages=message_list
            )
            return response["choices"][0]["message"]["content"]

        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

@dataclass
class EvalResult:
    """Stores the results of running an evaluation."""
    score: float | None
    metrics: dict[str, float] | None
    htmls: list[str]
    convos: list[MessageList]


@dataclass
class SingleEvalResult:
    """Stores the results of a single evaluation sample."""
    score: float | None
    metrics: dict[str, float] = field(default_factory=dict)
    html: str | None = None
    convo: MessageList | None = None


class Eval:
    """Base evaluation class."""
    def __call__(self, sampler: SamplerBase) -> EvalResult:
        raise NotImplementedError


EQUALITY_TEMPLATE = """
Look at the following two expressions and determine whether they are equivalent. Perform only trivial simplifications.

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Yes

    Expression 1: 3/2
    Expression 2: 1.5

Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $(x+1)^2$

Yes

---

Respond with only "Yes" or "No".

    Expression 1: {expression1}
    Expression 2: {expression2}
""".strip()


def check_equality(sampler: SamplerBase, expr1: str, expr2: str) -> bool:
    """Checks if two mathematical expressions are equivalent using the model."""
    prompt = EQUALITY_TEMPLATE.format(expression1=expr1, expression2=expr2)
    response = sampler([{"content": prompt, "role": "user"}])
    return response.lower().strip() == "yes"


class MathEval(Eval):
    """Math evaluation class for testing model accuracy on math problems."""

    def __init__(
        self,
        equality_checker: SamplerBase,
        num_examples: int | None = None,
        n_repeats: int = 16,
        split: Literal["math_test", "math_500_test"] = "math_test",
    ):
        # Load dataset
        df = pd.read_csv(f"https://openaipublic.blob.core.windows.net/simple-evals/{split}.csv")
        examples = df.to_dict(orient="records")

        if num_examples:
            assert n_repeats == 1, "n_repeats only supported when num_examples=None"
            rng = random.Random(0)
            examples = rng.sample(examples, num_examples)

        self.examples = examples * n_repeats
        self.equality_checker = equality_checker  # Must be a SamplerBase instance!

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        results = []

        for row in self.examples:
            prompt_messages = [{"content": row["Question"], "role": "user"}]
            response_text = sampler(prompt_messages)

            match = re.search(ANSWER_PATTERN, response_text)
            extracted_answer = match.group(1) if match else None

            if extracted_answer is None:
                score = 0  # If no valid answer was extracted, it's incorrect
            else:
                score = float(check_equality(self.equality_checker, row["Answer"], extracted_answer))

            results.append(
                SingleEvalResult(
                    score=score,
                    convo=prompt_messages + [{"content": response_text, "role": "assistant"}],
                )
            )

        valid_scores = [r.score for r in results if r.score is not None]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0

        return EvalResult(
            score=avg_score,
            metrics={"accuracy": avg_score},
            htmls=[],
            convos=[r.convo for r in results],
        )


if __name__ == "__main__":
    # List of models from your promptfoo config
    models = [
        "ollama:chat:llama3.2",
        "ollama:chat:deepseek-r1:1.5b",
        "ollama:chat:qwen:1.8b",
        "ollama:chat:gemma2:2b",
        "ollama:chat:phi3:3.8b",
        "ollama:chat:mistral",
        "ollama:chat:wizardlm2",
        "openai:gpt-4o"
    ]

    results = {}

    for model in models:
        sampler = ModelSampler(model)
        math_eval = MathEval(equality_checker=sampler)
        eval_result = math_eval(sampler)

        results[model] = {
            "pass": eval_result.score > 0.8,
            "score": eval_result.score,
            "reason": "High accuracy in math problems" if eval_result.score > 0.8 else "Low accuracy in math problems"
        }

    # Print all results
    print(json.dumps(results, indent=4))

