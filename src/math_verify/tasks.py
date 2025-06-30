import logging
from textwrap import dedent
from typing import Callable, Optional

import numpy as np
from lighteval.metrics.dynamic_metrics import SampleLevelMetric
from lighteval.metrics.utils.metric_utils import MetricCategory, MetricUseCase
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics import Metrics

from math_verify.few_shots import GSM8K_FEW_SHOTS, MATH_HARD_FEW_SHOTS
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

logger = logging.getLogger(__name__)


def as_lighteval_metric(
    metric: Callable[
        [list[str], list[str]], tuple[float, Optional[tuple[list[str], list[str]]]]
    ],
) -> SampleLevelMetric:
    def sample_level_fn(
        formatted_doc: Doc, golds: list[str], predictions: list[str]
    ) -> float:
        result, extracted_predictions = metric(golds, predictions)
        if extracted_predictions is not None:
            if not formatted_doc.specific:
                formatted_doc.specific = {}
            formatted_doc.specific["extracted_predictions"] = extracted_predictions
        return result

    return SampleLevelMetric(
        metric_name="extractive_match",
        sample_level_fn=sample_level_fn,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )


def create_avg_at_n_metric(
    n: int,
    gold_extraction_target: tuple,
    pred_extraction_target: tuple,
) -> SampleLevelMetric:
    """Create a custom avg@n metric that averages correctness across n generations per problem."""
    
    def sample_level_fn(
        formatted_doc: Doc, golds: list[str], predictions: list[str]
    ) -> float:
        if len(golds) > 1:
            raise Exception("Cannot compute avg@n with several golds")
        
        # Ensure we have exactly n predictions
        if len(predictions) != n:
            logger.warning(f"Expected {n} predictions for avg@{n}, got {len(predictions)}")
            # Pad with empty strings if needed, or truncate
            if len(predictions) < n:
                predictions = predictions + [""] * (n - len(predictions))
            else:
                predictions = predictions[:n]
        
        # Use the math_metric function to evaluate each prediction
        math_eval_fn = math_metric(
            gold_extraction_target=gold_extraction_target,
            pred_extraction_target=pred_extraction_target,
        )
        
        correct_count = 0
        all_extracted_predictions = []
        all_extracted_golds = []
        
        for pred in predictions:
            result, extracted_predictions = math_eval_fn(golds, [pred])
            correct_count += result
            if extracted_predictions:
                # extracted_predictions is a tuple of (golds, preds)
                all_extracted_golds.extend(extracted_predictions[0])  # gold extractions
                all_extracted_predictions.extend(extracted_predictions[1])  # pred extractions
        
        # Store extracted predictions for debugging
        if all_extracted_predictions or all_extracted_golds:
            if not formatted_doc.specific:
                formatted_doc.specific = {}
            formatted_doc.specific["extracted_predictions"] = (all_extracted_golds, all_extracted_predictions)
        
        # Return average correctness across n generations
        return correct_count / n
    
    return SampleLevelMetric(
        metric_name=f"avg@{n}:{n}_samples",  # Include sample count in name for lighteval
        sample_level_fn=sample_level_fn,
        category=MetricCategory.GENERATIVE_SAMPLING,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )


def math_hard_prompt_function(x: dict, task_name: str) -> Doc:
    answer = str(x["solution"])
    question = x["problem"]

    query = dedent(
        f"""\
Question: {question}
Step-by-Step Answer:\
"""
    ).strip()

    choices = [answer]
    return Doc(query=query, choices=choices, gold_index=0)


def math_prompt_function(x: dict, task_name: str) -> Doc:
    answer = str(x["answer"])
    question = x["problem"]

    query = dedent(
        f"""\
Question: {question}
Step-by-Step Answer:\
"""
    ).strip()

    # Always wrap the answer in LaTeX delimiters for consistent extraction
    if not (answer.startswith("$") and answer.endswith("$")):
        answer = f"${answer}$"
    
    choices = [answer]
    return Doc(query=query, choices=choices, gold_index=0)


def math_aime24_prompt_function(x: dict, task_name: str) -> Doc:
    answer = str(x["reference_solution"])
    question = x["problem"]

    query = dedent(
        f"""\
Question: {question}
Step-by-Step Answer:\
"""
    ).strip()

    choices = [f" {answer}"]
    return Doc(query=query, choices=choices, gold_index=0)


def math_amc23_prompt_function(x: dict, task_name: str) -> Doc:
    answer = str(x["answer"])
    question = x["question"]

    query = dedent(
        f"""\
Question: {question}
Step-by-Step Answer:\
"""
    ).strip()
    choices = [f" {answer}"]
    return Doc(query=query, choices=choices, gold_index=0)


def gsm8k_prompt_function(x: dict, task_name: str) -> Doc:
    answer = f"{x['answer'].split('####')[-1].strip()}"
    question = x["question"]

    query = dedent(
        f"""\
Question: {question}
Step-by-Step Answer:\
"""
    ).strip()

    choices = [f" {answer}"]
    return Doc(query=query, choices=choices, gold_index=0)


def create_few_shot_examples():
    """Create few-shot examples that lighteval can use"""
    few_shot_docs = []
    for shot in MATH_HARD_FEW_SHOTS:
        doc = Doc(
            query=f"Question: {shot['question']}\nStep-by-Step Answer:",
            choices=[shot['answer']],
            gold_index=0
        )
        few_shot_docs.append(doc)
    return few_shot_docs


math_hard_lighteval = [
    LightevalTaskConfig(
        name=f"math_hard:{subset}",
        suite=["lighteval", "math"],
        prompt_function=math_hard_prompt_function,
        hf_repo="lighteval/MATH-Hard",
        hf_subset=subset,
        evaluation_splits=["test"],
        generation_size=1024,
        metric=[
            as_lighteval_metric(
                math_metric(
                    gold_extraction_target=(
                        LatexExtractionConfig(boxed_match_priority=0),
                    ),
                    pred_extraction_target=(
                        LatexExtractionConfig(),
                        ExprExtractionConfig(),
                    ),
                )
            ),
        ],
        stop_sequence=["\nQuestion:", "\nProblem:", "\nquestion:", "\nproblem:"],
        trust_dataset=True,
        version=0,
    )
    for subset in [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
]

# Add avg@n versions of math_hard tasks
math_hard_avg_at_n_lighteval = [
    LightevalTaskConfig(
        name=f"math_hard:{subset}_avg@{n}",
        suite=["lighteval", "math"],
        prompt_function=math_hard_prompt_function,
        hf_repo="lighteval/MATH-Hard", 
        hf_subset=subset,
        evaluation_splits=["test"],
        generation_size=1024,
        metric=[
            create_avg_at_n_metric(
                n=n,
                gold_extraction_target=(LatexExtractionConfig(boxed_match_priority=0),),
                pred_extraction_target=(LatexExtractionConfig(), ExprExtractionConfig()),
            )
        ],
        stop_sequence=["\nQuestion:", "\nProblem:", "\nquestion:", "\nproblem:"],
        trust_dataset=True,
        version=0,
    )
    for subset in [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
    for n in [4, 8, 16]
]

math_500_lighteval = [
    LightevalTaskConfig(
        name="math_500",
        suite=["lighteval", "math"],
        prompt_function=math_prompt_function,
        hf_repo="HuggingFaceH4/MATH-500",
        hf_subset="default",
        evaluation_splits=["test"],
        generation_size=1024,
        metric=[
            as_lighteval_metric(
                math_metric(
                    gold_extraction_target=(
                        LatexExtractionConfig(try_extract_without_anchor=True),
                        ExprExtractionConfig(),
                    ),
                    pred_extraction_target=(
                        LatexExtractionConfig(),
                        ExprExtractionConfig(),
                    ),
                )
            ),
        ],
        stop_sequence=["\nQuestion:", "\nProblem:", "\nquestion:", "\nproblem:"],
        trust_dataset=True,
        version=0,
    )
]

# Add avg@n versions of math_500 tasks
math_500_avg_at_n_lighteval = [
    LightevalTaskConfig(
        name=f"math_500_avg@{n}",
        suite=["lighteval", "math"],
        prompt_function=math_prompt_function,
        hf_repo="HuggingFaceH4/MATH-500",
        hf_subset="default",
        evaluation_splits=["test"],
        generation_size=1024,
        metric=[
            create_avg_at_n_metric(
                n=n,
                gold_extraction_target=(LatexExtractionConfig(try_extract_without_anchor=True), ExprExtractionConfig()),
                pred_extraction_target=(LatexExtractionConfig(), ExprExtractionConfig()),
            )
        ],
        stop_sequence=["\nQuestion:", "\nProblem:", "\nquestion:", "\nproblem:"],
        trust_dataset=True,
        version=0,
    )
    for n in [4, 8, 16]
]


aime24_lighteval = [
    LightevalTaskConfig(
        name="aime24",
        suite=["lighteval", "math"],
        prompt_function=math_aime24_prompt_function,
        hf_repo="zwhe99/aime24",
        hf_subset="default",
        evaluation_splits=["test"],
        generation_size=1024,
        metric=[
            as_lighteval_metric(
                math_metric(
                    gold_extraction_target=(LatexExtractionConfig(),),
                    pred_extraction_target=(
                        LatexExtractionConfig(),
                        ExprExtractionConfig(),
                    ),
                )
            ),
        ],
        stop_sequence=["\nQuestion:", "\nProblem:", "\nquestion:", "\nproblem:"],
        trust_dataset=True,
        version=0,
    )
]

# Add avg@n versions of aime24 tasks
aime24_avg_at_n_lighteval = [
    LightevalTaskConfig(
        name=f"aime24_avg@{n}",
        suite=["lighteval", "math"],
        prompt_function=math_aime24_prompt_function,
        hf_repo="zwhe99/aime24",
        hf_subset="default",
        evaluation_splits=["test"],
        generation_size=1024,
        metric=[
            create_avg_at_n_metric(
                n=n,
                gold_extraction_target=(LatexExtractionConfig(),),
                pred_extraction_target=(LatexExtractionConfig(), ExprExtractionConfig()),
            )
        ],
        stop_sequence=["\nQuestion:", "\nProblem:", "\nquestion:", "\nproblem:"],
        trust_dataset=True,
        version=0,
    )
    for n in [4, 8, 16]
]

amc23_lighteval = [
    LightevalTaskConfig(
        name="amc23",
        suite=["lighteval", "math"],
        prompt_function=math_amc23_prompt_function,
        hf_repo="zwhe99/amc23",
        hf_subset="default",
        hf_filter=lambda x: len(x["question"].strip()) > 0,
        evaluation_splits=["test"],
        generation_size=1024,
        metric=[
            as_lighteval_metric(
                math_metric(
                    gold_extraction_target=(ExprExtractionConfig(),),
                    pred_extraction_target=(
                        LatexExtractionConfig(),
                        ExprExtractionConfig(),
                    ),
                )
            ),
        ],
        stop_sequence=["\nQuestion:", "\nProblem:", "\nquestion:", "\nproblem:"],
        trust_dataset=True,
        version=0,
    )
]

# Add avg@n versions of amc23 tasks
amc23_avg_at_n_lighteval = [
    LightevalTaskConfig(
        name=f"amc23_avg@{n}",
        suite=["lighteval", "math"],
        prompt_function=math_amc23_prompt_function,
        hf_repo="zwhe99/amc23",
        hf_subset="default",
        hf_filter=lambda x: len(x["question"].strip()) > 0,
        evaluation_splits=["test"],
        generation_size=1024,
        metric=[
            create_avg_at_n_metric(
                n=n,
                gold_extraction_target=(ExprExtractionConfig(),),
                pred_extraction_target=(LatexExtractionConfig(), ExprExtractionConfig()),
            )
        ],
        stop_sequence=["\nQuestion:", "\nProblem:", "\nquestion:", "\nproblem:"],
        trust_dataset=True,
        version=0,
    )
    for n in [4, 8, 16]
]

gsm8k_lighteval = [
    LightevalTaskConfig(
        name="gsm8k",
        suite=["lighteval", "math"],
        prompt_function=gsm8k_prompt_function,
        hf_repo="openai/gsm8k",
        hf_subset="main",
        hf_filter=lambda x: len(x["question"].strip()) > 0,
        evaluation_splits=["test"],
        generation_size=1024,
        stop_sequence=["\nQuestion:", "\nProblem:", "\nquestion:", "\nproblem:"],
        metric=[
            as_lighteval_metric(
                math_metric(
                    gold_extraction_target=(ExprExtractionConfig(),),
                    pred_extraction_target=(
                        LatexExtractionConfig(),
                        ExprExtractionConfig(),
                    )
                )
            ),
        ],
        trust_dataset=True,
        version=0,
    )
]

# Add avg@n versions of gsm8k tasks
gsm8k_avg_at_n_lighteval = [
    LightevalTaskConfig(
        name=f"gsm8k_avg@{n}",
        suite=["lighteval", "math"],
        prompt_function=gsm8k_prompt_function,
        hf_repo="openai/gsm8k",
        hf_subset="main",
        hf_filter=lambda x: len(x["question"].strip()) > 0,
        evaluation_splits=["test"],
        generation_size=1024,
        stop_sequence=["\nQuestion:", "\nProblem:", "\nquestion:", "\nproblem:"],
        metric=[
            create_avg_at_n_metric(
                n=n,
                gold_extraction_target=(ExprExtractionConfig(),),
                pred_extraction_target=(LatexExtractionConfig(), ExprExtractionConfig()),
            )
        ],
        trust_dataset=True,
        version=0,
    )
    for n in [4, 8, 16]
]


TASKS_TABLE = [
    *gsm8k_lighteval,
    *gsm8k_avg_at_n_lighteval,
    *math_hard_lighteval,
    *math_hard_avg_at_n_lighteval,
    *math_500_lighteval,
    *math_500_avg_at_n_lighteval,
    *aime24_lighteval,
    *aime24_avg_at_n_lighteval,
    *amc23_lighteval,
    *amc23_avg_at_n_lighteval,
]
