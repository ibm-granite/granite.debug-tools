"""
Helper functions shared between generate_variations.py and test_variations.py
"""

import json
import re
import logging
import os
import ast
import uuid
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from prompts import (
    prompt_decomposition,
    prompt_variation,
    prompt_correctness_value,
    prompt_sequential_segments_answers_reasoning,
    prompt_segment
)
from math_verify import parse, verify


@dataclass
class UniversalSample:
    instruction: str  # Question
    final_answer: str  # Final answer
    original_data: dict  # Store the complete original object
    sample_id: str
    file_format: str  # Track original format: 'jsonl', 'json', 'txt'
    sub_task: Optional[str]
    sub_task_answer: Optional[List[Dict[str, str]]]
    scaffolding: Optional[List[str]]
    evaluation: Optional[str]
    decomposition_evaluation: Optional[List[Dict[str, str]]]
    scaffolding_evaluation: Optional[List[Dict[str, str]]]
    evaluation_score: Optional[int]
    skill: Optional[str]
    decompositions: Optional[List[str]]
    verification: Optional[List[Dict[str, str]]]
    decomposition_score: Optional[Dict[str, List[int]]]


def load_universal_samples(mode: str, file_path: str) -> List[UniversalSample]:
    """Load samples from JSONL file with instruction/response format."""
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == '.jsonl':
        return load_jsonl_samples(mode, file_path)
    else:
        raise ValueError(f"Only JSONL format is supported. Got: {file_ext}")


def load_jsonl_samples(mode: str, file_path: str) -> List[UniversalSample]:
    """Load and parse samples from JSONL file with instruction/response format."""
    try:
        samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line.strip())

                    # Extract instruction
                    instruction = item.get("question", "")
                    ground_truth = item.get("answer", "")
                    sub_task = item.get("sub-task", [])
                    sub_task_answer = item.get("sub-task-answer", [])
                    scaffolding = item.get("scaffolding", [])
                    decompositions = item.get("decompositions", [])
                    evaluation = item.get("evaluation", '')
                    evaluation_score = item.get("evaluation_score", 0)
                    verification = item.get("scaffolding_verification", [])
                    decomposition_evaluation = item.get("decomposition_evaluation", [])
                    scaffolding_evaluation = item.get("scaffolding_evaluation", [])
                    skill = item.get("skill", [])
                    decomposition_score = item.get("decomposition_score", {})

                    mode_conditions = {
                        'answers': bool(sub_task),
                        'scaffolding': bool(sub_task_answer),
                        'verify': bool(scaffolding),
                        'decomposition': len(verification) > 0 and all(v.get('score', 0) == 1 for v in verification),
                        'debugging': bool(decomposition_evaluation),
                        'var_eval': bool(decompositions) and evaluation_score == 0,
                        'var_judge': bool(decomposition_evaluation),
                        'var_scaff_eval': bool(scaffolding) and evaluation_score == 0,
                        'var_scaff_judge': bool(scaffolding_evaluation),
                    }

                    # Skip if mode is known and condition is False
                    if mode in mode_conditions and not mode_conditions[mode]:
                        print(f"⏩ Skipping sample — mode='{mode}' | "
                              f"sub_task={len(sub_task)} | "
                              f"sub_task_answer={len(sub_task_answer)} | "
                              f"scaffolding={len(scaffolding)} | "
                              f"scaffolding_eval={len(verification)}")
                        continue

                    samples.append(UniversalSample(
                        instruction=instruction,
                        final_answer=ground_truth,
                        original_data=item,
                        sample_id=uuid.uuid4(),
                        file_format='jsonl',
                        sub_task=sub_task,
                        sub_task_answer=sub_task_answer,
                        scaffolding=scaffolding,
                        decompositions=decompositions,
                        evaluation=evaluation,
                        evaluation_score=evaluation_score,
                        decomposition_evaluation=decomposition_evaluation,
                        scaffolding_evaluation=scaffolding_evaluation,
                        skill=skill,
                        verification=verification,
                        decomposition_score=decomposition_score
                    ))

                except json.JSONDecodeError as e:
                    logging.warning(f"Skipping malformed JSON at line {i}: {e}")
                    continue

        logging.info(f"Loaded {len(samples)} samples from JSONL file {file_path}")
        return samples

    except Exception as e:
        logging.error(f"Error loading JSONL file: {str(e)}")
        raise


def get_prompt_segment(question: str) -> str:
    return prompt_segment.replace('{{ question }}', question)


def get_prompt_segment_answer_reasoning(question: str, segments: str) -> str:
    return prompt_sequential_segments_answers_reasoning.replace('{{ question }}', question).replace('{{ sequential_segments }}', segments)


def get_prompt_correctness_value(model_prediction: str, ground_truth: str) -> str:
    # lines = [line.strip() for line in model_prediction.strip().split("\n") if line.strip()]
    # last_line = lines[-1] if lines else model_prediction
    return prompt_correctness_value.replace('{{validator_final_answer}}', model_prediction.strip()).replace('{{ground_truth}}', ground_truth)


def get_prompt_variation(question: str, solved_segments: List[Dict[str, str]]) -> str:
    solved_segments = to_json_safe(solved_segments)
    return prompt_variation.replace('{{ question }}', question).replace('{{ solved_sequential_segments }}', json.dumps(solved_segments, indent=2))


def get_prompt_decomposition(question: str, specific_step_name: str, prior_info_as_list: List[Dict[str, str]]) -> str:
    prior_info_as_list = to_json_safe(prior_info_as_list)
    return prompt_decomposition.replace('{{ question }}', question).replace('{{ specific_step_name }}', specific_step_name).replace('{{ prior_info_as_list }}', json.dumps(prior_info_as_list, indent=2))


def parse_llm_output(key: str, llm_output: str):
    """
    Extract a list of dicts from LLM output like:
    [{"segment": "..."} , {"segment": "..."}]
    Handles messy backslashes, quotes, and newlines.
    """
    pattern = rf'"{re.escape(key)}"\s*:\s*(?:"([^"]*?)"|(-?\d+))'
    matches = re.findall(pattern, llm_output, re.DOTALL)

    if not matches:
        print("LLM output:\n", llm_output)
        print(f"No occurrences of key '{key}' found.")
        return []

    values = [m[0] if m[0] else m[1] for m in matches]

    # Then apply replace only on strings
    result = [{key: v.replace('\\n', '\n').replace('\\t', '\t')}
          if isinstance(v, str) else {key: v}
          for v in values]
    return result

def parse_llm_output_multiple(key1: str, key2: str, llm_output: str) -> List[Dict[str, str]]:
    # Escape keys for use in regex
    key1_pattern = re.escape(key1)
    key2_pattern = re.escape(key2)

    # Match JSON array of dicts with both keys, in any order
    pattern = (
        rf'\[\s*(?:{{'
        rf'\s*"{key1_pattern}"\s*:\s*".*?"\s*,\s*"{key2_pattern}"\s*:\s*".*?"\s*'
        rf'}}\s*,?\s*)+\]'
    )

    matches = re.findall(pattern, llm_output, re.DOTALL)
    # If regex found candidates
    if matches:
        last_match = matches[-1]
        safe_json = sanitize_json_string(clean_possible_json_block(last_match))
        try:
            parsed = json.loads(safe_json)
        except json.JSONDecodeError as e:
            print("JSON decode error:", e)
            return []
    else:
        # Fallback attempt on entire output
        llm_output_clean = sanitize_json_string(clean_possible_json_block(llm_output))
        try:
            parsed = json.loads(llm_output_clean)
        except Exception:
            return []

    # Validate structure
    if not isinstance(parsed, list):
        return []

    valid_items = []
    for item in parsed:
        if isinstance(item, dict) and sorted(item.keys()) == sorted([key1, key2]):
            valid_items.append(item)

    return valid_items


def clean_possible_json_block(text: str) -> str:
    """Removes Markdown fences and extracts the likely JSON substring."""
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    text = text.strip()
    match = re.search(r'(\[.*\]|\{.*\})', text, flags=re.DOTALL)
    return match.group(1).strip() if match else text

def sanitize_json_string(s: str) -> str:
    """Prepare JSON-like text for parsing without breaking structure."""
    return s.strip()

# def sanitize_json_string(s: str) -> str:
#     """Escapes unescaped backslashes and control characters for safe JSON parsing."""
#     s = re.sub(r'(?<!\\)\\(?![\\nt"r])', r'\\\\', s)
#     s = s.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
#     return s


def to_json_safe(obj, replace_ellipsis: bool = False):
    """
    Unified function to make objects JSON-serializable.

    Handles:
    - Sets → Lists
    - Ellipsis (...) → None (if replace_ellipsis=True)
    - Recursively processes dicts, lists, tuples

    Args:
        obj: Object to sanitize
        replace_ellipsis: If True, replaces ellipsis with None

    Returns:
        JSON-serializable version of obj
    """
    # Handle ellipsis
    if replace_ellipsis and (obj is ... or obj == "..."):
        return None

    # Handle sets
    if isinstance(obj, set):
        return [to_json_safe(v, replace_ellipsis) for v in obj]

    # Handle dicts
    if isinstance(obj, dict):
        return {k: to_json_safe(v, replace_ellipsis) for k, v in obj.items()}

    # Handle lists
    if isinstance(obj, list):
        return [to_json_safe(v, replace_ellipsis) for v in obj]

    # Handle tuples (convert to list for JSON)
    if isinstance(obj, tuple):
        return [to_json_safe(v, replace_ellipsis) for v in obj]

    # Return as-is for primitives and other types
    return obj


# Backward compatibility aliases
def replace_ellipsis(obj):
    """Replace ellipsis with None recursively. Legacy wrapper for to_json_safe."""
    return to_json_safe(obj, replace_ellipsis=True)


def sanitize_for_json(obj):
    """Sanitize objects for JSON serialization. Legacy wrapper for to_json_safe."""
    return to_json_safe(obj, replace_ellipsis=False)


def extract_json_from_string(text):
    """Extract the last valid JSON object from a string. Returns {} if none found."""
    text = text.strip('\"')
    brace_stack = []
    start_idx = None
    last_valid = None

    for i, char in enumerate(text):
        if char == '{':
            if not brace_stack:
                start_idx = i
            brace_stack.append('{')
        elif char == '}':
            if brace_stack:
                brace_stack.pop()
                if not brace_stack and start_idx is not None:
                    json_str = text[start_idx:i+1]
                    json_str = json_str.replace("\n", " ").strip()

                    # Try strict JSON first
                    try:
                        parsed = json.loads(json_str)
                        last_valid = parsed
                        start_idx = None
                        continue
                    except json.JSONDecodeError:
                        # Try Python literal safely
                        try:
                            if not json_str.startswith("{{"):
                                parsed = ast.literal_eval(json_str)
                                if isinstance(parsed, (dict, list)):
                                    last_valid = parsed
                        except Exception:
                            pass

                    start_idx = None

    # Optional ellipsis cleanup
    if last_valid and 'replace_ellipsis' in globals():
        last_valid = replace_ellipsis(last_valid)

    return last_valid or {}


def extract_rewritten_question(text: str) -> str:
    """Extract rewritten question from LLM response"""
    match = re.search(r"rewritten question:\s*(.+?)(?:\n\s*\n|$)", text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


def sub_task_answer_consistency(math_verifier: bool, samples: List[UniversalSample], client: Any = None, judge_model: str = "phi-4",
                              batch_size: int = 100, max_workers: int = 25) -> List[UniversalSample]:
    """Evaluate universal samples using the LLM judge - OPTIMIZED."""
    evaluated_samples = []

    if math_verifier:
        for sample in samples:
            gold_expr = sample.final_answer
            last_subtask = sample.sub_task_answer[-1]

            # Use 'answer' if it exists, else 'explanation', else skip
            if 'answer' in last_subtask and last_subtask['answer']:
                answer_expr = last_subtask['answer']
            elif 'explanation' in last_subtask and last_subtask['explanation']:
                answer_expr = last_subtask['explanation']
            else:
                continue

            try:
                gold = parse(gold_expr)
                answer = parse(answer_expr)
                result = verify(gold, answer)
                if result:
                    evaluated_samples.append(sample)
            except Exception as e:
                logging.warning(f"Skipping sample due to parse/verify error: {e}")
                continue
    else:
        if client is None:
            raise ValueError("Client must be provided when math_verifier is False")

        print(f"Evaluating {len(samples)} universal samples using {judge_model}...")
        print(f"OPTIMIZED SETTINGS: batch_size={batch_size}, max_workers={max_workers}")

        # Process in batches
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(samples) + batch_size - 1)//batch_size} ({len(batch)} samples)")

            user_prompts = [get_prompt_correctness_value(sample.sub_task_answer[-1]['answer'], str(sample.final_answer)) for sample in batch]

            try:
                responses = client.get_model_response(
                    user_prompts=user_prompts,
                    max_new_tokens=1024,
                    temperature=0.2,
                    top_k=0,
                    top_p=1.0
                )

                for sample, response in zip(batch, responses):
                    score = extract_json_from_string(response)['score']

                    if score == 1:
                        evaluated_samples.append(sample)

                    if len(evaluated_samples) % 100 == 0:
                        print(f"  Processed {len(evaluated_samples)}/{len(samples)} samples")

            except Exception as e:
                logging.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")

    return evaluated_samples


# Unified Save Function
def save_universal_samples(
    evaluated_samples: List[Tuple[UniversalSample, Any]],
    output_file: str,
    field_name: str,
    process_fn = None
):
    """
    Unified function to save evaluated samples with flexible field processing.

    Args:
        evaluated_samples: List of (sample, evaluation) tuples
        output_file: Path to output JSONL file
        field_name: Name of the field to add/update in the output
        process_fn: Optional function to process evaluation before saving.
                   If None, evaluation is saved directly.

    Examples:
        # Save sub-tasks (parse LLM output)
        save_universal_samples(samples, "output.jsonl", "sub-task",
                              lambda x: parse_llm_output("segment", x))

        # Save sub-task answers (parse multiple keys)
        save_universal_samples(samples, "output.jsonl", "sub-task-answer",
                              lambda x: parse_llm_output_multiple("explanation", "answer", x))

        # Save scaffolding (direct)
        save_universal_samples(samples, "output.jsonl", "scaffolding")

        # Save with dynamic field name (like variation_type + '_score')
        save_universal_samples(samples, "output.jsonl", "decomposition_score")
    """
    with open(output_file, 'w', encoding='utf-8', buffering=8192) as f:
        for sample, evaluation in evaluated_samples:
            scored_item = sample.original_data.copy()

            # Process evaluation if process_fn is provided, otherwise use directly
            if process_fn is not None:
                scored_item[field_name] = process_fn(evaluation)
            else:
                scored_item[field_name] = evaluation

            safe_json = to_json_safe(scored_item)
            f.write(json.dumps(safe_json, ensure_ascii=False) + '\n')


# Backward compatibility wrappers (for existing code)
def save_scored_universal_samples_first(evaluated_samples: List[Tuple[UniversalSample, str]],
                                output_file: str):
    """Save sub-tasks by parsing LLM output."""
    save_universal_samples(
        evaluated_samples,
        output_file,
        'sub-task',
        lambda x: parse_llm_output("segment", x)
    )

