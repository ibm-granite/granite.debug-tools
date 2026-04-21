import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from dotenv import load_dotenv

from model_client import ModelClientFactory, BaseModelClient
from prompts import (
    prompt_decomposition,
    prompt_variation,
    prompt_correctness_value,
    prompt_sequential_segments_answers_reasoning,
    prompt_segment
)
from math_verify import parse, verify

# Load environment variables from .env file
load_dotenv()
from helpers import (
    UniversalSample,
    load_universal_samples,
    load_jsonl_samples,
    get_prompt_segment,
    get_prompt_segment_answer_reasoning,
    get_prompt_correctness_value,
    get_prompt_variation,
    get_prompt_decomposition,
    parse_llm_output,
    parse_llm_output_multiple,
    to_json_safe,  
    sanitize_for_json,  
    extract_json_from_string,
    extract_rewritten_question,
    sub_task_answer_consistency,
    save_universal_samples, 
    clean_possible_json_block
)

def evaluate_benchmark(samples: List[UniversalSample], client: BaseModelClient, judge_model: str = "phi-4", 
                              client_type: str = "vllm", batch_size: int = 100, max_workers: int = 25) -> List[Tuple[UniversalSample, str]]:
    evaluated_samples = []
    
    print(f"Evaluating {len(samples)} universal samples using {judge_model} via {client_type}...")
    print(f"OPTIMIZED SETTINGS: batch_size={batch_size}, max_workers={max_workers}")
    
    # Process in batches
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(samples) + batch_size - 1)//batch_size} ({len(batch)} samples)")
        
        # Prepare prompts
        # system_prompts = [JUDGE_SYSTEM_PROMPT] * len(batch)
        user_prompts = [sample.instruction for sample in batch]

        try:
            
            responses = client.get_model_response(
                    user_prompts=user_prompts,
                    max_new_tokens=1024,  
                    temperature=0.0,   
                    top_k=None,           
                    top_p=1.0          
            )
            
            # Parse results
            for sample, response in zip(batch, responses):
                evaluated_samples.append((sample, response))
                # OPTIMIZED: Less verbose logging
                if len(evaluated_samples) % 100 == 0:
                    print(f"  Processed {len(evaluated_samples)}/{len(samples)} samples")
        
        except Exception as e:
            logging.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
            # Add failed evaluations for this batch with neutral scores
            for sample in batch:
                # failed_evaluation = EvaluationResult(5, 5, 5, 5, "Batch failed", "Batch failed", "Batch failed", "Batch failed")
                evaluated_samples.append((sample, 'Error parsing final answer'))
    
    return evaluated_samples 

    
#     return evaluated_samples 
def judge_evaluation(math_verifier:bool, samples: List[UniversalSample], client: BaseModelClient, judge_model: str = "phi-4", 
                              client_type: str = "vllm", batch_size: int = 100, max_workers: int = 25) -> List[Tuple[UniversalSample, int]]:
    """Evaluate universal samples using the LLM judge via {client_type} - OPTIMIZED."""
    # OPTIMIZED: Increased max_workers and better error handling
    evaluated_samples = []

    if math_verifier:
        for sample in samples:
            gold_expr = sample.final_answer
            evaluation = sample.evaluation

            # Determine answer_expr safely
            if isinstance(evaluation, dict):
                if evaluation.get("answer"):
                    answer_expr = evaluation["answer"]
                elif evaluation.get("explanation"):
                    answer_expr = evaluation["explanation"]
                else:
                    answer_expr = evaluation
            else:
                # evaluation is already a string or number
                answer_expr = evaluation

            try:
                gold = parse(gold_expr)
                answer = parse(answer_expr)
                result = verify(gold, answer)
                evaluated_samples.append((sample, int(result)))
            except Exception as e:
                evaluated_samples.append((sample, -1))
                logging.warning(f"Skipping sample due to parse/verify error: {e}")
                continue



    
    else:

        print(f"Evaluating {len(samples)} universal samples using {judge_model} via {client_type}...")
        print(f"OPTIMIZED SETTINGS: batch_size={batch_size}, max_workers={max_workers}")
    
        # Process in batches
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(samples) + batch_size - 1)//batch_size} ({len(batch)} samples)")
        
            # Prepare prompts
            # system_prompts = [JUDGE_SYSTEM_PROMPT] * len(batch)

            parsed_answers = []

            # Parse each sample in the batch
            for sample in batch:
                evaluation = str(sample.evaluation or "").strip()

                # Skip empty evaluations
                if not evaluation:
                    parsed_answers.append("")
                    continue

                parsed_answer = extract_json_from_string(evaluation)

                # Default to raw evaluation text
                answer_expr = evaluation

                if isinstance(parsed_answer, dict) and parsed_answer:

                    # Case 1: Has explicit 'answer' field
                    if parsed_answer.get("answer"):
                        answer_expr = json.dumps(to_json_safe(parsed_answer["answer"]))
                        # answer_expr = json.dumps(parsed_answer["answer"])

                    # Case 2: Has only 'explanation' → remove it if possible
                    elif parsed_answer.get("explanation"):
                        filtered = {k: v for k, v in parsed_answer.items() if k != "explanation"}
                        if filtered:
                            answer_expr = json.dumps(to_json_safe(filtered))  # or json.dumps(filtered) if you need a string
                        else:
                            answer_expr = str(evaluation)

                # Always append a valid string or object
                parsed_answers.append(answer_expr)
        

            user_prompts = [get_prompt_correctness_value(str(answer), str(sample.final_answer)) for answer, sample in zip (parsed_answers, batch)]
            try:
            
                responses = client.get_model_response(
                    user_prompts=user_prompts,
                    max_new_tokens=1024,  
                    temperature=0.0,     
                    top_k=None,          
                    top_p=1.0           
                )
            
                # Parse results
                for sample, response in zip(batch, responses):

                    answer = sample.evaluation
                    parsed_answer = extract_json_from_string(sample.evaluation)

                    # Safely get the answer as a string, fallback to empty string
                    if isinstance(parsed_answer, dict) and parsed_answer:
                        if 'answer' in parsed_answer:
                            answer = str(parsed_answer.get("answer", ""))
                        else:
                            answer = str(parsed_answer)



                    # If answer is empty or <final answer only>, score = 0
                    if answer == "" or answer == "<final answer only>":
                        score = 0
                    else:
                        # Otherwise, try to get score from the response
                        parsed_response = extract_json_from_string(response)
                        if isinstance(parsed_response, dict) and 'score' in parsed_response:
                            score = parsed_response['score']
                        else:
                            score = 0  # fallback

                    evaluated_samples.append((sample, score))

                    if len(evaluated_samples) % 100 == 0:
                        print(f"  Processed {len(evaluated_samples)}/{len(samples)} samples")
        
            except Exception as e:
                logging.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                # Add failed evaluations for this batch with neutral scores
                for sample in batch:
                    # failed_evaluation = EvaluationResult(5, 5, 5, 5, "Batch failed", "Batch failed", "Batch failed", "Batch failed")
                    evaluated_samples.append((sample, -1))
    
    return evaluated_samples
def save_scored_universal_samples_scored(evaluated_samples: List[Tuple[UniversalSample, List[int]]],
                                output_file: str):
    """Save evaluation scores directly."""
    save_universal_samples(evaluated_samples, output_file, 'evaluation_score')

def evaluate_decompositions(samples: List[UniversalSample], client: BaseModelClient, judge_model: str = "phi-4", 
                     batch_size: int = 100, max_workers: int = 25) -> List[Tuple[UniversalSample, List[str]]]:
    """Evaluate universal samples using the LLM judge via {client_type} - supports variable number of subtasks."""
    
    # client = ModelClientFactory.create_client(client_type, judge_model, max_workers=max_workers)
    evaluated_samples = []
    
    # print(f"Evaluating {len(samples)} universal samples using {judge_model} via {client_type}...")
    print(f"OPTIMIZED SETTINGS: batch_size={batch_size}, max_workers={max_workers}")
    
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(samples) + batch_size - 1)//batch_size} ({len(batch)} samples)")

        # Prepare prompts for all subtasks in the batch
        user_prompts = []
        meta_info = []  # keep track of which sample + subtask index
        
        # formatting_instruction = '''Only output valid JSON, strictly following the format: {"explanation": "<step-by-step reasoning in plain text>", "answer": "<final answer only>"}'''

        for sample_idx, sample in enumerate(batch):
            num_subtasks = len(sample.decompositions)  # variable number of subtasks
            for j in range(num_subtasks):
                prompt = sample.decompositions[j]
                user_prompts.append(prompt)
                meta_info.append(sample_idx)
        
        # Run LLM inference
        try:
            responses = client.get_model_response(
                user_prompts=user_prompts,
                max_new_tokens=2000,
                temperature=0.0,
                top_k=None,
                top_p=1.0
            )
        except Exception as e:
            logging.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
            responses = [""] * len(user_prompts)
        
        # Non JSON version
        # Collect responses per sample
        variation_answers = [[] for _ in range(len(batch))]
        for response, sample_idx in zip(responses, meta_info):
            variation_answers[sample_idx].append(response)
     
        # Store evaluated samples
        for sample, variation_answer in zip(batch, variation_answers):
            evaluated_samples.append((sample, variation_answer))
            if len(evaluated_samples) % 100 == 0:
                print(f"  Processed {len(evaluated_samples)}/{len(samples)} samples")

 
    return evaluated_samples 
def evaluate_scaffoldings (samples: List[UniversalSample], client: BaseModelClient, judge_model: str = "phi-4", 
                     batch_size: int = 100, max_workers: int = 25) -> List[Tuple[UniversalSample, List[str]]]:
    """Evaluate universal samples using the LLM judge via {client_type} - supports variable number of subtasks."""
    
    # client = ModelClientFactory.create_client(client_type, judge_model, max_workers=max_workers)
    evaluated_samples = []
    
    # print(f"Evaluating {len(samples)} universal samples using {judge_model} via {client_type}...")
    print(f"OPTIMIZED SETTINGS: batch_size={batch_size}, max_workers={max_workers}")
    
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(samples) + batch_size - 1)//batch_size} ({len(batch)} samples)")

        # Prepare prompts for all subtasks in the batch
        user_prompts = []
        meta_info = []  # keep track of which sample + subtask index
        
        # formatting_instruction = '''Only output valid JSON, strictly following the format: {"explanation": "<step-by-step reasoning in plain text>", "answer": "<final answer only>"}'''

        for sample_idx, sample in enumerate(batch):
            num_subtasks = len(sample.scaffolding)  # variable number of subtasks
            for j in range(num_subtasks):
                prompt = sample.scaffolding[j]
                user_prompts.append(prompt)
                meta_info.append(sample_idx)
        
        # Run LLM inference
        try:
            responses = client.get_model_response(
                user_prompts=user_prompts,
                max_new_tokens=2000,
                temperature=0.0,
                top_k=None,
                top_p=1.0
            )
        except Exception as e:
            logging.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
            responses = [""] * len(user_prompts)
        
        # Non JSON version
        # Collect responses per sample
        variation_answers = [[] for _ in range(len(batch))]
        for response, sample_idx in zip(responses, meta_info):
            variation_answers[sample_idx].append(response)
     
        # Store evaluated samples
        for sample, variation_answer in zip(batch, variation_answers):
            evaluated_samples.append((sample, variation_answer))
            if len(evaluated_samples) % 100 == 0:
                print(f"  Processed {len(evaluated_samples)}/{len(samples)} samples")


    
    return evaluated_samples 
def judge_variations(
    math_verifier:bool,
    samples: List[UniversalSample],
    client: BaseModelClient,
    judge_model: str = "phi-4",
    client_type: str = "vllm",
    batch_size: int = 100,
    max_workers: int = 25
) -> List[Tuple[UniversalSample, Dict[str, List[int]]]]:
    """
    Evaluate decomposition_evaluation for each sample using the LLM judge via {client_type}.
    Produces one score per decomposition question, aligned with the input order.
    Empty or '<final answer only>' answers get score = 0.
    """
    evaluated_samples = []

    if math_verifier:
        for sample in samples or []:
            decomp = getattr(sample, "decomposition_evaluation", None)
            sub_tasks = getattr(sample, "sub_task_answer", None)
            if not decomp or not sub_tasks:
                continue

            score_dict = []
            if(len(decomp) != len(sub_tasks)):
                continue

            for ix in range(len(sub_tasks)):
                d = decomp[ix]
                gold_expr = sub_tasks[ix].get('answer', '')
                evaluation = d.get('answer', '')
                gold = parse(gold_expr)
                answer = parse(evaluation)
                result = verify(gold, answer)
                score_dict.append(1 if result else 0)

            evaluated_samples.append((sample, score_dict))


    else:
        # client = ModelClientFactory.create_client(client_type, judge_model, max_workers=max_workers)

        print(f"Evaluating {len(samples)} universal samples using {judge_model} via {client_type}...")
        print(f"OPTIMIZED SETTINGS: batch_size={batch_size}, max_workers={max_workers}")

        # Flatten decomposition evaluations into one queue
        to_judge = []
        index_map = []  # (sample_idx, j)

        for sample_idx, sample in enumerate(samples):
            if not sample.decomposition_evaluation:
                continue

            # Initialize aligned score array
            sample._scores = {'decomposition': [0] * len(sample.decomposition_evaluation)}

            for j, decomp_eval in enumerate(sample.decomposition_evaluation):
                model_answer = str(decomp_eval).strip()
                expected_answer = ""
                if sample.sub_task_answer and j < len(sample.sub_task_answer):
                    expected_answer = str(sample.sub_task_answer[j].get('answer', '')).strip()

                # Skip generating prompts for empty or placeholder answers (keep default 0)
                if not model_answer or model_answer == "<final answer only>":
                    continue

                # Queue valid comparison
                parsed_answer = extract_json_from_string(model_answer)

                # Parsing answer
                answer_expr = model_answer

                if isinstance(parsed_answer, dict):
                    # Case 1: Has explicit 'answer' field
                    if parsed_answer.get("answer") is not None:
                        #answer_expr = json.dumps(parsed_answer["answer"])
                        
                        
                        answer_expr = json.dumps(to_json_safe(parsed_answer["answer"])) if not isinstance(parsed_answer["answer"], str) else parsed_answer["answer"]


                    # Case 2: Has only 'explanation' → remove it if possible
                    elif parsed_answer.get("explanation"):
                        filtered = {k: v for k, v in parsed_answer.items() if k != "explanation"}
                        if filtered:
                            answer_expr = json.dumps(to_json_safe(filtered))  # or json.dumps(filtered) if you need a string
                        else:
                            answer_expr = str(model_answer)


                
                prompt = get_prompt_correctness_value(str(answer_expr), str(expected_answer))
                to_judge.append(prompt)
                index_map.append((sample_idx, j))

        print(f"Total decomposition judgments to make: {len(to_judge)}")

        # Process in batches
        for i in range(0, len(to_judge), batch_size):
            batch_prompts = to_judge[i:i + batch_size]
            batch_indices = index_map[i:i + batch_size]
            print(f"Processing batch {i // batch_size + 1}/{(len(to_judge) + batch_size - 1)//batch_size}")

            try:
                responses = client.get_model_response(
                    user_prompts=batch_prompts,
                    max_new_tokens=1024,
                    temperature=0.0,
                    top_k=None,
                    top_p=1.0
                )

                # Parse results and assign back to proper index
                for (sample_idx, j), response in zip(batch_indices, responses):
                    sample = samples[sample_idx]
                    try:
                        parsed = extract_json_from_string(response)
                        score = int(parsed.get('score', 0))
                    except Exception as parse_err:
                        logging.warning(f"Parsing error for sample {sample.sample_id}: {parse_err}")
                        score = -1
                    print("score", score)
                    sample._scores['decomposition'][j] = score

            except Exception as e:
                logging.error(f"Error processing batch {i // batch_size + 1}: {e}")
                for (sample_idx, j) in batch_indices:
                    sample = samples[sample_idx]
                    sample._scores['decomposition'][j] = -1

        # Collect results
        for sample in samples:
            scores_dict = getattr(sample, "_scores", {'decomposition': []})
            evaluated_samples.append((sample, scores_dict))

    print("Decomposition evaluation completed successfully.")
    return evaluated_samples 
def safe_to_int(score_val) -> int:
    # Coerce safely to int
    if isinstance(score_val, (int, float)):
        return int(score_val)
    elif isinstance(score_val, str):
        try:
            return int(float(score_val.strip()))
        except:
            return -1
    elif isinstance(score_val, (list, tuple, set)):
        try:
            # Take the first element as fallback
            return safe_to_int(list(score_val)[0])
        except:
            return -1
    else:
        # Any other type → default to -1
        return -1

def judge_variations_scaff (
    math_verifier:bool,
    samples: List[UniversalSample],
    client: BaseModelClient,
    judge_model: str = "phi-4",
    client_type: str = "vllm",
    batch_size: int = 100,
    max_workers: int = 25
) -> List[Tuple[UniversalSample, Dict[str, List[int]]]]:
    """
    Evaluate decomposition_evaluation for each sample using the LLM judge via {client_type}.
    Produces one score per decomposition question, aligned with the input order.
    Empty or '<final answer only>' answers get score = 0.
    """
    evaluated_samples = []

    # this needs to be fixed
    if math_verifier:
        for sample in samples or []:
            decomp = getattr(sample, "scaffolding_evaluation", None)
            sub_tasks = [getattr(sample, "final_answer", None)] * len(decomp)
            if not decomp or not sub_tasks:
                continue

            score_dict = []
            if(len(decomp) != len(sub_tasks)):
                continue

            for ix in range(len(sub_tasks)):
                d = decomp[ix]
                gold_expr = sub_tasks[ix].get('answer', '')
                evaluation = d.get('answer', '')
                gold = parse(gold_expr)
                answer = parse(evaluation)
                result = verify(gold, answer)
                score_dict.append(1 if result else 0)

            evaluated_samples.append((sample, score_dict))


    else:
        # client = ModelClientFactory.create_client(client_type, judge_model, max_workers=max_workers)

        print(f"Evaluating {len(samples)} universal samples using {judge_model} via {client_type}...")
        print(f"OPTIMIZED SETTINGS: batch_size={batch_size}, max_workers={max_workers}")

        # Flatten decomposition evaluations into one queue
        to_judge = []
        index_map = []  # (sample_idx, j)

        for sample_idx, sample in enumerate(samples):
            if not sample.scaffolding_evaluation:
                continue

            # Initialize aligned score array
            sample._scores = {'scaffolding': [0] * len(sample.scaffolding_evaluation)}

            for j, decomp_eval in enumerate(sample.scaffolding_evaluation):
                model_answer = str(decomp_eval).strip()
                expected_answer = str(sample.final_answer).strip()

                # Skip generating prompts for empty or placeholder answers (keep default 0)
                if not model_answer or model_answer == "<final answer only>":
                    continue

                # Queue valid comparison
                parsed_answer = extract_json_from_string(model_answer)

                # Parsing answer
                answer_expr = model_answer

                if isinstance(parsed_answer, dict):
                    # Case 1: Has explicit 'answer' field
                    if parsed_answer.get("answer"):
                        # answer_expr = json.dumps(parsed_answer["answer"])
                        answer_expr = json.dumps(sanitize_for_json(parsed_answer["answer"]))


                    # Case 2: Has only 'explanation' → remove it if possible
                    elif parsed_answer.get("explanation"):
                        filtered = {k: v for k, v in parsed_answer.items() if k != "explanation"}
                        if filtered:
                            #answer_expr = json.dumps(filtered)  # or json.dumps(filtered) if you need a string
                            answer_expr = json.dumps(sanitize_for_json(filtered))


                prompt = get_prompt_correctness_value(answer_expr, expected_answer)
                to_judge.append(prompt)
                index_map.append((sample_idx, j))

        print(f"Total scaffolding judgments to make: {len(to_judge)}")

        # Process in batches
        for i in range(0, len(to_judge), batch_size):
            batch_prompts = to_judge[i:i + batch_size]
            batch_indices = index_map[i:i + batch_size]
            print(f"Processing batch {i // batch_size + 1}/{(len(to_judge) + batch_size - 1)//batch_size}")

            try:
                responses = client.get_model_response(
                    user_prompts=batch_prompts,
                    max_new_tokens=1024,
                    temperature=0.0,
                    top_k=None,
                    top_p=1.0
                )

                # Parse results and assign back to proper index
                for (sample_idx, j), response in zip(batch_indices, responses):
                    sample = samples[sample_idx]
                    try:
                        parsed = extract_json_from_string(response)
                        score_val = parsed.get('score', 0)
                        score = safe_to_int(score_val)
                    except Exception as parse_err:
                        logging.warning(f"Parsing error for sample {sample.sample_id}: {parse_err}")
                        score = -1
                    sample._scores['scaffolding'][j] = safe_to_int(score)

            except Exception as e:
                logging.error(f"Error processing batch {i // batch_size + 1}: {e}")
                for (sample_idx, j) in batch_indices:
                    sample = samples[sample_idx]
                    sample._scores['scaffolding'][j] = -1

        # Collect results
        for sample in samples:
            scores_dict = getattr(sample, "_scores", {'scaffolding': []})
            evaluated_samples.append((sample, scores_dict))

    print("Scaffolding evaluation completed successfully.")
    return evaluated_samples 
def save_scored_universal_samples_variation_scored(variation_type: str, evaluated_samples: List[Tuple[UniversalSample, List[int]]],
                                output_file: str):
    """Save variation scores with dynamic field name and validation."""
    # Validation checks
    if evaluated_samples is None:
        raise ValueError("evaluated_samples is None!")

    for idx, item in enumerate(evaluated_samples):
        if item is None:
            raise ValueError(f"evaluated_samples[{idx}] is None!")
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"evaluated_samples[{idx}] has wrong structure: {item}")

    # Use unified save function
    save_universal_samples(evaluated_samples, output_file, f'{variation_type}_score')

def find_sets(obj, path="root"):
    if isinstance(obj, set):
        print("❌ Found set at:", path)
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            find_sets(v, f"{path}.{k}")
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            find_sets(v, f"{path}[{i}]")

def main():

    parser = argparse.ArgumentParser(description="Generate sub-tasks for dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to config.json")
    args = parser.parse_args()

    # Resolve config path
    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    base_dir = config_path.parent

    input_stem = (base_dir / config.get("input_file", "data/dataset.jsonl")).resolve()
    
    
    model_name = config.get("debugging_model_name", "gpt-4o-mini")  # default to OpenAI model
    judge_model_name = config.get("judge_model_name", "gpt-4o-mini")
    client_type = config.get("client_type", "openai")
    math_verifier = config.get("Math-Verify", False)

    
    try:

        
        batch_size = 100 
        max_workers = 30 

        client = ModelClientFactory.create_client(client_type, model_name, tensor_parallel_size=1, max_workers=max_workers)
        judge_client = ModelClientFactory.create_client(client_type, judge_model_name, tensor_parallel_size=1, max_workers=max_workers)


        print("Debugging With Your Model")
        mode = 'bench_eval'
        input_file = input_stem
        samples = load_universal_samples(mode, input_file)
        if not samples:
           print(f"No valid samples found in {input_file}")
        print(len(samples))
        print("Step 1: Evaluate Benchmark")
        evaluated_samples = evaluate_benchmark(samples, client, model_name, client_type, batch_size, max_workers)
        output_file = input_file.with_name(input_file.stem + "_evaluation.jsonl")
        save_universal_samples(evaluated_samples, output_file, 'evaluation')
        print(f"✅ evaluation saved to {output_file}")

        
        print("Step 2: Judge Benchmark")
        mode = 'bench_judge'
        input_file = input_stem.with_name(input_stem.stem + "_evaluation.jsonl")
        samples = load_universal_samples(mode, input_file)
        if not samples:
           print(f"No valid samples found in {input_file}")
        judged_samples = judge_evaluation(math_verifier, samples, judge_client, judge_model_name, client_type, batch_size, max_workers)
        output_file = input_file.with_name(input_file.stem + "_scored.jsonl")
        save_scored_universal_samples_scored(judged_samples, output_file)
        print(f"✅ score saved to {output_file}")


        print("Step 3: Evaluate Variations")
        mode = 'var_eval'
        input_file = input_stem.with_name(input_stem.stem + "_evaluation_scored.jsonl")
        samples = load_universal_samples(mode, input_file)
        if not samples:
           print(f"No valid samples found in {input_file}")
        evaluated_samples = evaluate_decompositions(samples, client, model_name, batch_size, max_workers)
        output_file = input_file.with_name(input_file.stem + "_decomposition_evaluation.jsonl")
        save_universal_samples(evaluated_samples, output_file, 'decomposition_evaluation')
        print(f"✅ variation evaluation saved to {output_file}")

        print("Step 4: Evaluate Variations (scaffolding)")
        mode = 'var_scaff_eval'
        input_file = input_stem.with_name(input_stem.stem + "_evaluation_scored_decomposition_evaluation.jsonl")
        samples = load_universal_samples(mode, input_file)
        if not samples:
           print(f"No valid samples found in {input_file}")
        evaluated_samples = evaluate_scaffoldings(samples, client, model_name, batch_size, max_workers)
        output_file = input_file.with_name(input_file.stem + "_scaffolding_evaluation.jsonl")
        save_universal_samples(evaluated_samples, output_file, 'scaffolding_evaluation')
        print(f"✅ variation evaluation saved to {output_file}")


        print("Step 5: Judge Variations (decomposition)")
        mode = 'var_judge'
        input_file = input_stem.with_name(input_stem.stem + "_evaluation_scored_decomposition_evaluation_scaffolding_evaluation.jsonl")
        samples = load_universal_samples(mode, input_file)
        if not samples:
           print(f"No valid samples found in {input_file}")
        judged_samples = judge_variations(math_verifier, samples, judge_client, judge_model_name, client_type, batch_size, max_workers)
        output_file = input_file.with_name(input_file.stem + "_scored.jsonl")
        save_scored_universal_samples_variation_scored('decomposition', judged_samples, output_file)
        print(f"✅ score saved to {output_file}")

         
        print("Step 6: Judge Variations (scaffolding)")
        mode = 'var_scaff_judge'
        input_file = input_stem.with_name(input_stem.stem + "_evaluation_scored_decomposition_evaluation_scaffolding_evaluation_scored.jsonl")
        samples = load_universal_samples(mode, input_file)
        if not samples:
           print(f"No valid samples found in {input_file}")
        judged_samples = judge_variations_scaff(math_verifier, samples, judge_client, judge_model_name, client_type, batch_size, max_workers)
        output_file = input_file.with_name(input_file.stem + "_scored.jsonl")
        save_scored_universal_samples_variation_scored('scaffolding', judged_samples, output_file)
        print(f"✅ score saved to {output_file}")


    except Exception as e:
        print(f"Error during processing: {str(e)}")
        logging.error(f"Main process error: {str(e)}")


if __name__ == "__main__":
    main()