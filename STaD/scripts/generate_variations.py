import json
import argparse
import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from dotenv import load_dotenv
import regex as re

from model_client import ModelClientFactory, BaseModelClient
from prompts import (
    prompt_decomposition,
    prompt_variation,
    prompt_correctness_value,
    prompt_sequential_segments_answers_reasoning,
    prompt_segment
)
try:
    from math_verify import parse, verify
except ImportError:
    def parse(x): raise NotImplementedError("math_verify not installed")
    def verify(a, b): raise NotImplementedError("math_verify not installed")

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
    extract_json_from_string,
    extract_rewritten_question,
    sub_task_answer_consistency,
    save_universal_samples, 
    clean_possible_json_block
)

def generate_segment_answers(samples: List[UniversalSample], client: BaseModelClient, judge_model: str = "phi-4",
                              batch_size: int = 100, max_workers: int = 25) -> List[Tuple[UniversalSample, str]]:
    """Evaluate universal samples using the LLM judge - OPTIMIZED."""
    # OPTIMIZED: Increased max_workers and better error handling
    evaluated_samples = []
    
    print(f"Evaluating {len(samples)} universal samples using {judge_model}")
    print(f"OPTIMIZED SETTINGS: batch_size={batch_size}, max_workers={max_workers}")
    
    # Process in batches
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(samples) + batch_size - 1)//batch_size} ({len(batch)} samples)")
        
        # Prepare prompts
        user_prompts = [get_prompt_segment_answer_reasoning(sample.instruction, str(sample.sub_task)) for sample in batch]

        try:
            # OPTIMIZED: Shorter responses and faster temperature
            responses = client.get_model_response(
                user_prompts=user_prompts,
                max_new_tokens=1024,  
                temperature=0.2,     
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

def generate_scaffolding(
    samples: List[UniversalSample],
    client: BaseModelClient,
    judge_model: str = "phi-4",
    batch_size: int = 100,
    max_workers: int = 25
) -> List[Tuple[UniversalSample, List[str]]]:
    """Evaluate universal samples using the LLM judge, supporting variable sub-task counts."""
    evaluated_samples = []
    
    print(f"Evaluating {len(samples)} universal samples using {judge_model}")
    print(f"OPTIMIZED SETTINGS: batch_size={batch_size}, max_workers={max_workers}")
    
    # Process in batches
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(samples) + batch_size - 1)//batch_size} ({len(batch)} samples)")

        # Store generated prompts and bookkeeping info
        user_prompts = []
        meta_info = []  # (sample_index_in_batch, subtask_index)
        
        # Build prompts for all subtask lengths of each sample
        for idx, sample in enumerate(batch):
            merged = [
                {"segment": a.get("segment"), "answer": b.get("answer")}
                for a, b in zip(sample.sub_task, sample.sub_task_answer)
            ]
            num_subtasks = len(merged)
            for j in range(1, num_subtasks):
                merged_subset = merged[:j]
                prompt = get_prompt_variation(sample.instruction, merged_subset)
                user_prompts.append(prompt)
                meta_info.append((idx, j))  # keep track of which sample + subtask length this belongs to
        # Run inference for all prompts in this batch
        try:
            responses = client.get_model_response(
                user_prompts=user_prompts,
                max_new_tokens=2000,
                temperature=0.2,
                top_k=None,
                top_p=1.0
            )

            # Collect variations per sample
            variations = [[] for _ in range(len(batch))]
            for response, (sample_idx, j) in zip(responses, meta_info):
                rewritten = extract_rewritten_question(response)
                variations[sample_idx].append(rewritten)

        except Exception as e:
            logging.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
            # If error, fill with empty strings
            variations = [["" for _ in sample.sub_task] for sample in batch]

        # Save results
        for sample, variation in zip(batch, variations):
            evaluated_samples.append((sample, variation))
            if len(evaluated_samples) % 100 == 0:
                print(f"  Processed {len(evaluated_samples)}/{len(samples)} samples")

    return evaluated_samples

def judge_variation_samples(math_verifier: bool, evaluated_samples: List[Tuple[UniversalSample, List[str]]],
                            client: BaseModelClient,
                            judge_model: str = "phi-4",
                            batch_size: int = 100,
                            max_workers: int = 25) -> List[Tuple[UniversalSample, List[str]]]:
    """Evaluate universal samples using the LLM judge - supports variable subtask lengths."""


    judged_samples = []
    samples = [sample for sample, _ in evaluated_samples]
    final_answers = [answer for _, answer in evaluated_samples]

    if(math_verifier):
        variation_consistency = [[] for _ in range(len(samples))]
        for i, (sample, answers) in enumerate(zip(samples, final_answers)):
            gold_expr = sample.final_answer
            gold = parse(gold_expr)
            num_subtasks = len(answers)
            for j in range(num_subtasks):
                answer_expr =  str(answers[j]['answer'])
                answer_expr_last_line = answer_expr.split("\n")[-1]
                answer = parse(answer_expr_last_line)
                result = verify(gold, answer)
                if(result):
                    variation_consistency[i].append({'score': 1, 'justification': 'Math Verifier'})
                else:
                    variation_consistency[i].append({'score': 0, 'justification': 'Math Verifier'})
        # Merge scores with answers
        merged_answers = [
            [{**a, **s} for a, s in zip(ans_sublist, score_sublist)]
            for ans_sublist, score_sublist in zip(final_answers, variation_consistency)
        ]

        for sample, answer in zip(samples, merged_answers):
            judged_samples.append((sample, answer))
            if len(judged_samples) % 100 == 0:
                print(f"  Processed {len(judged_samples)}/{len(samples)} samples")

    else:
        print(f"Evaluating {len(samples)} universal samples using {judge_model}...")
        print(f"OPTIMIZED SETTINGS: batch_size={batch_size}, max_workers={max_workers}")

        # Process in batches
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            batch_answers = final_answers[i:i + batch_size]
            print(f"Processing batch {i // batch_size + 1}/{(len(samples) + batch_size - 1) // batch_size} ({len(batch)} samples)")

            # Prepare prompts for all subtasks in the batch
            user_prompts = []
            meta_info = []  # (sample_idx, subtask_idx)

            for sample_idx, (sample, answers) in enumerate(zip(batch, batch_answers)):
                num_subtasks = len(answers)
                for j in range(num_subtasks):
                    user_prompt = get_prompt_correctness_value(str(answers[j]['answer']), str(sample.final_answer))
                    user_prompts.append(user_prompt)
                    meta_info.append(sample_idx)

            # Run LLM inference
            try:
                responses = client.get_model_response(
                    user_prompts=user_prompts,
                    max_new_tokens=1024,
                    temperature=0.0,
                    top_k=None,
                    top_p=1.0
                )
            except Exception as e:
                logging.error(f"Error processing batch {i // batch_size + 1}: {str(e)}")
                responses = [{} for _ in user_prompts]

            # Collect results per sample
            variation_consistency = [[] for _ in range(len(batch))]
            for response, sample_idx in zip(responses, meta_info):
                print(response)
                parsed = extract_json_from_string(response)
                if not parsed:
                    parsed = {'score': 0, 'justification': 'Parsing error'}
                variation_consistency[sample_idx].append(parsed)

            # Merge scores with answers
            merged_answers = [
                [{**a, **s} for a, s in zip(ans_sublist, score_sublist)]
                for ans_sublist, score_sublist in zip(batch_answers, variation_consistency)
            ]

            for sample, answer in zip(batch, merged_answers):
                judged_samples.append((sample, answer))
                if len(judged_samples) % 100 == 0:
                    print(f"  Processed {len(judged_samples)}/{len(samples)} samples")

    return judged_samples


def generate_sub_task(samples: List[UniversalSample], client: BaseModelClient, judge_model: str = "phi-4",
                              batch_size: int = 100, max_workers: int = 25) -> List[Tuple[UniversalSample, str]]:
    """Evaluate universal samples using the LLM judge - OPTIMIZED."""
    # OPTIMIZED: Increased max_workers and better error handling
    evaluated_samples = []
    
    print(f"Evaluating {len(samples)} universal samples using {judge_model}")
    print(f"OPTIMIZED SETTINGS: batch_size={batch_size}, max_workers={max_workers}")
    
    # Process in batches
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(samples) + batch_size - 1)//batch_size} ({len(batch)} samples)")
        
        # Prepare prompts
        user_prompts = [get_prompt_segment(sample.instruction) for sample in batch]

        try:
            
            responses = client.get_model_response(
                    user_prompts=user_prompts,
                    max_new_tokens=1024,  
                    temperature=0.2,     
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
   
def solve_variations(samples: List[UniversalSample], client: BaseModelClient, judge_model: str = "phi-4",
                     batch_size: int = 100, max_workers: int = 25) -> List[Tuple[UniversalSample, List[str]]]:
    """Evaluate universal samples using the LLM judge - supports variable number of subtasks."""
    evaluated_samples = []
    
    print(f"Evaluating {len(samples)} universal samples using {judge_model}")
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
            variation_answers[sample_idx].append({'answer': response})
     
        # Store evaluated samples
        for sample, variation_answer in zip(batch, variation_answers):
            evaluated_samples.append((sample, variation_answer))
            if len(evaluated_samples) % 100 == 0:
                print(f"  Processed {len(evaluated_samples)}/{len(samples)} samples")
    
    return evaluated_samples


def generate_decomposition(samples: List[UniversalSample],
                           client: BaseModelClient,
                           judge_model: str = "phi-4",
                           batch_size: int = 100,
                           max_workers: int = 25) -> List[Tuple[UniversalSample, List[str]]]:
    """Generate decompositions for universal samples using the LLM judge - supports variable subtask lengths."""
    evaluated_samples = []

    print(f"Evaluating {len(samples)} universal samples using {judge_model}")
    print(f"OPTIMIZED SETTINGS: batch_size={batch_size}, max_workers={max_workers}")

    # Process in batches
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(samples) + batch_size - 1) // batch_size} ({len(batch)} samples)")

        # Collect prompts across all subtasks in the batch
        user_prompts = []
        meta_info = []  # (sample_idx, subtask_idx)

        for sample_idx, sample in enumerate(batch):
            merged = [
                {"segment": a.get("segment"), "answer": b.get("answer")}
                for a, b in zip(sample.sub_task, sample.sub_task_answer)
            ]
            num_subtasks = len(merged)

            for j in range(num_subtasks):
                specific_step = merged[j]
                prior_info_as_list = merged[:j]
                prompt = get_prompt_decomposition(sample.instruction,
                                                  specific_step["segment"],
                                                  prior_info_as_list)
                user_prompts.append(prompt)
                meta_info.append((sample_idx, j))

        # Query model
        try:
            responses = client.get_model_response(
                user_prompts=user_prompts,
                max_new_tokens=2000,
                temperature=0.2,
                top_k=None,
                top_p=1.0
            )
        except Exception as e:
            logging.error(f"Error processing batch {i // batch_size + 1}: {str(e)}")
            responses = ["" for _ in user_prompts]

        # Allocate empty list for each sample
        variations = [[] for _ in batch]

        # Distribute responses back to correct sample/subtask
        for response, (sample_idx, j) in zip(responses, meta_info):
            rewritten = extract_rewritten_question(response)
            variations[sample_idx].append(rewritten if rewritten else "")

        # Append results
        for sample, variation in zip(batch, variations):
            evaluated_samples.append((sample, variation))
            if len(evaluated_samples) % 100 == 0:
                print(f"  Processed {len(evaluated_samples)}/{len(samples)} samples")

    return evaluated_samples  

def main():

    parser = argparse.ArgumentParser(description="Generate sub-tasks for dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to config.json")
    args = parser.parse_args()

    # Resolve config path
    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    # Base directory = folder where config.json lives
    base_dir = config_path.parent

    # Resolve input and output paths relative to config.json
    input_stem = (base_dir / config.get("input_file", "data/dataset.jsonl")).resolve()
    
    
    model_name = config.get("model_name", "gpt-4o-mini")  # default to OpenAI model
    judge_model_name = config.get("judge_model_name", "gpt-4o-mini")
    client_type = config.get("client_type", "openai")
    math_verifier = config.get("Math-Verify", False)

    try:

        batch_size = 150
        max_workers = 30

        model_client = ModelClientFactory.create_client(client_type, model_name, tensor_parallel_size=1, max_workers=max_workers)
        judge_client = ModelClientFactory.create_client(client_type, judge_model_name, tensor_parallel_size=1, max_workers=max_workers)

        print("Step 1: Generate Sub-Tasks")
        mode = 'segment'
        input_file = input_stem
        samples = load_universal_samples(mode, input_file)
        if not samples:
           print(f"No valid samples found in {input_file}")
        evaluated_samples = generate_sub_task(samples, model_client, model_name, batch_size, max_workers)
        output_file = input_file.with_name(input_file.stem + "_segment.jsonl")
        save_universal_samples(evaluated_samples,output_file, 'sub-task', lambda x: parse_llm_output("segment", x))
        print(f"✅ Sub-tasks saved to {output_file}")

        print("Step 2: Generate Answers to Sub-Tasks")
        mode = 'answers'
        input_file = input_stem.with_name(input_stem.stem + "_segment.jsonl")
        samples = load_universal_samples(mode, input_file)
        if not samples:
            print(f"No valid samples found in {input_file}")
        evaluated_samples = generate_segment_answers(samples, model_client, model_name, batch_size, max_workers)
        output_file = input_file.with_name(input_file.stem + "_answers.jsonl")
        save_universal_samples(evaluated_samples, output_file, 'sub-task-answer', lambda x: parse_llm_output_multiple("explanation", "answer", x))
        print(f"✅ Sub-task answers saved to {output_file}")

        print("Step 3: Generate Scaffolding Variations")
        mode = 'scaffolding'
        input_file = input_stem.with_name(input_stem.stem + "_segment_answers.jsonl")
        samples = load_universal_samples(mode, input_file)
        if not samples:
            print(f"No valid samples found in {input_file}")
        filtered_samples = sub_task_answer_consistency(math_verifier, samples, judge_client, judge_model_name, batch_size, max_workers)
        evaluated_samples = generate_scaffolding(filtered_samples, model_client, model_name, batch_size, max_workers)
        output_file = input_file.with_name(input_file.stem + "_scaffolding.jsonl")
        save_universal_samples(evaluated_samples, output_file, 'scaffolding')
        print(f"✅ Scaffolding saved to {output_file}")

        print("Step 4: Verify")
        mode = 'verify'
        input_file = input_stem.with_name(input_stem.stem + "_segment_answers_scaffolding.jsonl")
        samples = load_universal_samples(mode, input_file)
        if not samples:
            print(f"No valid samples found in {input_file}")
        evaluated_samples = solve_variations(samples, model_client, model_name, batch_size, max_workers)
        evaluated_samples = judge_variation_samples(math_verifier, evaluated_samples, judge_client, judge_model_name, batch_size, max_workers)
        output_file = input_file.with_name(input_file.stem + "_verify.jsonl")
        save_universal_samples(evaluated_samples, output_file, 'scaffolding_verification')
        print(f"✅ Verification saved to {output_file}")
  
        print("Step 5: Generate Decomposition Variations")
        mode = 'decomposition'
        input_file = input_stem.with_name(input_stem.stem + "_segment_answers_scaffolding_verify.jsonl")
        samples = load_universal_samples(mode, input_file)
        if not samples:
            print(f"No valid samples found in {input_file}")
        evaluated_samples = generate_decomposition(samples, model_client, model_name, batch_size, max_workers)
        output_file = input_stem.with_name(input_stem.stem + "_final.jsonl")
        save_universal_samples(evaluated_samples, output_file, 'decompositions')
        print(f"✅ Decomposition saved to {output_file}")
        

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        logging.error(f"Main process error: {str(e)}")


if __name__ == "__main__":
    main()
