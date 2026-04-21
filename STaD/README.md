# STaD: Scaffolded Task Design

TL;DR: Scaffolding step by step reveals reasoning gaps in LLMs that aggregate benchmark scores often fail to find.

**STaD** (Scaffolded Task Design) is a framework for generating scaffolded variations of multi-step reasoning tasks to enable systematic LLM debugging, evaluation, and training.

[**Code**](https://github.com/ibm-granite/scaffolded-task-design) | [**Dataset**](https://huggingface.co/datasets/ibm-research/STaD) | [**Paper**](https://arxiv.org/pdf/2604.18177)

## News

| Date       | Update |
|------------|--------|
| 2026-04-16 | STaD is released and open-sourced. |
| 2026-04-06 | STaD is accepted to [ACL](https://2026.aclweb.org/) Findings 2026. |


## Overview

STaD breaks down complex multi-step questions into scaffolded variations where intermediate steps are revealed. This allows you to:
- 🔍 **Debug LLMs systematically** - Identify individual skills and skill combinations that lead to failures.
- 📈 **Test latent capabilities** - distinguish skill-level weaknesses from other types of difficulty.
- 🎯 **Target skill gaps** - Highlighting skill-level areas for further training or targeted interventions.

<img src="scaffolded_competence.png" alt="drawing" width="750"/>

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/ibm-granite/scaffolded-task-design.git
cd scaffolded-task-design
```
### 2. Install Dependencies

Create a Python virtual environment (Python 3.8+ required, 3.10+ recommended):

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Using conda
conda create -n stad python=3.10
conda activate stad
```

Install required packages:

```bash
pip install -r requirements.txt
```

For development (includes testing, linting, documentation tools):

```bash
pip install -r requirements-dev.txt
```

### 3. Set Up API Keys

Configure your API keys as environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-key"

# Google Gemini (optional)
export GEMINI_API_KEY="your-gemini-key"
```

Or create a `.env` file:

```bash
# .env
OPENAI_API_KEY=your-openai-key
GEMINI_API_KEY=your-gemini-key
```
### Paper Version (ACL Findings 2026)

The results in the paper correspond to the following immutable snapshot:

- **Commit:** `XXXXX`
- **Tag:** `v1.0-paper`
- **Branch state:** `main` at time of release

**To reproduce paper results:**

```bash
git checkout v1.0-paper
```
or
```bash
git checkout a1b2c3d4
```

## Quick Start

### Option A: Test Your Model on Existing Scaffolded Benchmarks

If you want to evaluate a model on pre-generated scaffolded tasks, start by creating a debugging configuration file (for example, `config_debugging.json`):

```json
{
  "input_file": "data/tot_arithmetic_scaffolded.jsonl",
  "debugging_model_name": "path/to/your/target_model",
  "judge_model_name": "path/to/your/judge_model",
  "client_type": "vllm",
  "Math-Verify": false
}
```

#### Available Scaffolded Benchmarks

Currently, three scaffolded benchmarks are available in the `data/` directory:

1. **ToT (Test of Time) Arithmetic**  
   Evaluates temporal reasoning involving arithmetic operations on times and durations  
   (Fatemi et al., 2024).  
   **1.45K** scaffolded tasks.

2. **GSM8K (Grade School Math 8K)**  
   Grade-school level math word problems  
   (Cobbe et al., 2021).  
   **1.17K** scaffolded tasks.

3. **Math-Hard**  
   Hard problems drawn from mathematics competitions, retaining only Level-5 questions  
   (Hendrycks et al., 2021).  
   **773** scaffolded tasks.

To test on a specific benchmark, set `input_file` accordingly, for example:

- `data/tot_arithmetic_scaffolded.jsonl`
- `data/gsm8k_scaffolded.jsonl`
- `data/math_hard_scaffolded.jsonl`

#### Run the Evaluation

```bash
python scripts/test_variations.py --config config_debugging.json
```


## Option B: Generate Scaffolded Tasks from Your Own Dataset

### 1. Prepare Your Dataset

Create a JSONL file under the `data/` directory. Each line should contain a JSON object with a `question` and a corresponding `answer`.

Example (`data/sample_data.jsonl`):

```json
{"question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins with four. She sells the remainder at $2 per egg. How much does she make daily?", "answer": "18"}
{"question": "The conference began at 09:30:00 AM and concluded at 04:45:00 PM. Calculate the total elapsed time.", "answer": "7 hours 15 minutes"}
```


### 2. Generate Scaffolded Variations

Create a configuration file (for example, `config.json`):

```json
{
  "input_file": "data/sample_data.jsonl",
  "model_name": "gpt-4o-mini",
  "judge_model_name": "gpt-4o-mini",
  "client_type": "openai",
  "Math-Verify": false
}
```

Run the generation pipeline:

```bash
python scripts/generate_variations.py --config config.json
```


## Configuration

### Model Configuration

The framework supports three model backends:

#### **OpenAI**
```json
{
  "model_name": "gpt-4o-mini",  // or "gpt-4o", "gpt-3.5-turbo"
  "client_type": "openai"
}
```

#### **Google Gemini**
```json
{
  "model_name": "gemini-1.5-flash",  // or "gemini-1.5-pro"
  "client_type": "gemini"
}
```

#### **VLLM (Local Models)**
```json
{
  "model_name": "/path/to/model",  // e.g., "microsoft/phi-4"
  "client_type": "vllm"
}
```

### Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `input_file` | Path to input JSONL dataset | Required |
| `model_name` | Model to use for generation | `"phi-4"` |
| `judge_model_name` | Model to use for evaluation | `"llama-3-3-70b"` |
| `debugging_model_name` | Model to test/debug | Required for testing |
| `client_type` | Backend: `"openai"`, `"gemini"`, or `"vllm"` | `"vllm"` |
| `Math-Verify` | Use symbolic math verification | `false` |

### Advanced Settings

Edit the script files to customize:
- `batch_size`: Number of samples to process concurrently (default: 100-150)
- `max_workers`: Thread pool size for parallel processing (default: 25-30)
- `temperature`: Model sampling temperature (default: 0.1-0.2)
- `max_new_tokens`: Maximum tokens to generate (default: 1000-2000)


## Output Format

### Final Output Structure

Each line in the output JSONL contains:

```json
{
  "question": "Original question",
  "answer": "Ground truth answer",
  "sub-task": [
    {"segment": "First sub-task"},
    {"segment": "Second sub-task"}
  ],
  "sub-task-answer": [
    {"explanation": "...", "answer": "..."},
    {"explanation": "...", "answer": "..."}
  ],
  "scaffolding": [
    "Scaffolding variation 1",
    "Scaffolding variation 2"
  ],
  "scaffolding_verification": [
    {"score": 1},
    {"score": 1}
  ],
  "decompositions": [
    "Decomposition variation 1",
    "Decomposition variation 2"
  ],
  "evaluation": "Model's response to original question",
  "evaluation_score": 0,
  "decomposition_evaluation": [
    "Response to decomposition 1",
    "Response to decomposition 2"
  ],
  "scaffolding_evaluation": [
    "Response to scaffolding 1",
    "Response to scaffolding 2"
  ],
  "decomposition_score": [1, 0],
  "scaffolding_score": [1, 1]
}
```


## Troubleshooting and Common Issues

### 1. Out-of-Memory (OOM) Errors with VLLM

A common issue when using VLLM locally is GPU OOM because both the target model and judge model may be loaded simultaneously.

**Workaround:** run the target model and judge model separately to reduce memory usage.

### 2. Empty Output Files After Evaluation

STaD only evaluates scaffolding and decompositions when the model fails the original task. If your model solves all original questions, empty outputs are expected.

### 3. Scaffold Generation Fails Verification

If the teacher model is weak on the chosen benchmark, generated scaffolding may fail verification. Use a strong teacher model for reliable results.



## Citation

If you use STaD in your research, please cite:

```bibtex
@inproceedings{an2026stad,
  title     = {STaD: Scaffolded Task Design for Identifying Compositional Skill Gaps in LLMs},
  author    = {An, Sungeun and Kadhe, Swanand Ravindra and Thakur, Shailja and DeLuca, Chad and Patel, Hima},
  year      = {2026},
  eprint    = {2604.18177},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url={https://arxiv.org/abs/2604.18177}
}

```
