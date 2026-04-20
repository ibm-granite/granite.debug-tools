# Dependencies Documentation

## Overview

This document describes all dependencies used in the STaD (Scaffolded Task Decomposition) project.

## Installation

### Basic Installation
```bash
pip install -r requirements.txt
```

### Development Installation
```bash
pip install -r requirements-dev.txt
```

## Core Dependencies

### Required Packages

1. **openai** (>=1.0.0)
   - Purpose: OpenAI API client for GPT models
   - Used in: `model_client.py`
   - Installation: `pip install openai`

2. **google-generativeai** (>=0.3.0)
   - Purpose: Google Gemini API client
   - Used in: `model_client.py`
   - Installation: `pip install google-generativeai`
   - Note: Optional if you only use OpenAI or VLLM

3. **regex** (>=2023.0.0)
   - Purpose: Advanced regular expression features
   - Used in: `generate_variations.py`
   - Installation: `pip install regex`
   - Note: Provides features beyond standard `re` module

4. **requests** (>=2.31.0)
   - Purpose: HTTP library for API calls
   - Used in: `model_client.py` 
   - Installation: `pip install requests`

## Optional Dependencies

### VLLM (Local Model Inference)

**vllm** (>=0.3.0)
- Purpose: Run large language models locally
- Used in: `model_client.py`
- Installation: `pip install vllm`
- Requirements:
  - CUDA-capable GPU recommended
  - Significant RAM (depends on model size)
- Note: Only needed if you want to run models locally

To enable VLLM, uncomment the line in `requirements.txt`:
```bash
# vllm>=0.3.0  → vllm>=0.3.0
```

## Local Modules

These modules are part of the project and don't require separate installation:

1. **math_verify**
   - Purpose: Mathematical verification utilities
   - Used in: `generate_variations.py`, `test_variations.py`
   - Location: Should be in project directory

2. **helpers**
   - Purpose: Shared utility functions
   - Used in: `generate_variations.py`, `test_variations.py`

3. **prompts**
   - Purpose: Prompt templates
   - Used in: All variation scripts

4. **model_client**
   - Purpose: Unified model client interface
   - Used in: All scripts that interact with LLMs

## Standard Library Modules

The following are part of Python's standard library (no installation needed):

- `json` - JSON encoding/decoding
- `argparse` - Command-line argument parsing
- `ast` - Abstract syntax tree operations
- `concurrent.futures` - Concurrent execution
- `logging` - Logging functionality
- `os` - Operating system interface
- `re` - Basic regular expressions
- `time` - Time-related functions
- `uuid` - UUID generation
- `pathlib` - Object-oriented filesystem paths
- `dataclasses` - Data classes
- `typing` - Type hints
- `abc` - Abstract base classes
- `threading` - Threading primitives

## Environment Variables

### OpenAI
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Google Gemini
```bash
export GEMINI_API_KEY="your-api-key-here"
# OR
export GOOGLE_API_KEY="your-api-key-here"
```

## Python Version

- **Minimum:** Python 3.8
- **Recommended:** Python 3.10+

Reason: The code uses modern type hints and dataclasses that work best with Python 3.10+.

## Troubleshooting

### Common Issues

1. **Import Error: No module named 'openai'**
   ```bash
   pip install openai
   ```

2. **Import Error: No module named 'google.generativeai'**
   ```bash
   pip install google-generativeai
   ```

3. **VLLM not available**
   - This is expected if VLLM is not installed
   - Only install if you need local model inference
   - Requires GPU support

4. **math_verify module not found**
   - Ensure `math_verify.py` is in the same directory as other scripts
   - Or add the directory to your PYTHONPATH

## Development Dependencies

For development and testing, install additional packages:

```bash
pip install -r requirements-dev.txt
```

### Development Tools

- **pytest** - Testing framework
- **pytest-cov** - Code coverage
- **black** - Code formatter
- **flake8** - Linting
- **mypy** - Type checking
- **isort** - Import sorting
- **jupyter** - Interactive notebooks
- **sphinx** - Documentation generation

## Verifying Installation

Test your installation:

```python
# Test OpenAI
from openai import OpenAI
client = OpenAI()
print("✓ OpenAI installed")

# Test Gemini (optional)
import google.generativeai as genai
print("✓ Google Generative AI installed")

# Test other packages
import regex
import requests
print("✓ All core packages installed")
```

## Updates

To update all packages to their latest versions:

```bash
pip install --upgrade -r requirements.txt
```

## License Compatibility

All listed dependencies are compatible with open-source projects:
- OpenAI: MIT License
- Google GenerativeAI: Apache 2.0
- regex: Apache 2.0
- requests: Apache 2.0
- VLLM: Apache 2.0
