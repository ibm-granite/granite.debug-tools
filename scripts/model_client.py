"""
Unified Model Client Interface
Supports VLLM, Gemini, and OpenAI models
"""

import os
import time
import logging
import concurrent.futures
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod

from openai import OpenAI, OpenAIError


# VLLM imports
try:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    VLLM_AVAILABLE = True
except ImportError:
    print("VLLM not available. Only Gemini client will work.")
    VLLM_AVAILABLE = False

# Gemini imports
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    print("Google GenerativeAI not available. Only VLLM client will work.")
    GEMINI_AVAILABLE = False


class BaseModelClient(ABC):
    """Abstract base class for model clients"""
    
    @abstractmethod
    def get_model_response(self, user_prompts: List[str], 
                          max_new_tokens: int = 1000, temperature: float = 0.1, 
                          **kwargs) -> List[str]:
        """Generate responses for given prompts"""
        pass
    
    @abstractmethod
    def get_single_response(self, user_prompt: str, 
                           max_new_tokens: int = 1000, temperature: float = 0.1,
                           **kwargs) -> str:
        """Generate single response"""
        pass

class VLLMClient(BaseModelClient):
    """Client for VLLM model inference"""
    
    def __init__(self, model_name: str = "microsoft/phi-4", tensor_parallel_size: int = 1):
        if not VLLM_AVAILABLE:
            raise ImportError("VLLM is not available. Please install vllm and transformers.")
        
        print(f"Initializing VLLMClient with model: {model_name}")
        try:
            self.model_name = model_name
            self.tensor_parallel_size = tensor_parallel_size
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.llm = LLM(
                model=model_name, 
                disable_log_stats=True, 
                tensor_parallel_size=tensor_parallel_size, #8 for big model, otherwise 1
                max_model_len=8192,
                gpu_memory_utilization=0.90,
                trust_remote_code=True
            )
            print("VLLMClient initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize VLLMClient: {str(e)}")
            raise

    def get_model_response(self, user_prompts: List[str], 
                          max_new_tokens: int = 2000, temperature: float = 0.1, 
                          top_k: int = 40, top_p: float = 0.9, **kwargs) -> List[str]:
        """Generate responses for batch of prompts"""
        print(f"Generating responses for {len(user_prompts)} prompts...")
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens
        )
        
        # Prepare messages
        # messages_list = [
        #     [{"role": "system", "content": sys}, {"role": "user", "content": usr}]
        #     for sys, usr in zip(system_prompts, user_prompts)
        # ]
        messages_list = [
            [{"role": "user", "content": usr}]
            for usr in user_prompts
        ]
        
        # Apply chat template
        texts = [
            self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) 
            for messages in messages_list
        ]
        
        # Generate responses
        outputs = self.llm.generate(texts, sampling_params)
        return [output.outputs[0].text for output in outputs]
    
    def get_single_response(self, user_prompt: str, 
                           max_new_tokens: int = 1000, temperature: float = 0.1,
                           **kwargs) -> str:
        """Generate single response"""
        responses = self.get_model_response(
            [user_prompt], max_new_tokens, temperature, **kwargs
        )
        return responses[0]

class GeminiClient(BaseModelClient):
    """Client for Google Gemini model inference"""
    
    def __init__(self, model_name: str = "gemini-1.5-flash", api_key: Optional[str] = None):
        if not GEMINI_AVAILABLE:
            raise ImportError("Google GenerativeAI is not available. Please install google-generativeai.")
        
        self.model_name = model_name
        
        # Configure API key
        if api_key:
            genai.configure(api_key=api_key)
        elif os.getenv("GEMINI_API_KEY"):
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        elif os.getenv("GOOGLE_API_KEY"):
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        else:
            raise ValueError("Gemini API key not provided. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")
        
        try:
            self.model = genai.GenerativeModel(model_name)
            print(f"GeminiClient initialized successfully with model: {model_name}")
        except Exception as e:
            print(f"Failed to initialize GeminiClient: {str(e)}")
            raise
    
    def get_model_response(self, user_prompts: List[str], 
                          max_new_tokens: int = 1000, temperature: float = 0.1, 
                          **kwargs) -> List[str]:
        """Generate responses for batch of prompts"""
        print(f"Generating responses for {len(user_prompts)} prompts...")
        
        responses = []
        
        # Configure generation parameters
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_new_tokens,
            temperature=temperature,
            top_p=kwargs.get('top_p', 0.9),
            top_k=kwargs.get('top_k', 40)
        )
        
        for user_prompt in user_prompts:
            try:
                # Combine system and user prompts
                # combined_prompt = f"System: {sys_prompt}\n\nUser: {user_prompt}"
                combined_prompt = f"User: {user_prompt}"

                
                # Generate response with retry logic
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = self.model.generate_content(
                            combined_prompt,
                            generation_config=generation_config
                        )
                        
                        if response.text:
                            responses.append(response.text)
                            break
                        else:
                            raise ValueError("Empty response generated")
                            
                    except Exception as e:
                        if attempt < max_retries - 1:
                            print(f"Retry {attempt + 1} for prompt due to: {e}")
                            time.sleep(1)  # Brief delay before retry
                        else:
                            print(f"Failed to generate response after {max_retries} attempts: {e}")
                            responses.append(f"[ERROR: Failed to generate response - {str(e)}]")
                        
            except Exception as e:
                print(f"Error processing prompt: {e}")
                responses.append(f"[ERROR: {str(e)}]")
        
        return responses
    
    def get_single_response(self, user_prompt: str, 
                           max_new_tokens: int = 1000, temperature: float = 0.1,
                           **kwargs) -> str:
        """Generate single response"""
        responses = self.get_model_response(
            [user_prompt], max_new_tokens, temperature, **kwargs
        )
        return responses[0]

class OpenAIClient(BaseModelClient):
    """Client for OpenAI API model inference"""

    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None, max_workers: int = 50):
        self.model_name = model_name
        self.max_workers = max_workers

        # Configure API key
        if api_key:
            self.client = OpenAI(api_key=api_key)
        elif os.getenv("OPENAI_API_KEY"):
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")

        print(f"OpenAIClient initialized with model: {model_name}")

    def call_openai_api(self, user_prompt: str, max_new_tokens: int = 1000,
                        temperature: float = 0.5, max_retries: int = 3, **kwargs) -> str:
        """Call OpenAI API with retries"""
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=temperature,
                    top_p=kwargs.get('top_p', 1.0),
                    frequency_penalty=kwargs.get('frequency_penalty', 0.0),
                    presence_penalty=kwargs.get('presence_penalty', 0.0),
                    max_tokens=max_new_tokens
                )
                return response.choices[0].message.content

            except OpenAIError as e:
                logging.error(f"OpenAI API call failed (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                else:
                    return "[ERROR: OpenAI API call failed after retries]"

    def call_openai_api_messages(self, messages: List[Dict[str, str]], max_new_tokens: int = 1000,
                                 temperature: float = 0.5, max_retries: int = 3, **kwargs) -> str:
        """Call OpenAI API with custom message format"""
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=kwargs.get('top_p', 1.0),
                    frequency_penalty=kwargs.get('frequency_penalty', 0.0),
                    presence_penalty=kwargs.get('presence_penalty', 0.0),
                    max_tokens=max_new_tokens
                )
                return response.choices[0].message.content

            except OpenAIError as e:
                logging.error(f"OpenAI API call failed (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                else:
                    return "[ERROR: OpenAI API call failed after retries]"

    def get_model_response(self, user_prompts: List[str],
                          max_new_tokens: int = 1000, temperature: float = 0.1,
                          **kwargs) -> List[str]:
        """Generate responses using concurrent execution for OpenAI"""
        print(f"Generating responses for {len(user_prompts)} prompts using OpenAI...")
        responses = []

        # Use concurrent execution for OpenAI
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for user_prompt in user_prompts:
                future = executor.submit(
                    self.call_openai_api, user_prompt, max_new_tokens, temperature, **kwargs
                )
                futures.append(future)

            # Maintain original order by iterating through futures list directly
            for future in futures:
                try:
                    response = future.result()
                    responses.append(response if response is not None else "")
                except Exception as e:
                    logging.error(f"OpenAI concurrent call failed: {e}")
                    responses.append("[ERROR: Concurrent call failed]")

        return responses

    def get_model_response_messages(self, messages_list: List[List[Dict[str, str]]],
                                   max_new_tokens: int = 1000, temperature: float = 0.1,
                                   **kwargs) -> List[str]:
        """Generate responses using concurrent execution with custom messages"""
        print(f"Generating responses for {len(messages_list)} prompts using OpenAI...")
        responses = []

        # Use concurrent execution for OpenAI
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for messages in messages_list:
                future = executor.submit(
                    self.call_openai_api_messages, messages, max_new_tokens, temperature, **kwargs
                )
                futures.append(future)

            # Maintain original order by iterating through futures list directly
            for future in futures:
                try:
                    response = future.result()
                    responses.append(response if response is not None else "")
                except Exception as e:
                    logging.error(f"OpenAI concurrent call failed: {e}")
                    responses.append("[ERROR: Concurrent call failed]")

        return responses

    def get_single_response(self, user_prompt: str,
                           max_new_tokens: int = 1000, temperature: float = 0.1,
                           **kwargs) -> str:
        """Generate single response"""
        return self.call_openai_api(user_prompt, max_new_tokens, temperature, **kwargs)



class ModelClientFactory:
    """Factory for creating model clients"""

    @staticmethod
    def create_client(client_type: str, model_name: Optional[str] = None, tensor_parallel_size: int = 1, **kwargs) -> BaseModelClient:
        """
        Create a model client of specified type

        Args:
            client_type: 'vllm', 'gemini', or 'openai'
            model_name: Model name (optional, uses defaults)
            **kwargs: Additional arguments for client initialization

        Returns:
            BaseModelClient instance
        """
        client_type = client_type.lower()

        if client_type == 'vllm':
            if not VLLM_AVAILABLE:
                raise ImportError("VLLM is not available")
            default_model = "microsoft/phi-4"
            kwargs.pop("max_workers", None)
            return VLLMClient(
                model_name=model_name or default_model,
                tensor_parallel_size=tensor_parallel_size,
                **kwargs
            )

        elif client_type == 'gemini':
            if not GEMINI_AVAILABLE:
                raise ImportError("Gemini is not available")
            default_model = "gemini-1.5-flash"
            kwargs.pop("max_workers", None)
            return GeminiClient(model_name or default_model, **kwargs)

        elif client_type == 'openai':
            default_model = "gpt-4o-mini"
            max_workers = kwargs.get('max_workers', 50)
            api_key = kwargs.get('api_key', None)
            return OpenAIClient(model_name or default_model, api_key=api_key, max_workers=max_workers)

        else:
            raise ValueError(f"Unknown client type: {client_type}. Use 'vllm', 'gemini', or 'openai'")

    @staticmethod
    def get_available_clients() -> List[str]:
        """Get list of available client types"""
        available = []
        if VLLM_AVAILABLE:
            available.append('vllm')
        if GEMINI_AVAILABLE:
            available.append('gemini')
        available.append('openai')
        return available

# Convenience class for backward compatibility
class VLLMClient_samelength(VLLMClient):
    """Backward compatibility wrapper for existing code"""
    pass

def test_clients():
    """Test function to verify clients work"""
    available_clients = ModelClientFactory.get_available_clients()
    print(f"Available clients: {available_clients}")

    test_user = "What is 2+2?"

    for client_type in available_clients:
        try:
            print(f"\nTesting {client_type} client...")

            if client_type == 'vllm':
                client = ModelClientFactory.create_client('vllm', 'microsoft/phi-4')
            elif client_type == 'gemini':
                # Skip Gemini test if no API key available
                if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
                    print(f"Skipping Gemini test - no API key available")
                    continue
                client = ModelClientFactory.create_client('gemini')
            elif client_type == 'openai':
                # Skip OpenAI test if no API key available
                if not os.getenv("OPENAI_API_KEY"):
                    print(f"Skipping OpenAI test - no API key available")
                    continue
                client = ModelClientFactory.create_client('openai')

            response = client.get_single_response(test_user, max_new_tokens=50)
            print(f"Response: {response[:100]}...")
            print(f"{client_type} client working correctly!")

        except Exception as e:
            print(f"Error testing {client_type} client: {e}")

if __name__ == "__main__":
    # Test the clients
    test_clients()

    # Example usage
    print("\n" + "="*50)
    print("EXAMPLE USAGE:")
    print("="*50)

    example_code = '''
# Using VLLM client
vllm_client = ModelClientFactory.create_client('vllm', 'microsoft/phi-4')
response = vllm_client.get_single_response(
    "What is 323 BC + 938 years?",
    temperature=0.1
)

# Using Gemini client (requires API key)
gemini_client = ModelClientFactory.create_client('gemini', api_key="your-api-key")
response = gemini_client.get_single_response(
    "What is 323 BC + 938 years?",
    temperature=0.1
)

# Using OpenAI client (requires API key)
openai_client = ModelClientFactory.create_client('openai', model_name='gpt-4o-mini', api_key="your-api-key")
response = openai_client.get_single_response(
    "What is 323 BC + 938 years?",
    temperature=0.1
)

# Batch processing
responses = client.get_model_response(
    ["User prompt 1", "User prompt 2"]
)

# Custom message format for OpenAI
messages_list = [
    [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is 2+2?"}],
    [{"role": "user", "content": "What is the capital of France?"}]
]
responses = openai_client.get_model_response_messages(messages_list)
    '''

    print(example_code)
