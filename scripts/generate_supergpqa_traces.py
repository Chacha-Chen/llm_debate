"""
Generate reasoning traces for SuperGPQA dataset.

Adapted from the GPQA trace generation script to work with SuperGPQA.

Usage:
    python generate_supergpqa_traces.py --model_name openai/gpt-4o
    python generate_supergpqa_traces.py --model_name anthropic/claude-sonnet-4
    python generate_supergpqa_traces.py --model_name z-ai/glm-4.7
    python generate_supergpqa_traces.py --model_name moonshotai/kimi-k2-thinking
    python generate_supergpqa_traces.py --model_name openai/gpt-4o --debug
"""

import os
import requests
import json
import re
from pathlib import Path
import random
from pydantic import BaseModel, ValidationError
from argparse import ArgumentParser
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from tqdm import tqdm

# Configuration
SUPERGPQA_DATA_FILE = "data/supergpqa/SuperGPQA_raw_500.json"
OUTPUT_DIR = "data/supergpqa"

# Get API key from environment variables or SECRETS file
def load_api_key():
    """Load OpenRouter API key from environment or SECRETS file"""
    # Try environment variable first
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        return api_key
    
    # Try SECRETS file
    secrets_path = Path("SECRETS")
    if secrets_path.exists():
        with open(secrets_path, 'r') as f:
            for line in f:
                if line.startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip()
    
    error_msg = """
OPENROUTER_API_KEY not found!

Please set up your OpenRouter API key using one of these methods:

Method 1 - Create SECRETS file (recommended):
    Create a file named 'SECRETS' in the project root with:
    OPENROUTER_API_KEY=your_key_here
    
Method 2 - Set environment variable:
    Windows: set OPENROUTER_API_KEY=your_key_here
    Linux/Mac: export OPENROUTER_API_KEY=your_key_here

Get your API key from: https://openrouter.ai/keys
"""
    raise ValueError(error_msg)

openrouter_api_key = load_api_key()

# Pydantic model for the reasoning trace
class ReasoningTrace(BaseModel):
    answer: str
    reasoning: list[str]

def load_prompt():
    """Load prompt template for reasoning trace generation"""
    prompt_template = """You are a PhD-level expert. Answer this multiple choice question and explain your reasoning.

Question: {question_text}

Options:
{choices_text}

Respond in strict JSON only with these keys:
- "answer": One choice (A, B, C, or D etc.)
- "reasoning": Array of 4-5 steps explaining your thought process

Only respond in JSON; do not include any text outside of the JSON object."""
    return prompt_template

def load_supergpqa_data(data_file: str, debug: bool = False):
    """
    Load SuperGPQA dataset from JSON file.
    
    Args:
        data_file: Path to SuperGPQA JSON file
        debug: Whether in debug mode (only process first 5 questions)
    
    Returns:
        List of question dictionaries
    """
    print(f"Loading SuperGPQA data from {data_file}")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} questions")
    
    if debug:
        print("DEBUG MODE: Processing only the first 5 questions")
        data = data[:5]
    
    # Show distribution
    if 'difficulty' in data[0]:
        difficulties = {}
        for item in data:
            diff = item.get('difficulty', 'unknown')
            difficulties[diff] = difficulties.get(diff, 0) + 1
        print(f"Questions per difficulty: {difficulties}")
    
    if 'discipline' in data[0]:
        disciplines = {}
        for item in data:
            disc = item.get('discipline', 'unknown')
            disciplines[disc] = disciplines.get(disc, 0) + 1
        print(f"Questions per discipline (top 5):")
        for disc, count in sorted(disciplines.items(), key=lambda x: -x[1])[:5]:
            print(f"  {disc}: {count}")
    
    return data

def is_retryable_exception(e):
    """Check if exception is retryable"""
    if isinstance(e, ValidationError):
        return True
    if isinstance(e, requests.exceptions.RequestException):
        return True
    if hasattr(e, "response") and e.response is not None:
        return e.response.status_code in [429, 500, 502, 503]
    return False

@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    retry=retry_if_exception_type(Exception)
)
def generate_reasoning_trace_with_retry(question, choices, model_name):
    """Generate reasoning trace with retry logic"""
    return generate_reasoning_trace(question, choices, model_name)

def generate_reasoning_trace(question: str, choices: str, model_name: str) -> ReasoningTrace:
    """Generate reasoning trace for a given question and choices"""
    prompt_template = load_prompt()
    
    # Format the prompt with question and choices
    formatted_prompt = prompt_template.format(
        question_text=question,
        choices_text=choices
    )
    
    response = get_response(formatted_prompt, model_name)
    return response

def get_response(prompt: str, model_name: str) -> ReasoningTrace:
    """Call OpenRouter API and parse response"""
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {openrouter_api_key}",
        },
        data=json.dumps({
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a very intelligent assistant, who follows instructions directly."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0,
            "n": 1
        }),
        timeout=(30, 120)  # (connection timeout, read timeout) - prevents hanging during response reading
    )

    # Check the response status
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
        raise requests.exceptions.RequestException(f"HTTP {response.status_code}")
    
    # Parse the JSON response
    response_data = response.json()
    
    # Extract the message content
    if 'choices' not in response_data or len(response_data['choices']) == 0:
        print("No choices found in response")
        print(f"Full response: {response_data}")
        raise ValueError("No choices in API response")
    
    message_content = response_data['choices'][0]['message']['content']
    
    try:
        # Extract JSON from markdown code block if present
        json_match = re.search(r'```json\s*\n(.*?)\n```', message_content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Try to find JSON object directly
            json_match = re.search(r'\{.*\}', message_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0).strip()
            else:
                raise ValueError("No JSON found in response")
        
        # Parse JSON
        json_data = json.loads(json_str)
        
        # Validate with Pydantic
        validated_response = ReasoningTrace(**json_data)
        return validated_response
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Trying to parse: {json_str[:200]}...")
        raise
    except ValidationError as e:
        print(f"Pydantic validation error: {e}")
        print(f"Received data: {json_data}")
        raise
    except Exception as e:
        print(f"Error parsing response: {e}")
        print(f"Raw response: {message_content[:200]}...")
        raise

def format_choices(options_dict, shuffled_keys):
    """Format choices as 'A. option1, B. option2, ...'"""
    formatted = []
    for i, key in enumerate(shuffled_keys):
        letter = chr(65 + i)  # A, B, C, D
        formatted.append(f"{letter}. {options_dict[key]}")
    return ", ".join(formatted)

def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, 
                       help="Model to use (e.g., openai/gpt-4o, anthropic/claude-sonnet-4)")
    parser.add_argument("--debug", action="store_true", 
                       help="Debug mode: only process first 5 questions")
    parser.add_argument("--data_file", type=str, default=SUPERGPQA_DATA_FILE,
                       help="Path to SuperGPQA JSON file")
    args = parser.parse_args()
    
    model_name = args.model_name
    data = load_supergpqa_data(args.data_file, debug=args.debug)
    
    # Determine output filename
    output_filename = f"SuperGPQA_Reasoning_Traces_{model_name.replace('/', '_')}_all_{len(data)}.json"
    output_path = Path(OUTPUT_DIR) / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing results if file exists
    results = []
    if output_path.exists():
        print(f"Found existing file: {output_path}")
        with open(output_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results")
    
    # Get processed indices to skip
    processed_indices = {result['index'] for result in results}
    remaining_data = [item for item in data if item.get('index', -1) not in processed_indices]
    
    if len(remaining_data) == 0:
        print("All questions already processed!")
        return
    
    print(f"Processing {len(remaining_data)} remaining questions out of {len(data)} total")
    
    # Process each remaining question
    total_processed = len(results)
    total_to_process = len(data)
    
    for item in tqdm(remaining_data, desc=f"Processing questions ({total_processed}/{total_to_process} done)"):
        index = item.get('index', -1)
        question = item.get('question', '')
        
        # Parse options (SuperGPQA format: list of options)
        options = item.get('options', [])
        correct_answer_letter = item.get('answer_letter', '').strip().upper()
        
        if not options or not correct_answer_letter:
            tqdm.write(f"Skipping question {index}: missing options or answer")
            continue
        
        # Convert options list to dict: {A: option1, B: option2, ...}
        options_dict = {chr(65 + i): opt for i, opt in enumerate(options)}
        correct_answer_text = options_dict.get(correct_answer_letter, '')
        
        # Shuffle choices (deterministic per question)
        random.seed(42 + index)
        shuffled_keys = list(options_dict.keys())
        random.shuffle(shuffled_keys)
        
        # Get new correct choice letter after shuffling
        correct_choice = chr(65 + shuffled_keys.index(correct_answer_letter))
        
        # Format choices for the prompt
        choices_str = format_choices(options_dict, shuffled_keys)
        
        tqdm.write(f"\nProcessing question {index}...")
        if 'discipline' in item:
            tqdm.write(f"Discipline: {item['discipline']}")
        
        try:
            response = generate_reasoning_trace_with_retry(question, choices_str, model_name)
            if response:
                result = {
                    'index': index,
                    'Question': question,
                    'choices': choices_str,
                    'correct_choice': correct_choice,
                    'model_answer': response.answer,
                    'model_reasoning': response.reasoning,
                    'model_correct': correct_choice == response.answer.strip().upper(),
                    # Include all original metadata
                    **{k: v for k, v in item.items() if k not in ['index', 'question', 'options']}
                }
                results.append(result)
                tqdm.write(f"✓ Successfully processed question {index}")
                
                # Save partial results after each successful question
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            else:
                tqdm.write(f"✗ Failed to process question {index}")
        except Exception as e:
            tqdm.write(f"✗ Error processing question {index}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"{'='*60}")
    print(f"Total results: {len(results)}")
    print(f"Results saved to: {output_path}")
    
    # Calculate accuracy
    if results:
        correct_count = sum(1 for r in results if r.get('model_correct', False))
        accuracy = correct_count / len(results) * 100
        print(f"Model accuracy: {correct_count}/{len(results)} ({accuracy:.1f}%)")

if __name__ == "__main__":
    main()
