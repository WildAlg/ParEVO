"""
Generate code with Vertex AI fine-tuned Gemini models.
Robust version with rate limiting, retries, and caching.
"""
# std imports
import argparse
import json
import os
import re
import time
from typing import Optional, List, Tuple

# tpl imports
from alive_progress import alive_bar
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, HarmCategory, HarmBlockThreshold
from google.api_core import exceptions

# --- Configuration ---
SYSTEM_TEMPLATE = """You are a **helpful** coding assistant.
You are helping a programmer write a C++ function. Write the body of the function and put it in a markdown code block. 
**Requirements**:
- **DO NOT WRITE ANY COMMENTS OR EXPLANATIONS** in the code!!! Generate **PURE** code!!!
- Before you return the code, make sure to **remove any comments or explanations** that you may have added.
"""

PROMPT_TEMPLATE = """Complete the C++ function {function_name}. Only write the body of the function {function_name}. 
```cpp
{prompt}
```
"""

# --- Argument Parser ---
def get_args():
    parser = argparse.ArgumentParser(description="Generate code with Vertex AI")
    parser.add_argument("-m", "--model", required=True, 
        help="Vertex AI model endpoint (full resource name or model ID)")
    parser.add_argument("-p", "--prompts", type=str, required=True, help="Path to prompts json")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to output json")
    parser.add_argument("--project", type=str, required=True, help="GCP project ID")
    parser.add_argument("--location", type=str, default="us-central1", help="GCP location")
    parser.add_argument("--max-requests", type=int, help="Max total requests to make")
    parser.add_argument("--max-requests-per-minute", type=int, default=60,
        help="Rate limit: max requests per minute")
    parser.add_argument("--dry", action="store_true", help="Don't make any requests")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top p for sampling")
    parser.add_argument("--top-k", type=int, default=40, help="Top k for sampling")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Max tokens to generate")
    parser.add_argument("--num-samples-per-prompt", type=int, default=20, 
        help="Number of samples per prompt")
    parser.add_argument("--candidate-count", type=int, default=5,
        help="Candidates per request (max 8 for Vertex AI)")
    parser.add_argument("--raw-prompt", action="store_true",
        help="Use raw prompt without any formatting")
    parser.add_argument("--cache", type=str, help="JSONL cache file for intermediate results")
    return parser.parse_args()


# --- Utility Functions ---
GPU_FUNCTION_NAME_PATTERN = re.compile(r"__global__ void ([a-zA-Z0-9_]+)\(")
CPU_FUNCTION_NAME_PATTERN = re.compile(r"\s*[a-zA-Z_]+ ([a-zA-Z0-9_]+)\(")

def get_function_name(prompt: str, execution_model: str) -> str:
    if execution_model in ['cuda', 'hip']:
        match = GPU_FUNCTION_NAME_PATTERN.match(prompt.splitlines()[-1])
    else:
        match = CPU_FUNCTION_NAME_PATTERN.match(prompt.splitlines()[-1])
    if match is None:
        raise ValueError(f"Could not find function name in prompt: {prompt}")
    return match.group(1)


def postprocess(prompt: str, output: str) -> str:
    """Postprocess the output - remove markdown fences and prompt echo."""
    output = output.strip()
    # Remove markdown code fences
    output = output.removeprefix("```cpp").removeprefix("```c++").removeprefix("```")
    output = output.removesuffix("```")
    output = output.strip()
    
    # Remove prompt if echoed
    if output.startswith(prompt):
        output = output[len(prompt):]
    
    return output


def clean_output(output: str, prompt: str) -> str:
    """
    Clean LLM output: remove prompt prefix and truncate at matching closing brace.
    """
    # Find and remove prompt
    prompt_loc = output.find(prompt)
    if prompt_loc != -1:
        output = output[prompt_loc + len(prompt):].strip()
    
    # Add opening brace temporarily to find matching close
    output = '{' + output
    
    # Find matching brace
    stack = []
    index = 0
    while index < len(output):
        token = output[index]
        if token == '{':
            stack.append(token)
        elif token == '}':
            if stack:
                stack.pop()
            if len(stack) == 0:
                break
        index += 1
    
    # Truncate at matching brace
    output = output[1:index+1]
    return output


# --- Rate Limiter ---
class RateLimiter:
    """Simple rate limiter using sliding window."""
    
    def __init__(self, max_requests_per_minute: int):
        self.max_rpm = max_requests_per_minute
        self.request_times: List[float] = []
    
    def wait_if_needed(self):
        """Block until we're under the rate limit."""
        now = time.time()
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        if len(self.request_times) >= self.max_rpm:
            # Need to wait
            oldest = min(self.request_times)
            sleep_time = 60 - (now - oldest) + 0.1  # Small buffer
            if sleep_time > 0:
                print(f"Rate limit reached. Sleeping for {sleep_time:.1f}s...")
                time.sleep(sleep_time)
        
        self.request_times.append(time.time())
    
    def record_request(self):
        """Record a request was made."""
        self.request_times.append(time.time())


# --- Generation Function ---
def generate_with_retry(
    model: GenerativeModel,
    prompt: str,
    generation_config: GenerationConfig,
    safety_settings: dict,
    rate_limiter: RateLimiter,
    max_retries: int = 5,
) -> Tuple[List[str], List[int], Optional[str]]:
    """
    Generate content with automatic retry on rate limits.
    
    Returns:
        Tuple of (outputs, finish_reasons, error_message)
    """
    outputs = []
    finish_reasons = []
    error = None
    
    for attempt in range(max_retries):
        try:
            rate_limiter.wait_if_needed()
            
            response = model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings,
            )
            
            # Process candidates
            for candidate in response.candidates:
                # Vertex AI finish reasons: FINISH_REASON_STOP=1, FINISH_REASON_MAX_TOKENS=2
                finish_reason = candidate.finish_reason
                
                if candidate.content and candidate.content.parts:
                    text = candidate.content.parts[0].text
                    outputs.append(text)
                    finish_reasons.append(finish_reason)
                else:
                    print(f"Warning: Candidate had no content (finish_reason={finish_reason})")
            
            return outputs, finish_reasons, None
            
        except exceptions.ResourceExhausted as e:
            wait_time = min(2 ** attempt * 5, 60)  # Exponential backoff, max 60s
            print(f"Rate limit hit (attempt {attempt + 1}/{max_retries}). "
                  f"Waiting {wait_time}s...")
            time.sleep(wait_time)
            error = str(e)
            
        except exceptions.InvalidArgument as e:
            # Usually means bad input, don't retry
            print(f"Invalid argument error: {e}")
            return [], [], str(e)
            
        except Exception as e:
            wait_time = min(2 ** attempt * 2, 30)
            print(f"Error (attempt {attempt + 1}/{max_retries}): {e}. "
                  f"Waiting {wait_time}s...")
            time.sleep(wait_time)
            error = str(e)
    
    return outputs, finish_reasons, error


# --- Cache Functions ---
def load_cache(cache_path: str) -> dict:
    """Load cache from JSONL file. Returns dict keyed by (name, parallelism_model, prompt)."""
    cache = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                key = (entry["name"], entry["parallelism_model"], entry["prompt"])
                cache[key] = entry
        print(f"Loaded {len(cache)} entries from cache")
    return cache


def append_to_cache(cache_path: str, entry: dict):
    """Append a single entry to cache file."""
    if cache_path:
        with open(cache_path, 'a') as f:
            f.write(json.dumps(entry) + "\n")


# --- Main ---
def main():
    args = get_args()
    
    # Initialize Vertex AI
    print(f"Initializing Vertex AI (project={args.project}, location={args.location})")
    vertexai.init(project=args.project, location=args.location)
    
    # Create model
    model = GenerativeModel(args.model)
    
    # Safety settings (match original script - block none)
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    
    # Generation config
    # Note: candidate_count > 1 requires temperature > 0
    effective_temp = args.temperature if args.temperature > 0 else 0.1
    
    # Rate limiter
    rate_limiter = RateLimiter(args.max_requests_per_minute)
    
    # Load prompts
    with open(args.prompts, 'r') as f:
        prompts = json.load(f)
    print(f"Loaded {len(prompts)} prompts")
    
    # Load existing outputs if not overwriting
    if not args.overwrite and os.path.exists(args.output):
        with open(args.output, 'r') as f:
            existing_outputs = json.load(f)
        
        # Copy existing outputs into prompts
        copy_count = 0
        for prompt in prompts:
            for o in existing_outputs:
                if (o["prompt"] == prompt["prompt"] and
                    o["name"] == prompt["name"] and
                    o["parallelism_model"] == prompt["parallelism_model"] and
                    "outputs" in o and
                    len(o["outputs"]) >= args.num_samples_per_prompt and
                    o.get("temperature") == args.temperature and
                    o.get("top_p") == args.top_p):
                    for col in ["temperature", "top_p", "do_sample", "max_new_tokens", 
                                "outputs", "raw_outputs", "finish_reasons"]:
                        if col in o:
                            prompt[col] = o[col]
                    copy_count += 1
                    break
        print(f"Copied {copy_count} existing outputs")
    
    # Load cache
    cache = load_cache(args.cache)
    
    # Generate outputs
    request_counter = 0
    with alive_bar(len(prompts), title="Generating outputs", dual_line=True) as bar:
        for prompt in prompts:
            # Skip if already have outputs
            if not args.overwrite and "outputs" in prompt:
                bar(skipped=True)
                continue
            
            # Check cache
            cache_key = (prompt["name"], prompt["parallelism_model"], prompt["prompt"])
            if cache_key in cache:
                cached = cache[cache_key]
                if len(cached.get("outputs", [])) >= args.num_samples_per_prompt:
                    for col in ["temperature", "top_p", "do_sample", "max_new_tokens",
                                "outputs", "raw_outputs", "finish_reasons"]:
                        if col in cached:
                            prompt[col] = cached[col]
                    bar.text("~> Loaded from cache")
                    bar()
                    continue
            
            # Format prompt
            original_prompt = prompt["prompt"]
            
            if args.raw_prompt:
                # For raw prompts: use prompt as-is
                prompt_text = original_prompt
            else:
                # Default: use template with system prompt (matches original script)
                function_name = get_function_name(original_prompt, prompt["parallelism_model"])
                prompt_text = PROMPT_TEMPLATE.format(
                    prompt=original_prompt, 
                    function_name=function_name
                )
                prompt_text = SYSTEM_TEMPLATE + "\n" + prompt_text
            
            # Dry run
            if args.dry:
                print(f"[DRY] Prompt:\n{prompt_text[:500]}...")
                bar()
                continue
            
            # Set metadata
            prompt["temperature"] = args.temperature
            prompt["top_p"] = args.top_p
            prompt["top_k"] = args.top_k
            prompt["do_sample"] = True
            prompt["max_new_tokens"] = args.max_new_tokens
            
            # Generate samples
            outputs = []
            raw_outputs = []
            finish_reasons = []
            errors = []
            
            while len(outputs) < args.num_samples_per_prompt:
                remaining = args.num_samples_per_prompt - len(outputs)
                
                # Adjust candidate count for final batch
                current_candidate_count = min(args.candidate_count, remaining, 8)
                current_config = GenerationConfig(
                    candidate_count=current_candidate_count,
                    max_output_tokens=args.max_new_tokens,
                    temperature=effective_temp,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )
                
                batch_outputs, batch_reasons, error = generate_with_retry(
                    model=model,
                    prompt=prompt_text,
                    generation_config=current_config,
                    safety_settings=safety_settings,
                    rate_limiter=rate_limiter,
                )
                
                if error:
                    errors.append(error)
                    print(f"Error after retries: {error}")
                    # Continue trying unless we've had too many errors
                    if len(errors) > 10:
                        print("Too many errors, skipping this prompt")
                        break
                
                for text, reason in zip(batch_outputs, batch_reasons):
                    raw_outputs.append(text)
                    finish_reasons.append(reason)
                    try:
                        cleaned = postprocess(original_prompt, text)
                    except Exception:
                        cleaned = text
                    outputs.append(cleaned)
                
                bar.text(f"~> Got {len(outputs)}/{args.num_samples_per_prompt} samples")
                request_counter += 1
                
                # Check max requests
                if args.max_requests and request_counter >= args.max_requests:
                    print(f"Reached max requests ({args.max_requests})")
                    break
            
            # Store results
            prompt["outputs"] = outputs[:args.num_samples_per_prompt]
            prompt["raw_outputs"] = raw_outputs[:args.num_samples_per_prompt]
            prompt["finish_reasons"] = finish_reasons[:args.num_samples_per_prompt]
            if errors:
                prompt["errors"] = errors
            
            # Update cache
            append_to_cache(args.cache, prompt)
            
            bar()
            
            # Write intermediate outputs
            with open(args.output, 'w') as f:
                json.dump(prompts, f, indent=2)
            
            # Check max requests (outer loop)
            if args.max_requests and request_counter >= args.max_requests:
                break
    
    # Final save
    with open(args.output, 'w') as f:
        json.dump(prompts, f, indent=2)
    
    print(f"\nDone! Made {request_counter} requests.")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()