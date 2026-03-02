# std imports
from argparse import ArgumentParser
import json
import os
import time
from typing import Optional, List, Tuple, Any

# tpl imports
from alive_progress import alive_bar
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, HarmCategory, HarmBlockThreshold
from google.api_core import exceptions


""" Prompt template: """
SYSTEM_TEMPLATE = """You are an expert C++ programmer with extensive experience in parallel programming. 
Write a parallel {} procedure in C++ that is correct and is the fastest parallel {} program you can generate.
Return the code between `// --- Start of file:` and `// --- End of file:` markers. 
"""


def get_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model", required=True, 
        help="Vertex AI model endpoint (full resource name)")
    parser.add_argument("--project", type=str, required=True, help="GCP project ID")
    parser.add_argument("--location", type=str, default="us-central1", help="GCP location")
    parser.add_argument("-p", "--prompts", type=str, required=True, help="Path to prompts json")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to output json")
    parser.add_argument("--max-requests", type=int, help="If provided, then only makes this many requests.")
    parser.add_argument("--dry", action="store_true", help="If provided, then don't make any requests.")
    parser.add_argument("--overwrite", action="store_true", help="If provided, then overwrite outputs already in file.")
    parser.add_argument("--temperature", type=float, default=0.7, help="The temperature to use for sampling.")
    parser.add_argument("--top-p", type=float, default=0.95, help="The top p to use for sampling.")
    parser.add_argument("--max-new-tokens", type=int, default=20000, help="The maximum number of tokens to generate.")
    parser.add_argument("--num-samples-per-prompt", type=int, default=20, help="The number of samples to generate " +
        "per prompt.")
    parser.add_argument("--candidate-count", type=int, default=1,
        help="Candidates per request (max 8 for Vertex AI)")
    return parser.parse_args()


def postprocess(prompt: str, output: str) -> str:
    """Extract code between explicit Start/End markers."""

    # Normalize whitespace
    output = output.strip()

    # Remove markdown fences if present
    for prefix in ("```cpp", "```c++", "```"):
        if output.startswith(prefix):
            output = output[len(prefix):].strip()
    if output.endswith("```"):
        output = output[:-3].strip()

    # Remove prompt if echoed
    if output.startswith(prompt):
        output = output[len(prompt):].strip()

    start_marker = "// --- Start of file:"
    end_marker = "// --- End of file:"

    start_idx = output.find(start_marker)
    end_idx = output.find(end_marker)

    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        print("Postprocess Warning: Could not find Start/End markers in model output, using raw output.", output)
        return output

    # Extract code block
    code = output[start_idx:end_idx]

    # Drop the start marker line itself
    code = code.split("\n", 1)[1]

    return code.rstrip()


def generate_with_retry(
    model: GenerativeModel,
    prompt: str,
    generation_config: GenerationConfig,
    safety_settings: dict,
    max_retries: int = 5,
) -> Tuple[List[str], List[Any], Optional[str]]:
    """
    Generate content with automatic retry on rate limits.
    Returns: (outputs, finish_reasons, error_message)
    """
    outputs = []
    finish_reasons = []
    error = None
    
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            response = model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings,
            )
            print(f"API call seconds: {time.time() - t0:.2f}")
            
            for candidate in response.candidates:
                finish_reason = candidate.finish_reason
                if candidate.content and candidate.content.parts:
                    text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
                    outputs.append(text)
                    finish_reasons.append(finish_reason)
                else:
                    print(f"Warning: Candidate had no content (finish_reason={finish_reason})")
            
            return outputs, finish_reasons, None
            
        except exceptions.ResourceExhausted as e:
            wait_time = min(2 ** attempt * 5, 60)
            print(f"Rate limit hit (attempt {attempt + 1}/{max_retries}). Waiting {wait_time}s...")
            time.sleep(wait_time)
            error = str(e)
            
        except exceptions.InvalidArgument as e:
            print(f"Invalid argument error: {e}")
            return [], [], str(e)
            
        except Exception as e:
            wait_time = min(2 ** attempt * 2, 30)
            print(f"Error (attempt {attempt + 1}/{max_retries}): {e}. Waiting {wait_time}s...")
            time.sleep(wait_time)
            error = str(e)
    
    return outputs, finish_reasons, error


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

    # get the prompts
    with open(args.prompts, 'r') as prompts_json:
        prompts = json.load(prompts_json)
    print(f"Loaded {len(prompts)} prompts")

    # read in outputs
    if not args.overwrite and os.path.exists(args.output):
        with open(args.output, 'r') as output_json:
            outputs = json.load(output_json)

        # copy existing outputs into prompts
        copy_count = 0
        for prompt in prompts:
            for o in outputs:
                if o["prompt"] == prompt["prompt"] and \
                   o["name"] == prompt["name"] and \
                   o["parallelism_model"] == prompt["parallelism_model"] and \
                   "outputs" in o and \
                   len(o["outputs"]) == args.num_samples_per_prompt and \
                   o["temperature"] == args.temperature and \
                   o["top_p"] == args.top_p:
                    for col in ["temperature", "top_p", "do_sample", "max_new_tokens", "outputs"]:
                        if col in o:
                            prompt[col] = o[col]
                    copy_count += 1
                    break
        print(f"Copied {copy_count} existing outputs.")

    # Generation config
    effective_temp = args.temperature if args.temperature > 0 else 0.1
    gen_config = GenerationConfig(
        candidate_count=min(args.candidate_count, 8),
        max_output_tokens=args.max_new_tokens,
        temperature=effective_temp,
        top_p=args.top_p,
    )

    # generate outputs
    request_counter = 0
    with alive_bar(len(prompts), title="Generating outputs", dual_line=True) as bar:
        for prompt in prompts:
            # see if we can skip this
            if not args.overwrite and "outputs" in prompt:
                bar(skipped=True)
                continue

            # get the prompt
            original_prompt = prompt["prompt"]
            system_prompt = SYSTEM_TEMPLATE.format(prompt["name"], prompt["name"])

            # generate the outputs
            if args.dry:
                print("system:", system_prompt)
                print("prompt:", original_prompt[:500], "...")
                bar()
                continue

            # set metadata
            prompt["temperature"] = args.temperature
            prompt["top_p"] = args.top_p
            prompt["do_sample"] = True
            prompt["max_new_tokens"] = args.max_new_tokens
            prompt["generate_model"] = args.model

            # generate the outputs
            outputs = []
            while len(outputs) < args.num_samples_per_prompt:
                input_text = system_prompt + "\n" + original_prompt
                
                batch_outputs, batch_reasons, error = generate_with_retry(
                    model=model,
                    prompt=input_text,
                    generation_config=gen_config,
                    safety_settings=safety_settings,
                )
                
                if error:
                    print(f"Error after retries: {error}")
                    time.sleep(5)
                    continue

                for text, reason in zip(batch_outputs, batch_reasons):
                    # Check finish reason (1=STOP, 2=MAX_TOKENS are acceptable)
                    if reason in (1, 2):
                        outputs.append(text)
                        bar.text(f"~> Received output {len(outputs)} of {args.num_samples_per_prompt}.")
                    else:
                        print(f"Got a completion with finish_reason={reason}.")
                        time.sleep(5)

            # postprocess outputs
            outputs = [postprocess(original_prompt, o) for o in outputs]
            prompt["outputs"] = outputs
            bar()

            # update counters
            request_counter += 1

            # check if we should stop
            if args.max_requests is not None and request_counter >= args.max_requests:
                print(f"Stopping after {request_counter} requests.")
                break

            # write intermediate outputs
            with open(args.output, 'w') as output_json:
                json.dump(prompts, output_json, indent=2)

    # summary stats
    print(f"Submitted {request_counter} requests.")

    # write outputs
    with open(args.output, 'w') as output_json:
        json.dump(prompts, output_json, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()