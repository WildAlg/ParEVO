# std imports
from argparse import ArgumentParser
import json
import os
import re
import time
from typing import Optional

# tpl imports
from tqdm import tqdm
from openai import OpenAI

""" Prompt template: """
SYSTEM_TEMPLATE = """You are an expert C++ programmer with extensive experience in parallel programming. 
Write a parallel {} procedure in C++ that is correct and is the fastest parallel {} program you can generate.
Return the code between #start and #end markers. 
"""

# """ Prompt template: """
# SYSTEM_TEMPLATE = """You are a helpful coding assistant.
# You are helping a programmer write a C++ function. Write the body of the function and put it in a markdown code block.
# Do not write any other code or explanations.
# """

# PROMPT_TEMPLATE = """Complete the C++ function {function_name}. Only write the body of the function {function_name}.

# ```cpp
# {prompt}
# ```
# """


def get_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model", choices=["gpt-5", "gpt-4-1106-preview", "gpt-3.5-turbo-1106"], required=True, help="The model to use.")
    parser.add_argument("--effort", choices=["minimal", "low", "mediam", "high"], required=True, help="Reasoning tokens the model generates before producing a response.")
    parser.add_argument("-p", "--prompts", type=str, required=True, help="Path to prompts json")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to output json")
    parser.add_argument("--api-key", type=str, help="OpenAI API key. " +
        "If not provided, then uses environment variable OPENAI_API_KEY.")
    # parser.add_argument("--openai-organization", type=str, help="OpenAI organization. " +
        # "If not provided, then uses environment variable OPENAI_ORGANIZATION.")
    parser.add_argument("--max-requests", type=int, help="If provided, then only makes this many requests.")
    parser.add_argument("--max-tokens-per-second", help="Limit the rate of token generation.")
    parser.add_argument("--max-requests-per-second", help="Limit the rate of request generation.")
    parser.add_argument("--dry", action="store_true", help="If provided, then don't make any requests.")
    parser.add_argument("--overwrite", action="store_true", help="If provided, then overwrite outputs already in file.")
    parser.add_argument("--temperature", type=float, default=0.7, help="The temperature to use for sampling.")
    parser.add_argument("--top-p", type=float, default=0.95, help="The top p to use for sampling.")
    parser.add_argument("--max-new-tokens", type=int, default=20000, help="The maximum number of tokens to generate.")
    parser.add_argument("--num-samples-per-prompt", type=int, default=1, help="The number of samples to generate per prompt.")
    return parser.parse_args()


def get_env_var(name: str) -> str:
    """ Get an environment variable. """
    if name not in os.environ:
        raise ValueError(f"Environment variable {name} not set.")
    return os.environ[name]

# GPU_FUNCTION_NAME_PATTERN = re.compile(r"__global__ void ([a-zA-Z0-9_]+)\(")
# CPU_FUNCTION_NAME_PATTERN = re.compile(r"\s*[a-zA-Z_]+ ([a-zA-Z0-9_]+)\(")
# def get_function_name(prompt: str, execution_model: str) -> str:
#     if execution_model in ['cuda', 'hip']:
#         match = GPU_FUNCTION_NAME_PATTERN.match(prompt.splitlines()[-1])
#     else:
#         match = CPU_FUNCTION_NAME_PATTERN.match(prompt.splitlines()[-1])
#     if match is None:
#         raise ValueError(f"Could not find function name in prompt: {prompt}")
#     return match.group(1)

def get_max_tokens_per_second(model: str) -> Optional[float]:
    """Rate limits—RPM/TPM values valid as of mid-2025 (exact values pending for newer models)."""
    if model in ("gpt-3.5-turbo", "gpt-3.5-turbo-1106"):
        # Example: ~200,000 tokens per minute for high-tier usage
        return 200_000 / 60  # ≈ 3,333 tokens/sec
    elif model in ("gpt-4", "gpt-4o"):
        # Example limited estimate: 40,000 tokens per minute
        return 40_000 / 60  # ≈ 667 tokens/sec
    elif model.startswith("gpt-4.1"):
        # No official rate data published yet
        return None
    elif model.startswith("gpt-5"):
        # Rate limits not released yet
        return None
    else:
        return None

def get_max_requests_per_second(model: str) -> Optional[float]:
    """Rate limit requests/sec based on mid-2025 data or TBD for newer models."""
    if model in ("gpt-3.5-turbo", "gpt-3.5-turbo-1106"):
        # Example: ~3,500 RPM → ~58 req/sec
        return 3_500 / 60
    elif model in ("gpt-4", "gpt-4o"):
        # Example: ~200 RPM → ~3.3 req/sec
        return 200 / 60
    elif model.startswith("gpt-4.1"):
        # To be updated when published
        return None
    elif model.startswith("gpt-5"):
        # Unknown
        return None
    else:
        return None

def get_max_requests_per_day(model: str) -> Optional[int]:
    """Daily rate cap estimates as of mid-2025; updated for new models when known."""
    if model in ("gpt-3.5-turbo", "gpt-3.5-turbo-1106"):
        return 10_000
    elif model in ("gpt-4", "gpt-4o"):
        return 10_000
    elif model.startswith("gpt-4.1"):
        return None
    elif model.startswith("gpt-5"):
        return None
    else:
        return None


def postprocess(prompt: str, output: str) -> str:
    """ Postprocess the output. """
    # remove leading ```, ```cpp, and trailing ```
    output = output.strip().lstrip("```cpp").lstrip("```").rstrip("```")

    # remove prompt if it included it
    if output.startswith(prompt):
        output = output[len(prompt):]

    return output

def main():
    args = get_args()

    # get the prompts
    with open(args.prompts, 'r') as prompts_json:
        prompts = json.load(prompts_json)

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
                   o["alg_name"] == prompt["alg_name"] and \
                   o["parallelism_model"] == prompt["parallelism_model"] and \
                   "outputs" in o and \
                   len(o["outputs"]) == args.num_samples_per_prompt and \
                   o["temperature"] == args.temperature and \
                   o["top_p"] == args.top_p:
                    for col in ["temperature", "top_p", "do_sample", "max_new_tokens", "outputs"]:
                        prompt[col] = o[col]
                    copy_count += 1
                    break
        print(f"Copied {copy_count} existing outputs.")

    # get the keys
    api_key = args.api_key or get_env_var("OPENAI_API_KEY")
    # organization = args.openai_organization or get_env_var("OPENAI_ORGANIZATION")

    # create the client
    client = OpenAI(api_key=api_key)

    # generation metadata
    MAX_TOKENS_PER_SECOND = args.max_tokens_per_second or get_max_tokens_per_second(args.model)
    MAX_REQUESTS_PER_SECOND = args.max_requests_per_second or get_max_requests_per_second(args.model)
    MAX_REQUESTS = args.max_requests or get_max_requests_per_day(args.model)

    # generate outputs
    request_counter = 0
    request_rate_counter = 0
    token_counter = 0
    token_rate_counter = 0
    token_timer = time.time()
    request_timer = time.time()
    for prompt in tqdm(prompts, desc="Generating outputs"):
        # see if we can skip this
        if not args.overwrite and "outputs" in prompt:
            continue

        # get the prompt
        original_prompt = prompt["prompt"]
        # function_name = get_function_name(original_prompt, prompt["parallelism_model"])
        # prompt_text = PROMPT_TEMPLATE.format(prompt=original_prompt, function_name=function_name)

        system_prompt = SYSTEM_TEMPLATE.format(prompt["name"], prompt["name"])
        # prompt_text = PROMPT_TEMPLATE.format(prompt["name"], prompt["name"])
        # generate the outputs
        if args.dry:
            print("system", system_prompt)
            # print("prompt", prompt_text)
            continue

        # set metadata
        if args.model.startswith("gpt-5") or args.model.startswith("gpt-4.1"):
            # reasoning models: no sampling parameters
            prompt["temperature"] = None
            prompt["top_p"] = None
            prompt["do_sample"] = False
        else:
            prompt["temperature"] = args.temperature
            prompt["top_p"] = args.top_p
            prompt["do_sample"] = True

        prompt["max_new_tokens"] = args.max_new_tokens
        prompt["generate_model"] = args.model

        # generate the outputs
        outputs = []
        input = system_prompt + "\n" + original_prompt
        for _ in range(args.num_samples_per_prompt):
            # build kwargs depending on model
            kwargs = dict(
                model=args.model,
                input=input,
                max_output_tokens=args.max_new_tokens,
            )
            if not (args.model.startswith("gpt-5") or args.model.startswith("gpt-4.1")):
                kwargs["temperature"] = args.temperature
                kwargs["top_p"] = args.top_p

            # always safe for reasoning models
            kwargs["reasoning"] = {"effort": args.effort}

            resp = client.responses.create(**kwargs)

            # update counters
            request_counter += 1
            request_rate_counter += 1
            token_counter += resp.usage.total_tokens
            token_rate_counter += resp.usage.total_tokens

            text_out = resp.output_text
            outputs.append(postprocess(original_prompt, text_out))

        prompt["outputs"] = outputs

        # check if we should stop
        if MAX_REQUESTS is not None and request_counter >= MAX_REQUESTS:
            print(f"Stopping after {request_counter} requests.")
            break
    
        # check if we should sleep
        tokens_per_second = token_rate_counter / (time.time() - token_timer)
        if MAX_TOKENS_PER_SECOND is not None and tokens_per_second > (MAX_TOKENS_PER_SECOND*0.9):
            sleep_time = 30
            print(f"Sleeping for {sleep_time} seconds.")
            time.sleep(sleep_time)
            token_timer = time.time()
            token_rate_counter = 0
        
        requests_per_second = request_rate_counter / (time.time() - request_timer)
        if MAX_REQUESTS_PER_SECOND is not None and requests_per_second > (MAX_REQUESTS_PER_SECOND*0.95):
            sleep_time = 60
            print(f"Sleeping for {sleep_time} seconds.")
            time.sleep(sleep_time)
            request_timer = time.time()
            request_rate_counter = 0

        # write intermediate outputs
        with open(args.output, 'w') as output_json:
            json.dump(prompts, output_json, indent=2)

    # summary stats
    print(f"Submitted {request_counter} requests.")
    print(f"Used {token_counter} tokens.")

    # write outputs
    with open(args.output, 'w') as output_json:
        json.dump(prompts, output_json, indent=2)
    

if __name__ == "__main__":
    main()