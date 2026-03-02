# std imports
from argparse import ArgumentParser
import json
import os
import re
import time
from typing import Optional

# tpl imports
from alive_progress import alive_bar
# import google.generativeai as genai
from google import genai
from google.genai import types


""" Prompt template: """
SYSTEM_TEMPLATE = """You are an expert C++ programmer with extensive experience in parallel programming. 
Write a parallel {} procedure in C++ that is correct and is the fastest parallel {} program you can generate.
Return the code between `// --- Start of file:` and `// --- End of file:` markers. 
"""

# PROMPT_TEMPLATE = """Complete the C++ function {function_name}. Only write the body of the function {function_name}. 
# ```cpp
# {prompt}
# ```
# """


def get_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model", choices=["gemini-2.5-flash", "gemini-2.5-pro"], required=True, help="The model to use.")
    parser.add_argument("-t", "--tier", choices=["free", "tier1", "tier2", "tier3"], default="tier3", help="Rate limits for Gemini API.")
    parser.add_argument("-p", "--prompts", type=str, required=True, help="Path to prompts json")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to output json")
    parser.add_argument("--api-key", type=str, help="Google AI API key. " +
        "If not provided, then uses environment variable GOOGLE_API_KEY.")
    parser.add_argument("--max-requests", type=int, help="If provided, then only makes this many requests.")
    parser.add_argument("--max-tokens-per-second", help="Limit the rate of token generation.")
    parser.add_argument("--max-requests-per-second", help="Limit the rate of request generation.")
    parser.add_argument("--dry", action="store_true", help="If provided, then don't make any requests.")
    parser.add_argument("--overwrite", action="store_true", help="If provided, then overwrite outputs already in file.")
    parser.add_argument("--temperature", type=float, default=0.7, help="The temperature to use for sampling.")
    parser.add_argument("--top-p", type=float, default=0.95, help="The top p to use for sampling.")
    parser.add_argument("--max-new-tokens", type=int, default=20000, help="The maximum number of tokens to generate.")
    parser.add_argument("--num-samples-per-prompt", type=int, default=20, help="The number of samples to generate " +
        "per prompt.")
    return parser.parse_args()


def get_env_var(name: str) -> str:
    """ Get an environment variable. """
    if name not in os.environ:
        raise ValueError(f"Environment variable {name} not set.")
    return os.environ[name]

# CPU_FUNCTION_NAME_PATTERN = re.compile(r"\s*[a-zA-Z_]+ ([a-zA-Z0-9_]+)\(")
# def get_function_name(prompt: str, execution_model: str) -> str:
#     if execution_model in ['cuda', 'hip']:
#         match = GPU_FUNCTION_NAME_PATTERN.match(prompt.splitlines()[-1])
#     else:
#         match = CPU_FUNCTION_NAME_PATTERN.match(prompt.splitlines()[-1])
#     if match is None:
#         raise ValueError(f"Could not find function name in prompt: {prompt}")
#     return match.group(1)

def get_max_tokens_per_second(model: str, tier: str) -> Optional[int]:
    """ rates limites as of August 2025 """
    if tier == "free":
        if model == "gemini-2.5-flash":
            tokens_per_minute = 250000
            return tokens_per_minute / 60
        elif model == "gemini-2.5-pro":
            tokens_per_minute = 250000
            return tokens_per_minute / 60
        else:
            return None
    elif tier == "tier1":
        if model == "gemini-2.5-flash":
            tokens_per_minute = 1_000_000
            return tokens_per_minute / 60
        elif model == "gemini-2.5-pro":
            tokens_per_minute = 2_000_000
            return tokens_per_minute / 60
        else:
            return None
    elif tier == "tier2":
        if model == "gemini-2.5-flash":
            tokens_per_minute = 3_000_000
            return tokens_per_minute / 60
        elif model == "gemini-2.5-pro":
            tokens_per_minute = 5_000_000
            return tokens_per_minute / 60
        else:
            return None
    elif tier == "tier3":
        if model == "gemini-2.5-flash":
            tokens_per_minute = 8_000_000
            return tokens_per_minute / 60
        elif model == "gemini-2.5-pro":
            tokens_per_minute = 8_000_000
            return tokens_per_minute / 60
        else:
            return None
    else:
        return None
    
def get_max_requests_per_second(model: str, tier: str) -> Optional[int]:
    """ rates limites as of August 2025 """
    if tier == "free":
        if model == "gemini-2.5-flash":
            requests_per_minute = 10
            return requests_per_minute / 60
        elif model == "gemini-2.5-pro":
            requests_per_minute = 5
            return requests_per_minute / 60
        else:
            return None
    elif tier == "tier1":
        if model == "gemini-2.5-flash":
            requests_per_minute = 1000
            return requests_per_minute / 60
        elif model == "gemini-2.5-pro":
            requests_per_minute = 150
            return requests_per_minute / 60
        else:
            return None
    elif tier == "tier2":
        if model == "gemini-2.5-flash":
            requests_per_minute = 2000
            return requests_per_minute / 60
        elif model == "gemini-2.5-pro":
            requests_per_minute = 1000
            return requests_per_minute / 60
        else:
            return None
    elif tier == "tier3":
        if model == "gemini-2.5-flash":
            requests_per_minute = 10000
            return requests_per_minute / 60
        elif model == "gemini-2.5-pro":
            requests_per_minute = 2000
            return requests_per_minute / 60
        else:
            return None
    else:
        return None

def get_max_requests_per_day(model: str, tier: str) -> Optional[int]:
    """ rates limites as of August 2025 (Tier 2)"""
    if tier == "free":
        if model == "gemini-2.5-flash":
            return 250
        elif model == "gemini-2.5-pro":
            return 100
        else:
            return None
    elif tier == "tier1":
        if model == "gemini-2.5-flash":
            return 10000
        elif model == "gemini-2.5-pro":
            return 10000
        else:
            return None
    elif tier == "tier2":
        if model == "gemini-2.5-flash":
            return 100_000
        elif model == "gemini-2.5-pro":
            return 50_000
        else:
            return None
    elif tier == "tier3":
        if model == "gemini-2.5-flash":
            return 200_000
        elif model == "gemini-2.5-pro":
            return 100_000
        else:
            return None
    else: 
        return None

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
        raise ValueError("Could not find Start/End markers in model output")

    # Extract code block
    code = output[start_idx:end_idx]

    # Drop the start marker line itself
    code = code.split("\n", 1)[1]

    return code.rstrip()


# def postprocess(prompt: str, output: str) -> str:
#     """ Postprocess the output. """
#     # remove leading ```, ```cpp, and trailing ```
#     output = output.strip().removeprefix("```cpp").removeprefix("```").removesuffix("```")

#     # remove prompt if it included it
#     if output.startswith(prompt):
#         output = output[len(prompt):]

#     return output

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

    api_key = args.api_key or get_env_var("GOOGLE_API_KEY")

    client = genai.Client(api_key=api_key)  # Gemini Developer API via API key

    gen_config = types.GenerateContentConfig(
        candidate_count=1,
        max_output_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            # Optional (supported per Gemini API docs):
            # {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
        ]
    )

    # # get the keys
    # api_key = args.api_key or get_env_var("GOOGLE_API_KEY")
    # genai.configure(api_key=api_key)

    # # create the client
    # config = genai.types.GenerationConfig(
    #     candidate_count=3,
    #     max_output_tokens=args.max_new_tokens,
    #     temperature=args.temperature,
    #     top_p=args.top_p
    # )
    # safety_settings = [
    #     {
    #         "category": "HARM_CATEGORY_DANGEROUS",
    #         "threshold": "BLOCK_NONE",
    #     },
    #     {
    #         "category": "HARM_CATEGORY_HARASSMENT",
    #         "threshold": "BLOCK_NONE",
    #     },
    #     {
    #         "category": "HARM_CATEGORY_HATE_SPEECH",
    #         "threshold": "BLOCK_NONE",
    #     },
    #     {
    #         "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    #         "threshold": "BLOCK_NONE",
    #     },
    #     {
    #         "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    #         "threshold": "BLOCK_NONE",
    #     },
    # ]
    # model = genai.GenerativeModel(args.model, generation_config=config, safety_settings=safety_settings)

    # generation metadata
    MAX_TOKENS_PER_SECOND = args.max_tokens_per_second or get_max_tokens_per_second(args.model, args.tier)
    MAX_REQUESTS_PER_SECOND = args.max_requests_per_second or get_max_requests_per_second(args.model, args.tier)
    MAX_REQUESTS = args.max_requests or get_max_requests_per_day(args.model, args.tier)

    # generate outputs
    request_counter = 0
    request_rate_counter = 0
    request_timer = time.time()
    with alive_bar(len(prompts), title="Generating outputs", dual_line=True) as bar:
        for prompt in prompts:
            # see if we can skip this
            if not args.overwrite and "outputs" in prompt:
                bar(skipped=True)
                continue

            # get the prompt
            original_prompt = prompt["prompt"]
            # function_name = get_function_name(original_prompt, prompt["parallelism_model"])
            # prompt_text = PROMPT_TEMPLATE.format(prompt=original_prompt, function_name=function_name)

            system_prompt = SYSTEM_TEMPLATE.format(prompt["name"], prompt["name"])
            # generate the outputs
            if args.dry:
                print("system", system_prompt)
                # print("prompt", prompt_text)
                continue

            # set metadata
            prompt["temperature"] = args.temperature
            prompt["top_p"] = args.top_p
            prompt["do_sample"] = True
            prompt["max_new_tokens"] = args.max_new_tokens
            prompt["generate_model"] = args.model

            # generate the outputs
            # completions = []
            outputs = []
            while len(outputs) < args.num_samples_per_prompt:
                input = system_prompt + "\n" + original_prompt
                # print("input: ", input)
                t0 = time.time()
                resp = client.models.generate_content(
                    model=args.model,
                    contents=input,
                    config=gen_config,
                )
                print("API call seconds: ", time.time() - t0)

                for cand in (resp.candidates or []):
                    # finish reasons are strings/enums depending on version; be tolerant:
                    finish = getattr(cand, "finish_reason", None)
                    if str(finish).upper().endswith("STOP"):
                        # Most robust way: drill into content.parts
                        text = ""
                        content = getattr(cand, "content", None)
                        if content and getattr(content, "parts", None):
                            text = "".join(getattr(p, "text", "") for p in content.parts)
                        else:
                            # Fallbacks some versions provide:
                            text = getattr(cand, "text", "") or getattr(resp, "text", "")

                        outputs.append(text)
                        bar.text(f"~> Received output {len(outputs)} of {args.num_samples_per_prompt}.")
                    else:
                        print(f"Got a completion with finish_reason={finish}.")
                        time.sleep(5)

                # completion = model.generate_content(input)
                # for j, candidate in enumerate(completion.candidates):
                #     if candidate.finish_reason == 1: # STOP
                #         outputs.append(candidate.content.parts[0].text)
                #         # print("text: ", completion.candidates[0].content.parts[0].text)
                #         bar.text(f"~> Received output {len(outputs)} of {args.num_samples_per_prompt}.")
                #     else:
                #         print(f"Got a completion with finish_reason={candidate.finish_reason}.")
                #         # print("text: ", completion.candidates[0].content.parts[0].text)
                #         time.sleep(5)

            # outputs = [c.text for c in completions]
            outputs = [postprocess(original_prompt, o) for o in outputs]
            prompt["outputs"] = outputs
            bar()

            # update counters
            request_counter += 1
            request_rate_counter += 1

            # check if we should stop
            if MAX_REQUESTS is not None and request_counter >= MAX_REQUESTS:
                print(f"Stopping after {request_counter} requests.")
                break
        
            # check if we should sleep
            requests_per_second = request_rate_counter / (time.time() - request_timer)
            if MAX_REQUESTS_PER_SECOND is not None and requests_per_second > (MAX_REQUESTS_PER_SECOND*0.95):
                sleep_time = 5
                print(f"Sleeping for {sleep_time} seconds.")
                time.sleep(sleep_time)
                request_timer = time.time()
                request_rate_counter = 0

            # write intermediate outputs
            with open(args.output, 'w') as output_json:
                json.dump(prompts, output_json, indent=2)

    # summary stats
    print(f"Submitted {request_counter} requests.")

    # write outputs
    with open(args.output, 'w') as output_json:
        json.dump(prompts, output_json, indent=2)
    client.close()
    

if __name__ == "__main__":
    main()