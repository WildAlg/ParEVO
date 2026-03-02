#!/usr/bin/env python3
"""
Add missing fields from the original prompts JSON to the vLLM output JSON.

This script compares the output JSON with the original prompts JSON and adds
any missing fields (like 'alg_name') that should have been copied over.

Usage:
    python add_missing_fields.py --prompts <prompts.json> --output <output.json> [--dry-run]
    
Example:
    python add_missing_fields.py --prompts prompts.json --output vllm_output.json
    python add_missing_fields.py --prompts prompts.json --output vllm_output.json --dry-run
"""

import argparse
import json
import sys


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", required=True, help="Path to the original prompts JSON file")
    parser.add_argument("--output", required=True, help="Path to the output JSON file to update")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without modifying files")
    parser.add_argument("--out-file", help="Path to write updated output (default: overwrite --output)")
    return parser.parse_args()


def find_matching_prompt(output_entry, prompts):
    """
    Find the matching prompt entry based on identifying fields.
    
    Match criteria (same as used in the scripts):
    - name
    - parallelism_model  
    - prompt
    """
    for prompt in prompts:
        if (prompt.get("alg_dir") == output_entry.get("name")):
            return prompt
    return None


def get_missing_fields(output_entry, prompt_entry):
    """
    Get fields that are in prompt_entry but not in output_entry.
    
    Excludes fields that are generation-specific (added during generation).
    """
    # Fields that are added during generation, not from original prompt
    generation_fields = {
        "temperature", "top_p", "do_sample", "max_new_tokens", 
        "prompted", "outputs", "raw_outputs", "generate_model"
    }
    
    missing = {}
    for key, value in prompt_entry.items():
        if key not in output_entry and key not in generation_fields:
            missing[key] = value
    
    return missing


def main():
    args = get_args()
    
    # Load prompts
    print(f"Loading prompts from: {args.prompts}")
    with open(args.prompts, 'r') as f:
        prompts = json.load(f)
    print(f"  Loaded {len(prompts)} prompts")
    
    # Load outputs
    print(f"Loading outputs from: {args.output}")
    with open(args.output, 'r') as f:
        outputs = json.load(f)
    print(f"  Loaded {len(outputs)} outputs")
    
    # Track statistics
    updated_count = 0
    not_found_count = 0
    already_complete_count = 0
    all_missing_fields = set()
    
    # Process each output entry
    print("\nProcessing outputs...")
    for i, output_entry in enumerate(outputs):
        output_entry["generate_model"] = "gemini-2.5-pro-finetuned"
        # Find matching prompt
        prompt_entry = find_matching_prompt(output_entry, prompts)
        
        if prompt_entry is None:
            not_found_count += 1
            print(f"  WARNING: No matching prompt found for output {i} "
                  f"(name={output_entry.get('name', 'N/A')})")
            continue
        
        # Get missing fields
        missing = get_missing_fields(output_entry, prompt_entry)
        
        if not missing:
            already_complete_count += 1
            continue
        
        # Track all missing field names
        all_missing_fields.update(missing.keys())
        
        # Add missing fields
        if args.dry_run:
            print(f"  Would add to output {i} (name={output_entry.get('name', 'N/A')}): "
                  f"{list(missing.keys())}")
        else:
            output_entry.update(missing)
        
        updated_count += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Total outputs: {len(outputs)}")
    print(f"  Updated: {updated_count}")
    print(f"  Already complete: {already_complete_count}")
    print(f"  No matching prompt found: {not_found_count}")
    
    if all_missing_fields:
        print(f"\n  Missing fields added: {sorted(all_missing_fields)}")
    
    # Write output
    if not args.dry_run and updated_count > 0:
        out_path = args.out_file or args.output
        print(f"\nWriting updated outputs to: {out_path}")
        with open(out_path, 'w') as f:
            json.dump(outputs, f, indent=4)
        print("Done!")
    elif args.dry_run:
        print("\n[DRY RUN] No files were modified.")
    else:
        print("\nNo updates needed.")


if __name__ == "__main__":
    main()