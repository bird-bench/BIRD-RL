#!/usr/bin/env python3
"""
Call LLM API with system and user prompts for SFT data generation.

Reads JSONL input with fields: instance_id, system_prompt, prompt
Calls the specified LLM API and writes responses back to JSONL.

Supports AWS Bedrock API with multiple model providers.

Input JSONL format:
{
    "instance_id": "...",
    "prompt": "user prompt content",
    "system_prompt": "system prompt content"
}

Output JSONL format (appends raw_response):
{
    "instance_id": "...",
    "prompt": "...",
    "system_prompt": "...",
    "raw_response": "LLM response"
}

Usage:
    python -m bird_rl.data.call_api \
        --input_path <prompts.jsonl> \
        --output_path <responses.jsonl> \
        --model_name deepseek-r1 \
        --api_key <your_api_key>
"""

import argparse
import json
import os
import time
import threading
import traceback
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


# Model name to Bedrock model ID mapping
MODEL_MAPPING = {
    "claude-opus-4": "us.anthropic.claude-opus-4-20250514-v1:0",
    "claude-sonnet-4": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "claude-3-7-sonnet": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "claude-3-5-haiku": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "deepseek-r1": "us.deepseek.r1-v1:0",
    "deepseek-v3": "deepseek.v3-v1:0",
    "llama-3-1-70b": "meta.llama3-1-70b-instruct-v1:0",
    "mistral-large-2407": "mistral.mistral-large-2407-v1:0",
}


def bedrock_api_request(
    system_prompt,
    user_prompt,
    model_id,
    api_key,
    max_tokens=4096,
    temperature=0,
    retries=10,
    initial_retry_delay=10,
):
    """Make a request to AWS Bedrock API with system and user prompts."""
    url = f"https://bedrock-runtime.us-west-2.amazonaws.com/model/{model_id}/invoke"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if "anthropic" in model_id:
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
    else:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

    retry_delay = initial_retry_delay

    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)

            if response.status_code == 200:
                result = response.json()
                if "anthropic" in model_id:
                    return result["content"][0]["text"]
                elif "choices" in result:
                    return result["choices"][0]["message"]["content"]
                elif "content" in result:
                    return result["content"][0]["text"]
                else:
                    return str(result)

            elif response.status_code in [429, 500, 503]:
                error_message = f"Retryable error {response.status_code}: {response.text}"
            else:
                error_message = f"Error {response.status_code}: {response.text}"
                return f"Error: {error_message}"

            if attempt < retries - 1:
                print(f"Attempt {attempt + 1}/{retries} failed: {error_message}")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 1.5, 100)
            else:
                return f"Error: Max retries exceeded - {error_message}"

        except requests.exceptions.Timeout:
            if attempt < retries - 1:
                print(f"Timeout on attempt {attempt + 1}/{retries}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 1.5, 100)
            else:
                return "Error: Request timeout after max retries"

        except Exception as e:
            error_message = f"{type(e).__name__}: {str(e)}"
            if attempt < retries - 1:
                print(f"Error on attempt {attempt + 1}: {error_message}. Retrying...")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 1.5, 100)
            else:
                return f"Error: {error_message}"

    return "Error: Max retries exceeded"


def worker_function(task, data_list, output_path, lock, api_key, max_tokens, temperature):
    """Process a single task and write result to output file."""
    idx, model_id = task
    item = data_list[idx]

    system_prompt = item.get('system_prompt', '')
    user_prompt = item.get('prompt', '')

    raw_response = f"Error: Processing failed for index {idx}"
    success = False

    try:
        raw_response = bedrock_api_request(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_id=model_id,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        success = not raw_response.startswith("Error:")
    except Exception as e:
        raw_response = f"Error: {type(e).__name__}: {str(e)}"

    with lock:
        with open(output_path, "a", encoding="utf-8") as f:
            row = item.copy()
            row["raw_response"] = raw_response
            row["_index"] = idx
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return idx, success


def sort_jsonl_by_index(file_path):
    """Sort JSONL file by _index field and remove _index."""
    all_data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    row = json.loads(line)
                    if "_index" in row:
                        all_data.append(row)
                except json.JSONDecodeError:
                    continue

    if not all_data:
        return

    all_data.sort(key=lambda x: x["_index"])
    with open(file_path, "w", encoding="utf-8") as f:
        for row in all_data:
            row.pop("_index", None)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Sorted {len(all_data)} records in {file_path}")


def collect_responses(
    data_list, model_name, output_path, api_key,
    num_threads=8, start_index=0, max_tokens=4096, temperature=0,
):
    """Process prompts concurrently using ThreadPoolExecutor."""
    model_id = MODEL_MAPPING.get(model_name)
    if not model_id:
        print(f"Error: Unknown model '{model_name}'. Available: {list(MODEL_MAPPING.keys())}")
        return

    print(f"Using model: {model_name} -> {model_id}")

    tasks = [(i, model_id) for i in range(start_index, len(data_list))]
    if not tasks:
        print("No tasks to process.")
        return

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if start_index == 0 and os.path.exists(output_path):
        open(output_path, "w").close()

    lock = threading.Lock()

    print(f"Processing {len(tasks)} tasks with {num_threads} threads...")
    successful = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(
                worker_function, t, data_list, output_path, lock, api_key, max_tokens, temperature
            ): t
            for t in tasks
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                _, success = future.result()
                if success:
                    successful += 1
                else:
                    failed += 1
            except Exception:
                failed += 1

    print(f"Done. Successful: {successful}, Failed: {failed}")

    if successful + failed > 0:
        sort_jsonl_by_index(output_path)


def main():
    parser = argparse.ArgumentParser(description="Call LLM API with system and user prompts")
    parser.add_argument("--input_path", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output_path", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--model_name", type=str, default="deepseek-r1",
                        help=f"Model name. Available: {list(MODEL_MAPPING.keys())}")
    parser.add_argument("--api_key", type=str, required=True, help="API key")
    parser.add_argument("--num_threads", type=int, default=8, help="Number of threads")
    parser.add_argument("--start_index", type=int, default=0, help="Starting index")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of prompts")
    parser.add_argument("--max_tokens", type=int, default=15000, help="Max tokens in response")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature")
    args = parser.parse_args()

    data_list = []
    with open(args.input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data_list.append(json.loads(line))

    print(f"Loaded {len(data_list)} instances from {args.input_path}")

    if args.limit and args.limit > 0:
        data_list = data_list[:args.limit]
        print(f"Limited to {args.limit} instances")

    collect_responses(
        data_list=data_list,
        model_name=args.model_name,
        output_path=args.output_path,
        api_key=args.api_key,
        num_threads=args.num_threads,
        start_index=args.start_index,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
