#!/usr/bin/env python3
"""
vLLM batch inference with system prompt support.

Reads JSONL input with fields: idx, prompt, system_prompt
Runs vLLM inference and writes results back to JSONL.
"""

import json
import argparse
import os
from pathlib import Path

from vllm import LLM, SamplingParams


def run_inference(
    model_path: str,
    prompt_path: str,
    output_path: str,
    gpu: str = "0",
    batch_size: int = 100,
    max_model_len: int = 20000,
    max_tokens: int = 3000,
    temperature: float = 0.0,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    gpu_count = len(gpu.split(","))

    print(f"Loading model from: {model_path}")
    print(f"Using {gpu_count} GPU(s): {gpu}")

    llm = LLM(
        model=model_path,
        tensor_parallel_size=gpu_count,
        max_model_len=max_model_len,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["<tool_response>", "</tool_response>"],
    )

    # Load prompts
    print(f"Loading prompts from: {prompt_path}")
    data = []
    with open(prompt_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"  Loaded {len(data)} prompts")

    # Format prompts using chat template
    formatted_prompts = []
    valid_indices = []

    for i, item in enumerate(data):
        system_prompt = item.get("system_prompt", "")
        user_prompt = item.get("prompt", "")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        try:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            token_count = len(tokenizer.encode(formatted))
            max_prompt_len = max_model_len - max_tokens

            if token_count > max_prompt_len:
                print(f"  Skipping idx={item.get('idx', i)}: {token_count} tokens > {max_prompt_len} limit")
                data[i]["raw_response"] = ""
                data[i]["skipped"] = True
                continue

            formatted_prompts.append(formatted)
            valid_indices.append(i)
        except Exception as e:
            print(f"  Error formatting idx={item.get('idx', i)}: {e}")
            data[i]["raw_response"] = ""
            data[i]["skipped"] = True

    print(f"  {len(formatted_prompts)} prompts ready for inference")

    # Run inference in batches
    all_outputs = []
    for batch_start in range(0, len(formatted_prompts), batch_size):
        batch_end = min(batch_start + batch_size, len(formatted_prompts))
        batch = formatted_prompts[batch_start:batch_end]
        print(f"  Processing batch {batch_start}-{batch_end} / {len(formatted_prompts)}")
        outputs = llm.generate(batch, sampling_params)
        all_outputs.extend(outputs)

    # Map outputs back to data
    for output_idx, data_idx in enumerate(valid_indices):
        response = all_outputs[output_idx].outputs[0].text
        data[data_idx]["raw_response"] = response

    # Write results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  Saved {len(data)} results to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="vLLM batch inference with system prompt")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--prompt_path", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output_path", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--gpu", type=str, default="0", help="GPU IDs (e.g., '0,1,2,3')")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    parser.add_argument("--max_model_len", type=int, default=20000, help="Max model context length")
    parser.add_argument("--max_tokens", type=int, default=3000, help="Max generation tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    args = parser.parse_args()

    run_inference(
        model_path=args.model_path,
        prompt_path=args.prompt_path,
        output_path=args.output_path,
        gpu=args.gpu,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
