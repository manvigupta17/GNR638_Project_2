"""
GNR638 — MCQ Visual Reasoning: Inference Script

Usage:
    python inference.py --test_dir <absolute_path_to_test_dir>

The test directory must contain:
    images/      - folder of PNG MCQ images
    test.csv     - CSV with column: image_name

Output:
    submission.csv written to the current working directory (NOT test_dir).

Citations:
    [1] Qwen2.5-VL: Bai et al., arXiv:2308.12966
        https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct
    [2] QLoRA / BitsAndBytes: Dettmers et al., arXiv:2305.14314
        https://github.com/TimDettmers/bitsandbytes
    [3] HuggingFace Transformers: Wolf et al., 2020
        https://github.com/huggingface/transformers
    [4] Self-Consistency Prompting: Wang et al., ICLR 2023, arXiv:2203.11171
"""

import os
import re
import argparse
from collections import Counter
from typing import Optional

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)

# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="GNR638 MCQ Visual Reasoning Inference")
    parser.add_argument(
        "--test_dir",
        type=str,
        required=True,
        help="Absolute path to the test directory containing test.csv and images/",
    )
    return parser.parse_args()


# ── GPU detection and model selection ────────────────────────────────────────

def select_model():
    """Select model size based on available GPU VRAM.

    Returns a tuple of (model_name, local_dir, num_passes, max_tokens).
    NUM_PASSES is set to 1 for the 72B model to stay within the 1-hour runtime
    limit (50 questions x ~50s each = ~42 min on L40s with 4-bit quantization).
    """
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name} | VRAM: {vram_gb:.1f} GB")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if vram_gb >= 35:
        model_name  = "Qwen/Qwen2.5-VL-72B-Instruct"
        local_dir   = os.path.join(script_dir, "Qwen2.5-VL-72B-Instruct")
        num_passes  = 1
        max_tokens  = 512
        print("-> 72B model selected (grading mode)")
    elif vram_gb >= 7:
        model_name  = "Qwen/Qwen2.5-VL-7B-Instruct"
        local_dir   = os.path.join(script_dir, "Qwen2.5-VL-7B-Instruct")
        num_passes  = 3
        max_tokens  = 512
        print("-> 7B model selected (testing mode)")
    else:
        model_name  = "Qwen/Qwen2.5-VL-3B-Instruct"
        local_dir   = os.path.join(script_dir, "Qwen2.5-VL-3B-Instruct")
        num_passes  = 3
        max_tokens  = 512
        print("-> 3B model selected (fallback mode)")

    print(f"Weights expected at: {local_dir}")

    # Hard check — internet is blocked at grading time; no download attempted.
    if not os.path.exists(local_dir):
        raise FileNotFoundError(
            f"Model weights not found at: {local_dir}\n"
            f"Please place the downloaded weights folder inside the project directory."
        )

    print(f"Weights found: {local_dir}")
    return model_name, local_dir, num_passes, max_tokens


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(local_dir: str, model_name: str):
    """Load Qwen2.5-VL with 4-bit NF4 quantization.

    NF4 quantization (Dettmers et al., NeurIPS 2023) reduces VRAM usage by ~60%
    while preserving reasoning quality. SDPA attention is used for memory efficiency.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,  # second quantization further reduces footprint
        bnb_4bit_quant_type="nf4",       # NF4 suits normally distributed weight tensors
    )

    processor = AutoProcessor.from_pretrained(local_dir)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        local_dir,
        quantization_config=bnb_config,
        device_map="auto",           # distributes layers across available GPUs
        torch_dtype=torch.float16,
        attn_implementation="sdpa",  # scaled dot-product attention for memory efficiency
    )

    print(f"Model loaded: {model_name}")
    return model, processor


# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert in deep learning, computer vision, and PyTorch.

You will be shown an image with a multiple-choice question (MCQ).
Four options are labelled A (1), B (2), C (3), D (4).

CRITICAL PyTorch nn.Sequential RULES:
- nn.Sequential processes layers LEFT-TO-RIGHT in order.
- Each layer's output becomes the next layer's input.
- Example: nn.Sequential(Linear, ReLU, Linear, ReLU, Linear) means:
    input -> Linear -> ReLU -> Linear -> ReLU -> Linear -> output
- This correctly implements: h1=ReLU(W1*x+b1), h2=ReLU(W2*h1+b2), y=W3*h2+b3
- ReLU placed AFTER Linear in Sequential is CORRECT (it activates that layer's output).
- ReLU placed BEFORE the first Linear is WRONG (nothing to activate yet).
- A missing ReLU between two Linears is WRONG (no activation function applied).

SOLVING STRATEGY:
1. READ the question and ALL four options carefully from the image.
2. For CALCULATION questions: apply formulas step by step, show all working.
3. For CODE/CONCEPT questions: evaluate each option (A,B,C,D) one by one.
   State whether each option is CORRECT or INCORRECT and WHY.
4. ELIMINATE wrong options. Pick the BEST remaining option.

RULE: You MUST answer 1, 2, 3, or 4. Only answer 5 if the image is unreadable.
An educated guess is ALWAYS better than skipping.

End your response with exactly this line:
FINAL ANSWER: X
where X is 1, 2, 3, or 4."""


# ── Answer extraction ─────────────────────────────────────────────────────────

def _extract_answer(text: str) -> Optional[str]:
    """Parse the model's raw output and extract a numeric answer (1-5).

    Uses a three-layer fallback strategy:
      1. Explicit 'FINAL ANSWER: X' tag (highest confidence).
      2. Natural-language patterns such as 'answer is B' or 'option 3'.
      3. Last standalone digit 1-4 anywhere in the response (last-resort heuristic).

    Returns None if no valid answer is found.
    """
    # Layer 1: Preferred explicit tag enforced by the system prompt.
    m = re.search(r'FINAL\s+ANSWER\s*:\s*([1-5])', text, re.IGNORECASE)
    if m:
        return m.group(1)

    # Layer 2: Natural-language patterns; maps letters A-D to digits 1-4.
    m = re.search(
        r'(?:answer\s+is|correct\s*(?:is|option)|option)\s*[:\s]+(?:\(?([A-D])\)?|\(?([1-4])\)?)',
        text, re.IGNORECASE,
    )
    if m:
        letter = m.group(1)
        digit  = m.group(2)
        if letter:
            return str(ord(letter.upper()) - ord('A') + 1)
        if digit:
            return digit

    # Layer 3: Scan from end of response for most recent answer digit.
    m = re.search(r'\b([1-4])\b', text[::-1])
    if m:
        return m.group(1)

    return None


# ── Inference ─────────────────────────────────────────────────────────────────

def _single_pass(
    image_path: str,
    model,
    processor,
    max_tokens: int,
    temperature: float = 0.0,
) -> str:
    """Run one model inference pass on the given image.

    Args:
        image_path:  Path to the MCQ image file.
        model:       Loaded Qwen2.5-VL model.
        processor:   Corresponding processor.
        max_tokens:  Maximum tokens to generate.
        temperature: 0.0 = greedy; > 0.0 = stochastic (self-consistency voting).

    Returns:
        A string digit '1'-'4', or '5' if unreadable or no answer found.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[WARN] Could not open {image_path}: {e}")
        return "5"

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text",  "text": SYSTEM_PROMPT},
        ],
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # High pixel budget preserves code readability in images with source snippets.
    inputs = processor(
        text=[text], images=[image], return_tensors="pt",
        min_pixels=1_000_000, max_pixels=4_000_000,
    ).to(next(model.parameters()).device)

    gen_kwargs = dict(
        max_new_tokens=max_tokens,
        pad_token_id=processor.tokenizer.eos_token_id,
    )

    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        gen_kwargs.update(do_sample=False)

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    # Slice off prompt tokens to isolate the generated response only.
    trimmed     = out[:, inputs["input_ids"].shape[1]:]
    answer_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    ans = _extract_answer(answer_text)
    return ans if ans else "5"


def predict_answer(
    image_path: str,
    model,
    processor,
    num_passes: int,
    max_tokens: int,
    verbose: bool = False,
) -> str:
    """Predict the MCQ answer using greedy decoding + optional self-consistency voting.

    One greedy pass is always run first. When num_passes > 1, additional stochastic
    passes are run and the plurality vote is returned. If the majority voted '5'
    (skip) but at least one pass returned a valid answer, the skip is overridden
    to avoid unnecessary zero-score questions.

    Args:
        image_path: Path to the MCQ image file.
        model:      Loaded model.
        processor:  Corresponding processor.
        num_passes: Total number of inference passes.
        max_tokens: Maximum tokens to generate per pass.
        verbose:    If True, prints per-pass details.

    Returns:
        A string digit '1'-'5'.
    """
    if verbose:
        print(f"\n{'='*50}\n{os.path.basename(image_path)}\n{'='*50}")

    greedy = _single_pass(image_path, model, processor, max_tokens, temperature=0.0)
    if verbose:
        print(f"  Greedy -> {greedy}")

    if num_passes == 1:
        return greedy

    votes = [greedy]
    for i in range(num_passes - 1):
        ans = _single_pass(image_path, model, processor, max_tokens, temperature=0.7)
        votes.append(ans)
        if verbose:
            print(f"  Pass {i+2} -> {ans}")

    vote_counts = Counter(votes)
    best, count = vote_counts.most_common(1)[0]

    if verbose:
        print(f"  Votes: {dict(vote_counts)} -> {best}")

    # Override skip if any substantive answer exists in the minority.
    if best == "5" and count < num_passes:
        non5 = Counter(v for v in votes if v != "5")
        if non5:
            best = non5.most_common(1)[0][0]
            if verbose:
                print(f"  Override skip -> {best}")

    return best


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    test_dir = args.test_dir

    # Validate test directory structure.
    test_csv   = os.path.join(test_dir, "test.csv")
    images_dir = os.path.join(test_dir, "images")

    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"test.csv not found at: {test_csv}")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"images/ folder not found at: {images_dir}")

    print(f"test.csv  : {test_csv}")
    print(f"images/   : {images_dir}")

    # Load model.
    model_name, local_dir, num_passes, max_tokens = select_model()
    model, processor = load_model(local_dir, model_name)

    # Run predictions.
    test_df     = pd.read_csv(test_csv)
    image_names = test_df["image_name"].tolist()
    print(f"\nQuestions: {len(image_names)} | Model: {model_name} | Passes: {num_passes}")

    results = []
    for img_name in tqdm(image_names, desc="Predicting"):
        path = os.path.join(images_dir, f"{img_name}.png")
        if not os.path.exists(path):
            # Mark missing images as unanswered (no penalty) rather than guessing.
            results.append({"id": img_name, "image_name": img_name, "option": "5"})
            continue
        ans = predict_answer(path, model, processor, num_passes, max_tokens)
        results.append({"id": img_name, "image_name": img_name, "option": ans})

    # Write submission.csv to current working directory (not test_dir).
    submission     = pd.DataFrame(results)
    submission_path = os.path.join(os.getcwd(), "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"\nsubmission.csv saved to {submission_path} ({len(submission)} rows)")
    print(submission.to_string(index=False))


if __name__ == "__main__":
    main()
