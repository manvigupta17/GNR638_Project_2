# GNR638 — MCQ Visual Reasoning Pipeline

Automatically answers deep-learning multiple-choice questions presented as PNG images using a quantized vision-language model and self-consistency decoding.

---

## Requirements

| Item | Specification |
|------|--------------|
| OS | Linux (Ubuntu recommended) |
| Python | 3.11 (via conda) |
| CUDA | 12.6 |
| GPU | L40s 48GB VRAM (grading) / T4 16GB (testing) |
| Disk | ~40 GB free for model weights |
| Conda environment | `gnr_project_env` |

---

## Project Structure

```
.
├── inference.py                   # Main inference script (accepts --test_dir)
├── setup.bash                     # One-time setup: env creation + weight download
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── Qwen2.5-VL-72B-Instruct/      # Model weights (downloaded by setup.bash)
    └── ...
```

**Test directory structure** (provided by TA at grading time):
```
<test_dir>/
├── images/
│   ├── image_1.png
│   ├── image_2.png
│   └── ...
├── test.csv
└── sample_submission.csv
```

---

## Setup (Run Once — Requires Internet)

```bash
bash setup.bash
```

This will:
1. Create the conda environment `gnr_project_env` with Python 3.11
2. Clone the project repository from GitHub
3. Install all Python dependencies from `requirements.txt`
4. Download `Qwen2.5-VL-72B-Instruct` weights (~40 GB) from HuggingFace

> **Note:** Internet is required only during setup. Inference runs fully offline.

---

## Running Inference (No Internet Required)

```bash
conda activate gnr_project_env
python inference.py --test_dir <absolute_path_to_test_dir>
```

**Output:** `submission.csv` is written to the **current working directory** (not `test_dir`).

### Grading commands (exactly as specified):

```bash
cd ./your_directory
bash setup.bash
conda activate gnr_project_env
python inference.py --test_dir <absolute_path_to_test_dir>
python <grading_script> --submission_file submission.csv
conda remove --name gnr_project_env --all -y
```

---

## How It Works

### Model selection

VRAM is queried at startup. The pipeline automatically selects:

| VRAM | Model | Passes | Est. runtime (50 q) |
|------|-------|--------|---------------------|
| >= 35 GB (L40s) | Qwen2.5-VL-72B-Instruct | 1 | ~40-50 min |
| >= 7 GB (T4) | Qwen2.5-VL-7B-Instruct | 3 | ~15-25 min |
| < 7 GB | Qwen2.5-VL-3B-Instruct | 3 | ~10-15 min |

All comfortably within the 1-hour runtime limit.

### Quantization

BitsAndBytes NF4 4-bit quantization (Dettmers et al., NeurIPS 2023) reduces VRAM from ~144 GB (bf16) to ~40-45 GB, making the 72B model runnable on a single L40s.

### Prompt design

A domain-aware system prompt instructs the model to reason step-by-step and emit a structured FINAL ANSWER: X tag. Explicit rules for nn.Sequential layer ordering are embedded to handle a common MCQ topic where default model reasoning is unreliable.

### Answer extraction

Three-layer regex fallback:
1. FINAL ANSWER: X tag (highest confidence)
2. Natural-language patterns (answer is B, option 3)
3. Last standalone digit 1-4 in the response (last-resort heuristic)

### Self-consistency voting

When num_passes > 1: one greedy pass + stochastic passes (temperature=0.7). Plurality vote is returned. If majority voted 5 (skip) but a minority returned a valid answer, the skip is overridden.

---

## Output Format

`submission.csv` (written to current working directory):

```
id,image_name,option
image_1,image_1,3
image_2,image_2,1
```

| Value | Meaning |
|-------|---------|
| 1 | Option A |
| 2 | Option B |
| 3 | Option C |
| 4 | Option D |
| 5 | Unanswered (image missing or unreadable) |

---

## Scoring

```
score = (correct) - 0.25 x (incorrect) - 1 x (hallucinated)
```

Values 1-4 are attempted answers. Value 5 is unanswered (0 points, no penalty). Any other value is hallucinated (-1 point). The pipeline only ever outputs 1-5.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| FileNotFoundError for weights | Run setup.bash first with internet; ensure Qwen2.5-VL-72B-Instruct/ is in the project directory |
| FileNotFoundError for test.csv | Check --test_dir points to directory containing test.csv and images/ |
| All predictions are 5 | Verify filenames in images/ match image_name column in test.csv |
| CUDA out of memory | Should not occur on L40s with 4-bit quant; model will auto-select smaller tier |
| bitsandbytes import error | Ensure CUDA 12.6 drivers are installed and GPU runtime is active |

---

## Citations

1. Qwen2.5-VL: Bai et al., arXiv:2308.12966 — https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct
2. QLoRA / BitsAndBytes: Dettmers et al., arXiv:2305.14314 — https://github.com/TimDettmers/bitsandbytes
3. HuggingFace Transformers: Wolf et al., 2020 — https://github.com/huggingface/transformers
4. Self-Consistency Prompting: Wang et al., ICLR 2023, arXiv:2203.11171
