# LLM Quantization Playbook

Hands-on quantization experiments on **Llama 3.2 1B** — what worked, what didn't, and the memory/accuracy trade-offs measured rather than assumed.

## Published models

| Model | Hugging Face |
|---|---|
| LoRA fine-tuned | [`Mubinmodi007/Llama-3.2-1B-finetuned`](https://huggingface.co/Mubinmodi007/Llama-3.2-1B-finetuned) |
| AWQ 4-bit quantized | [`Mubinmodi007/Llama-3.2-1B-AWQ-4bit`](https://huggingface.co/Mubinmodi007/Llama-3.2-1B-AWQ-4bit) |

## Why quantization

Quantization reduces the numerical precision of model weights, which buys three things: models that fit on limited hardware, faster inference, and lower serving cost — ideally without meaningfully degrading output quality. Measuring that last "ideally" is the point of this repo.

## Techniques explored

| Technique | Status |
|---|---|
| **AWQ** (Activation-aware Weight Quantization, 4-bit) | ✅ Implemented — post-training, memory stats captured |
| **LoRA** (low-rank adapters, quantization-aware training) | ✅ Implemented — fine-tuned on PIQA |
| **Unsloth** | ✅ Implemented |
| **GPTQ** | ⚠️ Attempted — could not get usable output |
| **Flash Attention** | ⚠️ Attempted — could not get usable output |

## Notebooks

| Notebook | Contents |
|---|---|
| `LLMquant_AWQ.ipynb` | Post-training quantization with AWQ, start to finish |
| `llama3.2_QAT.ipynb` | Quantization-aware training: LoRA fine-tune on PIQA, then quantize |
| `LLM_Quant_testing_accuarcy.ipynb` | Benchmarking harness — AWQ and LoRA variants vs. base Llama, at 100 test cases and full set, plus notes on what went wrong |

## Findings

**Memory:** AWQ delivers a clear reduction in footprint (see `awq_stats.jfif` for the measured before/after).

**Accuracy — honest result:** both quantized variants scored *low* in the benchmarking harness. Based on the analysis in the testing notebook, the likely cause is the evaluation setup rather than the quantization itself — prompt formatting and answer-matching in the harness — not a genuine collapse in model capability. This is written up rather than hidden because the debugging trail is the useful part; treat the accuracy numbers here as unresolved, not as a verdict on AWQ or LoRA.

## Stack

`autoawq` · `transformers` · `peft` · `trl` · `unsloth` · `datasets` · `torch` · `pynvml` (GPU memory measurement)

## Running it

A CUDA-capable GPU is required for the quantization and training notebooks.

```bash
pip install torch transformers datasets peft trl autoawq unsloth pynvml huggingface_hub
jupyter notebook
```
