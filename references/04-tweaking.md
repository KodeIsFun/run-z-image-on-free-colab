# 04 — Tweaking: what helps, what does not (all measured)

Hardware: free Colab T4 (15 360 MB VRAM, ~12.7 GB RAM), Colab image
`torch 2.11.0+cu128`, Python 3.13. Same seed (42) and prompts across rows;
speed ratios come from same-session pairs only. Numbers trace to the GPUTests
lab repo, `projects/2026-W40-zimage-turbo-t4/` (`results/timings_v7_zimage.json`
+ `results/timings_v7b_warm.json`), measured 2026-09-28 and re-verified on a
second VM (Qwen anchor rows identical to the second; Z rows ±10 %).

## The measured head-to-head table

| Row (sign canary, seed 42) | Z-Image bf16, 8 steps | Qwen r256, 6 steps |
|---|---|---|
| warm repeat, 768² (encode cached) | **18.0 s** (1.8 s/step) | 27.0 s (3.8 s/step) |
| first render with a NEW prompt, 768² | 42.0 s | 30.0 s |
| warm, 1024×1024 (Z's training size) | 39.0 s (4.0 s/step) | 42.0 s (6.2 s/step) |
| session's first image (stack load incl.) | 102.1 s | 84.1 s |
| z int8 split, 768² (incl. model swap) | 60.0 s (4.1 s/step) | — |

## Why the numbers look like this (the memory math)

Z-Image's bf16 unet is 12.31 GB; its text encoder (a Qwen3-4B) is 8.04 GB.
Together 20.7 GB on a 15.6 GB card → ComfyUI keeps the unet resident and
unloads the encoder, reloading it when a new prompt needs encoding. Hence:

- **repeats of one prompt**: encoder stays cached → pure 8-step denoise ≈ 18 s
  — the fastest thing measured on a free T4 (1.5× Qwen turbo's 27.0 s);
- **a fresh prompt**: + encoder reload ≈ 42 s — slower than Qwen's 30.0 s;
- **1024²**: 39.0 s vs Qwen's 42.0 s — Z wins even against the reload tax.

Qwen's GGUF stack (4.2 GB unet + 9.35 GB int8 encoder) fits resident together
and never reloads — steady across prompt changes.

**Workload rule: batches of variations on one prompt → Z-Image; mixed
prompts / interactive → Qwen.**

## The Z-Image recipe (official template, load-bearing parts)

1. **CLIPLoader `type: lumina2`** for the Qwen3-4B text encoder. There is no
   z_image type in stock ComfyUI's loader enum — the official template uses
   lumina2; anything else fails to encode.
2. **`ModelSamplingAuraFlow shift=3`** between the unet and the sampler. The
   official template ships it; dropping it changes the sampling geometry.
3. **8 steps, cfg 1.0, `res_multistep`/`simple`, ConditioningZeroOut
   negative.** The turbo checkpoint is distilled for exactly 8 NFEs with CFG
   off (guidance_scale 0.0 per Tongyi's card).
4. **fp16 flag pair on T4**: `--force-fp16 --disable-comfy-compiler` (the
   card has no bf16; the compiler crashes the fp16 forward without the
   second flag).
5. **Not needed for t2i**: the `z_image_turbo_distill_patch_lora` in the
   Comfy-Org repo targets the base/Omni path. The turbo split is
   self-contained.

## Variants

| Split | Size | Verdict on T4 |
|---|---|---|
| `z_image_turbo_bf16` (default) | 12.31 GB | fastest (1.8 s/step); triggers the encoder reload dance |
| `z_image_turbo_int8_convrot` | 6.20 GB | fits easiest, **slower per step on T4**: 4.1 s/step (60.0 s warm incl. swap) |
| `z_image_turbo_nvfp4` | — | skip: nvfp4 wants Blackwell-era hardware |

Untested here (labeled honestly): the fp8 text-encoder split
(`qwen_3_4b_fp8_mixed`, 5.63 GB — would shrink the reload), 2K upscale
workflows, image editing via `TextEncodeZImageOmni`.

## Quality

Text rendering: "FREE GPU LAB" neon sign letter-perfect on Z bf16 at 768²
and 1024², on the int8 split, and on Qwen — a tie on this canary. Z-Image's
card additionally claims bilingual (English/Chinese) text rendering.
