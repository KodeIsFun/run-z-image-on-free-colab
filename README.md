# run-z-image-on-free-colab

Run **Z-Image-Turbo** (Tongyi-MAI's 6B S3-DiT diffusion model, Apache-2.0) as
a real text-to-image service on a **free** Colab T4 — and call it **from any
machine** through a public API URL. The repo also ships the **Qwen-Image
turbo** GGUF engine beside it, because we raced the two on the same card and
the race is the point. Every number here was measured on an actual free T4;
every gotcha was paid for in a real failed run. Nothing is aspirational.

Written to be **executed by an agent** (any coding agent — Bash + Python is
enough) while keeping the human's manual part to a couple of clicks. If you
are an agent: read [SKILL.md](SKILL.md) first.

## What you get

| Path | What | Human effort |
|---|---|---|
| **Notebook** ([colab/run-z-image-t4.ipynb](colab/run-z-image-t4.ipynb)) | open in Colab, pick T4, Run all → installs ComfyUI, downloads both stacks, generates a sample, prints a public API URL | ~2 min of clicking + ~15 min waiting |
| **Public text-to-image API** | `clients/txt2img.py` (Python stdlib only) from any machine against the printed URL | zero — the URL is the token |
| **Two engines** | `--engine z` (default, Z-Image-Turbo) and `--engine qwen` (the comparison turbo) | a flag |
| **Tuning receipts** | the measured head-to-head and which knobs do/don't help | reading |

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KodeIsFun/run-z-image-on-free-colab/blob/main/colab/run-z-image-t4.ipynb)

## The measured head-to-head (free Colab T4, same canary + seed 42)

| Row (768×768) | Z-Image bf16, 8 steps, cfg off | Qwen turbo GGUF, 6 steps, cfg off |
|---|---|---|
| warm repeat of a prompt | **18.0 s** | 27.0 s |
| first render with a NEW prompt | 42.0 s | 30.0 s |
| warm at 1024×1024 (Z-Image's training size) | 39.0 s | 42.0 s |
| session's first image (stack load included) | 102.1 s | 84.1 s |
| memory-friendly variant (z int8 split) | 60.0 s | — |

**Workload rule: batches of variations on one prompt → Z-Image; mixed
prompts / interactive → Qwen.** The reason is pure memory arithmetic: Z's
12.3 GB unet + 8 GB text encoder cannot both stay resident on a 15.6 GB T4,
so ComfyUI reloads the encoder for every fresh prompt. Qwen's compact GGUF
stack (4.2 + 9.4 GB) fits together and never reloads. At 1024² Z-Image wins
even against the reload tax.

Text rendering is a tie on our neon-sign canary: both spell "FREE GPU LAB"
letter-perfect, at 768² and 1024², in both precision variants.

Why the Z-Image row is fast at all, and what is load-bearing in its recipe
(drop any part and it degrades):

- The **official template wiring**: `CLIPLoader` with `type: lumina2` for the
  Qwen3-4B text encoder (there is no z_image type), `ModelSamplingAuraFlow
  shift=3`, `res_multistep`/`simple`, 8 steps, CFG fully off.
- The **fp16 flag pair** on T4: `--force-fp16 --disable-comfy-compiler`
  (the card has no bf16; the compiler crashes the fp16 path without the
  second flag).
- The **bf16 split**, not nvfp4 (Blackwell-era hardware) and not int8 unless
  you need the VRAM headroom — int8 runs ~2.3× slower per step on T4.

Numbers trace to the GPUTests lab repo
(`projects/2026-W40-zimage-turbo-t4/`, files `timings_v7_zimage.json` +
`timings_v7b_warm.json`), measured 2026-09-28 on Colab T4s and re-verified
across two VMs (Qwen rows identical to the second, Z rows within ±10%).

## Quickstart (browser, no tools)

1. Click the Colab badge above → *Runtime → Change runtime type → T4 GPU →
   Save* → *Runtime → Run all*.
2. Wait ~15 minutes (installs 1 min, ~36 GB of downloads ≈ 6 min, boot
   ~2 min, first image ~2 min, tunnel 15 s). Cells print `OK:` as they go.
3. Copy the printed `API URL: https://….trycloudflare.com` and from any
   machine:

```bash
python3 clients/txt2img.py --url https://…trycloudflare.com \
    --prompt 'a neon shop sign that reads "FREE GPU LAB", rainy night' \
    --out sign.png
python3 clients/txt2img.py --url https://…trycloudflare.com --engine qwen \
    --prompt 'a red fox in snow, film photo' --out fox-qwen.png
```

The tunnel URL changes on every notebook re-run (free quick tunnel); the VM
sleeps when idle. When done: *Runtime → Manage sessions → TERMINATE*.

## Samples (real outputs, this exact recipe)

| | |
|---|---|
| ![Z-Image sample](samples/z_p2_freegpulab_sign.png) | ![Qwen sample](samples/qwen_p2_freegpulab_sign.png) |

Both generated at 768×768, seed 42, through the public tunnel from a machine
outside Colab.

## Repo map

- `colab/run-z-image-t4.ipynb` — the notebook (build artifact; edit
  `tools/build_notebook.py` and rebuild)
- `workflows/t2i-z-turbo.json`, `workflows/t2i-qwen-turbo.json` — the exact
  graphs, canonical
- `clients/txt2img.py` — stdlib-only client for both engines
- `tools/check_workflows.py` — consistency gate (graphs must not drift
  between json/notebook/client); run after any change, must print ALL GREEN
- `references/01…05` — manual steps, CLI lane, API usage, tuning receipts,
  troubleshooting
- `colab/verification-*.log` — fresh-T4 verification runs of this notebook

## Licenses and credit

- **Z-Image-Turbo**: Apache-2.0 (Tongyi-MAI); split files repackaged by
  Comfy-Org.
- **Qwen-Image-2.1**: Qwen Research License (experiments/research/demos;
  check before commercial use). Turbo LoRA by Viggle (use the author's
  runtime-hook node — merging is lossy). GGUF by unsloth; TE/VAE splits by
  abenzerps.
- Sibling repo: [run-qwen-image-on-free-colab](https://github.com/KodeIsFun/run-qwen-image-on-free-colab)
  — the Qwen recipe in depth.
