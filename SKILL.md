---
name: run-z-image-on-free-colab
description: Run Z-Image-Turbo (Tongyi-MAI 6B S3-DiT, Apache-2.0) text-to-image on a
  free Colab T4 with stock ComfyUI headless, plus the bundled Qwen-Image turbo GGUF
  engine for the measured head-to-head, and expose the ComfyUI API through a public
  tunnel consumable from anywhere. Use when the user wants free-GPU image generation,
  a text-to-image API without paying, or to compare Z-Image-Turbo vs Qwen turbo on
  the same card. Includes the Colab CLI lane, a verified notebook, a stdlib-only
  two-engine client, the official 8-step Z-Image recipe (~18 s per warm repeat,
  fastest measured on a free T4), and every gotcha paid for in real failed runs.
---

# Run Z-Image-Turbo on a free Colab T4 (and race it against Qwen turbo)

Everything here was **measured on real free T4s** (2026-09-28, Colab
`torch 2.11.0+cu128`, Python 3.13), not assembled from forum posts. The
numbers live in [references/04-tweaking.md](references/04-tweaking.md) and
trace to `projects/2026-W40-zimage-turbo-t4/` in the GPUTests lab repo
(<https://github.com/KodeIsFun/GPUTests>).

## The one decision that routes everything

| They want | Path | First action |
|---|---|---|
| Generate an image and see it work | Notebook in browser | the README's "Open in Colab" badge — zero installed tools |
| A text-to-image API from anywhere | Same notebook | cells 6–7 print a public `trycloudflare.com` URL; then [clients/txt2img.py](clients/txt2img.py) from ANY machine |
| The Qwen comparison engine | Same API | `--engine qwen` on the client (the notebook installs both stacks) |
| Agent-driven / headless runs | Colab CLI lane | [references/02-colab-cli-lane.md](references/02-colab-cli-lane.md) |
| Faster / bigger / different trade-off | Tuning | [references/04-tweaking.md](references/04-tweaking.md) |
| Something broke | Troubleshooting | [references/05-troubleshooting.md](references/05-troubleshooting.md) |

**Default config (verified):** Z-Image-Turbo **bf16 split** (Comfy-Org
repackage of Tongyi-MAI/Z-Image-Turbo, Apache-2.0) + qwen_3_4b text encoder +
ae VAE, stock ComfyUI, `--force-fp16 --disable-comfy-compiler`, **768×768,
8 steps, cfg 1.0, res_multistep/simple, ModelSamplingAuraFlow shift=3** →
**~1.8 s/step, 18.0 s per warm repeat** — fastest measured on a free T4.
Text canary ("FREE GPU LAB" neon sign) letter-perfect.

**The memory catch (invariant, not a bug):** Z's 12.31 GB unet + 8.04 GB text
encoder cannot both stay resident on 15.6 GB. ComfyUI reloads the encoder for
each NEW prompt → first render with fresh text ≈ 42 s; repeats ≈ 18 s; the
session's first image ≈ 102 s. The bundled Qwen turbo engine (GGUF stack,
both models resident) holds 27–30 s regardless of prompt changes. Route by
workload: variations → Z, mixed prompts → Qwen.

## Invariants — never violate these

1. **Never leave a Colab VM running.** `colab stop -s <name>`, then
   `colab sessions` must be empty. Download every artifact and verify each
   with `ls -la` BEFORE the stop — the VM and `/content` die together, and a
   full artifact set was lost exactly this way once.
2. **The T4 flags are a pair.** `--force-fp16` without
   `--disable-comfy-compiler` crashes the first forward (`aimdo memory
   compile error`); without fp16 the DiT crawls in an fp32 cast.
3. **The Z-Image recipe is the official template, verbatim.** CLIPLoader
   `type: lumina2` for the Qwen3-4B encoder (no z_image type exists),
   `ModelSamplingAuraFlow shift=3`, KSampler 8 steps cfg 1.0
   res_multistep/simple, ConditioningZeroOut negative. The
   `distill_patch_lora` in the Comfy-Org repo is NOT part of t2i — do not add
   it.
4. **The graphs must not drift.** `python3 tools/check_workflows.py` must
   print ALL GREEN after any change to workflows/, the notebook, or the
   client — the same graph lives in three places by design.
5. **Verify from OUTSIDE the VM** before claiming anything works: read
   `nvidia-smi` output, and `curl <tunnel-url>/system_stats` from your own
   machine. A URL printed on the VM is not proof it is reachable.
6. **`colab` CLI timeouts default to 30 SECONDS**; exec `--timeout` is
   seconds. Every uploaded script must be fresh-VM safe (own install +
   downloads with cached skips) — a fresh VM has an empty `/content`, and a
   script that assumes a sibling session's files dies instantly (this cost a
   rerun once: FileNotFoundError /content/ComfyUI).
7. **Colab VMs recycle mid-conversation.** Weights cached in /tmp or
   /content can vanish between two execs minutes apart; a job that assumed
   cache re-downloaded 14.5 GB. Always check the log's `cached:` lines
   before assuming state survived.
8. **Quant/precision swaps trade differently on T4.** z int8 fits easiest
   but runs ~2.3× slower per step (4.1 vs 1.8 s/step); nvfp4 does not belong
   on Turing at all. bf16 is the measured default.
9. **Licenses**: Z-Image-Turbo is Apache-2.0; the Qwen engine's base model is
   Qwen Research License — flag commercial use.
10. **Every number you quote traces to the measured table** in
    references/04-tweaking.md; untested configs are labeled there. Speed
    ratios come from same-session pairs only (VM variance is ±10% across
    sessions; Qwen anchor rows reproduced to the second across VMs).

## The 60-second version (agent driving the CLI lane)

```bash
colab new -s zimg --gpu T4
colab exec -s zimg -f colab/run-z-image-t4.ipynb --timeout 1500
# grep for: OK: gpu / install / download / launch / generate / tunnel + "API URL:"
colab stop -s zimg && colab sessions   # must be empty
```

Full details: [references/02-colab-cli-lane.md](references/02-colab-cli-lane.md).
