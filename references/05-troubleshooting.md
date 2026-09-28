# 05 — Troubleshooting

Every entry below was hit for real during the runs behind this repo (or its
sibling lab). Ordered by how likely you are to meet it.

## 1. `RuntimeError: aimdo memory compile error` (under `--force-fp16`)

ComfyUI's model compiler cannot compile the fp16 forward on Turing. **Fix:
launch with both flags** (the notebook's cell 4 already does):

```bash
python main.py --listen 127.0.0.1 --port 8188 --force-fp16 --disable-comfy-compiler
```

Without `--force-fp16` at all, the model silently runs an fp32 manual cast
(much slower) — check the log line `weight dtype ..., manual cast: ...`.

## 2. `CLIPLoader lacks the lumina2 type` (cell 5 assert)

ComfyUI is too old for Z-Image. `git -C /content/ComfyUI pull` and re-run
cell 4 (or restart the runtime and re-run all). There is no `z_image` type —
`lumina2` IS the correct value for Z-Image's Qwen3-4B encoder.

## 3. `UnetLoaderGGUF` / `ViggleTurboLora` missing (cell 5 assert)

The Qwen engine's custom nodes did not install or did not load: re-run
cell 2, then *Runtime → Restart runtime*, then *Run all* again — custom
nodes only load at server boot.

## 4. `Unknown model architecture!` when the Qwen GGUF loads

You cloned the **city96** fork of ComfyUI-GGUF. Use the **leejet** fork
(the notebook's cell 2 does):

```bash
git clone --depth 1 https://github.com/leejet/ComfyUI-GGUF \
  ComfyUI/custom_nodes/ComfyUI-GGUF
```

## 5. A Z-Image render with a new prompt takes ~42 s while repeats take ~18 s

Working as measured, not a bug: the 8 GB text encoder cannot stay resident
beside the 12.3 GB unet on 15.6 GB, so each NEW prompt pays one encoder
reload. Batches of variations on one prompt are the fast path; mixed prompts
should use `--engine qwen` (steady ~27–30 s). The int8 unet split avoids the
reload but runs ~2.3× slower per step on T4.

## 6. `FileNotFoundError: /content/ComfyUI` on a fresh VM

The script assumed a previous session's files. Every uploaded script must
carry its own setup with `if not os.path.exists(...)` skips. Also: Colab VMs
recycle **mid-conversation** — two execs minutes apart can be different
machines under the same session name; check the log's `cached:` lines
before assuming state survived.

## 7. Notebook prints `WARNING: no GPU`

The runtime is CPU. Menu *Runtime → Change runtime type → T4 GPU → Save*,
then *Run all* again. If Colab refuses with a quota message, the free GPU
allowance is exhausted — retry later; do not run image gen on CPU.

## 8. Port already in use / server "already running" on re-run

Cell 4 is idempotent: if `/system_stats` answers, it skips the boot. To
force a fresh server: *Runtime → Restart runtime* (or `pkill -f main.py`)
then re-run from cell 4.

## 9. Client times out after ~25 min

- Check the server is alive: `curl $URL/system_stats`.
- First render after boot includes model loads (~2 min on Z, ~2–3 min when
  switching to the Qwen engine for the first time).
- Quick tunnels occasionally reset connections; the bundled client retries
  transient failures automatically.

## 10. `download failed:` in cell 3

A truncated model must never render. Delete the named partial file and
re-run cell 3 — `wget -c` resumes. Cell 3 asserts every file's size for
this reason.
