# 03 — The API, consumable from anywhere

ComfyUI runs headless on the VM with its HTTP API on `127.0.0.1:8188`. The
notebook tunnels it through a **Cloudflare quick tunnel** (no account, no
token) and prints:

```
API URL: https://<random-words>.trycloudflare.com
```

The API surface used here is three endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/prompt` | POST | queue a workflow JSON; returns `{"prompt_id": ...}` |
| `/history/<prompt_id>` | GET | `{}` while running; on completion has `outputs` with image filenames and `status.status_str` = `success` or `error` |
| `/view?filename=X&subfolder=&type=output` | GET | the generated PNG bytes |

Generation is **asynchronous by design**: submit, poll, fetch.

## The easy way: the bundled client (stdlib only, any machine)

```bash
python3 clients/txt2img.py \
  --url https://<random-words>.trycloudflare.com \
  --prompt 'a neon shop sign that reads "FREE GPU LAB", rainy night' \
  --out my-image.png
```

`--engine z` is the default (Z-Image turbo, 8 steps, CFG off — ~18 s on warm
repeats of a prompt, ~42 s on the first render with fresh text).
`--engine qwen` runs the bundled Qwen turbo graph (~27 s warm, steady across
prompt changes). Both recipes run CFG fully off, so `--negative` and `--cfg`
are accepted and ignored. `--width/--height/--seed` work on both (defaults
768/768/42). Requires only Python 3 — no pip installs.

## The curl way

```bash
# 1. submit (workflow JSON: see workflows/t2i-z-turbo.json for the full shape)
curl -s -X POST $URL/prompt -H 'Content-Type: application/json' -d '{
  "prompt": {...}, "client_id": "curl"}'          # -> {"prompt_id": "..."}

# 2. poll until outputs appear
curl -s $URL/history/<prompt_id>

# 3. fetch the image named in outputs."10".images[0].filename
curl -s "$URL/view?filename=<name>&subfolder=&type=output" -o out.png
```

## Engine cheat sheet

| | `--engine z` (default) | `--engine qwen` |
|---|---|---|
| graph | `workflows/t2i-z-turbo.json` | `workflows/t2i-qwen-turbo.json` |
| steps / cfg | 8 / off | 6 / off |
| warm repeat (768²) | ~18 s | ~27 s |
| new prompt (768²) | ~42 s (encoder reload) | ~30 s |
| needs server-side | stock ComfyUI only | leejet ComfyUI-GGUF + viggle_turbo.py (notebook installs both) |
