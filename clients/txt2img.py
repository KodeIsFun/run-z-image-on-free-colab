#!/usr/bin/env python3
"""txt2img — generate an image on a remote ComfyUI (e.g. the free-T4 notebook
tunnel) using only the Python standard library. No pip installs, any OS.

Tested end-to-end from outside Colab through a trycloudflare tunnel.

Two engines, both measured on the same free T4 (see the README table):

  --engine z (default)   Z-Image-Turbo bf16, official 8-step recipe, CFG off.
                         FASTEST on repeats of a prompt (~18 s warm at 768²),
                         but a NEW prompt pays a one-time text-encoder reload
                         (~42 s) — its 8 GB encoder cannot stay resident
                         beside the 12.3 GB unet on a 15.6 GB card.
  --engine qwen          Qwen-Image turbo (Viggle LoRA, GGUF, 6 steps, CFG
                         off) — steady ~27 s warm regardless of prompt
                         changes; needs the GGUF stack the notebook also
                         downloads.

Examples:
    python3 txt2img.py --url https://x-y-z.trycloudflare.com --prompt "a red fox in snow"
    python3 txt2img.py --url $URL --engine qwen --prompt "..." --out fox-qwen.png
    python3 txt2img.py --url $URL --prompt "..." --width 1024 --height 1024 --seed 7 --out big.png
"""

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request

Z_UNET = "z_image_turbo_bf16.safetensors"
Z_TE = "qwen_3_4b.safetensors"
Z_VAE = "ae.safetensors"

GGUF_FILE = "qwen-image-2.1-Q4_K_M.gguf"
TE_FILE = "qwen3vl_8b_int8_convrot.safetensors"
VAE_FILE = "qwen_image_2.1_vae_bf16.safetensors"
LORA_FILE = "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors"
SIGMAS_6 = "1.0, 0.9375, 0.875, 0.75, 0.5, 0.25"

TIMEOUT_SUBMIT = 60
TIMEOUT_IMAGE = 1800  # generous: includes queue + cold model load (~5 min)


def build_workflow_z(prompt, width, height, seed):
    """The exact Z-Image graph the notebook runs (workflows/t2i-z-turbo.json).

    Official template recipe, verbatim: lumina2 clip type for the Qwen3-4B
    encoder, ModelSamplingAuraFlow shift=3, 8 steps, cfg 1.0,
    res_multistep/simple, ConditioningZeroOut negative.
    """
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": Z_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": Z_TE, "type": "lumina2", "device": "default"}},
        "3": {"class_type": "VAELoader",
              "inputs": {"vae_name": Z_VAE}},
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["2", 0], "text": prompt}},
        "5": {"class_type": "ConditioningZeroOut",
              "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "EmptySD3LatentImage",
              "inputs": {"width": width, "height": height, "batch_size": 1}},
        "7": {"class_type": "ModelSamplingAuraFlow",
              "inputs": {"model": ["1", 0], "shift": 3}},
        "8": {"class_type": "KSampler",
              "inputs": {"model": ["7", 0], "positive": ["4", 0],
                         "negative": ["5", 0], "latent_image": ["6", 0],
                         "seed": seed, "steps": 8, "cfg": 1.0,
                         "sampler_name": "res_multistep",
                         "scheduler": "simple", "denoise": 1.0}},
        "9": {"class_type": "VAEDecode",
              "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveImage",
               "inputs": {"images": ["9", 0], "filename_prefix": "txt2img"}},
    }


def build_workflow_qwen(prompt, width, height, seed):
    """The exact Qwen turbo graph (workflows/t2i-qwen-turbo.json): Viggle
    runtime-hook LoRA (never merged), resolution-shifted sigmas, CFG fully
    off via BasicGuider."""
    return {
        "1": {"class_type": "UnetLoaderGGUF",
              "inputs": {"unet_name": GGUF_FILE}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": TE_FILE, "type": "qwen_image"}},
        "3": {"class_type": "VAELoader",
              "inputs": {"vae_name": VAE_FILE}},
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["2", 0], "text": prompt}},
        "5": {"class_type": "ViggleTurboLora",
              "inputs": {"model": ["1", 0], "lora_name": LORA_FILE,
                         "strength": 1.0}},
        "6": {"class_type": "EmptySD3LatentImage",
              "inputs": {"width": width, "height": height, "batch_size": 1}},
        "7": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "8": {"class_type": "KSamplerSelect",
              "inputs": {"sampler_name": "euler"}},
        "9": {"class_type": "ViggleTurboSigmas",
              "inputs": {"latent": ["6", 0], "nodes": SIGMAS_6}},
        "10": {"class_type": "BasicGuider",
               "inputs": {"model": ["5", 0], "conditioning": ["4", 0]}},
        "11": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["7", 0], "guider": ["10", 0],
                          "sampler": ["8", 0], "sigmas": ["9", 0],
                          "latent_image": ["6", 0]}},
        "12": {"class_type": "VAEDecode",
               "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "SaveImage",
               "inputs": {"images": ["12", 0], "filename_prefix": "txt2img"}},
    }


def build_workflow(prompt, steps, cfg, width, height, seed, engine="z"):
    if engine == "z":
        return build_workflow_z(prompt, width, height, seed)
    if engine == "qwen":
        return build_workflow_qwen(prompt, width, height, seed)
    raise ValueError(f"unknown engine: {engine}")


def api(url, path, payload=None, timeout=60):
    req = urllib.request.Request(url.rstrip("/") + path, method="GET")
    if payload is not None:
        req = urllib.request.Request(url.rstrip("/") + path,
                                     data=json.dumps(payload).encode(),
                                     method="POST",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def api_retry(url, path, payload=None, timeout=60, tries=4):
    """api() with retries — quick tunnels occasionally reset a connection;
    one flaky reset must not kill a client that is willing to wait 25 min."""
    last = None
    for attempt in range(tries):
        try:
            return api(url, path, payload, timeout)
        except Exception as e:  # noqa: BLE001 — anything transient counts
            last = e
            time.sleep(min(2 ** attempt, 8))
    raise last


def main():
    p = argparse.ArgumentParser(description="txt2img on a remote ComfyUI")
    p.add_argument("--url", required=True, help="e.g. https://x-y.trycloudflare.com")
    p.add_argument("--prompt", required=True)
    p.add_argument("--engine", choices=["z", "qwen"], default="z",
                   help="z = Z-Image-Turbo (default, fastest on repeats); "
                        "qwen = Qwen turbo GGUF (steady across prompt changes)")
    p.add_argument("--negative", default="",
                   help="ignored: both engines run CFG fully off")
    p.add_argument("--steps", type=int, default=None,
                   help="default: fixed by the recipe (8 for z, 6 for qwen)")
    p.add_argument("--cfg", type=float, default=None,
                   help="ignored: CFG is fully off in both recipes")
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="generated.png")
    a = p.parse_args()

    url = a.url
    try:
        api_retry(url, "/system_stats", timeout=15)
        print(f"server alive: {url}")
    except Exception as e:
        sys.exit(f"server not reachable at {url} — is the notebook running? ({e})")

    wf = build_workflow(a.prompt, a.steps, a.cfg, a.width, a.height, a.seed, a.engine)
    pid = api_retry(url, "/prompt", {"prompt": wf, "client_id": "txt2img.py"},
                    timeout=TIMEOUT_SUBMIT)["prompt_id"]
    mode = ("Z-Image turbo (8 steps, cfg off)" if a.engine == "z"
            else "Qwen turbo (6 steps, cfg off)")
    print(f"queued {a.width}x{a.height}, {mode}, seed {a.seed} — polling...")

    t0 = time.time()
    while time.time() - t0 < TIMEOUT_IMAGE:
        time.sleep(3)
        try:
            hist = api_retry(url, f"/history/{pid}", timeout=20)
        except Exception:
            continue  # transient tunnel reset — the loop timeout bounds this
        if pid not in hist:
            continue
        entry = hist[pid]
        status = entry.get("status", {})
        if status.get("status_str") == "error":
            sys.exit(f"generation failed: {json.dumps(status.get('messages', []))[:400]}")
        imgs = [im for out in entry.get("outputs", {}).values() for im in out.get("images", [])]
        if imgs:
            im = imgs[0]
            q = urllib.parse.urlencode({"filename": im["filename"],
                                        "subfolder": im.get("subfolder", ""),
                                        "type": im.get("type", "output")})
            data = None
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(
                            f"{url.rstrip('/')}/view?{q}", timeout=120) as r:
                        data = r.read()
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(min(2 ** attempt, 8))
            with open(a.out, "wb") as f:
                f.write(data)
            print(f"done in {time.time() - t0:.0f}s -> {a.out}")
            return
    sys.exit("timed out waiting for the image (~25 min) — see troubleshooting")


if __name__ == "__main__":
    main()
