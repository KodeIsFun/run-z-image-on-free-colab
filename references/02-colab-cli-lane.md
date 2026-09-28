# 02 — Colab CLI lane (agent-driven)

Install once, auth once (see [01-manual-steps.md](01-manual-steps.md) Path B
in the sibling repo if missing), then everything here is headless. Binary is
`colab` (package `google-colab-cli`; 0.7.4 measured — 0.6.0's `exec` is
broken by a `jupyter_kernel_client` dep drift).

## Install + auth (needs the human once)

```bash
uv tool install google-colab-cli==0.7.4 --force
colab --auth=oauth2 sessions            # browser consent on first use
```

## Run the notebook headlessly

`colab run` executes a **.py script** on a fresh VM and self-cleans.
Notebooks go through `colab new` + `colab exec -f` (exec's `--timeout` is
**seconds**, default 30 — always set it):

```bash
colab new --gpu T4 -s zimg
colab exec -s zimg -f colab/run-z-image-t4.ipynb --timeout 1500
colab stop -s zimg
colab sessions                          # must be empty
```

- `exec` streams cell output live — grep for the notebook's success markers:
  `OK: gpu`, `OK: install`, `OK: download`, `OK: launch`, `OK: generate`,
  `OK: tunnel`, and the `API URL:` line.
- If `colab new` 400s on the accelerator, the free tier has no T4 entitlement
  *right now*: retry later. An unrecognized `--gpu` value silently falls back
  to A100 — always pass `T4`.

## Long jobs: the detached-launch pattern

A second `exec` **while a cell runs** queues behind it. For long payloads,
run detached and poll the log:

```bash
colab upload -s zimg job.py /content/job.py
colab exec -s zimg -f launcher.py --timeout 60   # Popen nohup + return
# poll: colab download -s zimg /content/job.log ./job.log  every ~2 min
```

## Artifacts: download BEFORE stop, verify every one

`colab stop` is terminal — the VM and `/content` die together. Protocol:

```bash
colab download -s zimg /content/out/image.png ./image.png
ls -la ./image.png          # verify size > 0 — every single file
# only when ALL artifacts are verified locally:
colab stop -s zimg && colab sessions   # must be empty
```

Never bundle downloads and the stop into one command; never grep away a
download's error.

## Fresh-VM safety and mid-session recycling

- Any script uploaded to a VM must carry its own setup (clone/pip/download
  with `if not os.path.exists(...)` skips). One of this repo's verification
  runs died on boot with `FileNotFoundError: /content/ComfyUI` for exactly
  this reason.
- **VMs recycle mid-conversation**: two execs minutes apart can land on
  different VMs under the same session name (a 14.5 GB re-download was paid
  for this way). Check the log's `cached:` lines before assuming state
  survived.

## Surface detection on Colab VMs

Colab images ship stray `/kaggle` AND `/kaggle/working` directories — no
path proves Kaggle. Detect via env: `KAGGLE_KERNEL_RUN_TYPE` (Kaggle-only)
vs `COLAB_GPU` (Colab-only), and record both as `surface_proof` in any
timings output.
