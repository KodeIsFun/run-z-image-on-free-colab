# 01 — Manual steps (what the human does by hand)

Everything automatable is automated. The human's part is deliberately tiny.
As the agent, read the path that applies and **say these sentences to the
user** — do not assume they know the Colab UI.

## Path A — browser only (no tools, no CLI, works for anyone)

Total human effort: ~2 minutes of clicking, then ~15 minutes of waiting.

1. **Have a Google account.** That is the only prerequisite. Free tier gives
   a T4 GPU with usage limits that reset daily; a normal first run fits.
2. **Open the notebook in Colab** — the badge in the README, i.e.:

   ```
   https://colab.research.google.com/github/KodeIsFun/run-z-image-on-free-colab/blob/main/colab/run-z-image-t4.ipynb
   ```

   (Fallback: any Colab session → *File → Open notebook → GitHub* tab, paste
   `KodeIsFun/run-z-image-on-free-colab`.)
3. **Pick the GPU**: menu *Runtime → Change runtime type → T4 GPU → Save*.
   Cell 1 checks and warns if the runtime has no GPU.
4. **Run all**: menu *Runtime → Run all*, approve the "not authored by
   Google" warning. Cells print `OK: ...` as they finish.
5. **Wait ~15 minutes.** Install ≈ 1 min, downloads ≈ 6 min (~36 GB for both
   engines), boot ≈ 2 min, first image ≈ 2 min (model loads; warm repeats
   after it run ~18 s), tunnel ≈ 15 s.
6. **Copy the line that looks like**

   ```
   API URL: https://something-random-words.trycloudflare.com
   ```

   and generate from anywhere:

   ```bash
   python3 clients/txt2img.py \
     --url https://something-random-words.trycloudflare.com \
     --prompt "a neon shop sign that reads \"FREE GPU LAB\", rainy night" \
     --out my-image.png
   ```

   `--engine qwen` switches to the bundled Qwen turbo comparison engine.

### Tell the user these caveats up front

- The tunnel URL **changes every run** — re-run the notebook for a fresh one.
- The free VM **sleeps when idle** and is reclaimed eventually. A re-run
  costs one click and ~15 minutes.
- The first render with a **new prompt** on the Z engine pays a one-time
  text-encoder reload (~42 s); repeats of the same prompt run ~18 s. This is
  the memory math of a 15.6 GB card, not a bug — see 04-tweaking.md.
- **Licenses**: Z-Image-Turbo is Apache-2.0; the Qwen engine's base model is
  Qwen Research License — check before commercial use.
