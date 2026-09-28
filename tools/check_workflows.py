#!/usr/bin/env python3
"""Consistency check: the two engine graphs exist in the right places and
MUST not drift apart. Run after any change:

    python3 tools/check_workflows.py

The Z-Image turbo graph is compared node-by-node across THREE copies
(workflows/t2i-z-turbo.json, the notebook's generate cell,
clients/txt2img.py build_workflow(engine="z")). The Qwen turbo comparison
graph is checked json vs client (it is intentionally notebook-absent — the
notebook generates with Z-Image; the qwen engine is a client/CLI mode,
documented in references/). Graphs are compared after ignoring the two
fields that are *supposed* to differ between copies (the sample prompt text
and the SaveImage filename prefix). Everything else — loaders, sizes, seed,
sampler wiring, shift, schedule — must be identical.
"""

import ast
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
IGNORED_INPUTS = {"filename_prefix", "text"}


def normalize(graph):
    out = {}
    for node_id, node in graph.items():
        if node_id == "_comment":
            continue
        inputs = {k: v for k, v in node["inputs"].items() if k not in IGNORED_INPUTS}
        out[node_id] = {"class_type": node["class_type"], "inputs": inputs}
    return out


def wf_from_notebook(nb_path, cell_marker):
    nb = json.loads(nb_path.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] != "code" or cell_marker not in "".join(cell["source"]):
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                    getattr(t, "id", None) == "wf" for t in node.targets):
                return ast.literal_eval(node.value)
    raise SystemExit(f"no 'wf = ' assignment found in the {cell_marker!r} cell")


def wf_from_client(client_path, engine, steps, cfg):
    import importlib.util
    spec = importlib.util.spec_from_file_location("txt2img_client", client_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_workflow("PROMPT", steps=steps, cfg=cfg,
                              width=768, height=768, seed=42, engine=engine)


def main():
    fails = 0
    pairs = [
        ("z-turbo", ROOT / "workflows" / "t2i-z-turbo.json",
         "ModelSamplingAuraFlow", "z", 8, 1.0),
        ("qwen-turbo", ROOT / "workflows" / "t2i-qwen-turbo.json",
         None, "qwen", 6, 1.0),
    ]
    for name, json_path, nb_marker, engine, steps, cfg in pairs:
        canon = normalize(json.loads(json_path.read_text()))
        sources = [("client", wf_from_client(ROOT / "clients" / "txt2img.py",
                                             engine, steps, cfg))]
        if nb_marker is not None:
            sources.insert(0, ("notebook", wf_from_notebook(
                ROOT / "colab" / "run-z-image-t4.ipynb", nb_marker)))
        for label, other in sources:
            other = normalize(other)
            if canon == other:
                print(f"OK   {name}: json == {label}")
            else:
                fails += 1
                keys = set(canon) | set(other)
                for k in sorted(keys, key=str):
                    if canon.get(k) != other.get(k):
                        print(f"FAIL {name}: node {k} differs\n"
                              f"     json:    {canon.get(k)}\n"
                              f"     {label}: {other.get(k)}")
    # every model file a workflow loads must be fetched by the download cell
    dl_src = (ROOT / "tools" / "build_notebook.py").read_text()
    for wf_name in ("t2i-z-turbo.json", "t2i-qwen-turbo.json"):
        wf = json.loads((ROOT / "workflows" / wf_name).read_text())
        for node_id, node in wf.items():
            if node_id == "_comment":
                continue
            for key in ("unet_name", "clip_name", "vae_name", "lora_name"):
                if key in node["inputs"]:
                    fname = node["inputs"][key]
                    if fname not in dl_src:
                        fails += 1
                        print(f"FAIL download cell: {wf_name} loads {fname!r} "
                              f"but CELL_DOWNLOAD never fetches it")
                    else:
                        print(f"OK   download cell fetches {fname} "
                              f"({wf_name} {key})")
    print(f"\n{'ALL GREEN' if fails == 0 else f'{fails} CHECK(S) FAILED'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
