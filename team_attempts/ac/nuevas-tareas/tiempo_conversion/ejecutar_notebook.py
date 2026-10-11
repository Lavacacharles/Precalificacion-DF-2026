"""Construye y ejecuta el notebook desde analisis.py preservando sus salidas."""
import json
import os
import sys
from pathlib import Path
from jupyter_client import KernelManager

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
runtime = HERE / ".runtime"
runtime.mkdir(exist_ok=True)
for key, folder in [("IPYTHONDIR", "ipython"), ("JUPYTER_CONFIG_DIR", "jupyter"), ("JUPYTER_DATA_DIR", "data")]:
    os.environ[key] = str(runtime / folder)
cells = []
for i, block in enumerate((HERE / "analisis.py").read_text(encoding="utf-8").split("# %%")[1:]):
    header, _, source = block.partition("\n")
    if "[markdown]" in header:
        source = "\n".join(line[2:] if line.startswith("# ") else "" if line == "#" else line
                           for line in source.splitlines())
        cell = dict(cell_type="markdown", source=source.strip(), metadata={})
    else:
        cell = dict(cell_type="code", source=source.strip(), metadata={}, execution_count=None, outputs=[])
    cell["id"] = f"tiempos-{i:02d}"
    cells.append(cell)
nb = dict(nbformat=4, nbformat_minor=5, cells=cells, metadata={
    "kernelspec": {"name": "python3", "language": "python", "display_name": "Python 3.13"},
    "language_info": {"name": "python", "version": "3.13"}})
dest = HERE / "Modelos_tiempo_conversion.ipynb"


def collect(client, request):
    outputs, count = [], None
    while True:
        msg = client.get_iopub_msg(timeout=3600)
        if msg.get("parent_header", {}).get("msg_id") != request:
            continue
        kind, content = msg["msg_type"], msg["content"]
        if kind == "status" and content["execution_state"] == "idle":
            return outputs, count
        if kind == "execute_input":
            count = content["execution_count"]
        elif kind == "stream":
            outputs.append(dict(output_type="stream", name=content["name"], text=content["text"]))
            print(content["text"], end="", flush=True)
        elif kind in ["display_data", "execute_result"]:
            out = dict(output_type=kind, data=content["data"], metadata=content.get("metadata", {}))
            if kind == "execute_result":
                out["execution_count"] = content["execution_count"]
            outputs.append(out)
        elif kind == "error":
            raise RuntimeError("\n".join(content["traceback"]))


manager = KernelManager(kernel_name="python3")
manager.start_kernel(cwd=str(HERE))
client = manager.client()
client.start_channels()
try:
    client.wait_for_ready(timeout=90)
    collect(client, client.execute("%matplotlib inline"))
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            cell["outputs"], cell["execution_count"] = collect(client, client.execute(cell["source"]))
            dest.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"Celda {i+1}/{len(cells)} lista", flush=True)
finally:
    client.stop_channels()
    manager.shutdown_kernel(now=True)
print("Notebook completo:", dest)
