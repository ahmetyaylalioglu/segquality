"""Throwaway in-process notebook executor (bypasses the broken jupyter/pyzmq kernel).

Runs each code cell in a shared namespace, capturing stdout, matplotlib figures and
display() calls, and writes the outputs back into the .ipynb so it renders with
results embedded. Continue-on-error so one plotting bug does not abort the run.
"""
import base64
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nbformat  # noqa: E402

NB = Path(__file__).with_name("goruntu_kanali_raporu.ipynb")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    nb = nbformat.read(str(NB), as_version=4)
    ns: dict = {"__name__": "__main__"}
    captured: list = []

    def _display(*objs):
        for obj in objs:
            if hasattr(obj, "_repr_html_"):
                captured.append(nbformat.v4.new_output(
                    "display_data", data={"text/html": obj._repr_html_(),
                                          "text/plain": repr(obj)}))
            else:
                captured.append(nbformat.v4.new_output(
                    "display_data", data={"text/plain": repr(obj)}))

    ns["display"] = _display

    count = 0
    failures: list = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        count += 1
        captured.clear()
        plt.close("all")
        buf = io.StringIO()
        outputs: list = []
        src = "\n".join(
            "" if ln.lstrip().startswith(("%", "!")) else ln
            for ln in cell.source.splitlines()
        )
        try:
            with redirect_stdout(buf):
                exec(src, ns)
        except Exception as exc:  # noqa: BLE001
            import traceback
            tb = traceback.format_exc()
            text = buf.getvalue()
            if text:
                outputs.append(nbformat.v4.new_output("stream", name="stdout", text=text))
            outputs.append(nbformat.v4.new_output(
                "error", ename=type(exc).__name__, evalue=str(exc),
                traceback=tb.splitlines()))
            cell.outputs = outputs
            cell.execution_count = count
            failures.append((count, tb))
            print(f"CELL FAILED (exec#{count}):\n{tb}")
            continue

        text = buf.getvalue()
        if text:
            outputs.append(nbformat.v4.new_output("stream", name="stdout", text=text))
        outputs.extend(captured)
        for num in plt.get_fignums():
            fig = plt.figure(num)
            b = io.BytesIO()
            fig.savefig(b, format="png", dpi=120, bbox_inches="tight")
            b.seek(0)
            png = base64.b64encode(b.read()).decode("ascii")
            outputs.append(nbformat.v4.new_output(
                "display_data", data={"image/png": png}, metadata={}))
        cell.outputs = outputs
        cell.execution_count = count
        print(f"ok cell exec#{count}: {len(outputs)} output(s)")

    nbformat.write(nb, str(NB))
    print(f"DONE: executed {count} code cells, written to {NB.name}")
    if failures:
        print(f"FAILURES in {len(failures)} cell(s): {[f[0] for f in failures]}")
    else:
        print("NO FAILURES")


if __name__ == "__main__":
    main()
