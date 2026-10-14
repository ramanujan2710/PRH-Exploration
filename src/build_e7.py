import nbformat, re
src = open("e7_cells.py").read().replace("__E7_TEXT__", open("e7_text.py").read()).replace("__WORKER__", open("e3_worker.py").read())
chunks = re.split(r"^# %%(.*)$", src, flags=re.M); cells = []
for i in range(1, len(chunks), 2):
    kind = chunks[i].strip(); body = chunks[i + 1].strip("\n")
    if kind == "[md]":
        cells.append(nbformat.v4.new_markdown_cell("\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())))
    else:
        cells.append(nbformat.v4.new_code_cell(body))
nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata = {"accelerator": "GPU", "colab": {"machine_shape": "hm", "provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}
nbformat.validate(nb); nbformat.write(nb, "PRH_E7_rosetta_prompt.ipynb")
open("e7_flat.py", "w").write("\n\n".join(c.source for c in cells if c.cell_type == "code"))
print(len(cells), "cells")
