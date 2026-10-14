import nbformat, re
src = open("e6_cells.py").read().replace("__ENGINE__", open("prh_engine.py").read()).replace("__E6_EXTRACT__", open("e6_extract.py").read())
chunks = re.split(r"^# %%(.*)$", src, flags=re.M); cells = []
for i in range(1, len(chunks), 2):
    kind = chunks[i].strip(); body = chunks[i + 1].strip("\n")
    if kind == "[md]":
        cells.append(nbformat.v4.new_markdown_cell("\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())))
    else:
        cells.append(nbformat.v4.new_code_cell(body))
nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata = {"accelerator": "GPU", "colab": {"machine_shape": "hm", "provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}
nbformat.validate(nb); nbformat.write(nb, "PRH_E6_global_representations.ipynb")
open("e6_flat.py", "w").write("\n\n".join(c.source for c in cells if c.cell_type == "code"))
print(len(cells), "cells")
