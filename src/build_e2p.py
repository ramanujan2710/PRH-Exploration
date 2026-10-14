import nbformat, re
v4 = open("prh_extract_v4.py").read(); e2p = open("e2p_extract.py").read()
src = (open("e2p_cells.py").read().replace("__ENGINE__", open("prh_engine.py").read())
       .replace("__EXTRACT_V4_STR__", repr(v4)).replace("__E2P_STR__", repr(e2p)).replace("__EXTRACT_V4__", v4))
chunks = re.split(r"^# %%(.*)$", src, flags=re.M); cells = []
for i in range(1, len(chunks), 2):
    kind = chunks[i].strip(); body = chunks[i + 1].strip("\n")
    if kind == "[md]":
        cells.append(nbformat.v4.new_markdown_cell("\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())))
    else:
        cells.append(nbformat.v4.new_code_cell(body))
nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata = {"accelerator": "GPU", "colab": {"machine_shape": "hm", "provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}
nbformat.validate(nb); nbformat.write(nb, "PRH_E2prompt.ipynb")
open("e2p_flat.py", "w").write("\n\n".join(c.source for c in cells if c.cell_type == "code"))
print(len(cells), "cells")
