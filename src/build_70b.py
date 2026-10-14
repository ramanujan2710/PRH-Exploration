import nbformat, re, sys
src=open('cells_70b.py').read()
src=src.replace('__ENGINE__',open('prh_engine.py').read()).replace('__EXTRACT_V4_STR__',repr(open('prh_extract_v4.py').read())).replace('__EXTRACT_V4__',open('prh_extract_v4.py').read())
chunks=re.split(r'^# %%(.*)$', src, flags=re.M); cells=[]
for i in range(1,len(chunks),2):
    kind=chunks[i].strip(); body=chunks[i+1].strip('\n')
    if kind=='[md]':
        cells.append(nbformat.v4.new_markdown_cell("\n".join(l[2:] if l.startswith('# ') else l.lstrip('#') for l in body.splitlines())))
    else: cells.append(nbformat.v4.new_code_cell(body))
nb=nbformat.v4.new_notebook(cells=cells)
nb.metadata={"accelerator":"GPU","colab":{"gpuType":"H100","machine_shape":"hm","provenance":[]},"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}}
nbformat.validate(nb); nbformat.write(nb,'PRH_audit_v4_llama3_70b.ipynb')
open('dryrun_70b.py','w').write("\n\n".join(c.source for c in cells if c.cell_type=='code').replace('plt.show()','plt.close()'))
print(len(cells),'cells')
