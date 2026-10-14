import nbformat, re
src=open('e3_cells.py').read().replace('__WORKER__', open('e3_worker.py').read())
chunks=re.split(r'^# %%(.*)$', src, flags=re.M); cells=[]
for i in range(1,len(chunks),2):
    kind=chunks[i].strip(); body=chunks[i+1].strip('\n')
    cells.append(nbformat.v4.new_markdown_cell("\n".join(l[2:] if l.startswith('# ') else l.lstrip('#') for l in body.splitlines())) if kind=='[md]' else nbformat.v4.new_code_cell(body))
nb=nbformat.v4.new_notebook(cells=cells)
nb.metadata={"accelerator":"GPU","colab":{"gpuType":"H100","machine_shape":"hm","provenance":[]},"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}}
nbformat.validate(nb); nbformat.write(nb,'PRH_E3_rosetta_yardstick.ipynb')
open('dryrun_e3.py','w').write("\n\n".join(c.source for c in cells if c.cell_type=='code').replace('plt.show()','plt.close()'))
print(len(cells),'cells')
