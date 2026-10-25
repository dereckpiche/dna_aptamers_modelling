"""Run actual notebook code and save every completed cell, including inline figures."""
import os, sys, time
from pathlib import Path
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
import matplotlib
from matplotlib_inline.backend_inline import configure_inline_support

path=Path(sys.argv[1]).resolve()
os.chdir(path.parent)
sys.path.insert(0,str(path.parent))
notebook=nbformat.read(path,as_version=4)
matplotlib.use('module://matplotlib_inline.backend_inline')
shell=InteractiveShell.instance()
configure_inline_support(shell,'inline')
count=0
for i,cell in enumerate(notebook.cells):
    if cell.cell_type!='code':continue
    count+=1;print('START cell',i,flush=True);began=time.time()
    with capture_output() as captured:
        result=shell.run_cell(cell.source,store_history=True)
    cell.execution_count=count;cell.outputs=[]
    if captured.stdout:cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
    if captured.stderr:cell.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
    for out in captured.outputs:
        cell.outputs.append(nbformat.v4.new_output('display_data',data=out.data,metadata=out.metadata))
    nbformat.write(notebook,path)
    print(captured.stdout,flush=True)
    print('END cell',i,'seconds',round(time.time()-began,2),flush=True)
    if result.error_before_exec or result.error_in_exec:
        print(captured.stderr,flush=True)
        raise RuntimeError(f'Cell {i} failed: {result.error_before_exec or result.error_in_exec}')
print('COMPLETE',flush=True)
