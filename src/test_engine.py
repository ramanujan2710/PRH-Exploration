import sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,'.'); sys.path.insert(0,'../prh_src'); sys.path.insert(0,'../e1')
from prh_engine import *
import huh_metrics as huh          # Huh et al.'s own implementation
from e1lib import cka_debiased as ref_deb, center_gram, cka_from_centered
torch.manual_seed(0); n=200
Z=torch.randn(n,8)
fx=torch.stack([Z@torch.randn(8,64)+0.5*torch.randn(n,64) for _ in range(3)],1)*torch.linspace(1,3,64)
fy=torch.stack([Z@torch.randn(8,96)+1.0*torch.randn(n,96) for _ in range(4)],1)
fx[:,:,0]+=20*torch.randn(n,3)  # an outlier dimension
for v in VARIANTS:
    lx,ly=preprocess(fx,v),preprocess(fy,v)
    g=torch.Generator().manual_seed(1); perm=torch.randperm(n,generator=g); half=(perm[:n//2],perm[n//2:])
    sx,sy=SideStats(lx,half_idx=half),SideStats(ly,half_idx=half)
    cka=linear_cka_pairs(sx,sy); ucka=unbiased_cka_pairs(sx.hu,sy.hu); mk=mknn_pairs(sx,sy)
    err=[]
    for i in range(3):
        for j in range(4):
            X,Y=lx[i],ly[j]
            err.append(abs(cka[i,j].item()-huh.AlignmentMetrics.cka(X,Y)))
            err.append(abs(ucka[i,j].item()-huh.AlignmentMetrics.unbiased_cka(X,Y)))
            Kx,Ky=(X@X.T).double().numpy(),(Y@Y.T).double().numpy()
            err.append(abs(ucka[i,j].item()-ref_deb(Kx,Ky)))
            err.append(abs(mk[i,j].item()-huh.AlignmentMetrics.mutual_knn(X,Y,topk=10)))
    print(v,'max abs err vs Huh/reference:',f'{max(err):.2e}')
# huh clip check vs Huh remove_outliers(exact=False)
a=huh_clip(fx); b=huh.remove_outliers(fx,q=0.95,exact=False)
print('clip max diff',(a-b).abs().max().item())
perms=[torch.randperm(n) for _ in range(50)]
out,m=align_pair(sx,sy,perms); print({k:(round(v,4) if isinstance(v,float) else v) for k,v in out.items()})
# permutation correctness: null for a permuted copy of features should look like H0
pp=torch.randperm(n); sy_shuf=SideStats([l[pp] for l in ly])
o2,_=align_pair(SideStats(lx),sy_shuf,perms); print('shuffled pair: cka_cal',round(o2['cka_cal'],4),'mknn_cal',round(o2['mknn_cal'],4),'p',o2['cka_p'])
print(layer_diagnostics(fx)[0])
