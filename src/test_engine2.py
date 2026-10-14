import sys, numpy as np, torch
sys.path.insert(0,'.'); sys.path.insert(0,'../e1')
from prh_engine import *
import huh_metrics as huh
torch.manual_seed(0); n=512
Z=torch.randn(n,6)
def side(d, L, noise, const_first=False):
    ls=[]
    for l in range(L):
        f=Z@torch.randn(6,d)+noise*torch.randn(n,d)+30*torch.ones(n,d)*(l==0)
        if const_first and l==0: f=torch.ones(n,d)*5+1e-4*torch.randn(n,d)   # near-constant layer (ViT block-0 CLS)
        ls.append(f)
    return torch.stack(ls,1)
fx, fy = side(64,4,0.5), side(96,5,1.0,const_first=True)
g=torch.Generator().manual_seed(1); perm=torch.randperm(n,generator=g); half=(perm[:n//2],perm[n//2:])
perms=[torch.randperm(n) for _ in range(100)]; perms_half=[torch.randperm(n//2) for _ in range(100)]
for v in VARIANTS:
    lx,ly=preprocess(fx,v),preprocess(fy,v)
    sx,sy=SideStats(lx,half_idx=half),SideStats(ly,half_idx=half)
    u=unbiased_cka_pairs(sx.hu,sy.hu)
    err=max(abs(u[i,j].item()-huh.AlignmentMetrics.unbiased_cka(lx[i].double(),ly[j].double())) for i in range(4) for j in range(1,5))
    out,_=align_pair(sx,sy,perms,perms_half=perms_half)
    print(v,'ucka err vs Huh(fp64) on valid layers %.1e'%err,'| const layer -> NaN:',bool(torch.isnan(u[:,0]).all()),
          '| ucka_max %.3f heldout %.3f | cka_max %.3f agg-cal %.3f (tau %.3f) heldout-cal %.3f (tau %.3f) | nullmax pair %s'%(
          out['ucka_max'],out['ucka_heldout'],out['cka_max'],out['cka_cal'],out['cka_tau'],out['cka_heldout_cal'],out['cka_heldout_tau'],out['nullmax_pair']))
from e1lib import cka_debiased
lx,ly=preprocess(fx,'huh'),preprocess(fy,'huh'); sx,sy=SideStats(lx),SideStats(ly); u=unbiased_cka_pairs(sx.hu,sy.hu)
e=max(abs(u[i,j].item()-cka_debiased((lx[i]@lx[i].T).double().numpy(),(ly[j]@ly[j].T).double().numpy())) for i in range(4) for j in range(1,5))
print('vs epsilon-free fp64 reference: %.1e'%e)
X,Y=lx[0].double(),ly[1].double(); K,L=X@X.T,Y@Y.T
print('self-HSIC of layer 0 (huh):', huh.hsic_unbiased(K,K).item(), '-> Huh adds 1e-6 to the denominator')
