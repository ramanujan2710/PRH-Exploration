import numpy as np, time, sys
sys.path.insert(0,'.')
from e1lib import *
rng=np.random.default_rng(1); n=1024; r=32
for beta in [0,1.0]:
  X=Side(1024,r,rng,beta=beta,sigma=1.0)
  for d in [256,8192]:
    Y=Side(d,r,rng,beta=beta); tune_sigma(X,Y,0.30)
    # Monte Carlo check of pop CKA with large n
    Z=rng.standard_normal((20000,r)); Fx=X.sample(Z,rng); Fy=Y.sample(Z,rng)
    Cx=np.cov(Fx.T,dtype=np.float64) if d<=2048 else None
    t=time.time(); Z=rng.standard_normal((n,r)); m=measure(X.sample(Z,rng),Y.sample(Z,rng),rng,rbf=True)
    print(beta,d,'pop',round(pop_cka(X,Y),3),'PRx',round(part_ratio(X)),'PRy',round(part_ratio(Y)),
      {k:round(v,3) for k,v in m.items()}, 'pred_gap',round(predicted_gap(0.3,part_ratio(X),part_ratio(Y),n),3),'gap',round(m['raw']-m['null_mu'],3),'t',round(time.time()-t,1))
