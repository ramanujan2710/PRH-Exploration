import numpy as np, time
from e1lib import *
rng=np.random.default_rng(1)
for alpha in [0.0,1.0]:
    dx,dy,r,rp=256,512,32,32
    A,C,px=make_side(rng,dx,r,rp,alpha); B,P,py=make_side(rng,dy,r,rp,alpha)
    Dx=0.5*px; s=tune_noise(0.3,A,C,Dx,B,P,py); Dy=s**2*py
    print('alpha',alpha,'pop',round(pop_cka(A,C,Dx,B,P,Dy),4))
    # large-n empirical (feature-space linear CKA ~ population)
    n=60000; Z=rng.standard_normal((n,r))
    X=sample_side(rng,n,Z,A,C,Dx); Y=sample_side(rng,n,Z,B,P,Dy)
    X-=X.mean(0);Y-=Y.mean(0)
    Sxy=X.T@Y/n; Sxx=X.T@X/n; Syy=Y.T@Y/n
    print('  empirical n=60k', round(np.sum(Sxy**2)/np.sqrt(np.sum(Sxx**2)*np.sum(Syy**2)),4))
    n=1024; Z=rng.standard_normal((n,r))
    X=sample_side(rng,n,Z,A,C,Dx); Y=sample_side(rng,n,Z,B,P,Dy)
    t=time.time(); m=measure(X,Y,rng); print('  n=1024 measure', {k:round(float(v),3) for k,v in m.items()}, f'{time.time()-t:.1f}s')
# timing at big width
dy=8192; A,C,px=make_side(rng,1024,32,32,1.0); B,P,py=make_side(rng,dy,32,32,1.0)
Z=rng.standard_normal((1024,32)); X=sample_side(rng,1024,Z,A,C,0.5*px); Y=sample_side(rng,1024,Z,B,P,py)
t=time.time(); m=measure(X,Y,rng); print('d=8192 time',f'{time.time()-t:.1f}s')
