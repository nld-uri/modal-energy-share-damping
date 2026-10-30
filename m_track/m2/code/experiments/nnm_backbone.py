import sys, json; import os; _H=os.path.dirname(os.path.abspath(__file__)); sys.path[:0]=[os.path.join(_H,'..','models'),os.path.join(_H,'..','identification'),os.path.join(_H,'..','experiments'),_H,os.path.join(_H,'..','..','experiments'),os.path.join(_H,'..','..','models'),os.path.join(_H,'..','..','identification')]
import numpy as np, lattice_nl as ln, bias_map as bm
from scipy.integrate import solve_ivp
m=bm.MODE; n=24
L=ln.Lattice(n=n,j0=bm.J0,k3=1.0,zeta1=bm.ZETA1)
W2,PSI=L.wn**2,L.psi
def f(t,z):
    a,v=z[:n],z[n:]; return np.concatenate([v,-W2*a-PSI*((PSI@a)**3)])
def fvar(t,Z):
    a=Z[:n]; v=Z[n:2*n]; Phi=Z[2*n:].reshape(2*n,2*n)
    J=np.zeros((2*n,2*n)); J[:n,n:]=np.eye(n); J[n:,:n]=-np.diag(W2)-3*((PSI@a)**2)*np.outer(PSI,PSI)
    return np.concatenate([v,-W2*a-PSI*((PSI@a)**3),(J@Phi).ravel()])
RT=dict(method='DOP853',rtol=1e-11,atol=1e-13)
def half(a0,T):
    s=solve_ivp(f,(0,T/2),np.concatenate([a0,np.zeros(n)]),**RT); return s.y[n:,-1]
def solve(Am,guess_a,guess_T):
    # unknowns: a (all but m) and T; a[m]=Am fixed; residual v(T/2)=0
    idx=[i for i in range(n) if i!=m]
    x=np.concatenate([guess_a[idx],[guess_T]])
    def unpack(x):
        a=np.zeros(n); a[idx]=x[:-1]; a[m]=Am; return a,x[-1]
    for it in range(30):
        a,T=unpack(x); r=half(a,T)
        if np.linalg.norm(r)<1e-11*max(1,Am): break
        Jm=np.zeros((n,n)); h=1e-7*max(Am,1e-3)
        for k in range(n):
            xp=x.copy(); xp[k]+=h if k<n-1 else 1e-7*T
            ap,Tp=unpack(xp); Jm[:,k]=(half(ap,Tp)-r)/(xp[k]-x[k])
        x=x-np.linalg.solve(Jm,r)
    a,T=unpack(x)
    return a,T,float(np.linalg.norm(half(a,T)))
out=[]; a=np.zeros(n); a[m]=1e-3; T=2*np.pi/L.wn[m]
for Am in np.concatenate([np.linspace(0.02,1.2,60)]):
    a[m]=Am; a,T,res=solve(Am,a,T)
    Z0=np.concatenate([a,np.zeros(n),np.eye(2*n).ravel()])
    s=solve_ivp(fvar,(0,T),Z0,dense_output=False,**RT); M=s.y[2*n:,-1].reshape(2*n,2*n)
    ev,evec=np.linalg.eig(M)
    tt=np.linspace(0,T,801); so=solve_ivp(f,(0,T),np.concatenate([a,np.zeros(n)]),t_eval=tt,**RT)
    w=PSI@so.y[:n]; kap=float(np.abs(w).max()**2)
    i=int(np.argmax(np.abs(ev))); v=evec[:,i]; e=np.abs(v[:n])**2*W2+np.abs(v[n:])**2; e=e/e.sum()
    top=sorted([(int(j+1),float(e[j])) for j in range(n)],key=lambda t:-t[1])[:2]
    out.append(dict(Am=float(Am),T=float(T),omega=float(2*np.pi/T),kappa=kap,residual=res,max_mult=float(np.abs(ev).max()),top=top,
                    share7=float((W2[m]*a[m]**2)/(W2@(a**2)))))
    print(f"A7 {Am:.3f} w {2*np.pi/T:.5f} kap {kap:.4f} res {res:.1e} |lam| {np.abs(ev).max():.6f} top {top}",flush=True)
json.dump(out,open(os.path.join(_H,'..','..','checkpoints','nnm.json'),'w'),indent=1)
