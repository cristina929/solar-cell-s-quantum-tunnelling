import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# 1. CONSTANTES Y PARAMETROS 
q=1.60217663e-19; eps0=8.85418781e-12; kB=1.380649e-23; T=300.0
Vt=kB*T/q; eps_Si=11.9*eps0

W_emisor=0.5e-6; W_base=160e-6; W_total=W_emisor+W_base
Na=1e19*1e6                      # m^-3, emisor p+ (IZQUIERDA)
Nd=1e15*1e6                      # m^-3, base n    (DERECHA)
mu_n=1400e-4; mu_p=450e-4        # m^2/V.s
Dn=mu_n*Vt; Dp=mu_p*Vt
Nc,Nv,Eg=2.8e19*1e6,1.04e19*1e6,1.12
ni=np.sqrt(Nc*Nv*np.exp(-Eg/Vt))
tau_n=1e-3; tau_p=1e-3

# 2. MALLA GRADUADA 
def malla_graduada(W_e,W_b,dx_min=2e-9,r=1.10):
    """Refinamiento geometrico a ambos lados de la union."""
    seg=[];d=dx_min;s=0.0
    while s<W_e: seg.append(min(d,W_e-s)); s+=seg[-1]; d*=r
    x_em=np.concatenate([[0.0],np.cumsum(seg[::-1])]); x_em=x_em[x_em<=W_e]
    seg=[];d=dx_min;s=0.0
    while s<W_b: seg.append(min(d,W_b-s)); s+=seg[-1]; d*=r
    return np.unique(np.concatenate([x_em,[W_e],W_e+np.cumsum(seg)]))

x=malla_graduada(W_emisor,W_base); N=len(x)
dx=np.diff(x); dx_prom=(dx[:-1]+dx[1:])/2.0
N_dopaje=np.where(x<W_emisor,-Na,Nd)

Vbi=Vt*np.log(Na*Nd/ni**2)
W_zce=np.sqrt(2*eps_Si*Vbi/q*(1/Na+1/Nd))
print(f"ni  = {ni*1e-6:.4e} cm^-3   Vbi = {Vbi*1e3:.1f} mV   ZCE = {W_zce*1e6:.3f} um")
print(f"Malla: {N} nodos | dx_min = {dx.min()*1e9:.1f} nm | "
      f"dx_max = {dx.max()*1e6:.1f} um")
print(f"Nodos dentro de la ZCE: {np.sum((x>W_emisor)&(x<W_emisor+W_zce))}")

# 3. FUNCIONES AUXILIARES 
def Bernoulli(z):
    z=np.clip(z,-30,30); B=np.zeros_like(z); m=np.abs(z)<1e-5
    B[m]=1.0-z[m]/2.0+z[m]**2/12.0
    B[~m]=z[~m]/(np.exp(z[~m])-1.0)
    return B

def SRH_y_derivadas(n,p):
    num=n*p-ni**2; den=tau_p*(n+ni)+tau_n*(p+ni)
    return num/den,(p*den-num*tau_p)/den**2,(n*den-num*tau_n)/den**2

psi_p=-Vt*np.log(Na/ni)          # contacto izquierdo (emisor p+)
psi_n= Vt*np.log(Nd/ni)          # contacto derecho  (base n)

# 4. POISSON NO LINEAL — GUMMEL AUTENTICO 
def solve_poisson(psi,n,p,V):
    """
    Poisson con portadores expresados via cuasi-niveles de Fermi.
    phi_n, phi_p se congelan; n(psi) y p(psi) se recalculan en cada
    iteracion de Newton, y el jacobiano incluye d(p-n)/dpsi = -(n+p)/Vt.
    """
    phi_n=psi-Vt*np.log(np.maximum(n,1.0)/ni)
    phi_p=psi+Vt*np.log(np.maximum(p,1.0)/ni)
    for k in range(60):
        nn=ni*np.exp(np.clip((psi-phi_n)/Vt,-60,60))
        pp=ni*np.exp(np.clip((phi_p-psi)/Vt,-60,60))
        F=np.zeros(N); dg=np.ones(N); up=np.zeros(N-1); lo=np.zeros(N-1)
        i=np.arange(1,N-1)
        F[i]=((psi[i+1]-psi[i])/(dx[i]*dx_prom[i-1])
             -(psi[i]-psi[i-1])/(dx[i-1]*dx_prom[i-1]))
        F[i]+=(q/eps_Si)*(pp[i]-nn[i]+N_dopaje[i])
        dg[i]=(-1/(dx[i]*dx_prom[i-1])-1/(dx[i-1]*dx_prom[i-1])
               -(q/eps_Si)*(nn[i]+pp[i])/Vt)          # <-- termino clave
        up[i]=1/(dx[i]*dx_prom[i-1]); lo[i-1]=1/(dx[i-1]*dx_prom[i-1])
        F[0]=psi[0]-psi_p; F[-1]=psi[-1]-(psi_n-V)
        d=spsolve(diags([dg,lo,up],[0,-1,1],format='csr'),-F)
        d=np.clip(d,-2*Vt,2*Vt)                       # amortiguacion
        psi=psi+d
        if np.max(np.abs(d))<1e-10: break
    return psi

# 5. CONTINUIDAD (Scharfetter-Gummel) + BC OHMICA 
def solve_n(psi,n_old,p_old,V,G,v_rear):
    """Continuidad de electrones. BC trasera: Robin con velocidad de tunel v_rear."""
    dpsi=np.diff(psi)/Vt
    dg=np.ones(N); up=np.zeros(N-1); lo=np.zeros(N-1); Rhs=np.zeros(N)
    i=np.arange(1,N-1)
    Ai=(Dn/dx[i-1])*Bernoulli(-dpsi[i-1]); Bi=(Dn/dx[i-1])*Bernoulli(dpsi[i-1])
    Ad=(Dn/dx[i])*Bernoulli(-dpsi[i]);     Bd=(Dn/dx[i])*Bernoulli(dpsi[i])
    dg[i]=(Ad+Bi)/dx_prom[i-1]; up[i]=-Bd/dx_prom[i-1]; lo[i-1]=-Ai/dx_prom[i-1]
    R0,dR_dn,_=SRH_y_derivadas(n_old,p_old)
    dg[i]+=dR_dn[i]; Rhs[i]=-R0[i]+dR_dn[i]*n_old[i]+G[i]
    dg[0]=1.0; Rhs[0]=(ni**2)/Na           # frontal ohmico
    BdL=Bernoulli(np.array([dpsi[-1]]))[0]
    BmL=Bernoulli(np.array([-dpsi[-1]]))[0]; dxh=dx[-1]/2
    dg[-1]=(Dn/dx[-1])*BdL/dxh + v_rear/dxh + dR_dn[-1]
    lo[-1]=-(Dn/dx[-1])*BmL/dxh
    Rhs[-1]=-R0[-1]+dR_dn[-1]*n_old[-1]+G[-1]+v_rear*Nd/dxh
    return np.maximum(spsolve(diags([dg,lo,up],[0,-1,1],format='csr'),Rhs),1.0)

def solve_p(psi,n_new,p_old,V,G,S_rear):
    """Continuidad de huecos. BC trasera: Robin con S_rear (pasivacion)."""
    dpsi=np.diff(psi)/Vt
    dg=np.ones(N); up=np.zeros(N-1); lo=np.zeros(N-1); Rhs=np.zeros(N)
    i=np.arange(1,N-1)
    Ai=(Dp/dx[i-1])*Bernoulli(dpsi[i-1]); Bi=(Dp/dx[i-1])*Bernoulli(-dpsi[i-1])
    Ad=(Dp/dx[i])*Bernoulli(dpsi[i]);     Bd=(Dp/dx[i])*Bernoulli(-dpsi[i])
    dg[i]=(Ad+Bi)/dx_prom[i-1]; up[i]=-Bd/dx_prom[i-1]; lo[i-1]=-Ai/dx_prom[i-1]
    R0,_,dR_dp=SRH_y_derivadas(n_new,p_old)
    dg[i]+=dR_dp[i]; Rhs[i]=-R0[i]+dR_dp[i]*p_old[i]+G[i]
    dg[0]=1.0; Rhs[0]=Na                   # frontal ohmico
    BdL=Bernoulli(np.array([dpsi[-1]]))[0]
    BmL=Bernoulli(np.array([-dpsi[-1]]))[0]; dxh=dx[-1]/2
    peq=(ni**2)/Nd                         # CORREGIDO: sin exp(V/Vt)
    dg[-1]=(Dp/dx[-1])*BmL/dxh + S_rear/dxh + dR_dp[-1]
    lo[-1]=-(Dp/dx[-1])*BdL/dxh
    Rhs[-1]=-R0[-1]+dR_dp[-1]*p_old[-1]+G[-1]+S_rear*peq/dxh
    return np.maximum(spsolve(diags([dg,lo,up],[0,-1,1],format='csr'),Rhs),1.0)

def perfil_J(psi,n,p):
    """Perfil J(x) en A/cm^2 por el esquema Scharfetter-Gummel."""
    d=np.diff(psi)/Vt
    Jn= q*(Dn/dx)*(n[1:]*Bernoulli(d)-n[:-1]*Bernoulli(-d))
    Jp=-q*(Dp/dx)*(p[1:]*Bernoulli(-d)-p[:-1]*Bernoulli(d))
    return (Jn+Jp)*1e-4


#  ETAPA 5 — CONTACTO TOPCon: TUNEL WKB + BARRIDO DIRECTO
hbar=1.054571817e-34; m0=9.1093837015e-31; kT_eV=kB*T/q
from optica_scaps import cargar_optica, G_de_x
R=0.10; opt=cargar_optica(); G_broad=G_de_x(x,opt,R=R)

def T_WKB(E_eV,F,Phi_B=3.1,d=1.2e-9,m_ox=0.3):
    E=np.atleast_1d(E_eV).astype(float); m=m_ox*m0
    out=np.ones_like(E); s_=E<Phi_B; U0=(Phi_B-E[s_])*q
    if abs(F)<1.0: g=(2*d/hbar)*np.sqrt(2*m*U0)
    else:
        s=q*abs(F); xt=U0/s; L=np.minimum(xt,d); UL=np.clip(U0-s*L,0,None)
        g=(4*np.sqrt(2*m))/(3*hbar*s)*(U0**1.5-UL**1.5)
    out[s_]=np.exp(-g); return out

def v_t_tunnel(F=0.0,m_si=0.26,**kw):
    E=np.linspace(0,25*kT_eV,600); Tt=T_WKB(E,F,**kw); w=np.exp(-E/kT_eV)
    Tb=np.trapezoid(Tt*w,E)/np.trapezoid(w,E)
    vth=np.sqrt(8*kB*T/(np.pi*m_si*m0))
    return 0.25*vth*Tb, Tb

def estado_inicial():
    return (np.where(x<W_emisor,psi_p,psi_n).astype(float),
            np.where(N_dopaje>0,Nd,(ni**2)/Na).astype(float),
            np.where(N_dopaje<0,Na,(ni**2)/Nd).astype(float))

def gummel(psi,n,p,V,G,v_rear,S_rear,max_it=400,tol=1e-9):
    for it in range(max_it):
        n_o,p_o=n.copy(),p.copy()
        psi=solve_poisson(psi,n,p,V)
        n=solve_n(psi,n,p,V,G,v_rear); p=solve_p(psi,n,p,V,G,S_rear)
        err=max(np.max(np.abs(n-n_o)/np.maximum(n,1e12)),
                np.max(np.abs(p-p_o)/np.maximum(p,1e12)))
        if err<tol: break
    return psi,n,p,it

V_OHM=1e7*1e-2      # cm/s -> m/s

def barrido_directo(v_rear,S_rear,Vmax=0.75,dV=0.02):
    """Barrido J-V ILUMINADO completo (como SCAPS): G(x) activa en cada V."""
    psi,n,p=estado_inicial(); Vs=np.arange(0.0,Vmax+1e-9,dV); Js=[]
    for V in Vs:
        psi,n,p,it=gummel(psi,n,p,V,G_broad,v_rear,S_rear)
        Js.append(np.mean(perfil_J(psi,n,p)))
    Js=np.array(Js); Jg=-Js if Js[0]<0 else Js
    Jsc=Jg[0]
    cr=np.where(np.diff(np.sign(Jg))!=0)[0]
    if len(cr):
        k=cr[0]; Voc=Vs[k]+(Vs[k+1]-Vs[k])*Jg[k]/(Jg[k]-Jg[k+1])
    else: Voc=np.nan
    P=Vs*Jg; km=np.argmax(P)
    FF=P[km]/(Jsc*Voc)*100 if np.isfinite(Voc) else np.nan
    return Jsc*1e3, Voc, FF, P[km]/0.1*100, Vs, Jg

print("Caso                  Jsc(mA)   Voc(V)   FF(%)  eta(%)")
print("-"*56)
casos=[("Base ohmico (S=inf)",V_OHM,V_OHM)]
for Sp in [1e3,1e2,1e1,1e0]:
    casos.append((f"TOPCon Sp={Sp:.0f}",V_OHM,Sp*1e-2))
res=[]
for nom,vr,Sr in casos:
    Jsc,Voc,FF,eta,Vs,Jg=barrido_directo(vr,Sr)
    res.append((nom,Jsc,Voc,FF,eta,Vs,Jg))
    print(f"{nom:20s} {Jsc:8.2f} {Voc:8.4f} {FF:7.1f} {eta:7.2f}")

vt,Tb=v_t_tunnel(Phi_B=3.1,d=1.2e-9,m_ox=0.3)
rho=(kB*T)/(q**2*vt*Nd)*1e4
print(f"\nTunel 3.1eV/1.2nm/0.3m0: <T>={Tb:.2e}  v_t={vt*100:.1f} cm/s  "
      f"rho_c={rho:.2f} ohm.cm2")

fig,(a1,a2)=plt.subplots(1,2,figsize=(12,4.8))
E=np.linspace(0,3.0,300)
for Vox,c_ in [(0.0,'navy'),(0.3,'darkorange'),(0.6,'crimson')]:
    a1.semilogy(E,T_WKB(E,Vox/1.2e-9),color=c_,label=f'$V_{{ox}}$={Vox} V')
a1.set_xlabel('E sobre Ec (eV)'); a1.set_ylabel('T(E)'); a1.set_ylim(1e-12,2)
a1.set_title('WKB SiOx 3.1 eV / 1.2 nm'); a1.grid(True,which='both',ls=':'); a1.legend()
cols=['#888','#bcdfca','#7fbf9b','#3f9d6c','#1f6b43']
for (nom,Jsc,Voc,FF,eta,Vs,Jg),c_ in zip(res,cols):
    a2.plot(Vs,Jg*1e3,'o-',ms=3,color=c_,label=f"{nom.replace('TOPCon ','')} ({Voc*1e3:.0f} mV)")
a2.axhline(0,color='k',lw=0.8); a2.set_ylim(bottom=0)
a2.set_xlabel('V (V)'); a2.set_ylabel(r'J (mA/cm$^2$)')
a2.set_title('J-V iluminada directa vs pasivacion trasera')
a2.grid(True,ls=':'); a2.legend(fontsize=7)
plt.tight_layout(); plt.savefig('etapa5_topcon.png',dpi=150)
print("\nGrafica guardada: etapa5_topcon.png")