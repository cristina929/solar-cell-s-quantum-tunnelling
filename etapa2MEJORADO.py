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

# ══ 5. CONTINUIDAD (Scharfetter-Gummel) + BC OHMICA (CORRECCION 3) ══
def solve_n(psi,n_old,p_old,V):
    dpsi=np.diff(psi)/Vt
    dg=np.ones(N); up=np.zeros(N-1); lo=np.zeros(N-1); Rhs=np.zeros(N)
    i=np.arange(1,N-1)
    Ai=(Dn/dx[i-1])*Bernoulli(-dpsi[i-1]); Bi=(Dn/dx[i-1])*Bernoulli(dpsi[i-1])
    Ad=(Dn/dx[i])*Bernoulli(-dpsi[i]);     Bd=(Dn/dx[i])*Bernoulli(dpsi[i])
    dg[i]=(Ad+Bi)/dx_prom[i-1]; up[i]=-Bd/dx_prom[i-1]; lo[i-1]=-Ai/dx_prom[i-1]
    R0,dR_dn,_=SRH_y_derivadas(n_old,p_old)
    dg[i]+=dR_dn[i]; Rhs[i]=-R0[i]+dR_dn[i]*n_old[i]
    dg[0],dg[-1]=1.0,1.0
    Rhs[0]=(ni**2)/Na            # emisor p+: electrones minoritarios
    Rhs[-1]=Nd                   # base n:   electrones mayoritarios
    return np.maximum(spsolve(diags([dg,lo,up],[0,-1,1],format='csr'),Rhs),1.0)

def solve_p(psi,n_new,p_old,V):
    dpsi=np.diff(psi)/Vt
    dg=np.ones(N); up=np.zeros(N-1); lo=np.zeros(N-1); Rhs=np.zeros(N)
    i=np.arange(1,N-1)
    Ai=(Dp/dx[i-1])*Bernoulli(dpsi[i-1]); Bi=(Dp/dx[i-1])*Bernoulli(-dpsi[i-1])
    Ad=(Dp/dx[i])*Bernoulli(dpsi[i]);     Bd=(Dp/dx[i])*Bernoulli(-dpsi[i])
    dg[i]=(Ad+Bi)/dx_prom[i-1]; up[i]=-Bd/dx_prom[i-1]; lo[i-1]=-Ai/dx_prom[i-1]
    R0,_,dR_dp=SRH_y_derivadas(n_new,p_old)
    dg[i]+=dR_dp[i]; Rhs[i]=-R0[i]+dR_dp[i]*p_old[i]
    dg[0],dg[-1]=1.0,1.0
    Rhs[0]=Na                    # emisor p+: huecos mayoritarios
    Rhs[-1]=(ni**2)/Nd           # base n: huecos minoritarios EN EQUILIBRIO
                                 # (sin exp: el sesgo va solo en el potencial)
    return np.maximum(spsolve(diags([dg,lo,up],[0,-1,1],format='csr'),Rhs),1.0)

def perfil_J(psi,n,p):
    """Perfil J(x) en A/cm^2 por el esquema Scharfetter-Gummel."""
    d=np.diff(psi)/Vt
    Jn= q*(Dn/dx)*(n[1:]*Bernoulli(d)-n[:-1]*Bernoulli(-d))
    Jp=-q*(Dp/dx)*(p[1:]*Bernoulli(-d)-p[:-1]*Bernoulli(d))
    return (Jn+Jp)*1e-4

# 6. BARRIDO EN TENSION CON CONTINUACION 
voltajes=np.arange(0.0,0.701,0.02)
psi=np.where(x<W_emisor,psi_p,psi_n).astype(float)
n=np.where(N_dopaje>0,Nd,(ni**2)/Na).astype(float)
p=np.where(N_dopaje<0,Na,(ni**2)/Nd).astype(float)

print("\nBarrido J-V en oscuridad (Gummel no lineal + Scharfetter-Gummel)")
print(f"{'V(V)':>7} {'J(A/cm2)':>13} {'iters':>7} {'unif. J(x)':>12}")
corrientes=[]
for V in voltajes:
    for it in range(300):
        n_o,p_o=n.copy(),p.copy()
        psi=solve_poisson(psi,n,p,V)
        n=solve_n(psi,n,p,V)
        p=solve_p(psi,n,p,V)
        err=max(np.max(np.abs(n-n_o)/np.maximum(n,1e12)),
                np.max(np.abs(p-p_o)/np.maximum(p,1e12)))
        if err<1e-9: break                     # criterio relativo coherente
    Jx=perfil_J(psi,n,p)
    J_t=np.mean(Jx)                            # media: mas estable que Jx[0]
    unif=(np.max(np.abs(Jx))-np.min(np.abs(Jx)))/(abs(J_t)+1e-30)
    corrientes.append(abs(J_t))
    print(f"{V:7.3f} {abs(J_t):13.4e} {it:7d} {unif:12.2e}")
corrientes=np.array(corrientes)

# ══ 7. EXTRACCION DE J0 Y n  (ventana 0.44-0.56 V) ══════════════════
# Ventana justificada: por debajo de 0.44 V domina el ruido numerico;
# por encima de 0.56 V empieza la alta inyeccion y n se aparta de 1.
mask=(voltajes>=0.44)&(voltajes<=0.56)&(corrientes>0)
coef=np.polyfit(voltajes[mask],np.log(corrientes[mask]),1)
n_id=1.0/(Vt*coef[0]); J0=np.exp(coef[1])

# Referencia analitica de dos regiones con contactos ohmicos (S -> inf)
Ln=np.sqrt(Dn*tau_n); Lp=np.sqrt(Dp*tau_p)
coth=lambda z: np.cosh(z)/np.sinh(z)
J0e=q*ni**2*(Dn/Ln)/Na*coth(W_emisor/Ln)*1e-4
J0b=q*ni**2*(Dp/Lp)/Nd*coth(W_base  /Lp)*1e-4
J0_an=J0e+J0b

print("\n"+"="*58)
print("PARAMETROS EXTRAIDOS  (ventana 0.44-0.56 V)")
print("="*58)
print(f"  Factor de idealidad  n  = {n_id:.4f}      [analitico 1.0000]")
print(f"  Corriente saturacion J0 = {J0:.4e} A/cm2")
print(f"  J0 analitico            = {J0_an:.4e} A/cm2")
print(f"    (emisor {J0e:.3e} + base {J0b:.3e})")
print(f"  Desviacion en J0        = {(J0/J0_an-1)*100:+.1f} %")
print("="*58)

# 8. GRAFICA 
J_fit=J0*np.exp(voltajes/(n_id*Vt))
plt.figure(figsize=(7,5))
plt.semilogy(voltajes,corrientes,'o',color='darkgreen',ms=6,
             label='Solver DD (Gummel no lineal + SG)')
plt.semilogy(voltajes,J_fit,'--',color='crimson',lw=2,
             label=f'Ajuste ideal (n = {n_id:.3f})')
plt.axvspan(0.44,0.56,color='gold',alpha=0.25,label='Ventana de ajuste')
plt.xlabel('Voltaje aplicado V (V)')
plt.ylabel(r'Densidad de corriente |J| (A/cm$^2$)')
plt.title('Etapa 2: curva J-V en oscuridad')
plt.grid(True,which='both',ls=':')
plt.legend(); plt.tight_layout()
plt.savefig('etapa2_JV_oscuridad.png',dpi=150)
print("\nGrafica guardada: etapa2_JV_oscuridad.png")