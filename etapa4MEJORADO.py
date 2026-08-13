import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
# 1. CONSTANTES Y PARAMETROS
q=1.60217663e-19; eps0=8.85418781e-12; kB=1.380649e-23; T=300.0  # carga elemental, permitividad vacío, Boltzmann, T
Vt=kB*T/q; eps_Si=11.9*eps0  # voltaje térmico ≈25.85 mV; permitividad del silicio
W_emisor=0.5e-6; W_base=160e-6; W_total=W_emisor+W_base  # geometría: emisor 0.5 µm + base 160 µm
Na=1e19*1e6 # m^-3, emisor p+ (IZQUIERDA)
Nd=1e15*1e6 # m^-3, base n (DERECHA)
mu_n=1400e-4; mu_p=450e-4 # m^2/V.s  # movilidades de electrones y huecos
Dn=mu_n*Vt; Dp=mu_p*Vt  # difusividades (relación de Einstein)
Nc,Nv,Eg=2.8e19*1e6,1.04e19*1e6,1.12  # densidades de estados y gap del Si
ni=np.sqrt(Nc*Nv*np.exp(-Eg/Vt))  # concentración intrínseca a 300 K
tau_n=1e-3; tau_p=1e-3  # tiempos de vida SRH (1 ms → recombinación volumétrica baja)
# 2. MALLA GRADUADA
def malla_graduada(W_e,W_b,dx_min=2e-9,r=1.10):
    #Refinamiento geometrico a ambos lados de la union.
    # dx_min=2 nm cerca de la unión; r=1.10 → el paso crece un 10 % cada nodo
    seg=[];d=dx_min;s=0.0
    while s<W_e: seg.append(min(d,W_e-s)); s+=seg[-1]; d*=r  # construye pasos del emisor
    x_em=np.concatenate([[0.0],np.cumsum(seg[::-1])]); x_em=x_em[x_em<=W_e]  # invierte para refinar junto a la unión
    seg=[];d=dx_min;s=0.0
    while s<W_b: seg.append(min(d,W_b-s)); s+=seg[-1]; d*=r  # construye pasos de la base
    return np.unique(np.concatenate([x_em,[W_e],W_e+np.cumsum(seg)]))  # une emisor + unión + base
x=malla_graduada(W_emisor,W_base); N=len(x)  # vector de nodos y número total
dx=np.diff(x); dx_prom=(dx[:-1]+dx[1:])/2.0  # anchos de arista y anchos de control de volumen
N_dopaje=np.where(x<W_emisor,-Na,Nd)  # perfil de dopaje: -Na (p+) | +Nd (n)
Vbi=Vt*np.log(Na*Nd/ni**2)  # potencial de contacto de la unión
W_zce=np.sqrt(2*eps_Si*Vbi/q*(1/Na+1/Nd))  # ancho analítico de la zona de carga espacial
print(f"ni = {ni*1e-6:.4e} cm^-3 Vbi = {Vbi*1e3:.1f} mV ZCE = {W_zce*1e6:.3f} um")
print(f"Malla: {N} nodos | dx_min = {dx.min()*1e9:.1f} nm | "
      f"dx_max = {dx.max()*1e6:.1f} um")
print(f"Nodos dentro de la ZCE: {np.sum((x>W_emisor)&(x<W_emisor+W_zce))}")
# 3. FUNCIONES AUXILIARES
def Bernoulli(z):
    # B(z)=z/(exp(z)-1). Serie de Taylor cerca de 0 para evitar 0/0.
    # Esencial en el esquema Scharfetter-Gummel de las corrientes.
    z=np.clip(z,-30,30); B=np.zeros_like(z); m=np.abs(z)<1e-5
    B[m]=1.0-z[m]/2.0+z[m]**2/12.0  # expansión de Taylor
    B[~m]=z[~m]/(np.exp(z[~m])-1.0)
    return B
def SRH_y_derivadas(n,p):
    # Recombinación Shockley-Read-Hall y derivadas analíticas ∂R/∂n, ∂R/∂p
    # (se usan para linealizar las ecuaciones de continuidad en Gummel)
    num=n*p-ni**2; den=tau_p*(n+ni)+tau_n*(p+ni)
    return num/den,(p*den-num*tau_p)/den**2,(n*den-num*tau_n)/den**2
psi_p=-Vt*np.log(Na/ni) # contacto izquierdo (emisor p+)
psi_n= Vt*np.log(Nd/ni) # contacto derecho (base n)
# 4. POISSON NO LINEAL — GUMMEL AUTENTICO
def solve_poisson(psi,n,p,V):
    # Resuelve d/dx(ε dψ/dx) = -q(p-n+Ndop) con Newton.
    # Los portadores se expresan vía cuasi-Fermi congelados; el Jacobiano
    # incluye el término -(n+p)/Vt → convergencia cuadrática.
    phi_n=psi-Vt*np.log(np.maximum(n,1.0)/ni)  # cuasi-nivel de Fermi de electrones
    phi_p=psi+Vt*np.log(np.maximum(p,1.0)/ni)  # cuasi-nivel de Fermi de huecos
    for k in range(60):  # máximo 60 iteraciones de Newton
        nn=ni*np.exp(np.clip((psi-phi_n)/Vt,-60,60))  # n(ψ) actualizado
        pp=ni*np.exp(np.clip((phi_p-psi)/Vt,-60,60))  # p(ψ) actualizado
        F=np.zeros(N); dg=np.ones(N); up=np.zeros(N-1); lo=np.zeros(N-1)
        i=np.arange(1,N-1)  # nodos interiores
        # Residuo: discretización centrada de d²ψ/dx² + carga espacial
        F[i]=((psi[i+1]-psi[i])/(dx[i]*dx_prom[i-1])
             -(psi[i]-psi[i-1])/(dx[i-1]*dx_prom[i-1]))
        F[i]+=(q/eps_Si)*(pp[i]-nn[i]+N_dopaje[i])
        # Jacobiano: derivada del residuo respecto a ψ (término clave de Newton)
        dg[i]=(-1/(dx[i]*dx_prom[i-1])-1/(dx[i-1]*dx_prom[i-1])
               -(q/eps_Si)*(nn[i]+pp[i])/Vt) # término clave
        up[i]=1/(dx[i]*dx_prom[i-1]); lo[i-1]=1/(dx[i-1]*dx_prom[i-1])
        F[0]=psi[0]-psi_p; F[-1]=psi[-1]-(psi_n-V)  # BC Dirichlet (sesgo en el contacto derecho)
        d=spsolve(diags([dg,lo,up],[0,-1,1],format='csr'),-F)  # resuelve J·Δψ = -F
        d=np.clip(d,-2*Vt,2*Vt) # amortiguacion  # evita saltos > 2Vt
        psi=psi+d
        if np.max(np.abs(d))<1e-10: break  # convergencia alcanzada
    return psi
# 5. CONTINUIDAD (Scharfetter-Gummel) + BC OHMICA
def solve_n(psi,n_old,p_old,V,G):
    # Continuidad de electrones: (1/q) dJn/dx = R - G
    # Discretización Scharfetter-Gummel + BC Dirichlet óhmicas.
    dpsi=np.diff(psi)/Vt
    dg=np.ones(N); up=np.zeros(N-1); lo=np.zeros(N-1); Rhs=np.zeros(N)
    i=np.arange(1,N-1)
    # Coeficientes de Scharfetter-Gummel en cada arista
    Ai=(Dn/dx[i-1])*Bernoulli(-dpsi[i-1]); Bi=(Dn/dx[i-1])*Bernoulli(dpsi[i-1])
    Ad=(Dn/dx[i])*Bernoulli(-dpsi[i]); Bd=(Dn/dx[i])*Bernoulli(dpsi[i])
    dg[i]=(Ad+Bi)/dx_prom[i-1]; up[i]=-Bd/dx_prom[i-1]; lo[i-1]=-Ai/dx_prom[i-1]
    R0,dR_dn,_=SRH_y_derivadas(n_old,p_old)  # recombinación + derivada respecto a n
    dg[i]+=dR_dn[i]; Rhs[i]=-R0[i]+dR_dn[i]*n_old[i]+G[i]  # linealización + generación
    dg[0],dg[-1]=1.0,1.0  # BC Dirichlet
    Rhs[0]=(ni**2)/Na # emisor p+: electrones minoritarios
    Rhs[-1]=Nd # base n: electrones mayoritarios
    return np.maximum(spsolve(diags([dg,lo,up],[0,-1,1],format='csr'),Rhs),1.0)
def solve_p(psi,n_new,p_old,V,G):
    # Continuidad de huecos (análoga a solve_n). BC Dirichlet óhmicas.
    dpsi=np.diff(psi)/Vt
    dg=np.ones(N); up=np.zeros(N-1); lo=np.zeros(N-1); Rhs=np.zeros(N)
    i=np.arange(1,N-1)
    Ai=(Dp/dx[i-1])*Bernoulli(dpsi[i-1]); Bi=(Dp/dx[i-1])*Bernoulli(-dpsi[i-1])
    Ad=(Dp/dx[i])*Bernoulli(dpsi[i]); Bd=(Dp/dx[i])*Bernoulli(-dpsi[i])
    dg[i]=(Ad+Bi)/dx_prom[i-1]; up[i]=-Bd/dx_prom[i-1]; lo[i-1]=-Ai/dx_prom[i-1]
    R0,_,dR_dp=SRH_y_derivadas(n_new,p_old)
    dg[i]+=dR_dp[i]; Rhs[i]=-R0[i]+dR_dp[i]*p_old[i]+G[i]
    dg[0],dg[-1]=1.0,1.0
    Rhs[0]=Na # emisor p+: huecos mayoritarios
    Rhs[-1]=(ni**2)/Nd # base n: huecos minoritarios EN EQUILIBRIO
                                
    return np.maximum(spsolve(diags([dg,lo,up],[0,-1,1],format='csr'),Rhs),1.0)
def perfil_J(psi,n,p):
    #Perfil J(x) en A/cm^2 por el esquema Scharfetter-Gummel.
    d=np.diff(psi)/Vt
    Jn= q*(Dn/dx)*(n[1:]*Bernoulli(d)-n[:-1]*Bernoulli(-d))  # corriente de electrones
    Jp=-q*(Dp/dx)*(p[1:]*Bernoulli(-d)-p[:-1]*Bernoulli(d))  # corriente de huecos
    return (Jn+Jp)*1e-4  # A/m² → A/cm²
# ETAPA 4 — EFICIENCIA CUANTICA EXTERNA EQE(lambda)
_trapz=getattr(np,"trapezoid",getattr(np,"trapz",None))  # compatibilidad numpy antiguo/nuevo
hc_eV=1239.997  # constante hc en eV·nm (para λg = hc/Eg)
from optica_scaps import leer_abs, cargar_optica
_lam_abs,_al_abs=leer_abs()  # tabla α(λ) de SCAPS
def alpha_scaps_m(lam_nm):
    # Interpola el coeficiente de absorción α [1/m] a la λ pedida
    lam_nm=np.atleast_1d(lam_nm).astype(float)
    return np.interp(lam_nm,_lam_abs,_al_abs,left=_al_abs[0],right=0.0)
_opt=cargar_optica()
lam=_opt['lam']; al_lam=_opt['alpha']; Phi_bin=_opt['Phi_bin']  # espectro AM1.5G en bins
R_frontal=0.10  # reflexión frontal del 10 %
# Generación óptica de banda ancha (Beer-Lambert sumado en todos los bins)
G_broad=(1.0-R_frontal)*(np.exp(-np.outer(x,al_lam))*(al_lam*Phi_bin)).sum(axis=1)
def corriente_terminal(psi,n,p):
    return np.mean(perfil_J(psi,n,p)) # J(x) uniforme -> media directa
def estado_inicial():
    # Estado de equilibrio térmico (sin sesgo ni luz)
    return (np.where(x<W_emisor,psi_p,psi_n).astype(float),
            np.where(N_dopaje>0,Nd,(ni**2)/Na).astype(float),
            np.where(N_dopaje<0,Na,(ni**2)/Nd).astype(float))
def gummel(psi,n,p,V,G,max_it=400,tol=1e-9):
    # Bucle de Gummel: Poisson → n → p hasta convergencia relativa
    for it in range(max_it):
        n_o,p_o=n.copy(),p.copy()
        psi=solve_poisson(psi,n,p,V)
        n=solve_n(psi,n,p,V,G); p=solve_p(psi,n,p,V,G)
        err=max(np.max(np.abs(n-n_o)/np.maximum(n,1e12)),
                np.max(np.abs(p-p_o)/np.maximum(p,1e12)))
        if err<tol: break
    return psi,n,p,it
def resolver_V0(G):
    #Cortocircuito (V=0) con criterio de convergencia real.
    # Devuelve |Jsc|, iteraciones y medida de uniformidad de J(x)
    psi,n,p=estado_inicial()
    psi,n,p,it=gummel(psi,n,p,0.0,G)
    Jx=perfil_J(psi,n,p)
    unif=(np.max(np.abs(Jx))-np.min(np.abs(Jx)))/(abs(np.mean(Jx))+1e-30)
    return abs(np.mean(Jx)),it,unif
# ── Estado oscuro V=0 convergido (arranque caliente por lambda) ─────
_psi0,_n0,_p0=estado_inicial()
_psi0,_n0,_p0,_it0=gummel(_psi0,_n0,_p0,0.0,np.zeros(N))  # resuelve una vez en oscuridad
print(f"Estado oscuro V=0 convergido en {_it0} iteraciones")
def resolver_warm(G):
    #V=0 partiendo del estado oscuro convergido.
    # Acelera mucho el barrido espectral (warm start)
    psi,n,p=_psi0.copy(),_n0.copy(),_p0.copy()
    psi,n,p,it=gummel(psi,n,p,0.0,G)
    return abs(np.mean(perfil_J(psi,n,p))),it
# ── 1. Referencia: Jsc de banda ancha ──────────────────────────────
print("\nJsc de banda ancha (referencia)...")
Jsc_broad,it_b,unif_b=resolver_V0(G_broad)  # Jsc con todo el espectro AM1.5G
print(f" Jsc (banda ancha) = {Jsc_broad*1e3:.3f} mA/cm2 "
      f"(iters={it_b}, unif={unif_b:.1e})")
# ── 2. Barrido EQE(lambda) ─────────────────────────────────────────
print("\nBarrido EQE(lambda)...")
lam_eqe=np.linspace(300.,1100.,41)  # 41 longitudes de onda de 300 a 1100 nm
Phi_mono=1e19 # sonda lineal de baja intensidad  # flujo monocromático [fotones/m²/s]
den_eqe=q*Phi_mono*1e-4  # denominador de la EQE [A/cm²]
EQE=np.zeros_like(lam_eqe); iters=np.zeros_like(lam_eqe,dtype=int)
for k,lam0 in enumerate(lam_eqe):
    a_m=alpha_scaps_m(lam0)[0]  # α(λ) interpolado
    G_mono=(1.0-R_frontal)*a_m*Phi_mono*np.exp(-a_m*x)  # generación monocromática Beer-Lambert
    Jk,itk=resolver_warm(G_mono)  # resuelve con warm start
    EQE[k]=Jk/den_eqe; iters[k]=itk  # EQE = Jsc(λ) / (q·Φ)
    if k%4==0 or lam0>1050:
        print(f" lambda={lam0:6.0f} nm EQE={EQE[k]*100:6.2f} % (iters={itk})")
IQE=EQE/(1.0-R_frontal)  # eficiencia cuántica interna (corrige la reflexión)
# ── 3. Test de consistencia: integrar EQE x AM1.5G ─────────────────
# Si el solver es correcto, ∫ EQE(λ)·Φ(λ) dλ debe recuperar Jsc de banda ancha
EQE_bins=np.interp(lam,lam_eqe,EQE)  # interpola EQE sobre los bins del espectro
Jsc_recon=q*(EQE_bins*Phi_bin).sum()*1e-4*1e3  # Jsc reconstruida [mA/cm²]
err=(Jsc_recon-Jsc_broad*1e3)/(Jsc_broad*1e3)*100  # discrepancia relativa [%]
print("\n"+"="*58)
print("TEST DE CONSISTENCIA INTERNO")
print("="*58)
print(f" Jsc banda ancha (solver directo) : {Jsc_broad*1e3:7.3f} mA/cm2")
print(f" Jsc reconstruido (EQE x AM1.5G) : {Jsc_recon:7.3f} mA/cm2")
print(f" Discrepancia : {err:+7.2f} %")
print(f" Iteraciones por lambda: min={iters.min()}, max={iters.max()}")
print("="*58)
# ── 4. Comparacion con SCAPS ───────────────────────────────────────
lam_s=np.arange(300,910,10.)
eqe_s=np.array([3.443,3.614,3.782,3.896,3.998,4.062,4.131,4.915,9.151,13.676,
 20.302,26.633,34.229,40.112,46.613,52.179,57.530,61.736,66.446,69.743,
 72.956,76.244,79.006,80.692,82.863,84.694,86.077,87.110,88.150,89.524,
 90.302,90.966,91.544,92.123,92.713,93.280,93.831,94.363,94.779,95.155,
 95.525,95.831,96.152,96.428,96.678,96.913,97.121,97.302,97.459,97.585,
 97.682,97.742,97.761,97.730,97.636,97.456,97.167,96.732,96.106,95.213,93.967])
print("\nCOMPARACION CON SCAPS")
print(f" {'lam(nm)':>8} {'propio':>8} {'SCAPS':>8} {'dif(pp)':>9}")
for l in [300,400,500,600,700,800,900]:
    ep=np.interp(l,lam_eqe,EQE)*100; es=np.interp(l,lam_s,eqe_s)
    print(f" {l:8.0f} {ep:8.2f} {es:8.2f} {ep-es:+9.1f}")
# ── 5. Grafica ─────────────────────────────────────────────────────
fig,ax=plt.subplots(figsize=(8,5.2))
ax.plot(lam_eqe,EQE*100,'o-',color='navy',lw=2,ms=4,label='EQE (solver propio)')
ax.plot(lam_s,eqe_s,'s-',color='orange',ms=3,lw=1.5,alpha=0.8,label='EQE SCAPS')
ax.plot(lam_eqe,IQE*100,'--',color='gray',lw=1.2,label='IQE = EQE/(1-R)')
ax.axhline((1-R_frontal)*100,color='crimson',ls=':',lw=1,
           label=f'techo 1-R = {(1-R_frontal)*100:.0f}%')
ax.axvline(hc_eV/1.12,color='green',ls=':',lw=1,label=r'$\lambda_g$ = 1107 nm')
ax.set_xlabel(r'$\lambda$ (nm)'); ax.set_ylabel('Eficiencia cuantica (%)')
ax.set_ylim(0,100); ax.set_xlim(300,1150)
ax.set_title(f'Etapa 4: EQE($\\lambda$) | test Jsc: '
             f'{Jsc_broad*1e3:.2f} vs {Jsc_recon:.2f} mA/cm2 ({err:+.1f}%)')
ax.grid(True,ls=':'); ax.legend(loc='lower center',fontsize=8)
plt.tight_layout(); plt.savefig('etapa4_eqe.png',dpi=150)
print("\nGráfica guardada: etapa4_eqe.png")