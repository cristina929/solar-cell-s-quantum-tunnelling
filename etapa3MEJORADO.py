import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
# 1. CONSTANTES Y PARAMETROS
q=1.60217663e-19; eps0=8.85418781e-12; kB=1.380649e-23; T=300.0  # carga elemental, ε0, Boltzmann, T
Vt=kB*T/q; eps_Si=11.9*eps0  # voltaje térmico ≈25.85 mV; permitividad del Si
W_emisor=0.5e-6; W_base=160e-6; W_total=W_emisor+W_base  # geometría: emisor 0.5 µm + base 160 µm
Na=1e19*1e6 # m^-3, emisor p+ (IZQUIERDA)
Nd=1e15*1e6 # m^-3, base n (DERECHA)
mu_n=1400e-4; mu_p=450e-4 # m^2/V.s  # movilidades de electrones y huecos
Dn=mu_n*Vt; Dp=mu_p*Vt  # difusividades (Einstein: D = µ·Vt)
Nc,Nv,Eg=2.8e19*1e6,1.04e19*1e6,1.12  # densidades de estados y gap del Si
ni=np.sqrt(Nc*Nv*np.exp(-Eg/Vt))  # concentración intrínseca a 300 K
tau_n=1e-3; tau_p=1e-3  # tiempos de vida SRH (1 ms → recombinación volumétrica baja)
# 2. MALLA GRADUADA
def malla_graduada(W_e,W_b,dx_min=2e-9,r=1.10):
    """Refinamiento geometrico a ambos lados de la union."""
    # dx_min=2 nm cerca de la unión; r=1.10 → el paso crece un 10 % cada nodo
    seg=[];d=dx_min;s=0.0
    while s<W_e: seg.append(min(d,W_e-s)); s+=seg[-1]; d*=r  # pasos del emisor
    x_em=np.concatenate([[0.0],np.cumsum(seg[::-1])]); x_em=x_em[x_em<=W_e]  # invierte para refinar junto a la unión
    seg=[];d=dx_min;s=0.0
    while s<W_b: seg.append(min(d,W_b-s)); s+=seg[-1]; d*=r  # pasos de la base
    return np.unique(np.concatenate([x_em,[W_e],W_e+np.cumsum(seg)]))  # une emisor + unión + base
x=malla_graduada(W_emisor,W_base); N=len(x)  # vector de nodos y número total
dx=np.diff(x); dx_prom=(dx[:-1]+dx[1:])/2.0  # anchos de arista y de control de volumen
N_dopaje=np.where(x<W_emisor,-Na,Nd)  # perfil de dopaje: -Na (p+) | +Nd (n)
Vbi=Vt*np.log(Na*Nd/ni**2)  # potencial built-in de la unión
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
    """
    Poisson con portadores expresados via cuasi-niveles de Fermi.
    phi_n, phi_p se congelan; n(psi) y p(psi) se recalculan en cada
    iteracion de Newton, y el jacobiano incluye d(p-n)/dpsi = -(n+p)/Vt.
    """
    phi_n=psi-Vt*np.log(np.maximum(n,1.0)/ni)  # cuasi-Fermi de electrones
    phi_p=psi+Vt*np.log(np.maximum(p,1.0)/ni)  # cuasi-Fermi de huecos
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
               -(q/eps_Si)*(nn[i]+pp[i])/Vt) # <-- termino clave
        up[i]=1/(dx[i]*dx_prom[i-1]); lo[i-1]=1/(dx[i-1]*dx_prom[i-1])
        F[0]=psi[0]-psi_p; F[-1]=psi[-1]-(psi_n-V)  # BC Dirichlet (sesgo en el contacto derecho)
        d=spsolve(diags([dg,lo,up],[0,-1,1],format='csr'),-F)  # resuelve J·Δψ = -F
        d=np.clip(d,-2*Vt,2*Vt) # amortiguacion  # evita saltos > 2Vt
        psi=psi+d
        if np.max(np.abs(d))<1e-10: break  # convergencia alcanzada
    return psi
# 5. CONTINUIDAD (Scharfetter-Gummel) + BC OHMICA (CORRECCION 3)
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
                                 # (sin exp: el sesgo va solo en el potencial)
    return np.maximum(spsolve(diags([dg,lo,up],[0,-1,1],format='csr'),Rhs),1.0)
def perfil_J(psi,n,p):
    """Perfil J(x) en A/cm^2 por el esquema Scharfetter-Gummel."""
    d=np.diff(psi)/Vt
    Jn= q*(Dn/dx)*(n[1:]*Bernoulli(d)-n[:-1]*Bernoulli(-d))  # corriente de electrones
    Jp=-q*(Dp/dx)*(p[1:]*Bernoulli(-d)-p[:-1]*Bernoulli(d))  # corriente de huecos
    return (Jn+Jp)*1e-4  # A/m² → A/cm²
# ETAPA 3 — OPTICA ESPECTRAL, J-V OSCURA Y Jsc ILUMINADA
_trapz=getattr(np,"trapezoid",getattr(np,"trapz",None))  # compatibilidad numpy antiguo/nuevo
from optica_scaps import cargar_optica
_opt=cargar_optica()
lam=_opt['lam']; alpha_l=_opt['alpha']; Phi_bin=_opt['Phi_bin']  # espectro AM1.5G en bins
R_frontal=0.10  # reflexión frontal del 10 %
Phi_tot=Phi_bin.sum()  # flujo total de fotones sobre el gap
Jsc_techo=q*Phi_tot*1e-4*1e3  # techo óptico sin reflexión [mA/cm²]
# Generación óptica de banda ancha (Beer-Lambert sumado en todos los bins)
G_optica=(1.0-R_frontal)*(np.exp(-np.outer(x,alpha_l))*(alpha_l*Phi_bin)).sum(axis=1)
Jsc_abs=q*_trapz(G_optica,x)*1e-4*1e3  # generación absorbida con R [mA/cm²]
print("\nDATOS OPTICOS (ficheros nativos de SCAPS):")
print(f" Flujo de fotones sobre el gap : {Phi_tot:.3e} /m2/s")
print(f" Techo optico Jsc (sin R) : {Jsc_techo:.2f} mA/cm2")
print(f" Generacion absorbida (con R) : {Jsc_abs:.2f} mA/cm2")
def corriente_terminal(psi,n,p):
    return np.mean(perfil_J(psi,n,p)) # media, no el valor en x=0
def estado_inicial():
    # Estado de equilibrio térmico (sin sesgo ni luz)
    return (np.where(x<W_emisor,psi_p,psi_n).astype(float),
            np.where(N_dopaje>0,Nd,(ni**2)/Na).astype(float),
            np.where(N_dopaje<0,Na,(ni**2)/Nd).astype(float))
def gummel(psi,n,p,V,G,max_it=300,tol=1e-9):
    # Bucle de Gummel: Poisson → n → p hasta convergencia relativa
    for it in range(max_it):
        n_o,p_o=n.copy(),p.copy()
        psi=solve_poisson(psi,n,p,V)
        n=solve_n(psi,n,p,V,G); p=solve_p(psi,n,p,V,G)
        err=max(np.max(np.abs(n-n_o)/np.maximum(n,1e12)),
                np.max(np.abs(p-p_o)/np.maximum(p,1e12)))
        if err<tol: break
    return psi,n,p,it
# ── PARTE A: curva oscura -> J0 y n ────────────────────────────────
print("\nPARTE A: curva oscura para extraer J0 y factor de idealidad...")
voltajes_osc=np.arange(0.0,0.601,0.02)  # 0 → 0.60 V en pasos de 20 mV
G_cero=np.zeros(N)  # sin generación (oscuridad)
psi,n,p=estado_inicial(); J_osc=[]
for V in voltajes_osc:
    psi,n,p,it=gummel(psi,n,p,V,G_cero)  # resuelve en cada polarización
    J_osc.append(abs(corriente_terminal(psi,n,p)))
J_osc=np.array(J_osc)
# Ajuste semilogarítmico en la zona limpia (0.44–0.56 V) para extraer n y J0
mask=(voltajes_osc>=0.44)&(voltajes_osc<=0.56)&(J_osc>0)
coef=np.polyfit(voltajes_osc[mask],np.log(J_osc[mask]),1)
n_id=1.0/(Vt*coef[0]); J0=np.exp(coef[1])  # factor de idealidad y corriente de saturación
# Comparación con la expresión analítica de Shockley (baja inyección)
Ln=np.sqrt(Dn*tau_n); Lp=np.sqrt(Dp*tau_p)
coth=lambda z: np.cosh(z)/np.sinh(z)
J0_an=(q*ni**2*(Dn/Ln)/Na*coth(W_emisor/Ln)
      +q*ni**2*(Dp/Lp)/Nd*coth(W_base/Lp))*1e-4
print(f" n = {n_id:.4f} [analitico 1.0000]")
print(f" J0 = {J0:.3e} A/cm2 [analitico {J0_an:.3e}]")
# ── PARTE B: iluminado a V=0 -> Jsc ────────────────────────────────
print("\nPARTE B: iluminado a V=0 para extraer Jsc recolectada...")
psi,n,p=estado_inicial()
psi,n,p,it=gummel(psi,n,p,0.0,G_optica,max_it=400)  # cortocircuito con generación
Jsc=abs(corriente_terminal(psi,n,p))
Jx=perfil_J(psi,n,p)
unif=(np.max(np.abs(Jx))-np.min(np.abs(Jx)))/abs(np.mean(Jx))  # uniformidad de J(x)
print(f" Jsc (recolectada) = {Jsc*1e3:.3f} mA/cm2 (iters={it}, unif={unif:.1e})")
print(f" Eficiencia de coleccion = {Jsc*1e3/Jsc_abs*100:.1f} %")
# ── PARTE C: BARRIDO J-V ILUMINADO COMPLETO ────────────────────────
# Igual que SCAPS: en CADA voltaje se resuelve el sistema completo
# (Poisson + continuidad de n + continuidad de p) con G(x) activa.
# No se usa superposicion ni la formula del diodo.
print("\nPARTE C: barrido J-V ILUMINADO completo (como SCAPS)...")
V_light = np.arange(0.0, 0.661, 0.02)  # 0 → 0.66 V en pasos de 20 mV
psi,n,p = estado_inicial()
J_light_dir, its, unifs = [], [], []
for V in V_light:
    psi,n,p,it = gummel(psi,n,p,V,G_optica,max_it=400)  # resuelve completo en cada V
    Jx = perfil_J(psi,n,p); Jm = np.mean(Jx)
    J_light_dir.append(Jm)
    its.append(it)
    unifs.append((np.max(np.abs(Jx))-np.min(np.abs(Jx)))/(abs(Jm)+1e-30))
J_light_dir = np.array(J_light_dir) # A/cm2 (signo: negativo = generacion)
print(f" {'V(V)':>6} {'J(mA/cm2)':>12} {'iters':>7} {'unif':>10}")
for V,J,it,u in zip(V_light, J_light_dir, its, unifs):
    if abs(V*100 - round(V*100)) < 1e-6 and round(V*1000) % 100 == 0 or V>0.55:
        print(f" {V:6.2f} {J*1e3:12.4f} {it:7d} {u:10.1e}")
# Convenio generador: J positiva cuando la celula entrega corriente
J_gen = -J_light_dir if J_light_dir[0] < 0 else J_light_dir
# --- Voc: cruce por cero de la curva iluminada (interpolacion) ---
signo = np.sign(J_gen)
cruces = np.where(np.diff(signo) != 0)[0]
if len(cruces) > 0:
    k = cruces[0]
    Voc_dir = V_light[k] + (V_light[k+1]-V_light[k]) * J_gen[k]/(J_gen[k]-J_gen[k+1])
else:
    Voc_dir = np.nan
# --- MPP directo de la curva ---
P_dir = V_light * J_gen  # potencia
kmax = np.argmax(P_dir)
Pmax_dir, Vmpp_dir, Jmpp_dir = P_dir[kmax], V_light[kmax], J_gen[kmax]
Jsc_dir = J_gen[0]
FF_dir = Pmax_dir/(Jsc_dir*Voc_dir)*100 if np.isfinite(Voc_dir) else np.nan
eta_dir = Pmax_dir/0.1*100  # Pin = 0.1 W/cm² (1 sol)
# --- Superposicion (metodo anterior), para comparar ---
Voc_sup = n_id*Vt*np.log(Jsc/J0+1.0)  # Voc de la fórmula del diodo
Vg = np.linspace(0,Voc_sup,400)
Jl = Jsc - J0*(np.exp(Vg/(n_id*Vt))-1.0)  # J(V) por superposición
Ps = Vg*Jl; ks = np.argmax(Ps)
FF_sup = Ps[ks]/(Jsc*Voc_sup)*100
eta_sup = Ps[ks]/0.1*100
print("\n"+"="*70)
print("COMPARACION: barrido directo (como SCAPS) vs superposicion vs SCAPS")
print("="*70)
print(f"{'Magnitud':16} {'Directo':>11} {'Superpos.':>11} {'SCAPS':>10} {'dir-SCAPS':>11}")
print(f"{'Jsc (mA/cm2)':16} {Jsc_dir*1e3:11.3f} {Jsc*1e3:11.3f} {30.61:10.2f} "
      f"{(Jsc_dir*1e3/30.61-1)*100:+10.1f}%")
print(f"{'Voc (V)':16} {Voc_dir:11.4f} {Voc_sup:11.4f} {0.5755:10.4f} "
      f"{(Voc_dir/0.5755-1)*100:+10.1f}%")
print(f"{'FF (%)':16} {FF_dir:11.2f} {FF_sup:11.2f} {81.81:10.2f} "
      f"{(FF_dir/81.81-1)*100:+10.1f}%")
print(f"{'eta (%)':16} {eta_dir:11.2f} {eta_sup:11.2f} {14.41:10.2f} "
      f"{(eta_dir/14.41-1)*100:+10.1f}%")
print(f"{'Vmpp (V)':16} {Vmpp_dir:11.4f} {Vg[ks]:11.4f} {0.5000:10.4f}")
print("="*70)
# ── GRAFICAS ───────────────────────────────────────────────────────
fig,(ax0,ax1,ax2)=plt.subplots(1,3,figsize=(17,5))
ax0b=ax0.twinx()
ax0.semilogy(x*1e6,G_optica,color='orange',lw=2)  # perfil de generación
ax0b.semilogy(lam,alpha_l/100.0,color='steelblue',lw=1.5)  # α(λ)
ax0.axvline(W_emisor*1e6,color='gray',ls='--',lw=1)  # posición de la unión
ax0.set_xlabel(r'x ($\mu$m)'); ax0.set_ylabel(r'G(x) (m$^{-3}$s$^{-1}$)',color='orange')
ax0b.set_ylabel(r'$\alpha$ (cm$^{-1}$)',color='steelblue')
ax0.set_xlim(0,20); ax0.set_title('Optica espectral')
ax1.semilogy(voltajes_osc,J_osc,'o',color='gray',ms=4,label='Solver DD (oscura)')
Vfit=np.linspace(0.35,0.60,100)
ax1.semilogy(Vfit,J0*np.exp(Vfit/(n_id*Vt)),'--',color='crimson',lw=2,
             label=f'Ajuste (n={n_id:.3f})')
ax1.axvspan(0.44,0.56,alpha=0.20,color='gold',label='Ventana de ajuste')
ax1.set_xlabel('V (V)'); ax1.set_ylabel(r'|J| (A/cm$^2$)')
ax1.set_title('J-V oscura'); ax1.grid(True,which='both',ls=':'); ax1.legend(fontsize=8)
ax2.plot(V_light,J_gen*1e3,'o-',color='darkgreen',lw=2,ms=4,
         label='Barrido directo (como SCAPS)')
ax2.plot(Vg,Jl*1e3,'--',color='gray',lw=1.5,label='Superposicion')
ax2.axhline(0,color='k',lw=0.8)
ax2.plot([0],[Jsc_dir*1e3],'bo',ms=8,label=f'Jsc={Jsc_dir*1e3:.2f}')
ax2.plot([Voc_dir],[0],'r^',ms=8,label=f'Voc={Voc_dir:.3f} V')
ax2.plot([Vmpp_dir],[Jmpp_dir*1e3],'ks',ms=8,label='MPP')
ax2.set_xlabel('V (V)'); ax2.set_ylabel(r'J (mA/cm$^2$)')
ax2.set_title(f'J-V iluminada DIRECTA (FF={FF_dir:.1f}%, $\\eta$={eta_dir:.2f}%)')
ax2.grid(True,ls=':'); ax2.legend(fontsize=8); ax2.set_ylim(bottom=0)
plt.tight_layout(); plt.savefig('etapa3_espectral.png',dpi=150)
print("\nGrafica guardada: etapa3_espectral.png")
#actualizacion, para la celula base ohmica obtengo Voc de 0,5726 Jsc de 27,526922 FF de 81,74 y eta de 14,32, todo esto con R=0.1 , para que lo tengas en cuenta.