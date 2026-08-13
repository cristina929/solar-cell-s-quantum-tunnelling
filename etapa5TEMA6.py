import numpy as np
import matplotlib.pyplot as plt
import numpy as np

# 1. CONSTANTES E INPUTS (Idénticos)
q = 1.60217663e-19  # carga elemental [C]
eps0 = 8.85418781e-12  # permitividad del vacío [F/m]
kB = 1.380649e-23  # constante de Boltzmann [J/K]
T = 300.0  # temperatura [K]
Vt = kB * T / q  # voltaje térmico ≈ 25.85 mV
eps_Si = 11.9 * eps0  # permitividad del silicio
hbar = 1.054571817e-34  # h barra [J·s]
m0 = 9.1093837015e-31  # masa del electrón libre [kg]
kT_eV = kB * T / q  # kT en eV
W_emisor = 0.5e-6  # espesor del emisor p+ [m] = 0.5 µm
W_base = 160e-6  # espesor de la base n [m] = 160 µm
W_total = W_emisor + W_base
Na = 1e19 * 1e6  # aceptores en el emisor p+ [m^-3]
Nd = 1e15 * 1e6  # donores en la base n [m^-3]
mu_n = 1400e-4  # movilidad de electrones [m²/V·s]
mu_p = 450e-4  # movilidad de huecos [m²/V·s]
Dn = mu_n * Vt  # difusividad de electrones
Dp = mu_p * Vt  # difusividad de huecos
Nc = 2.8e19 * 1e6  # densidad de estados en la banda de conducción
Nv = 1.04e19 * 1e6  # densidad de estados en la banda de valencia
Eg = 1.12  # gap del silicio [eV]
ni = np.sqrt(Nc * Nv * np.exp(-Eg / Vt))  # concentración intrínseca
tau = 1e-3  # tiempo de vida SRH [s]
Ln = np.sqrt(Dn * tau)  # longitud de difusión de electrones [m]
Lp = np.sqrt(Dp * tau)  # longitud de difusión de huecos [m]
# Malla espacial
x = np.unique(np.concatenate([np.linspace(0, W_emisor, 120), np.linspace(W_emisor, W_total, 280)]))
# Óptica SCAPS
from optica_scaps import cargar_optica, G_de_x
R = 0.10  # reflexión frontal del 10 %
opt = cargar_optica()
G_broad = G_de_x(x, opt, R=R)  # generación óptica AM1.5G [m^-3 s^-1]

# 2. MODELO DE TÚNEL WKB (Capa de Óxido TOPCon)
def T_WKB(E_eV, F, Phi_B=3.1, d=1.2e-9, m_ox=0.3):
    # Probabilidad de transmisión túnel T(E) por WKB a través de SiOx
    # Phi_B = altura de barrera [eV], d = espesor [m], m_ox = masa efectiva
    E = np.atleast_1d(E_eV).astype(float)
    m = m_ox * m0
    out = np.ones_like(E)
    s_ = E < Phi_B  # solo túnel si E < barrera
    U0 = (Phi_B - E[s_]) * q  # altura de barrera [J]
    if abs(F) < 1.0:  # campo ≈ 0 → barrera rectangular
        g = (2 * d / hbar) * np.sqrt(2 * m * U0)
    else:  # campo finito → barrera trapezoidal / triangular
        s = q * abs(F)
        xt = U0 / s  # punto de salida clásico
        L = np.minimum(xt, d)
        UL = np.clip(U0 - s * L, 0, None)
        g = (4 * np.sqrt(2 * m)) / (3 * hbar * s) * (U0**1.5 - UL**1.5)
    out[s_] = np.exp(-g)
    return out
def v_t_tunnel(F=0.0, m_si=0.26, **kw):
    # Velocidad efectiva de túnel promediada térmicamente:
    # v_t = (1/4) · v_th · <T>
    E = np.linspace(0, 25 * kT_eV, 600)
    Tt = T_WKB(E, F, **kw)
    w = np.exp(-E / kT_eV)  # peso de Boltzmann
    Tb = np.trapezoid(Tt * w, E) / np.trapezoid(w, E)  # <T>
    vth = np.sqrt(8 * kB * T / (np.pi * m_si * m0))  # velocidad térmica
    return 0.25 * vth * Tb, Tb

# 3. SOLVER ANALÍTICO CON CONTORNO TRASERO S_rear
# Probabilidad de colección local (solución analítica de la ecuación de difusión)
eta_col = np.zeros_like(x)
m_emisor = x < W_emisor
m_base = x >= W_emisor
eta_col[m_emisor] = np.cosh(x[m_emisor] / Lp) / np.cosh(W_emisor / Lp)  # emisor: cosh
eta_col[m_base] = np.exp(-(x[m_base] - W_emisor) / Ln)  # base: exponencial
def jsc_de_analitico():
    # Jsc = q ∫ G(x) · eta_col(x) dx
    Jsc_m2 = q * np.trapezoid(G_broad * eta_col, x)
    return Jsc_m2 * 1e-4 # A/cm2
def J0_de_analitico(S_rear_m_s):
    """
    Saturación J0 analítica introduciendo S_rear (m/s) en la cara trasera.
    S_rear = infinity representa contacto óhmico sin pasivación.
    """
    # Término emisor (p+)
    J0_emisor = q * (Dp * ni**2) / (Lp * Na) * np.tanh(W_emisor / Lp)
   
    # Término base (n) con condición de frontera de velocidad S_rear
    if S_rear_m_s >= 1e6: # Óhmico / S = inf
        factor_base = np.tanh(W_base / Ln)
    else:
        # Fórmula general con velocidad de recombinación finita
        num = (S_rear_m_s * Ln / Dn) + np.tanh(W_base / Ln)
        den = 1.0 + (S_rear_m_s * Ln / Dn) * np.tanh(W_base / Ln)
        factor_base = num / den
       
    J0_base = q * (Dn * ni**2) / (Ln * Nd) * factor_base
    return (J0_emisor + J0_base) * 1e-4 # A/cm2
def parametros(v_rear, S_rear):
    # Extrae Jsc, J0, Voc, FF y eta a partir de S_rear
    Jsc = jsc_de_analitico()
    J0 = J0_de_analitico(S_rear)
    Voc = Vt * np.log(Jsc / J0 + 1.0)  # Voc del diodo ideal
    voc = Voc / Vt
    FF = (voc - np.log(voc + 0.72)) / (voc + 1.0)  # aproximación empírica de FF
    return Jsc * 1e3, J0, Voc, FF * 100, Jsc * Voc * FF / 0.1 * 100

# 4. EJECUCIÓN Y TABLA DE RESULTADOS
V_OHM = 1e7  # velocidad óhmica casi infinita [m/s]
print("Caso Jsc(mA) J0(A/cm2) Voc(V) FF(%) eta(%)")
print("-" * 64)
casos = [("Base ohmico (S=inf)", V_OHM, V_OHM, '#888')]
for Sp in [1e3, 1e2, 1e1, 1e0]:
    casos.append((f"TOPCon Sp={Sp:.0f}", V_OHM, Sp * 1e-2, None)) # cm/s a m/s
cols = ['#888', '#bcdfca', '#7fbf9b', '#3f9d6c', '#1f6b43']
curvas = []
for (nom, vr, Sr, _), c in zip(casos, cols):
    Jsc, J0, Voc, FF, eta = parametros(vr, Sr)
    curvas.append((nom, Jsc, J0, Voc, c))
    print(f"{nom:20s} {Jsc:6.2f} {J0:.2e} {Voc:.4f} {FF:5.1f} {eta:5.2f}")
# Evaluación del Túnel
vt, Tb = v_t_tunnel(Phi_B=3.1, d=1.2e-9, m_ox=0.3)  # velocidad de túnel y <T>
rho = (kB * T) / (q**2 * vt * Nd) * 1e4  # resistividad de contacto [Ω·cm²]
print(f"\nTunel 3.1eV/1.2nm/0.3m0: <T>={Tb:.2e} v_t={vt*100:.1f} cm/s rho_c={rho:.2f} ohm.cm2")

# 5. GRÁFICAS
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
# Transmisión WKB
E = np.linspace(0, 3.0, 300)
for Vox, c_ in [(0.0, 'navy'), (0.3, 'darkorange'), (0.6, 'crimson')]:
    a1.semilogy(E, T_WKB(E, Vox / 1.2e-9), color=c_, label=f'V_ox={Vox} V')
a1.set_xlabel('E sobre Ec (eV)')
a1.set_ylabel('T(E)')
a1.set_ylim(1e-12, 2)
a1.set_title('WKB SiOx 3.1 eV / 1.2 nm (trapez./triang.)')
a1.grid(True, which='both', ls=':')
a1.legend()
# Comparativa Voc
labs = [c[0].replace('TOPCon ', '').replace('Base ohmico (S=inf)', 'ohmico\nS=inf') for c in curvas]
vocs = [c[3] * 1e3 for c in curvas]
cs = [c[4] for c in curvas]
a2.bar(labs, vocs, color=cs)
for i, v in enumerate(vocs):
    a2.text(i, v + 1.5, f'{v:.0f}', ha='center', fontsize=9)
a2.set_ylabel('Voc (mV)')
a2.set_title('Voc vs pasivacion trasera Sp (cm/s)')
a2.set_ylim(min(vocs) - 25, max(vocs) + 15)
plt.tight_layout()
plt.savefig('etapa5_topcon_analitico.png', dpi=150)
plt.show()
# Verificación de corriente a V = 0.50 V
Jsc_val, J0_val, Voc_val, _, _ = parametros(V_OHM, 10 * 1e-2) # Sp = 10 cm/s -> 0.1 m/s
J_050 = (J0_val * (np.exp(0.50 / Vt) - 1.0) - Jsc_val * 1e-3) * 1e3 # mA/cm2
print(f"\nJ(0.50V) = {abs(J_050):.2f} mA/cm²")