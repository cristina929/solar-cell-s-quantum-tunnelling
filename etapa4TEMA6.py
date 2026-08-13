import numpy as np
import matplotlib.pyplot as plt

# 1. CONSTANTES E INPUTS (Identicos)
q = 1.60217663e-19  # carga elemental [C]
eps0 = 8.85418781e-12  # permitividad del vacío [F/m]
kB = 1.380649e-23  # constante de Boltzmann [J/K]
T = 300  # temperatura [K]
Vt = (kB * T) / q  # voltaje térmico ≈ 25.85 mV
eps_Si = 11.9 * eps0  # permitividad del silicio
W_emisor = 0.5e-6  # espesor del emisor p+ [m] = 0.5 µm
W_base = 160.e-6  # espesor de la base n [m] = 160 µm
W_total = W_emisor + W_base  # espesor total
Na = 1e19 * 1e6  # aceptores en el emisor p+ [m^-3]
Nd = 1e15 * 1e6  # donores en la base n [m^-3]
mu_n = 1400e-4; mu_p = 450e-4  # movilidades [m²/V·s]
Dn = mu_n * Vt; Dp = mu_p * Vt  # difusividades (Einstein)
Nc, Nv, Eg = 2.8e19 * 1e6, 1.04e19 * 1e6, 1.12  # densidades de estados y gap
ni = np.sqrt(Nc * Nv * np.exp(-Eg / Vt))  # concentración intrínseca
tau_n_val = 1e-3; tau_p_val = 1e-3  # tiempos de vida SRH [s]
# Longitudes de difusión (Ecuaciones de baja inyección)
Ln = np.sqrt(Dn * tau_n_val)  # longitud de difusión de electrones [m]
Lp = np.sqrt(Dp * tau_p_val)  # longitud de difusión de huecos [m]

# 2. MALLA
x = np.unique(np.concatenate([
    np.linspace(0, W_emisor, 120),  # más nodos en el emisor fino
    np.linspace(W_emisor, W_total, 280)]))  # malla en la base
N = len(x)

# 3. ÓPTICA (Identica a tu script)
hc_eV = 1239.997 # nm*eV  # constante hc para λg = hc/Eg
from optica_scaps import leer_abs, cargar_optica
_lam_abs, _al_abs = leer_abs()  # tabla α(λ) de SCAPS
def alpha_scaps_m(lam_nm):
    # Interpola el coeficiente de absorción α [1/m] a la λ pedida
    lam_nm = np.atleast_1d(lam_nm).astype(float)
    return np.interp(lam_nm, _lam_abs, _al_abs, left=_al_abs[0], right=0.0)
_opt = cargar_optica()
lam = _opt['lam'] # nm  # longitudes de onda del espectro AM1.5G
al_lam = _opt['alpha'] # 1/m  # α(λ) en cada bin
Phi_bin = _opt['Phi_bin'] # fotones/m2/s  # flujo de fotones por bin
R_frontal = 0.10  # reflexión frontal del 10 %
# G(x) de banda ancha (Beer-Lambert sumado en todos los bins)
G_broad = (1.0 - R_frontal) * (np.exp(-np.outer(x, al_lam)) * (al_lam * Phi_bin)).sum(axis=1)

# 4. SOLVER ANALÍTICO DE DIFUSIÓN (Probabilidad de colección eta_col)
# Función de transferencia / Eficiencia de recolección local eta_col(x)
# Se deriva directamente de la ecuación de difusión analítica en baja inyección
eta_col = np.zeros_like(x)
m_emisor = x < W_emisor
m_base = x >= W_emisor
# Región P (Emisor, x < W_emisor): solución cosh para contacto óhmico en x=0
eta_col[m_emisor] = np.cosh(x[m_emisor] / Lp) / np.cosh(W_emisor / Lp)
# Región N (Base, x >= W_emisor): decaimiento exponencial desde la unión
eta_col[m_base] = np.exp(-(x[m_base] - W_emisor) / Ln)
def resolver_difusion_analitica(G_perfil):
    """
    Calcula Jsc integrando G(x) ponderado por la probabilidad
    de colección de portadores eta_col(x).
    """
    # Integración trapezoidal de q * G(x) * eta_col(x) dx
    Jsc_m2 = q * np.trapezoid(G_perfil * eta_col, x)
    return Jsc_m2 * 1e-4 # Convertido a A/cm2

# 5. REFERENCIA: Jsc de banda ancha (V=0)
print("Jsc de banda ancha (referencia)...")
Jsc_broad = resolver_difusion_analitica(G_broad) # A/cm2  # Jsc con todo el espectro AM1.5G
print(f" Jsc (banda ancha) = {Jsc_broad*1e3:.3f} mA/cm2\n")

# 6. ETAPA 4: EQE por barrido monocromático (V=0)
print("Barrido EQE(lambda)...")
lam_eqe = np.linspace(300., 1100., 41) # nm  # 41 longitudes de onda
Phi_mono = 1e19 # fotones/m2/s  # flujo monocromático de sonda
den_eqe = q * Phi_mono * 1e-4 # A/cm2 incidentes  # denominador de la EQE
EQE = np.zeros_like(lam_eqe)
for k, lam0 in enumerate(lam_eqe):
    a_m = alpha_scaps_m(lam0)[0]  # α(λ) interpolado
    G_mono = (1.0 - R_frontal) * a_m * Phi_mono * np.exp(-a_m * x)  # generación monocromática
    Jk = resolver_difusion_analitica(G_mono)  # Jsc monocromática
    EQE[k] = Jk / den_eqe  # EQE = Jsc(λ) / (q·Φ)
    print(f" lambda={lam0:6.0f} nm EQE={EQE[k]*100:6.2f} %")
IQE = EQE / (1.0 - R_frontal)  # eficiencia cuántica interna (corrige la reflexión)

# 7. TEST DE CONSISTENCIA
# Si el modelo es correcto, ∫ EQE(λ)·Φ(λ) dλ debe recuperar Jsc de banda ancha
EQE_bins = np.interp(lam, lam_eqe, EQE)  # interpola EQE sobre los bins del espectro
Jsc_recon = q * (EQE_bins * Phi_bin).sum() * 1e-4 * 1e3 # mA/cm2  # Jsc reconstruida
print("\n" + "="*56)
print("TEST DE CONSISTENCIA INTERNO")
print("="*56)
print(f" Jsc banda ancha (solver directo) : {Jsc_broad*1e3:7.3f} mA/cm2")
print(f" Jsc reconstruido (EQE x AM1.5G) : {Jsc_recon:7.3f} mA/cm2")
err = (Jsc_recon - Jsc_broad*1e3)/(Jsc_broad*1e3)*100  # discrepancia relativa [%]
print(f" Discrepancia : {err:+7.2f} %")
print("="*56)

# 8. GRÁFICA DE RESULTADOS
fig, ax = plt.subplots(figsize=(8, 5.2))
ax.plot(lam_eqe, EQE*100, 'o-', color='navy', lw=2, ms=4, label='EQE (Modelo Difusión Analítico)')
ax.plot(lam_eqe, IQE*100, '--', color='gray', lw=1.2, label='IQE = EQE/(1-R)')
ax.axhline((1-R_frontal)*100, color='crimson', ls=':', lw=1, label=f'techo 1-R = {(1-R_frontal)*100:.0f}%')
ax.axvline(hc_eV/1.12, color='green', ls=':', lw=1, label=r'$\lambda_g$ = 1107 nm')
ax.set_xlabel(r'$\lambda$ (nm)')
ax.set_ylabel('Eficiencia cuántica (%)')
ax.set_ylim(0, 100)
ax.set_xlim(300, 1150)
ax.set_title(f'Etapa 4: EQE(λ) Analítico | Jsc {Jsc_broad*1e3:.2f} vs {Jsc_recon:.2f} mA/cm² ({err:+.1f}%)')
ax.grid(True, ls=':')
ax.legend(loc='lower center', fontsize=9)
plt.tight_layout()
plt.savefig('etapa4_eqe_analitico.png', dpi=150)
plt.show()