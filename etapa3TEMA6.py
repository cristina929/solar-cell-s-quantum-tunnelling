import numpy as np
import matplotlib.pyplot as plt

# 1. DATOS SPECTRALES Y ABSORCIÓN (de las fuentes)
# Coeficiente de absorción del Silicio: lambda [nm], alpha [1/m]
raw_abs = """
260 2.10E08 270 2.21E08 280 2.35E08 290 2.13E08 300 1.65E08 310 1.44E08
320 1.28E08 330 1.19E08 340 1.12E08 350 1.08E08 360 1.04E08 370 7.32E07
380 2.82E07 390 1.70E07 400 1.07E07 410 7.80E06 420 5.73E06 430 4.63E06
440 3.70E06 450 3.06E06 460 2.54E06 470 2.18E06 480 1.82E06 490 1.59E06
500 1.38E06 510 1.18E06 520 1.02E06 530 9.27E05 540 8.10E05 550 7.15E05
560 6.45E05 570 5.94E05 580 5.43E05 590 4.77E05 600 4.40E05 610 4.09E05
620 3.82E05 630 3.55E05 640 3.28E05 650 3.02E05 660 2.77E05 670 2.53E05
680 2.34E05 690 2.17E05 700 2.00E05 710 1.86E05 720 1.71E05 730 1.58E05
740 1.46E05 750 1.342E05 760 1.234E05 770 1.133E05 780 1.039E05 790 9.51E04
800 8.69E04 810 7.92E04 820 7.21E04 830 6.55E04 840 5.94E04 850 5.36E04
860 4.83E04 870 4.34E04 880 3.89E04 890 3.47E04 900 3.08E04 910 2.72E04
920 2.39E04 930 2.09E04 940 1.82E04 950 1.57E04 960 1.34E04 970 1.14E04
980 9.51E03 990 7.90E03 1000 6400.0 1010 5100.1 1020 3900.9 1030 3000.2
1040 2200.6 1050 1600.3 1060 1100.1 1070 800.00 1080 600.20 1090 400.70
1100 300.50 1110 200.70 1120 200.00 1130 100.50 1140 100.01 1150 68.0
1160 42.00 1170 22.00 1180 6.5 1190 3.6 1200 2.3 1210 1.3
1220 0.77 1230 0.38 1240 0.15 1250 0.0
"""
# Espectro de luz AM1.5G: lambda [nm], Potencia [W/m2]
raw_spe = """
305.0 0.04800 310.0 0.23238 315.0 0.54450 320.0 0.90135 325.0 1.28742 330.0 1.88289
335.0 1.98457 340.0 2.15306 345.0 2.22299 350.0 3.65008 360.0 5.34591 370.0 6.54581
380.0 7.08628 390.0 7.57170 400.0 9.95916 410.0 11.44670 420.0 11.68187 430.0 11.15997
440.0 13.02812 450.0 15.09041 460.0 15.89996 470.0 15.91177 480.0 16.13227 490.0 15.53414
500.0 15.54154 510.0 15.70997 520.0 15.10392 530.0 15.60647 540.0 15.56661 550.0 23.28738
570.0 29.95176 590.0 28.43396 610.0 29.38973 630.0 28.80925 650.0 28.39838 670.0 27.29172
690.0 23.75072 710.0 17.68235 718.0 7.41233 724.0 11.80100 740.0 17.21923 753.0 10.77465
758.0 5.56304 763.0 3.79513 768.0 8.67959 780.0 17.84488 800.0 19.15106 816.0 10.61303
824.0 6.53929 832.0 7.26152 840.0 13.45886 860.0 19.43966 880.0 20.55956 905.0 13.59171
915.0 6.81306 925.0 4.97575 930.0 2.47678 937.0 2.53067 948.0 4.77279 965.0 8.20990
980.0 9.33587 994.0 21.93074 1040.0 26.39676 1070.0 18.50298 1100.0 10.41107 1120.0 2.49641
1130.0 1.45884 1137.0 2.72242 1161.0 6.96365 1180.0 8.60153 1200.0 12.00256 1235.0 20.93583
"""
# Carga de arreglos numéricos
data_abs = np.fromstring(raw_abs, sep=' ').reshape(-1, 2)  # [λ, α] del silicio
data_spe = np.fromstring(raw_spe, sep=' ').reshape(-1, 2)  # [λ, potencia] AM1.5G
lam_abs, alpha_vals = data_abs[:, 0], data_abs[:, 1]
lam_spe, P_vals = data_spe[:, 0], data_spe[:, 1]
# Interpola alpha sobre la malla del espectro AM1.5G
alpha_interp = np.interp(lam_spe, lam_abs, alpha_vals)  # α(λ) en los bins del espectro

# 2. CONSTANTES FÍSICAS Y PARÁMETROS DE LA CÉLULA (De tu script original)
q = 1.60217663e-19 # C  # carga elemental
kB = 1.380649e-23 # J/K  # constante de Boltzmann
T = 300 # K
Vt = (kB * T) / q # V (~0.02586 V)  # voltaje térmico
h = 6.62607015e-34 # J*s  # constante de Planck
c = 2.99792458e8 # m/s  # velocidad de la luz
# Dimensiones geométricas
W_emisor = 0.5e-6 # 0.5 um (región p+)
W_base = 160.0e-6 # 160 um (región n)
W_total = W_emisor + W_base
# Dopajes
Na = 1e19 * 1e6 # 1e19 cm^-3 -> m^-3  # aceptores en el emisor p+
Nd = 1e15 * 1e6 # 1e15 cm^-3 -> m^-3  # donores en la base n
# Movilidades y difusividades
mu_n = 1400e-4 # m^2/V/s  # movilidad de electrones
mu_p = 450e-4 # m^2/V/s  # movilidad de huecos
Dn = mu_n * Vt # Diff de electrones  # relación de Einstein
Dp = mu_p * Vt # Diff de huecos
# Tiempos de vida y Longitudes de difusión L = sqrt(D * tau)
tau_n = 1e-3 # s  # tiempo de vida de electrones
tau_p = 1e-3 # s  # tiempo de vida de huecos
Ln = np.sqrt(Dn * tau_n) # Longitud de difusión de electrones en emisor/base
Lp = np.sqrt(Dp * tau_p) # Longitud de difusión de huecos
# Concentración intrínseca de portadores
Nc, Nv, Eg = 2.8e19 * 1e6, 1.04e19 * 1e6, 1.12  # densidades de estados y gap
ni = np.sqrt(Nc * Nv * np.exp(-Eg / Vt))  # ni a 300 K
# Parámetros ópticos de superficie
R_frontal = 0.10 # 10% de reflexión en x = 0
# Convertir potencia W/m2 a flujo de fotones Phi_0 por bin espectral
E_photon = (h * c) / (lam_spe * 1e-9)  # energía de un fotón [J]
Phi_bin = P_vals / E_photon # fotones / m^2 / s  # flujo de fotones por bin

# 3. MESH Y GENERACIÓN ÓPTICA G(x) (Beer-Lambert)
x = np.linspace(0, W_total, 1000)  # malla uniforme de 1000 nodos
# G(x) = sum_lambda (1 - R) * alpha(lambda) * Phi(lambda) * exp(-alpha(lambda) * x)
expo = np.exp(-np.outer(x, alpha_interp))  # matriz de atenuación exp(-αx)
G_x = (1.0 - R_frontal) * np.sum(expo * (alpha_interp * Phi_bin), axis=1)  # generación [m^-3 s^-1]

# 4. RESOLUCIÓN DE ECUACIONES DE DIFUSIÓN DE BASE Y EMISOR
# A) Componente de corriente de fotogeneración J_L (Fotocorriente)
# Integración de la contribución espectral por fotodiodo plano (Capa neutra n/p)
def calcular_J_iluminacion():
    # En aproximación de juntura estrecha y baja inyección:
    # J_L = q * int_0^{W_total} G(x) * eta_collection(x) dx
    # La probabilidad de colección eta(x) para una juntura en x = W_emisor:
    eta_col = np.zeros_like(x)
   
    # Emisor (x < W_emisor): solución cosh para contacto óhmico en x=0
    m_emisor = x < W_emisor
    eta_col[m_emisor] = np.cosh(x[m_emisor] / Lp) / np.cosh(W_emisor / Lp)
   
    # Base (x >= W_emisor): decaimiento exponencial desde la unión
    m_base = x >= W_emisor
    eta_col[m_base] = np.exp(-(x[m_base] - W_emisor) / Ln)
   
    # Integración numérica
    J_L_val = q * np.trapezoid(G_x * eta_col, x) # A/m^2
    return J_L_val * 1e-4, eta_col # Retorna A/cm^2 y perfil
J_L, eta_col = calcular_J_iluminacion()  # fotocorriente y perfil de colección
# B) Corriente de saturación en obscuridad (J0) mediante teoría de Shockley
# J0 = J0_p (emisor) + J0_n (base)
J0_p = q * (Dp * ni**2) / (Lp * Na) * np.tanh(W_emisor / Lp)  # contribución del emisor p+
J0_n = q * (Dn * ni**2) / (Ln * Nd) * np.tanh(W_base / Ln)  # contribución de la base n
J0_total = (J0_p + J0_n) * 1e-4 # Convertido a A/cm^2

# 5. CURVA J-V Y EXTRACTION DE PARÁMETROS
Voc = Vt * np.log((J_L / J0_total) + 1.0)  # tensión de circuito abierto (diodo ideal)
V_grid = np.linspace(0, Voc, 500)  # malla de voltajes de 0 a Voc
# Ecuación ideal del diodo iluminado: J(V) = J_L - J0 * (exp(V / Vt) - 1)
J_grid = J_L - J0_total * (np.exp(V_grid / Vt) - 1.0) # A/cm^2
P_grid = V_grid * J_grid  # potencia
idx_mpp = np.argmax(P_grid)  # índice del punto de máxima potencia
Pmax = P_grid[idx_mpp]
Vmpp = V_grid[idx_mpp]
Jmpp = J_grid[idx_mpp]
Pin = 0.1 # W/cm2 (1 sol = 100 mW/cm2)
FF = (Pmax / (J_L * Voc)) * 100  # factor de forma [%]
eta = (Pmax / Pin) * 100  # eficiencia [%]

# 6. MOSTRAR RESULTADOS
print("==============================================================")
print(" RESULTADOS MODELO DE DIFUSIÓN EN BAJA INYECCIÓN (ANALÍTICO)")
print("==============================================================")
print(f" J0 (Saturación) : {J0_total:.3e} A/cm^2")
print(f" Jsc / J_L (Fototerm) : {J_L * 1e3:.3f} mA/cm^2")
print(f" Voc (Tensión abierto) : {Voc:.4f} V")
print(f" FF (Factor de Forma) : {FF:.2f} %")
print(f" Eficiencia (eta) : {eta:.2f} %")
print("==============================================================")

# 7. GRÁFICAS
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# Gráfica 1: Perfil de Generación y Eficiencia de Colección
axes[0].plot(x * 1e6, G_x, color='darkorange', label='G(x) [Beer-Lambert]')
axes[0].set_xlabel(r'Profundidad $x$ ($\mu$m)')
axes[0].set_ylabel(r'Generación $G(x)$ (m$^{-3}$s$^{-1}$)', color='darkorange')
axes[0].set_yscale('log')
ax0_twin = axes[0].twinx()
ax0_twin.plot(x * 1e6, eta_col, color='navy', linestyle='--', label=r'$\eta_{col}(x)$')
ax0_twin.set_ylabel('Probabilidad de colección', color='navy')
axes[0].set_title('Generación Óptica y Colección de Portadores')
# Gráfica 2: Curva J-V Iluminada
axes[1].plot(V_grid, J_grid * 1e3, color='forestgreen', lw=2, label='Curva J-V (Difusión)')
axes[1].plot([0], [J_L * 1e3], 'bo', label=f'Jsc = {J_L*1e3:.2f} mA/cm²')
axes[1].plot([Voc], [0], 'ro', label=f'Voc = {Voc:.3f} V')
axes[1].plot([Vmpp], [Jmpp * 1e3], 'ks', label=f'MPP ({Pmax*1e3:.2f} mW/cm²)')
axes[1].set_xlabel('Voltaje (V)')
axes[1].set_ylabel('Corriente J (mA/cm²)')
axes[1].set_title(f'Característica J-V (FF = {FF:.1f}%, $\eta$ = {eta:.2f}%)')
axes[1].grid(True, ls=':')
axes[1].legend()
plt.tight_layout()
plt.show()