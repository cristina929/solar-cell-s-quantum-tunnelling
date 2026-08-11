import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# 1. CONSTANTES FÍSICAS E INPUTS DE LA CÉLULA
q = 1.60217663e-19        # Carga elemental (C)
eps0 = 8.85418781e-12     # Permitividad del vacío (F/m)
kB = 1.380649e-23         # Constante de Boltzmann (J/K)
T = 300                   # Temperatura (K)
Vt = (kB * T) / q         # Voltaje térmico (~0.0259 V)

# Datos la célula (convertidos a SI)
W_emisor = 0.5e-6         # 0.5 µm
W_base = 160.e-6          # 160 µm
W_total = W_emisor + W_base

Na = 1e19 * 1e6           # m^-3 (Emisor p+)
Nd = 1e15 * 1e6           # m^-3 (Base n)
eps_Si = 11.9 * eps0      # Permitividad del Silicio

Nc = 2.8e19 * 1e6         # m^-3
Nv = 1.04e19 * 1e6        # m^-3
Eg = 1.12                 # eV

# Calcular ni exacta (concentración en equilibrio)
ni = np.sqrt(Nc * Nv * np.exp(-Eg / Vt))

# 2. CONSTRUCCIÓN DE LA MALLA (GRID)
nodos_emisor = 150
nodos_base = 350

x_emisor = np.linspace(0, W_emisor, nodos_emisor)
x_base = np.linspace(W_emisor, W_total, nodos_base + 1)[1:]
x = np.concatenate((x_emisor, x_base))
N = len(x)

# Definir vector de dopaje Neto (N_D - N_A)
N_dopaje = np.zeros(N)
N_dopaje[:nodos_emisor] = -Na
N_dopaje[nodos_emisor:] = Nd

# Pasos espaciales locales (malla no uniforme)
dx = np.diff(x)

# 3. CONDICIONES DE CONTORNO EN EQUILIBRIO
# Potencial en los contactos neutros (Dirichlet)
psi_p = -Vt * np.log(Na / ni)  # Extremo p+ (x=0)
psi_n =  Vt * np.log(Nd / ni)  # Extremo n (x=W_total)
Vbi = psi_n - psi_p

# Estimación inicial (Aproximación de escalón paso-banda)
psi = np.zeros(N)
psi[:nodos_emisor] = psi_p
psi[nodos_emisor:] = psi_n

# 4. SOLVER DE POISSON NO LINEAL (NEWTON-RAPHSON)
max_iter = 100
tolerancia = 1e-6

for iteration in range(max_iter):
    # Vectores de carga libre según el potencial
    p = ni * np.exp(-psi / Vt)
    n = ni * np.exp(psi / Vt)
    
    # Residuos de la ecuación de Poisson: F(psi) = d^2(psi)/dx^2 + q*(p - n + N_dopaje)/eps
    F = np.zeros(N)
    # Jacobiano (derivada del residuo respecto a psi): Matriz tridiagonal
    diag_central = np.zeros(N)
    diag_sup = np.zeros(N - 1)
    diag_inf = np.zeros(N - 1)
    
    # Ecuaciones internas usando diferencias finitas para malla no uniforme
    for i in range(1, N - 1):
        dx_izq = dx[i - 1]
        dx_der = dx[i]
        dx_prom = (dx_izq + dx_der) / 2.0
        
        # Derivada segunda espacial aproximada
        F[i] = (psi[i + 1] - psi[i]) / (dx_der * dx_prom) - (psi[i] - psi[i - 1]) / (dx_izq * dx_prom)
        # Añadir término de densidad de carga rho/eps
        F[i] += (q / eps_Si) * (p[i] - n[i] + N_dopaje[i])
        
        # Elementos del Jacobiano para el nodo i
        diag_central[i] = -1.0 / (dx_der * dx_prom) - 1.0 / (dx_izq * dx_prom) - (q / eps_Si) * (p[i] + n[i]) / Vt
        diag_sup[i] = 1.0 / (dx_der * dx_prom)
        diag_inf[i - 1] = 1.0 / (dx_izq * dx_prom)
        
    # Condiciones de contorno en los extremos (F = 0, delta_psi = 0)
    F[0] = 0
    F[-1] = 0
    diag_central[0] = 1.0
    diag_central[-1] = 1.0
    
    # Ensamblar matriz tridiagonal del Jacobiano
    J = diags([diag_central, diag_inf, diag_sup], [0, -1, 1], format='csr')
    
    # Resolver el sistema lineal para el paso de Newton: J * delta_psi = -F
    delta_psi = spsolve(J, -F)
    
    # Actualizar el potencial
    psi += delta_psi
    
    # Criterio de parada: evaluar el error máximo del incremento
    error = np.max(np.abs(delta_psi))
    if error < tolerancia:
        print(f"¡Solver de Poisson en equilibrio convergió con éxito en la iteración {iteration}!")
        break
else:
    print("Alerta: El solver no alcanzó la tolerancia establecida, revisar los parámetros.")

# 5. CÁLCULO DEL CAMPO ELÉCTRICO Y POST-PROCESADO
# El Campo Eléctrico es E = -d(psi)/dx
campo_E = np.zeros(N)
campo_E[1:-1] = -(psi[2:] - psi[:-2]) / (dx[:-1] + dx[1:])

# 6. VISUALIZACIÓN DE RESULTADOS
fig, ax1 = plt.subplots(figsize=(10, 5))

# Graficar Potencial Electrostático (Eje izquierdo)
color = 'tab:blue'
ax1.set_xlabel('Posición x (µm)')
ax1.set_ylabel('Potencial Intrínseco ψ (V)', color=color)
ax1.plot(x * 1e6, psi, color=color, label='Potencial (ψ)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle=':')

# Graficar Campo Eléctrico (Eje derecho)
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Campo Eléctrico E (V/cm)', color=color)
ax2.plot(x * 1e6, campo_E * 1e-2, color=color, label='Campo Eléctrico (E)')
ax2.tick_params(axis='y', labelcolor=color)

# Enfocar la gráfica en la zona de la unión p-n (primeras 5 micras) para ver el detalle
ax1.set_xlim(0, 5) 

plt.title('Etapa 2: Validación del Perfil de Potencial y Campo E en Equilibrio')
fig.tight_layout()
plt.show()
print(f"Vbi = {Vbi:.4f} V")