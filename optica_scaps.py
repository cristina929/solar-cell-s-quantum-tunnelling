# -*- coding: utf-8 -*-
"""
optica_scaps.py - Optica comun para las etapas 3/4/5, leida de los
ficheros nativos de SCAPS para que ambas herramientas vean EXACTAMENTE
la misma iluminacion y absorcion.

Ficheros esperados (en la misma carpeta que el script):
  AM1_5G_1_sun.spe : espectro AM1.5G de SCAPS. 2a columna = potencia
                     W/m2 INTEGRADA EN CADA BIN de longitud de onda
                     (no densidad espectral). Total = 1000 W/m2.
  Si.abs           : absorcion del silicio (PC1D), lambda[nm] alfa[1/m].

Uso:
  from optica_scaps import cargar_optica, G_de_x
  opt = cargar_optica()                  # busca los ficheros junto al script
  G   = G_de_x(x, opt, R=0.10)           # perfil de generacion en la malla x [m]
"""
import os
import numpy as np

hc = 6.62607015e-34 * 2.99792458e8   # J*m
q  = 1.602176634e-19

def _ruta(fn):
    """Busca el fichero junto al script y, si no, en el directorio actual."""
    base = os.path.dirname(os.path.abspath(__file__))
    for carpeta in (base, os.getcwd()):
        p = os.path.join(carpeta, fn)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"No encuentro '{fn}'. Copialo a la carpeta del script: {base}")

def leer_spe(fn='AM1_5G_1_sun.spe'):
    """Devuelve (lambda[nm], P_bin[W/m2]) del espectro SCAPS."""
    lam, P = [], []
    for line in open(_ruta(fn), encoding='latin1'):
        line = line.strip()
        if not line or line.startswith('>'):
            continue
        t = line.split()
        try:
            l, p = float(t[0]), float(t[1])
        except (ValueError, IndexError):
            continue
        lam.append(l); P.append(p)
    return np.array(lam), np.array(P)

def leer_abs(fn='Si.abs'):
    """Devuelve (lambda[nm], alfa[1/m]) de la absorcion SCAPS/PC1D."""
    lam, al = [], []
    for line in open(_ruta(fn), encoding='latin1'):
        line = line.strip()
        if not line or line.startswith('/'):
            continue
        t = line.split()
        try:
            l, a = float(t[0]), float(t[1])
        except (ValueError, IndexError):
            continue
        lam.append(l); al.append(a)
    return np.array(lam), np.array(al)

def cargar_optica(lam_max=1250.0):
    """Carga espectro y absorcion y los pone en una malla comun.

    Devuelve un dict con:
      lam     : longitudes de onda de los bins del espectro [nm]
      Phi_bin : flujo de fotones POR BIN [fotones/m2/s]
      alpha   : alfa(lam) interpolada del Si.abs [1/m]
    Solo se retienen bins con alfa > 0 y lam <= lam_max.
    """
    lam_s, P_s = leer_spe()
    lam_a, al_a = leer_abs()
    Phi = P_s * (lam_s * 1e-9) / hc          # fotones/m2/s por bin
    alpha = np.interp(lam_s, lam_a, al_a, left=al_a[0], right=0.0)
    m = (alpha > 0) & (lam_s <= lam_max)
    return {'lam': lam_s[m], 'Phi_bin': Phi[m], 'alpha': alpha[m]}

def G_de_x(x, opt, R=0.10):
    """Perfil de fotogeneracion G(x) [1/m3/s] por Beer-Lambert, una pasada.

    x   : malla de posiciones [m] (0 = superficie frontal)
    opt : dict de cargar_optica()
    R   : reflectancia frontal (constante)
    """
    al = opt['alpha']; Phi = opt['Phi_bin']
    return (1.0 - R) * (np.exp(-np.outer(x, al)) * (al * Phi)).sum(axis=1)

def J_fotones_techo(opt, R=0.10, W=None):
    """Techo de corriente [mA/cm2]: fotones absorbidos (W finito) o
    incidentes utiles (W=None)."""
    if W is None:
        absorb = 1.0
    else:
        absorb = 1.0 - np.exp(-opt['alpha'] * W)
    return q * ((1 - R) * opt['Phi_bin'] * absorb).sum() * 0.1

if __name__ == '__main__':
    opt = cargar_optica()
    print(f"bins utiles: {len(opt['lam'])}  "
          f"({opt['lam'][0]:.0f}-{opt['lam'][-1]:.0f} nm)")
    print(f"techo incidente (R=0.10): {J_fotones_techo(opt):.2f} mA/cm2")
    print(f"techo absorbido 160um   : {J_fotones_techo(opt, W=160e-6):.2f} mA/cm2")
