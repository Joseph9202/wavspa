"""
================================================================================
WavSpA (Wavelet Space Attention) - Arquitectura de Transformers con Wavelets
================================================================================

Este archivo implementa la arquitectura WavSpA que combina:
1. Transformadas Wavelet Adaptativas
2. Mecanismos de Atención (Transformer, Performer, Linformer, etc.)
3. Aprendizaje End-to-End en JAX/Flax

Autor: Yufan Zhuang
Paper: https://arxiv.org/abs/2210.01989
Propósito Educativo: Seminario Dpto. Matemáticas - Univalle
================================================================================
"""

import numpy as np
from flax import linen as nn
import jax
import jax.numpy as jnp
from lra_benchmarks.models.layers import common_layers
from typing import Callable, Any, Optional
from lra_benchmarks.utils.fft_convolve import fftconvolve
import jax.scipy.signal as jsignal
import pywt
import wavspa
from jax.experimental import sparse
from wavspa.wavelet_lifting import liftdec, liftrec, liftdec_learn, liftrec_learn

# Importar diferentes tipos de mecanismos de atención
from lra_benchmarks.models.wavspa.middle_layer_linear import LinearAttention
from lra_benchmarks.models.wavspa.middle_layer_linformer import LinformerAttention
from lra_benchmarks.models.wavspa.middle_layer_performer import PerformerAttn
from lra_benchmarks.models.wavspa.middle_layer_longformer import LongformerAttention


# ==============================================================================
# SECCIÓN 1: FUNCIONES DE INICIALIZACIÓN DE WAVELETS
# ==============================================================================

def sinwave(N, decay):
    """
    Genera una wavelet basada en función seno con decaimiento.
    
    Matemáticamente:
        h_0(x) = √2 * cos(x-1) / x^decay
    
    donde x está normalizado en el intervalo [1, N*dt*π]
    
    Args:
        N (int): Longitud del filtro wavelet
        decay (float): Factor de decaimiento exponencial
    
    Returns:
        np.ndarray: Filtro wavelet normalizado de forma (N, 1)
    
    Intuición Matemática:
        - La función coseno proporciona oscilaciones
        - El decaimiento x^decay asegura que la wavelet se localice
        - La normalización √2 preserva la energía de la señal
    """
    # Paso de tiempo adaptativo basado en N
    dt = 1/N * np.pi * (np.sqrt(N))
    
    # Crear el dominio espacial
    x = np.linspace(1, N*dt*np.pi, num=N)
    
    # Aplicar la función seno con decaimiento
    y = np.cos(x-1) / x ** decay
    
    # Normalizar a √2 (convención de wavelets ortogonales)
    h_0 = np.sqrt(2) * y / np.sum(y)
    h_0.shape = (-1, 1)
    
    return h_0


def daubcqf(N):
    """
    Calcula los filtros de escalado y wavelet de Daubechies.
    
    TEORÍA MATEMÁTICA:
    ==================
    Las wavelets de Daubechies son soluciones al problema de construcción
    de wavelets ortogonales con soporte compacto y máxima suavidad.
    
    Propiedades clave:
    1. Ortogonalidad: ∫ φ(t)φ(t-k)dt = δ_k (delta de Kronecker)
    2. Soporte compacto: φ(t) = 0 para |t| > N-1
    3. Momentos nulos: ∫ t^k ψ(t)dt = 0, para k < N/2
    
    El algoritmo construye los coeficientes h_0 del filtro paso-bajo
    usando las raíces del polinomio Q(z) que satisface:
        |P(ω)|² + |P(ω+π)|² = 1  (condición de reconstrucción perfecta)
    
    Args:
        N (int): Longitud del filtro (debe ser par)
    
    Returns:
        np.ndarray: Coeficientes del filtro de Daubechies de forma (N, 1)
    
    Referencias:
        Daubechies, I. (1988). "Orthonormal bases of compactly supported wavelets."
        Communications on Pure and Applied Mathematics, 41(7), 909-996.
    """
    assert N % 2 == 0, 'No existe filtro de Daubechies para longitud impar'
    
    K = int(N/2)  # Orden de la wavelet
    
    # Inicialización de variables auxiliares
    a = 1
    p = 1
    q = 1
    
    # Comenzar con filtro Haar (caso base: db1)
    # Haar es la wavelet más simple: h_0 = [1, 1]/√2
    h_0 = np.array([1.0, 1.0])
    
    # ITERACIÓN PARA CONSTRUIR DAUBECHIES DE ORDEN SUPERIOR
    # Este bucle construye iterativamente los coeficientes usando
    # convoluciones y el polinomio característico
    for j in range(1, K):
        # Coeficiente binomial con signo alterno
        a = -a * 0.25 * (j + K - 1) / j
        
        # Expandir h_0 mediante convolución con [1, 1]
        h_0 = np.hstack((0, h_0)) + np.hstack((h_0, 0))
        
        # Construir polinomios auxiliares p y q
        # p representa términos de derivadas de orden superior
        p = np.hstack((0, -p)) + np.hstack((p, 0))
        p = np.hstack((0, -p)) + np.hstack((p, 0))
        
        # q acumula los términos del polinomio característico
        q = np.hstack((0, q, 0)) + a * p
    
    # SELECCIÓN DE RAÍCES
    # De las 2K-2 raíces de q, seleccionamos K-1 dentro del círculo unitario
    # Esto garantiza la estabilidad y causalidad del filtro
    q = np.sort(np.roots(q))
    qt = q[:K-1]
    
    # Construir h_0 a partir de las raíces seleccionadas
    h_0 = np.convolve(h_0, np.real(np.poly(qt)))
    
    # Normalización a √2 (conservación de energía)
    h_0 = np.sqrt(2) * h_0 / np.sum(h_0)
    
    # Reformatear y voltear (convención de procesamiento de señales)
    h_0.shape = (-1, 1)
    h_0 = np.flip(h_0, axis=0)
    
    # Verificar ortogonalidad: ||h_0||² = 1
    assert np.abs(np.sum(h_0 ** 2)) - 1 < 1e-4, \
        'Numéricamente inestable para este valor de N'
    
    return h_0


@jax.jit  # Compilación JIT para eficiencia GPU
def parametrized_wavelet(thetas, S, S_inv):
    """
    Construye wavelets ortogonales parametrizadas por ángulos de rotación.
    
    TEORÍA - DESCOMPOSICIÓN QR Y WAVELETS ORTOGONALES:
    ===================================================
    Este método implementa la construcción de wavelets mediante 
    factorización en productos de matrices de rotación Givens.
    
    Matemáticamente:
        C = e₁ · R(θ₁) · S · R(θ₂) · S · ... · R(θ_L) · S · S⁻¹
    
    donde:
    - e₁ = [1, 0, ..., 0] vector canónico
    - R(θ) = matriz de rotación Givens 2D
    - S = matriz de permutación cíclica
    
    Propiedades garantizadas:
    1. Ortogonalidad: C·C^T = I
    2. Conservación de energía: ||C|| = 1
    3. Diferenciabilidad: ∂C/∂θ existe (permite backpropagation)
    
    Args:
        thetas (jnp.ndarray): Ángulos de rotación de forma (L, wav_dim)
                              donde L = wlen/2
        S (sparse.BCOO): Matriz de permutación cíclica dispersa
        S_inv (jnp.ndarray): Inversa de la matriz de permutación
    
    Returns:
        jnp.ndarray: Coeficientes de wavelet ortogonal de forma (2L,)
    
    Visualización:
        θ₁     θ₂         θ_L
         ↓      ↓           ↓
        [1,0]→[R₁]→[S]→[R₂]→...→[R_L]→[S⁻¹] = wavelet
    
    Referencias:
        - Rippel, O., et al. (2015). "Spectral Representations for 
          Convolutional Neural Networks."
    """
    L = thetas.shape[0]  # Número de rotaciones = longitud/2
    
    # Vector inicial (impulso unitario)
    C = jnp.eye(N=L*2)[0, :]  # e₁ = [1, 0, 0, ..., 0]
    
    # CADENA DE ROTACIONES Y PERMUTACIONES
    for theta in thetas[:]:
        # Construir matriz de rotación Givens 2D:
        #     ⎡ sin(θ)   cos(θ) ⎤
        # A = ⎣ cos(θ)  -sin(θ) ⎦
        # 
        # Esta es una rotación que preserva norma
        A = jnp.array([
            [jnp.sin(theta), jnp.cos(theta)], 
            [jnp.cos(theta), -jnp.sin(theta)]
        ])
        
        # Construir matriz bloque-diagonal con L copias de A
        # Esto permite procesar múltiples canales en paralelo
        R = sparse.BCOO.fromdense(
            jax.scipy.linalg.block_diag(*[A for _ in range(L)]), 
            nse=4*L  # Número de elementos no nulos
        )
        
        # Aplicar rotación y permutación: C ← C · R · S
        C = C @ R @ S
    
    # Aplicar permutación inversa para completar la construcción
    C = jnp.matmul(C, S_inv)
    
    return C


# ==============================================================================
# SECCIÓN 2: FUNCIONES DE INICIALIZACIÓN DE PARÁMETROS
# ==============================================================================

def ortho_init(key, shape, dtype):
    """
    Inicializador para wavelets ortogonales parametrizadas.
    
    Genera ángulos θ ~ U[0, 2π] y construye wavelets ortogonales.
    Esta inicialización garantiza que el modelo comience con wavelets válidas.
    
    Args:
        key: Clave aleatoria JAX
        shape (tuple): (longitud_filtro, dimensión_canal)
        dtype: Tipo de dato (float32, bfloat16, etc.)
    
    Returns:
        jnp.ndarray: Wavelets ortogonales iniciales
    """
    def init(key, shape, dtype):
        # Ángulos uniformes en [0, 2π]
        thetas = nn.initializers.uniform(2*jnp.pi)(
            key, 
            shape=(int(shape[0]/2),), 
            dtype=dtype
        )
        # Construir wavelet y replicar para todos los canales
        return jnp.repeat(
            parametrized_wavelet(thetas), 
            repeats=shape[1], 
            axis=-1
        )
    return init(key, shape, dtype)


def sin_init(key, shape, dtype):
    """
    Inicializador para wavelets basadas en funciones seno.
    
    Útil para señales periódicas o cuando se desea un sesgo inductivo
    hacia patrones oscilatorios.
    """
    def init(key, shape, dtype):
        wav_vec = jnp.asarray(sinwave(N=shape[0], decay=1.0))
        return jnp.repeat(wav_vec, repeats=shape[1], axis=-1)
    return init(key, shape, dtype)


def eye_init():
    """
    Inicializador identidad (para debugging o baselines).
    """
    def init(key, shape, dtype):
        return jnp.eye(N=shape[0], M=shape[1], dtype=dtype)
    return init


def db_init(key, shape, dtype):
    """
    Inicializador usando wavelets de Daubechies estándar.
    
    Estrategia:
    - Si wlen/2 ≤ 20: usar PyWavelets (tablas precalculadas)
    - Si wlen/2 > 20: calcular numéricamente con daubcqf()
    
    Args:
        key: Clave aleatoria (no usada aquí, wavelets deterministas)
        shape (tuple): (longitud_filtro, num_canales)
        dtype: Tipo de dato
    
    Returns:
        jnp.ndarray: Coeficientes de Daubechies replicados para todos los canales
    """
    def init(key, shape, dtype):
        orden = int(shape[0] / 2)  # Orden de Daubechies: db{orden}
        
        if orden <= 20:
            # Usar biblioteca PyWavelets para órdenes estándar
            db_wavelet = pywt.Wavelet(f'db{orden}')
            h_0 = db_wavelet.filter_bank[0]  # Filtro paso-bajo
            wav_vec = jnp.expand_dims(jnp.asarray(h_0), axis=-1)
        else:
            # Calcular numéricamente para órdenes altos
            wav_vec = jnp.asarray(daubcqf(N=shape[0]))
        
        # Replicar para todos los canales
        return jnp.repeat(wav_vec, repeats=shape[1], axis=-1)
    
    return init(key, shape, dtype)


# ==============================================================================
# SECCIÓN 3: BLOQUE PRINCIPAL - WavspaBlock
# ==============================================================================

class WavspaBlock(nn.Module):
    """
    Bloque fundamental de WavSpA que combina:
    1. Transformada Wavelet Adaptativa (Forward)
    2. Atención Multi-Escala en el dominio wavelet
    3. Transformada Wavelet Inversa (Backward)
    4. Conexión Residual + MLP
    
    ARQUITECTURA:
    =============
    
    Input x ∈ ℝ^(B×L×D)
        ↓
    [LayerNorm] → x_norm
        ↓
    [Wavelet Forward Transform] → {z₀, z₁, ..., z_level}
        ↓
    [Self-Attention en cada nivel] → {z'₀, z'₁, ..., z'_level}
        ↓
    [Wavelet Inverse Transform] → x_reconstructed
        ↓
    [Residual Connection] → x + x_reconstructed
        ↓
    [MLP Block] → output
    
    VENTAJAS SOBRE TRANSFORMER ESTÁNDAR:
    =====================================
    1. Multi-escala: Captura patrones en diferentes frecuencias
    2. Eficiencia: O(L) en lugar de O(L²) para secuencias largas
    3. Localidad: Wavelets tienen soporte compacto
    4. Adaptabilidad: Parámetros de wavelet entrenables
    
    Attributes:
        qkv_dim (int): Dimensión de queries, keys, values
        mlp_dim (int): Dimensión de la capa feedforward
        num_heads (int): Número de cabezas de atención
        level (int): Niveles de descomposición wavelet (profundidad)
        wlen (int): Longitud del filtro wavelet
        wavelet (str): Tipo de wavelet ('db', 'ortho', 'lift', 'adalift')
        model_type (str): Tipo de atención ('transformer', 'performer', etc.)
    """
    
    # Hiperparámetros del bloque
    qkv_dim: int
    mlp_dim: int
    num_heads: int
    L: int                    # Longitud máxima de secuencia
    max_len: int
    nb_features: int          # Features para Performer (aproximación)
    dtype: Any = jnp.float32
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    
    # Hiperparámetros de wavelet
    level: int = 2            # Número de niveles de descomposición
    wlen: int = 32            # Longitud del filtro wavelet (potencia de 2)
    wav_dim: int = None       # Dimensión de embedding (auto-inferida)
    poly_dim: int = 1
    wavelet: str = 'db2'      # Tipo: 'db', 'sin', 'ortho', 'lift', 'adalift'
    model_type: str = 'transformer'  # 'transformer', 'performer', 'linformer', etc.
    
    def setup(self):
        """
        Inicialización de parámetros de wavelet.
        
        Esta función se ejecuta una vez al crear el módulo y configura
        los parámetros entrenables según el tipo de wavelet seleccionado.
        
        OPCIONES DE WAVELET:
        ====================
        
        1. 'adalift': Adaptive Lifting Scheme
           - Parámetros: adawave_est, adawave_pred
           - Ventaja: Flexibilidad máxima, puede aprender cualquier wavelet
        
        2. 'lift': Lifting Scheme estándar (no entrenable)
           - Sin parámetros
           - Ventaja: Eficiencia, buena baseline
        
        3. 'ortho': Orthogonal parametrization
           - Parámetros: thetas (ángulos de rotación)
           - Ventaja: Garantiza ortogonalidad estricta
        
        4. 'db': Daubechies adaptativa
           - Parámetros: adawave (coeficientes de filtro)
           - Ventaja: Inicialización matemáticamente fundamentada
        
        5. 'sin': Sinusoidal wavelet
           - Parámetros: adawave
           - Ventaja: Buena para señales periódicas
        
        6. Default: Daubechies fija (no entrenable)
           - Sin parámetros
           - Ventaja: Baseline matemáticamente sólida
        """
        # Validación: longitud debe ser par para descomposición binaria
        assert self.wlen % 2 == 0, "wlen debe ser par para descomposición wavelet"
        
        self.eps = 1e-4  # Epsilon para estabilidad numérica
        
        # CONFIGURACIÓN SEGÚN TIPO DE WAVELET
        
        if "lift" in self.wavelet:
            # ADAPTIVE LIFTING SCHEME
            # Parametriza directamente las operaciones de predict y update
            self.adawave_est = self.param(
                'adawave_est',
                nn.initializers.normal(stddev=0.02),
                (self.wlen, self.wav_dim),
                self.dtype
            )
            self.adawave_pred = self.param(
                'adawave_pred',
                nn.initializers.normal(stddev=0.02),
                (self.wlen, self.wav_dim),
                self.dtype
            )
            
        elif "ortho" in self.wavelet:
            # ORTHOGONAL PARAMETRIZATION
            # Construye wavelets mediante rotaciones Givens
            L = int(self.wlen / 2)
            
            # Matriz de permutación cíclica S
            # S_{i,j} = 1 si j = (i+1) mod 2L, 0 otherwise
            S = jnp.zeros(shape=[2*L, 2*L], dtype=int)
            i = jnp.asarray(range(2*L))
            j = jnp.asarray(range(1, 2*L+1)) % (2*L)
            S = S.at[i, j].set(1)
            
            # Convertir a formato disperso (eficiencia)
            self.S = sparse.BCOO.fromdense(S, nse=2*L)
            self.S_inv = jnp.linalg.inv(S)
            
            # Parámetros: ángulos de rotación θ ∈ [0, 2π]
            self.thetas = self.param(
                'thetas',
                nn.initializers.uniform(2*jnp.pi),
                (L, self.wav_dim),
                self.dtype
            )
            
        elif "db" in self.wavelet:
            # ADAPTIVE DAUBECHIES
            # Inicializa con Daubechies pero permite entrenamiento
            self.adawave = self.param(
                'adawave',
                db_init,
                (self.wlen, self.wav_dim),
                self.dtype
            )
            
        elif "sin" in self.wavelet:
            # SINUSOIDAL WAVELET
            self.adawave = self.param(
                'adawave',
                sin_init,
                (self.wlen, self.wav_dim),
                self.dtype
            )
            
        else:
            # DEFAULT: DAUBECHIES FIJA (NO ENTRENABLE)
            # Útil como baseline o cuando no se desea entrenar wavelets
            self.adawave = db_init(
                key=None,
                shape=(self.wlen, self.wav_dim),
                dtype=self.dtype
            )
    
    @nn.compact
    def __call__(self,
                 inputs,
                 inputs_segmentation=None,
                 padding_mask=None,
                 deterministic=False):
        """
        Forward pass del bloque WavSpA.
        
        FLUJO DE PROCESAMIENTO:
        =======================
        
        1. Enmascaramiento de padding
        2. Normalización de capa
        3. Transformada wavelet forward (multi-escala)
        4. Atención en cada escala wavelet
        5. Transformada wavelet inversa (reconstrucción)
        6. Conexión residual
        7. Bloque MLP con otra residual
        
        Args:
            inputs (jnp.ndarray): Tensor de entrada de forma (B, L, D)
                                  B = batch size
                                  L = longitud de secuencia
                                  D = dimensión de embedding
            inputs_segmentation: Información de segmentación (no usado actualmente)
            padding_mask (jnp.ndarray): Máscara de padding de forma (B, L, 1)
            deterministic (bool): Si False, aplica dropout (entrenamiento)
        
        Returns:
            jnp.ndarray: Salida del bloque de forma (B, L, D)
        
        MATEMÁTICA DETALLADA:
        =====================
        
        Sea x ∈ ℝ^(B×L×D) el input
        
        1. Normalización:
           x̃ = LayerNorm(x)
        
        2. Descomposición Wavelet (J niveles):
           {z₀, z₁, ..., z_J} = WaveletDec(x̃)
           
           donde:
           - z₀: aproximación (bajas frecuencias)
           - z_j: detalles nivel j (altas frecuencias)
           - Longitud: |z_j| ≈ L/2^j
        
        3. Atención Multi-Escala:
           z'_j = Attention(z_j, z_j, z_j)  ∀j ∈ {0, 1, ..., J}
           
           Cada escala captura patrones a diferente resolución:
           - z₀: tendencias globales, contexto largo
           - z_J: detalles finos, patrones locales
        
        4. Reconstrucción:
           x' = WaveletRec({z'₀, z'₁, ..., z'_J})
        
        5. Residual + MLP:
           y = x + x'
           output = y + MLP(LayerNorm(y))
        
        COMPLEJIDAD:
        ============
        - Wavelet Transform: O(L·log(L))
        - Attention por nivel: O(L_j²·D) donde L_j = L/2^j
        - Total: O(L·log(L) + Σ(L/2^j)²·D) ≈ O(L·D) para J grande
        
        Comparado con Transformer estándar O(L²·D), esto es O(L) más eficiente!
        """
        
        # Validar dimensiones
        assert inputs.ndim == 3, "Input debe ser 3D: (batch, length, dim)"
        
        # =====================================================================
        # PASO 1: ENMASCARAMIENTO DE PADDING
        # =====================================================================
        # Poner a cero las posiciones de padding para no contaminar la atención
        inputs = jnp.where(padding_mask, inputs, 0)
        
        # =====================================================================
        # PASO 2: NORMALIZACIÓN DE CAPA
        # =====================================================================
        # LayerNorm estabiliza el entrenamiento
        # x̃ = (x - μ) / √(σ² + ε) * γ + β
        x = nn.LayerNorm(dtype=self.dtype)(inputs)
        
        # =====================================================================
        # PASO 3: PREPARACIÓN DE WAVELETS
        # =====================================================================
        # Construir wavelets según el tipo especificado
        
        if "ortho" in self.wavelet:
            # Wavelets ortogonales parametrizadas
            # Aplicar vmap para procesar todos los canales en paralelo
            wavelet = jax.vmap(
                parametrized_wavelet,
                in_axes=(1, None, None),  # Vectorizar sobre dim=1 de thetas
                out_axes=1
            )(self.thetas, self.S, self.S_inv)
            
        elif "lift" not in self.wavelet:
            # Wavelets adaptativas estándar (db, sin)
            wavelet = self.adawave
        
        # =====================================================================
        # PASO 4: TRANSFORMADA WAVELET FORWARD (DESCOMPOSICIÓN)
        # =====================================================================
        # Aplicar descomposición multi-escala
        
        if "adalift" in self.wavelet:
            # Adaptive Lifting Scheme con parámetros entrenables
            z = liftdec_learn(
                x,
                self.adawave_est,
                self.adawave_pred,
                level=self.level
            )
        elif "lift" in self.wavelet:
            # Lifting Scheme estándar (no entrenable)
            z = wavspa.liftdec(x, level=self.level)
        else:
            # Descomposición wavelet estándar con filtros entrenables
            z = wavspa.wavedec_learn(x, wavelet, level=self.level)
        
        # z es ahora una lista: [z_0, z_1, ..., z_level]
        # donde z_j tiene forma (B, L_j, D) con L_j ≈ L/2^j
        
        # =====================================================================
        # PASO 5: ATENCIÓN MULTI-ESCALA EN DOMINIO WAVELET
        # =====================================================================
        # Aplicar mecanismo de atención en cada escala wavelet
        
        for level in range(len(z)):
            # Manejar caso especial de longitud 1 (compresión máxima)
            conv_flag = False
            if z[level].shape[1] == 1:
                z[level] = jnp.squeeze(z[level], axis=1)
                conv_flag = True
            
            # SELECCIÓN DE MECANISMO DE ATENCIÓN
            # Diferentes arquitecturas para diferentes trade-offs
            
            if self.model_type == 'transformer':
                # ====== TRANSFORMER ESTÁNDAR ======
                # Complejidad: O(L_j² · D)
                # Pros: Expresividad máxima
                # Contras: Costoso para secuencias largas
                z[level] = nn.SelfAttention(
                    num_heads=self.num_heads,
                    dtype=self.dtype,
                    qkv_features=self.qkv_dim,
                    kernel_init=nn.initializers.xavier_uniform(),
                    bias_init=nn.initializers.normal(stddev=1e-6),
                    use_bias=False,
                    broadcast_dropout=False,
                    dropout_rate=self.attention_dropout_rate,
                    decode=False
                )(z[level], deterministic=deterministic)
                
            elif self.model_type == 'performer':
                # ====== PERFORMER ======
                # Complejidad: O(L_j · D)
                # Método: Aproximación de kernel usando Random Features
                # Pros: Lineal en L, escalable
                # Contras: Aproximación, puede perder información
                z[level] = PerformerAttn(
                    num_heads=self.num_heads,
                    qkv_dim=self.qkv_dim,
                    lax_scan_unroll=16,
                    nb_features=self.nb_features,
                    dropout_rate=self.attention_dropout_rate,
                    qkv_normalizarion=True
                )(z[level], deterministic=deterministic)
                
            elif self.model_type == 'linformer':
                # ====== LINFORMER ======
                # Complejidad: O(L_j · k · D) donde k << L_j
                # Método: Proyección de baja dimensión de keys y values
                # Pros: Muy eficiente, buena aproximación
                # Contras: Requiere conocer max_len a priori
                z[level] = LinformerAttention(
                    num_heads=self.num_heads,
                    qkv_features=self.qkv_dim,
                    max_len=self.max_len,
                    dropout_rate=self.attention_dropout_rate,
                    low_rank_features=128
                )(z[level], deterministic=deterministic)
                
            elif self.model_type == 'linear_attention':
                # ====== LINEAR ATTENTION ======
                # Complejidad: O(L_j · D²)
                # Método: Kernel trick con función de activación
                # Pros: Lineal en L, matemáticamente elegante
                # Contras: Menos expresivo que softmax
                z[level] = LinearAttention(
                    num_heads=self.num_heads,
                    qkv_features=self.qkv_dim
                )(z[level])
                z[level] = nn.Dropout(rate=self.dropout_rate)(
                    z[level],
                    deterministic=deterministic
                )
                
            elif self.model_type == 'longformer':
                # ====== LONGFORMER ======
                # Complejidad: O(L_j · w · D) donde w = window_size
                # Método: Atención local con ventana deslizante
                # Pros: Lineal en L, captura dependencias locales bien
                # Contras: Puede perder dependencias muy largas
                z[level] = LongformerAttention(
                    num_heads=self.num_heads,
                    qkv_features=self.qkv_dim,
                    sliding_window_size=512,
                    broadcast_dropout=False,
                    bias=False,
                    dropout_rate=self.attention_dropout_rate,
                    dtype=self.dtype
                )(z[level], deterministic=deterministic)
                
            else:
                raise NotImplementedError(
                    f"Tipo de modelo '{self.model_type}' no implementado"
                )
            
            # Restaurar dimensión si fue comprimida
            if conv_flag:
                z[level] = jnp.expand_dims(z[level], axis=1)
        
        # =====================================================================
        # PASO 6: TRANSFORMADA WAVELET INVERSA (RECONSTRUCCIÓN)
        # =====================================================================
        # Reconstruir la señal desde las representaciones multi-escala
        
        if "adalift" in self.wavelet:
            # Reconstrucción con adaptive lifting
            z = liftrec_learn(z, self.adawave_est, self.adawave_pred)
        elif "lift" in self.wavelet:
            # Reconstrucción con lifting estándar
            z = wavspa.liftrec(z)
        else:
            # Reconstrucción con wavelets estándar
            z = wavspa.waverec_learn(z, wavelet)
        
        # Recortar a la longitud original (manejo de padding)
        z = z[:, :inputs.shape[1], :]
        
        # =====================================================================
        # PASO 7: PRIMERA CONEXIÓN RESIDUAL
        # =====================================================================
        # x_out = x_in + f(x_in)
        # Esto permite gradientes directos y estabiliza el entrenamiento
        x = z + inputs
        
        # Aplicar dropout
        y = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        
        # =====================================================================
        # PASO 8: BLOQUE MLP (FEEDFORWARD)
        # =====================================================================
        # MLP estándar:
        # FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
        # Expande a mlp_dim y luego proyecta de vuelta a qkv_dim
        y = common_layers.MlpBlock(
            mlp_dim=self.mlp_dim,
            dtype=self.dtype,
            dropout_rate=self.dropout_rate
        )(y, deterministic=deterministic)
        
        # =====================================================================
        # PASO 9: SEGUNDA CONEXIÓN RESIDUAL
        # =====================================================================
        return x + y


# ==============================================================================
# SECCIÓN 4: ENCODER COMPLETO - WavspaEncoder
# ==============================================================================

class WavspaEncoder(nn.Module):
    """
    Encoder completo basado en WavSpA para tareas de clasificación y secuencias.
    
    ARQUITECTURA COMPLETA:
    ======================
    
    Input tokens → [Embedding] → [Positional Encoding] → 
    [WavspaBlock × N] → [LayerNorm] → [Classification Head]
    
    COMPONENTES:
    ============
    1. Token Embedding: Mapea índices discretos a vectores densos
    2. Positional Encoding: Añade información de posición
    3. N capas de WavspaBlock: Procesamiento multi-escala
    4. LayerNorm final: Estabilización
    5. Classification Head: Proyección a clases (opcional)
    
    USO TÍPICO:
    ===========
    - Clasificación de texto (sentiment, topic, etc.)
    - Tareas de Long Range Arena (LRA)
    - Procesamiento de secuencias largas (hasta 16K tokens)
    
    Attributes:
        vocab_size (int): Tamaño del vocabulario
        emb_dim (int): Dimensión de embeddings
        num_layers (int): Número de capas WavSpA
        num_heads (int): Cabezas de atención por capa
        wavelet (str): Tipo de wavelet a usar
        level (int): Niveles de descomposición wavelet
    """
    
    # Configuración del vocabulario y embeddings
    vocab_size: int
    nb_features: int = 256
    use_bfloat16: bool = False
    emb_dim: int = 512
    
    # Configuración de atención
    num_heads: int = 8
    poly_dim: int = 1
    dtype: Any = jnp.float32
    num_layers: int = 6
    qkv_dim: int = 512
    mlp_dim: int = 2048
    max_len: int = 512
    
    # Regularización
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    
    # Configuración posicional
    learn_pos_emb: bool = False  # False = sinusoidal fijo
    
    # Configuración de clasificación
    classifier: bool = False
    classifier_pool: str = 'CLS'  # 'CLS', 'MEAN', 'MAX'
    num_classes: int = 10
    
    # Configuración de pesos
    tied_weights: bool = False
    shared_embedding: Optional[Callable] = None
    
    # Configuración de wavelet
    wavelet: str = 'db2'
    level: int = 2
    wlen: int = 32
    h: int = 16
    model_type: str = 'transformer'
    
    @nn.compact
    def __call__(self,
                 inputs,
                 inputs_positions=None,
                 inputs_segmentation=None,
                 train: bool = False):
        """
        Forward pass del encoder completo.
        
        FLUJO DETALLADO:
        ================
        
        1. Convertir tokens a embeddings
        2. Añadir [CLS] token si es clasificación
        3. Añadir positional encodings
        4. Aplicar dropout inicial
        5. Pasar por N capas WavSpA
        6. Normalización final
        7. Classification head (si aplica)
        
        Args:
            inputs (jnp.ndarray): Índices de tokens de forma (B, L)
            inputs_positions: Posiciones para packed sequences
            inputs_segmentation: Información de segmentos
            train (bool): Modo de entrenamiento (activa dropout)
        
        Returns:
            jnp.ndarray: 
                - Si classifier=False: encodings de forma (B, L, D)
                - Si classifier=True: logits de forma (B, num_classes)
        
        EJEMPLO DE USO:
        ===============
        ```python
        # Clasificación de sentimientos
        encoder = WavspaEncoder(
            vocab_size=50000,
            num_classes=2,  # positivo/negativo
            classifier=True,
            num_layers=6,
            wavelet='db2',
            level=3
        )
        
        logits = encoder(tokens, train=True)
        # logits: (batch, 2)
        ```
        """
        
        # Validar entrada
        assert inputs.ndim == 2, "Input debe ser 2D: (batch, length)"
        
        # =====================================================================
        # PASO 1: CREAR MÁSCARAS DE PADDING
        # =====================================================================
        # Asumimos que 0 = token de padding
        # src_padding_mask[i,j,0] = True si token j en batch i es válido
        src_padding_mask = (inputs > 0)[..., None]
        
        # =====================================================================
        # PASO 2: TOKEN EMBEDDING
        # =====================================================================
        # Mapear índices discretos a vectores densos
        # tokens ∈ ℤ^(B×L) → embeddings ∈ ℝ^(B×L×D)
        
        if self.shared_embedding is None:
            # Crear nueva capa de embedding
            input_embed = nn.Embed(
                num_embeddings=self.vocab_size,
                features=self.emb_dim,
                embedding_init=nn.initializers.normal(stddev=1.0)
            )
        else:
            # Usar embedding compartido (útil para encoder-decoder)
            input_embed = self.shared_embedding
        
        # Convertir a int32 y aplicar embedding
        x = inputs.astype('int32')
        x = input_embed(x)  # (B, L) → (B, L, emb_dim)
        
        # =====================================================================
        # PASO 3: AÑADIR [CLS] TOKEN PARA CLASIFICACIÓN
        # =====================================================================
        max_len = self.max_len
        
        if self.classifier and self.classifier_pool == 'CLS':
            # Crear token [CLS] aprendible
            # Este token agregará información de toda la secuencia
            cls = self.param(
                'cls',
                nn.initializers.zeros,
                (1, 1, self.emb_dim)
            )
            
            # Replicar para todo el batch
            cls = jnp.tile(cls, [x.shape[0], 1, 1])
            
            # Concatenar al inicio: [CLS, token_1, ..., token_L]
            x = jnp.concatenate([cls, x], axis=1)
            max_len += 1
            
            # Actualizar máscara de padding
            src_padding_mask = jnp.concatenate(
                [src_padding_mask[:, :1], src_padding_mask],
                axis=1
            )
        
        # =====================================================================
        # PASO 4: POSITIONAL ENCODING
        # =====================================================================
        # Añadir información de posición a los embeddings
        # 
        # Dos opciones:
        # a) Aprendible: posiciones como parámetros entrenables
        # b) Sinusoidal: PE(pos, 2i) = sin(pos/10000^(2i/d))
        #                PE(pos, 2i+1) = cos(pos/10000^(2i/d))
        
        pe_init = nn.initializers.normal(stddev=0.02) if self.learn_pos_emb else None
        
        x = common_layers.AddPositionEmbs(
            posemb_init=pe_init,
            max_len=max_len,
            name='posembed_input'
        )(x, inputs_positions=inputs_positions)
        
        # Dropout inicial
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        
        # =====================================================================
        # PASO 5: CONFIGURAR TIPO DE DATO
        # =====================================================================
        # bfloat16 puede acelerar entrenamiento en TPU/GPU modernas
        if self.use_bfloat16:
            x = x.astype(jnp.bfloat16)
            dtype = jnp.bfloat16
        else:
            dtype = jnp.float32
        
        # =====================================================================
        # PASO 6: STACK DE CAPAS WavSpA
        # =====================================================================
        # Aplicar N capas de procesamiento wavelet + atención
        
        for lyr in range(self.num_layers):
            x = WavspaBlock(
                wav_dim=self.emb_dim,
                qkv_dim=self.qkv_dim,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                poly_dim=self.poly_dim,
                wavelet=self.wavelet,
                level=self.level,
                model_type=self.model_type,
                wlen=self.wlen,
                nb_features=self.nb_features,
                max_len=max_len,
                L=max_len,
                dtype=self.dtype,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                name=f'encoderblock_{lyr}'
            )(
                x,
                inputs_segmentation=inputs_segmentation,
                padding_mask=src_padding_mask,
                deterministic=not train
            )
        
        # =====================================================================
        # PASO 7: NORMALIZACIÓN FINAL
        # =====================================================================
        encoded = nn.LayerNorm(dtype=dtype, name='encoder_norm')(x)
        
        # =====================================================================
        # PASO 8: CLASSIFICATION HEAD (OPCIONAL)
        # =====================================================================
        if self.classifier:
            # Aplicar pooling y proyección a clases
            # Soporta: 'CLS', 'MEAN', 'MAX'
            encoded = common_layers.classifier_head(
                encoded,
                self.num_classes,
                self.mlp_dim,
                pooling_mode=self.classifier_pool
            )
            # encoded: (B, num_classes)
        
        return encoded


# ==============================================================================
# SECCIÓN 5: DUAL ENCODER - Para tareas de similitud
# ==============================================================================

class WavspaDualEncoder(nn.Module):
    """
    Dual Encoder basado en WavSpA para tareas de similitud de pares.
    
    ARQUITECTURA:
    =============
    
            Input1                  Input2
              ↓                       ↓
        [WavspaEncoder]         [WavspaEncoder]
              ↓                       ↓
           encoding1              encoding2
              ↓                       ↓
              └───────[Interaction]────┘
                          ↓
                    [Classification]
                          ↓
                     Similarity Score
    
    APLICACIONES:
    =============
    - Natural Language Inference (NLI)
    - Paraphrase Detection
    - Semantic Textual Similarity
    - Question Answering Retrieval
    
    INTERACCIONES SOPORTADAS:
    =========================
    1. None: Concatenación simple [enc1; enc2]
    2. "NLI": [enc1; enc2; |enc1-enc2|; enc1⊙enc2]
       donde ⊙ es multiplicación elemento a elemento
    
    Attributes:
        interaction (str): Tipo de interacción entre encodings
    """
    
    vocab_size: int
    use_bfloat16: bool = False
    emb_dim: int = 512
    num_heads: int = 8
    poly_dim: int = 1
    dtype: Any = jnp.float32
    num_layers: int = 6
    qkv_dim: int = 512
    mlp_dim: int = 2048
    max_len: int = 512
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    learn_pos_emb: bool = False
    classifier: bool = False
    classifier_pool: str = 'CLS'
    num_classes: int = 10
    tied_weights: bool = False
    shared_embedding: Optional[Callable] = None
    interaction: str = None  # 'NLI', None
    
    wavelet: str = 'db2'
    level: int = 2
    wlen: int = 32
    model_type: str = 'transformer'
    
    @nn.compact
    def __call__(self,
                 inputs1,
                 inputs2,
                 inputs1_positions=None,
                 inputs2_positions=None,
                 inputs1_segmentation=None,
                 inputs2_segmentation=None,
                 train: bool = False):
        """
        Forward pass para dual encoding.
        
        FLUJO:
        ======
        1. Encodear inputs1 e inputs2 independientemente
        2. Aplicar función de interacción
        3. Clasificar la relación
        
        Args:
            inputs1 (jnp.ndarray): Primera secuencia (B, L1)
            inputs2 (jnp.ndarray): Segunda secuencia (B, L2)
            train (bool): Modo de entrenamiento
        
        Returns:
            jnp.ndarray: Logits de clasificación (B, num_classes)
        
        EJEMPLO:
        ========
        ```python
        # Natural Language Inference
        dual_encoder = WavspaDualEncoder(
            vocab_size=50000,
            num_classes=3,  # entailment, contradiction, neutral
            interaction='NLI'
        )
        
        premise = jnp.array([[1, 2, 3, ...]])     # "El gato está en el tejado"
        hypothesis = jnp.array([[4, 5, 6, ...]]) # "Un animal está afuera"
        
        logits = dual_encoder(premise, hypothesis, train=True)
        # logits: (1, 3)
        ```
        """
        
        # =====================================================================
        # CREAR ENCODER COMPARTIDO
        # =====================================================================
        # Ambas secuencias usan el mismo encoder (pesos compartidos)
        # Esto garantiza un espacio de representación consistente
        
        encoder = WavspaEncoder(
            vocab_size=self.vocab_size,
            use_bfloat16=self.use_bfloat16,
            emb_dim=self.emb_dim,
            num_heads=self.num_heads,
            poly_dim=self.poly_dim,
            num_layers=self.num_layers,
            qkv_dim=self.qkv_dim,
            mlp_dim=self.mlp_dim,
            max_len=self.max_len,
            wavelet=self.wavelet,
            level=self.level,
            wlen=self.wlen,
            model_type=self.model_type,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            name='encoder'
        )
        
        # =====================================================================
        # ENCODEAR AMBAS SECUENCIAS
        # =====================================================================
        inputs1_encoded = encoder(
            inputs=inputs1,
            inputs_positions=inputs1_positions,
            inputs_segmentation=inputs1_segmentation,
            train=train
        )
        
        inputs2_encoded = encoder(
            inputs=inputs2,
            inputs_positions=inputs2_positions,
            inputs_segmentation=inputs2_segmentation,
            train=train
        )
        
        # =====================================================================
        # INTERACCIÓN Y CLASIFICACIÓN
        # =====================================================================
        # Combinar los dos encodings y clasificar su relación
        encoded = common_layers.classifier_head_dual(
            inputs1_encoded,
            inputs2_encoded,
            self.num_classes,
            self.mlp_dim,
            pooling_mode=self.classifier_pool,
            interaction=self.interaction
        )
        
        return encoded


# ==============================================================================
# FIN DEL ARCHIVO - RESUMEN PARA EDUCACIÓN
# ==============================================================================
"""
RESUMEN CONCEPTUAL DE WavSpA
=============================

1. MOTIVACIÓN:
   - Transformers estándar: O(L²) complejidad cuadrática
   - Problema: No escalan a secuencias largas (L > 4096)
   - Solución: Procesar en dominio wavelet multi-escala

2. IDEA CENTRAL:
   - Descomponer señal en múltiples escalas (wavelets)
   - Aplicar atención en cada escala independientemente
   - Reconstruir señal procesada
   
   Ventaja: Complejidad reducida de O(L²) a ~O(L log L)

3. COMPONENTES CLAVE:
   a) Wavelets Adaptativas: Aprenden filtros óptimos para los datos
   b) Atención Multi-Escala: Captura patrones local y global
   c) Reconstrucción Perfecta: No se pierde información

4. VARIANTES IMPLEMENTADAS:
   - AdaWavSpA: Wavelets completamente adaptativas
   - OrthoWavSpA: Wavelets ortogonales parametrizadas
   - LiftWavSpA: Esquema de lifting adaptativo

5. RESULTADOS:
   - State-of-the-art en Long Range Arena benchmark
   - Mejoras de 10-20% sobre Transformer base
   - Escalable hasta 16K tokens

6. MATEMÁTICA FUNDAMENTAL:
   - Teoría de Wavelets: Base ortonormal multi-resolución
   - Análisis Multi-Escala: Descomposición espacio-frecuencia
   - Self-Attention: Mecanismo de contextualización global

REFERENCIAS:
============
- Paper: "WavSpA: Wavelet Space Attention" (2022)
- Daubechies Wavelets: Clásico de análisis wavelet
- Transformer: "Attention is All You Need" (Vaswani et al., 2017)
- LRA: "Long Range Arena" benchmark (Tay et al., 2021)

Para más detalles, ver:
- /wavspa/: Implementación core de wavelets
- /lra_benchmarks/: Experimentos y datasets
- README.md: Instrucciones de uso
"""
