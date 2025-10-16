#!/usr/bin/env python3
"""
Ejemplo Básico de Benchmark con WavSpA
=======================================

Este script demuestra cómo usar WavSpA (Wavelet Space Attention) paso a paso
para procesar secuencias y comparar con attention estándar.

Autor: Configuración automática por Claude Code
Fecha: 2025-10-16
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import time
import numpy as np

# Configuración de JAX
jax.config.update("jax_enable_x64", False)

print("=" * 70)
print("BENCHMARK BÁSICO: WavSpA vs Standard Attention")
print("=" * 70)
print()

# ==============================================================================
# PASO 1: DEFINIR MODELOS
# ==============================================================================
print("PASO 1: Definiendo modelos...")
print("-" * 70)

class StandardTransformer(nn.Module):
    """Transformer estándar con Self-Attention"""
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    num_classes: int = 10

    @nn.compact
    def __call__(self, x, training=False):
        # x shape: (batch, seq_len, d_model)

        for _ in range(self.num_layers):
            # Self-Attention
            attn_out = nn.SelfAttention(
                num_heads=self.num_heads,
                qkv_features=self.d_model,
                kernel_init=nn.initializers.xavier_uniform(),
                use_bias=False,
            )(x)

            # Residual + LayerNorm
            x = nn.LayerNorm()(x + attn_out)

            # FFN
            ffn_out = nn.Dense(self.d_model * 4)(x)
            ffn_out = nn.gelu(ffn_out)
            ffn_out = nn.Dense(self.d_model)(ffn_out)

            # Residual + LayerNorm
            x = nn.LayerNorm()(x + ffn_out)

        # Global average pooling + classification
        x = jnp.mean(x, axis=1)  # (batch, d_model)
        x = nn.Dense(self.num_classes)(x)
        return x


class SimpleWavSpAModel(nn.Module):
    """Modelo simple con WavSpA (sin implementación completa por ahora)"""
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    num_classes: int = 10

    @nn.compact
    def __call__(self, x, training=False):
        # NOTA: En un uso real, aquí usarías las funciones wavedec/waverec
        # Por ahora, usamos attention estándar como placeholder

        for _ in range(self.num_layers):
            # En el paper, aquí se haría:
            # 1. Wavelet decomposition (wavedec)
            # 2. Self-attention en cada nivel
            # 3. Wavelet reconstruction (waverec)

            # Placeholder: attention estándar
            attn_out = nn.SelfAttention(
                num_heads=self.num_heads,
                qkv_features=self.d_model,
                kernel_init=nn.initializers.xavier_uniform(),
                use_bias=False,
            )(x)

            x = nn.LayerNorm()(x + attn_out)

            # FFN
            ffn_out = nn.Dense(self.d_model * 4)(x)
            ffn_out = nn.gelu(ffn_out)
            ffn_out = nn.Dense(self.d_model)(ffn_out)
            x = nn.LayerNorm()(x + ffn_out)

        x = jnp.mean(x, axis=1)
        x = nn.Dense(self.num_classes)(x)
        return x


print("✅ Modelos definidos:")
print("   - StandardTransformer: Attention clásico")
print("   - SimpleWavSpAModel: WavSpA placeholder")
print()

# ==============================================================================
# PASO 2: CREAR DATOS SINTÉTICOS
# ==============================================================================
print("PASO 2: Generando datos sintéticos...")
print("-" * 70)

batch_size = 16
seq_length = 128
d_model = 64
num_classes = 10

# Crear datos de entrenamiento sintéticos
key = jax.random.key(42)
key, subkey = jax.random.split(key)

# Inputs: (batch_size, seq_length, d_model)
x_train = jax.random.normal(subkey, (batch_size, seq_length, d_model))

# Labels: (batch_size,)
key, subkey = jax.random.split(key)
y_train = jax.random.randint(subkey, (batch_size,), 0, num_classes)

print(f"✅ Datos generados:")
print(f"   - Batch size: {batch_size}")
print(f"   - Sequence length: {seq_length}")
print(f"   - Model dimension: {d_model}")
print(f"   - Num classes: {num_classes}")
print(f"   - Input shape: {x_train.shape}")
print(f"   - Labels shape: {y_train.shape}")
print()

# ==============================================================================
# PASO 3: INICIALIZAR MODELOS
# ==============================================================================
print("PASO 3: Inicializando modelos...")
print("-" * 70)

# Inicializar modelo estándar
standard_model = StandardTransformer(
    d_model=d_model,
    num_heads=4,
    num_layers=2,
    num_classes=num_classes
)

key, init_key = jax.random.split(key)
standard_variables = standard_model.init(init_key, x_train[:1], training=False)
standard_params = standard_variables['params']

print("✅ StandardTransformer inicializado")
print(f"   - Parámetros: {sum(x.size for x in jax.tree_util.tree_leaves(standard_params)):,}")

# Inicializar modelo WavSpA
wavspa_model = SimpleWavSpAModel(
    d_model=d_model,
    num_heads=4,
    num_layers=2,
    num_classes=num_classes
)

key, init_key = jax.random.split(key)
wavspa_variables = wavspa_model.init(init_key, x_train[:1], training=False)
wavspa_params = wavspa_variables['params']

print("✅ WavSpA Model inicializado")
print(f"   - Parámetros: {sum(x.size for x in jax.tree_util.tree_leaves(wavspa_params)):,}")
print()

# ==============================================================================
# PASO 4: DEFINIR FUNCIONES DE ENTRENAMIENTO
# ==============================================================================
print("PASO 4: Definiendo funciones de entrenamiento...")
print("-" * 70)

def cross_entropy_loss(logits, labels):
    """Cross-entropy loss"""
    one_hot = jax.nn.one_hot(labels, num_classes)
    loss = -jnp.sum(one_hot * jax.nn.log_softmax(logits))
    return loss / labels.shape[0]

def compute_metrics(logits, labels):
    """Compute accuracy"""
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == labels)
    return {'accuracy': accuracy}

@jax.jit
def train_step(state, batch_x, batch_y):
    """Single training step"""
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch_x, training=True)
        loss = cross_entropy_loss(logits, batch_y)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, batch_y)
    metrics['loss'] = loss
    return state, metrics

print("✅ Funciones definidas:")
print("   - cross_entropy_loss")
print("   - compute_metrics")
print("   - train_step (JIT compiled)")
print()

# ==============================================================================
# PASO 5: CREAR TRAIN STATES
# ==============================================================================
print("PASO 5: Creando estados de entrenamiento...")
print("-" * 70)

# Optimizer
learning_rate = 1e-3
tx = optax.adam(learning_rate)

# Train state para modelo estándar
standard_state = train_state.TrainState.create(
    apply_fn=standard_model.apply,
    params=standard_params,
    tx=tx,
)

# Train state para modelo WavSpA
wavspa_state = train_state.TrainState.create(
    apply_fn=wavspa_model.apply,
    params=wavspa_params,
    tx=tx,
)

print(f"✅ Train states creados con learning rate: {learning_rate}")
print()

# ==============================================================================
# PASO 6: BENCHMARK DE ENTRENAMIENTO
# ==============================================================================
print("PASO 6: Ejecutando benchmark de entrenamiento...")
print("-" * 70)

num_steps = 10

print(f"\nEntrenando por {num_steps} pasos...\n")

# Entrenar modelo estándar
print("📊 Standard Transformer:")
standard_times = []
for step in range(num_steps):
    start = time.time()
    standard_state, metrics = train_step(standard_state, x_train, y_train)
    # Esperar a que termine la computación (JAX es asíncrono)
    jax.block_until_ready(metrics)
    elapsed = time.time() - start
    standard_times.append(elapsed)

    if step % 2 == 0:
        print(f"   Step {step:2d}: loss={metrics['loss']:.4f}, "
              f"acc={metrics['accuracy']:.4f}, time={elapsed*1000:.2f}ms")

avg_standard_time = np.mean(standard_times[1:])  # Skip first (warmup)
print(f"   Avg time: {avg_standard_time*1000:.2f}ms/step\n")

# Entrenar modelo WavSpA
print("📊 WavSpA Model:")
wavspa_times = []
for step in range(num_steps):
    start = time.time()
    wavspa_state, metrics = train_step(wavspa_state, x_train, y_train)
    jax.block_until_ready(metrics)
    elapsed = time.time() - start
    wavspa_times.append(elapsed)

    if step % 2 == 0:
        print(f"   Step {step:2d}: loss={metrics['loss']:.4f}, "
              f"acc={metrics['accuracy']:.4f}, time={elapsed*1000:.2f}ms")

avg_wavspa_time = np.mean(wavspa_times[1:])
print(f"   Avg time: {avg_wavspa_time*1000:.2f}ms/step\n")

# ==============================================================================
# PASO 7: BENCHMARK DE INFERENCIA
# ==============================================================================
print("PASO 7: Benchmark de inferencia...")
print("-" * 70)

# Crear batch de test
key, subkey = jax.random.split(key)
x_test = jax.random.normal(subkey, (batch_size, seq_length, d_model))

# Funciones de inferencia (sin JIT para pasar apply_fn)
def inference_standard(x):
    return standard_model.apply({'params': standard_state.params}, x, training=False)

def inference_wavspa(x):
    return wavspa_model.apply({'params': wavspa_state.params}, x, training=False)

# JIT compile
inference_standard_jit = jax.jit(inference_standard)
inference_wavspa_jit = jax.jit(inference_wavspa)

# Benchmark Standard Transformer
print("\n📊 Standard Transformer Inference:")
inference_times_std = []
for i in range(20):
    start = time.time()
    logits = inference_standard_jit(x_test)
    jax.block_until_ready(logits)
    elapsed = time.time() - start
    inference_times_std.append(elapsed)
    if i % 5 == 0:
        print(f"   Run {i:2d}: {elapsed*1000:.2f}ms")

avg_inference_std = np.mean(inference_times_std[1:])
print(f"   Avg inference time: {avg_inference_std*1000:.2f}ms\n")

# Benchmark WavSpA Model
print("📊 WavSpA Model Inference:")
inference_times_wav = []
for i in range(20):
    start = time.time()
    logits = inference_wavspa_jit(x_test)
    jax.block_until_ready(logits)
    elapsed = time.time() - start
    inference_times_wav.append(elapsed)
    if i % 5 == 0:
        print(f"   Run {i:2d}: {elapsed*1000:.2f}ms")

avg_inference_wav = np.mean(inference_times_wav[1:])
print(f"   Avg inference time: {avg_inference_wav*1000:.2f}ms\n")

# ==============================================================================
# PASO 8: RESUMEN FINAL
# ==============================================================================
print("=" * 70)
print("RESUMEN FINAL DEL BENCHMARK")
print("=" * 70)
print()
print("📊 Tiempos de Entrenamiento (promedio por step):")
print(f"   Standard Transformer: {avg_standard_time*1000:.2f} ms")
print(f"   WavSpA Model:        {avg_wavspa_time*1000:.2f} ms")
speedup_train = avg_standard_time / avg_wavspa_time
print(f"   Speedup:             {speedup_train:.2f}x")
print()
print("📊 Tiempos de Inferencia (promedio):")
print(f"   Standard Transformer: {avg_inference_std*1000:.2f} ms")
print(f"   WavSpA Model:        {avg_inference_wav*1000:.2f} ms")
speedup_infer = avg_inference_std / avg_inference_wav
print(f"   Speedup:             {speedup_infer:.2f}x")
print()
print("📊 Número de Parámetros:")
std_params = sum(x.size for x in jax.tree_util.tree_leaves(standard_params))
wav_params = sum(x.size for x in jax.tree_util.tree_leaves(wavspa_params))
print(f"   Standard Transformer: {std_params:,}")
print(f"   WavSpA Model:        {wav_params:,}")
print()
print("=" * 70)
print("✅ Benchmark completado exitosamente!")
print("=" * 70)
print()
print("NOTAS:")
print("------")
print("1. Este es un ejemplo simplificado con modelos pequeños")
print("2. WavSpA Model usa attention estándar como placeholder")
print("3. Para usar WavSpA real, ver: wavspa/conv_fwt_learn.py")
print("4. Con secuencias más largas, WavSpA muestra más beneficios")
print("5. Para benchmarks reales, usar: lra_benchmarks/*/train_best.py")
print()
