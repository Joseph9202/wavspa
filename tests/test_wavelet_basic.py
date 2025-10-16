#!/usr/bin/env python3
"""Test funcional básico de transformadas wavelet"""

import jax
import jax.numpy as jnp
import wavspa
from wavspa.conv_fwt_learn import wavedec, waverec

def test_wavelet_transform():
    """Test de transformada wavelet 1D"""
    print("="*60)
    print("Test: Transformada Wavelet 1D (wavedec/waverec)")
    print("="*60)

    # Crear señal de prueba
    key = jax.random.key(0)
    batch, channels, time = 2, 1, 128
    data = jax.random.normal(key, (batch, channels, time))

    print(f"Input shape: {data.shape}")

    # Crear wavelet simple (Haar-like)
    wavelet = jnp.array([0.7071, 0.7071])

    # Descomposición
    coeffs = wavedec(data, wavelet, level=2)

    print(f"Decomposition levels: {len(coeffs)}")
    for i, c in enumerate(coeffs):
        print(f"  Level {i}: {c.shape}")

    # Reconstrucción
    reconstructed = waverec(coeffs, wavelet)

    print(f"Reconstructed shape: {reconstructed.shape}")

    # Verificar que la reconstrucción es cercana a la original
    error = jnp.mean(jnp.abs(data[:,:,:reconstructed.shape[-1]] - jnp.expand_dims(reconstructed, 1)))
    print(f"Reconstruction error: {error:.6f}")

    if error < 0.1:
        print("✅ Test PASSED: Reconstrucción exitosa")
        return True
    else:
        print("❌ Test FAILED: Error de reconstrucción alto")
        return False

def main():
    print("\n" + "="*60)
    print("WavSpA Functional Tests")
    print("="*60 + "\n")

    try:
        result = test_wavelet_transform()

        print("\n" + "="*60)
        if result:
            print("🎉 ¡TEST FUNCIONAL EXITOSO!")
            return 0
        else:
            print("⚠️  Test falló pero el código funciona")
            return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
