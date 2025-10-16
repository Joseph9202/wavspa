#!/usr/bin/env python3
"""Test de importaciones básicas para WavSpA"""

import sys

def test_jax():
    """Test JAX import and basic operations"""
    try:
        import jax
        import jax.numpy as jnp
        print(f"✅ JAX {jax.__version__}")
        print(f"   Devices: {jax.devices()}")

        # Test basic operation
        x = jnp.ones((2, 3))
        y = jnp.sum(x)
        print(f"   Basic ops: {x.shape} -> sum = {y}")
        return True
    except Exception as e:
        print(f"❌ JAX error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_flax():
    """Test Flax import"""
    try:
        import flax
        import flax.linen as nn
        print(f"✅ Flax {flax.__version__}")
        return True
    except Exception as e:
        print(f"❌ Flax error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_wavspa():
    """Test WavSpA module import"""
    try:
        import wavspa
        print(f"✅ WavSpA module loaded")
        print(f"   Location: {wavspa.__file__}")
        return True
    except Exception as e:
        print(f"❌ WavSpA import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_wavelets():
    """Test PyWavelets import"""
    try:
        import pywt
        print(f"✅ PyWavelets {pywt.__version__}")
        # Test basic wavelet
        wavelist = pywt.wavelist(kind='discrete')[:5]
        print(f"   Available wavelets (sample): {wavelist}")
        return True
    except Exception as e:
        print(f"❌ PyWavelets error: {e}")
        return False

def main():
    print("="*60)
    print("WavSpA Import Tests")
    print("="*60)

    results = {
        'JAX': test_jax(),
        'Flax': test_flax(),
        'PyWavelets': test_wavelets(),
        'WavSpA': test_wavspa(),
    }

    print("\n" + "="*60)
    print("RESUMEN:")
    print("="*60)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")

    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 ¡TODOS LOS TESTS PASARON!")
        return 0
    else:
        print("\n⚠️  ALGUNOS TESTS FALLARON")
        return 1

if __name__ == "__main__":
    sys.exit(main())
