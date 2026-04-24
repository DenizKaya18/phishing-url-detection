# ===== COLAB'DA İLK HÜCREYE EKLE - TÜM KÜTÜPHANELERI YÜK =====

# Bu hücreyi Colab'da çalıştır (ilk şey olarak)
# Run'a tıkla (▶️) ve bitmesini bekle
 
print("📦 TÜM KÜTÜPHANELER YÜKLENİYOR...\n")

# pip güncelleştir
import subprocess
import sys

def install_package(package_name, pip_name=None):
    """Kütüphane yükle"""
    if pip_name is None:
        pip_name = package_name

    print(f"⬇️ {pip_name} yükleniyor...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
    print(f"✓ {pip_name} yüklendi\n")

# 1. TensorFlow (GPU ile)
print("1️⃣  TensorFlow yükleniyor (GPU destekli)...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tensorflow[and-cuda]"])
print("✓ TensorFlow yüklendi\n")

# 2. Data Processing
install_package("pandas", "pandas")
install_package("numpy", "numpy")

# 3. Scikit-learn
install_package("scikit-learn", "scikit-learn")

# 4. wordsegment (URL'deki sözcükleri parçalamak için)
install_package("wordsegment", "wordsegment")

# 5. tldextract (TLD'yi ayıklamak için - Alan adı bilgisi)
install_package("tldextract", "tldextract")

# 6. Görselleştirme
install_package("matplotlib", "matplotlib")
install_package("seaborn", "seaborn")

# 7. İsteğe bağlı ama önerilir
install_package("psutil", "psutil")  # Bellek izleme için

print("\n" + "="*50)
print("✅ TÜM KÜTÜPHANELER YÜKLENDİ!")
print("="*50)

# Kontrol et
print("\n🔍 Kütüphaneleri kontrol ediyor...")
try:
    import tensorflow as tf
    print(f"✓ TensorFlow: {tf.__version__}")
except:
    print("❌ TensorFlow yüklenemedi")

try:
    import pandas as pd
    print(f"✓ Pandas: {pd.__version__}")
except:
    print("❌ Pandas yüklenemedi")

try:
    import numpy as np
    print(f"✓ NumPy: {np.__version__}")
except:
    print("❌ NumPy yüklenemedi")

try:
    from sklearn import __version__ as sklearn_version
    print(f"✓ Scikit-learn: {sklearn_version}")
except:
    print("❌ Scikit-learn yüklenemedi")

try:
    from wordsegment import load
    print(f"✓ wordsegment: yüklü")
except:
    print("❌ wordsegment yüklenemedi")

try:
    import tldextract
    print(f"✓ tldextract: yüklü")
except:
    print("❌ tldextract yüklenemedi")

try:
    import matplotlib
    print(f"✓ matplotlib: {matplotlib.__version__}")
except:
    print("❌ matplotlib yüklenemedi")

try:
    import seaborn
    print(f"✓ seaborn: {seaborn.__version__}")
except:
    print("❌ seaborn yüklenemedi")

print("\n" + "="*50)
print("✅ HAZIR OLABILIRSIN!")
print("="*50)