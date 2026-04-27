import subprocess
import sys

def install_package(package_name, pip_name=None):
    """Install library"""
    if pip_name is None:
        pip_name = package_name

    print(f"⬇️ Installing {pip_name}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
    print(f"✓ {pip_name} installed\n")

# 1. TensorFlow (with GPU)
print("1️⃣  Installing TensorFlow (with GPU support)...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tensorflow[and-cuda]"])
print("✓ TensorFlow installed\n")

# 2. Data Processing
install_package("pandas", "pandas")
install_package("numpy", "numpy")

# 3. Scikit-learn
install_package("scikit-learn", "scikit-learn")

# 4. wordsegment (for splitting words in URLs)
install_package("wordsegment", "wordsegment")

# 5. tldextract (for extracting TLD - domain information)
install_package("tldextract", "tldextract")

# 6. Visualization
install_package("matplotlib", "matplotlib")
install_package("seaborn", "seaborn")

# 7. Optional but recommended
install_package("psutil", "psutil")  # For memory monitoring

print("\n" + "="*50)
print("✅ ALL LIBRARIES INSTALLED!")
print("="*50)

# Check installations
print("\n🔍 Checking libraries...")
try:
    import tensorflow as tf
    print(f"✓ TensorFlow: {tf.__version__}")
except:
    print("❌ TensorFlow installation failed")

try:
    import pandas as pd
    print(f"✓ Pandas: {pd.__version__}")
except:
    print("❌ Pandas installation failed")

try:
    import numpy as np
    print(f"✓ NumPy: {np.__version__}")
except:
    print("❌ NumPy installation failed")

try:
    from sklearn import __version__ as sklearn_version
    print(f"✓ Scikit-learn: {sklearn_version}")
except:
    print("❌ Scikit-learn installation failed")

try:
    from wordsegment import load
    print(f"✓ wordsegment: installed")
except:
    print("❌ wordsegment installation failed")

try:
    import tldextract
    print(f"✓ tldextract: installed")
except:
    print("❌ tldextract installation failed")

try:
    import matplotlib
    print(f"✓ matplotlib: {matplotlib.__version__}")
except:
    print("❌ matplotlib installation failed")

try:
    import seaborn
    print(f"✓ seaborn: {seaborn.__version__}")
except:
    print("❌ seaborn installation failed")

print("\n" + "="*50)
print("✅ YOU ARE READY!")
print("="*50)
