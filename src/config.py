import os
import multiprocessing
import tensorflow as tf
from tensorflow.keras import mixed_precision

# ==================== PATH SETTINGS ====================
# Dynamic path detection for modular structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.txt")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "cv_checkpoints")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Ensure all necessary directories exist
for path in [CHECKPOINT_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(path, exist_ok=True)

# ==================== HARDWARE & PERFORMANCE ====================
# Get system CPU count for parallel processing
CPU_COUNT = multiprocessing.cpu_count()

# Worker counts for feature extraction and caching
# Optimized to prevent deadlocks while maximizing speed
CACHE_WORKERS = min(8, CPU_COUNT)
FEATURE_WORKERS = min(CPU_COUNT, 10)
EXTRACT_WORKERS = min(8, CPU_COUNT)

# GPU Settings: Memory growth and mixed precision
# Works on CPU as well (mixed_precision just won't add speed on CPU)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # mixed_float16 uses float16 for calculations but float32 for weights
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

# ==================== MODEL PARAMETERS ====================
N_FOLDS = 10
N_MODELS = 4
BATCH_SIZE = 512
EPOCHS = 15
RANDOM_SEEDS = [42, 123, 456, 789]

# ==================== VISUAL SETTINGS ====================
# Colors for tqdm progress bars to distinguish phases
COLOR_CACHE = 'green'     # Phase: Building RAM caches
COLOR_FEATURE = 'yellow'  # Phase: Computing weighted features
COLOR_EXTRACT = 'cyan'    # Phase: Vectorized URL extraction
COLOR_FOLD = 'magenta'   # Phase: Cross-validation progress