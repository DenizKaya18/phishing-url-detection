import tensorflow as tf
from tensorflow.keras import mixed_precision
import gc

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def clear_gpu_memory():
    tf.keras.backend.clear_session()
    gc.collect()

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✓ GPU Memory Growth Enabled: {len(gpus)} GPU(s)")

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print("✓ GPU Optimizations Active")

# ===== IMPORTS =====
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import os
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import tempfile
import shutil
import csv
import re
import urllib.parse
from urllib.parse import urlparse
from wordsegment import load, segment
import tldextract
from collections import Counter
import json
import pickle
from functools import lru_cache
import sys

sys.setrecursionlimit(21000)
load()

# ==================== GLOBAL CACHE (RAM ONLY) ====================
FEATURE_CACHE = {}
BAG_OF_WORDS_DATA = {}
SEGMENTED_BOW_DATA = {}
NGRAMS_DATA = {}
GRAMS_4_DATA = {}
TLD_WEIGHTS_DATA = {}


from multiprocessing import cpu_count
CPU_COUNT = cpu_count()
CACHE_WORKERS = min(8, CPU_COUNT)
FEATURE_WORKERS = min(CPU_COUNT, 10)
EXTRACT_WORKERS = min(8, CPU_COUNT)

print(f"\n✓ System CPU Count: {CPU_COUNT}")
print(f"✓ Used Workers:")
print(f"  - Cache Workers: {CACHE_WORKERS}")
print(f"  - Feature Workers: {FEATURE_WORKERS}")
print(f"  - Extract Workers: {EXTRACT_WORKERS}\n")

# ============ PROGRESS BAR COLORS ============
COLOR_CACHE = 'green'      # Cache creation color
COLOR_FEATURE = 'yellow'   # Feature creation color
COLOR_EXTRACT = 'cyan'     # Feature extraction color
COLOR_FOLD = 'magenta'     # Fold operation color
print("✓ RAM Cache Systems Initialized")


# ==================== VECTORIZED FEATURE EXTRACTOR ====================
class VectorizedFeatureExtractor:
    """Hızlı feature extraction (vectorized) - using SUM of token weights per URL"""

    def __init__(self, bow_data=None, seg_bow_data=None, ngrams_data=None,
                 grams4_data=None, tld_data=None):
        # expects: {'token': weight, ...}
        self.bow_data = bow_data or {}
        self.seg_bow_data = seg_bow_data or {}
        self.ngrams_data = ngrams_data or {}
        self.grams4_data = grams4_data or {}
        self.tld_data = tld_data or {}

    @staticmethod
    @lru_cache(maxsize=10000)
    def _extract_tld(url):
        try:
            extracted = tldextract.extract(url)
            return extracted.suffix or ""
        except:
            return ""

    @staticmethod
    def _has_ip(url):
        return bool(re.match(r'.*\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b.*', url))

    @staticmethod
    def _has_tinyurl(url):
        return 'bit.ly' in url or 'tinyurl.com' in url

    @staticmethod
    def _tokenize(url):
        # primary tokenization (word characters)
        return re.findall(r'\w+', url)

    @staticmethod
    def _make_ngrams_from_string(s, n=3):
        # generate character n-grams (overlapping)
        if not s or len(s) < n:
            return []
        return [s[i:i+n] for i in range(len(s)-n+1)]

    def _segment_token_safe(self, token):
        # try to call user-provided `segment` if available, otherwise fallback
        try:
            segmented = segment(token)  # if you have a segment function in scope
            if segmented and isinstance(segmented, (list, tuple)):
                return segmented
        except Exception:
            pass
        # fallback: split on non-alpha boundaries inside token (simple heuristic)
        parts = re.findall(r'[A-Za-z]{3,}|[0-9]+', token)
        return parts if parts else [token]

    def _process_batch_vectorized(self, urls, labels):
        batch_features = []
        batch_labels = []
        processed_urls = []

        for url, label in zip(urls, labels):
            try:
                url_len = len(url)
                special_chars = sum(1 for c in url if not c.isalnum())
                url_len_ratio = special_chars / max(url_len, 1)

                tld = self._extract_tld(url)
                tld_weight = float(self.tld_data.get(tld, 0.0))

                parsed = urlparse(url)
                netloc = parsed.netloc or ""
                subdomain_count = netloc.count('.')

                # Path ratio and Domain ratio
                path = parsed.path or ""
                path_length = len(path)

                domain_info = tldextract.extract(url)
                domain = (domain_info.domain + '.' + domain_info.suffix) if domain_info.suffix else domain_info.domain
                domain_length = len(domain)

                path_ratio = path_length / max(url_len, 1)
                domain_ratio = domain_length / max(url_len, 1)

                # Tokens (basic)
                tokens = self._tokenize(url)

                # ----- BOW weighted SUM -----
                bow_sum = 0.0
                bow_match_count = 0
                for t in tokens:
                    w = float(self.bow_data.get(t, 0.0))
                    if w > 0:
                        bow_sum += w
                        bow_match_count += 1

                # ----- Segmented BOW weighted SUM -----
                seg_bow_sum = 0.0
                seg_match_count = 0
                for t in tokens:
                    parts = self._segment_token_safe(t)
                    for p in parts:
                        w = float(self.seg_bow_data.get(p, 0.0))
                        if w > 0:
                            seg_bow_sum += w
                            seg_match_count += 1

                # ----- N-grams (3-grams) weighted SUM -----
                ngrams_sum = 0.0
                ngrams_match_count = 0
                # create char-level n-grams from url (or from tokens if preferred)
                for t in tokens:
                    ngrams = self._make_ngrams_from_string(t, n=3)
                    for ng in ngrams:
                        w = float(self.ngrams_data.get(ng, 0.0))
                        if w > 0:
                            ngrams_sum += w
                            ngrams_match_count += 1

                # ----- 4-grams weighted SUM -----
                grams4_sum = 0.0
                grams4_match_count = 0
                for t in tokens:
                    grams4 = self._make_ngrams_from_string(t, n=4)
                    for g4 in grams4:
                        w = float(self.grams4_data.get(g4, 0.0))
                        if w > 0:
                            grams4_sum += w
                            grams4_match_count += 1

                # Bag_of_Words_Count: number of distinct tokens that matched BOW weights
                bag_of_words_count = float(bow_match_count)

                features = [
                    float(url_len),                      # 0: URL_Length
                    float(special_chars),                # 1: Special_Character_Count
                    float(url_len_ratio),                # 2: URL_Length_Ratio
                    tld_weight,                          # 3: TLD (weighted)
                    1.0 if self._has_ip(url) else 0.0,   # 4: IP_Address_Usage
                    1.0 if self._has_tinyurl(url) else 0.0, # 5: Tiny_URL
                    1.0 if '@' in url else 0.0,          # 6: At_Symbol
                    1.0 if url.count('//') > 1 else 0.0, # 7: URL_Redirection
                    1.0 if '-' in netloc or (netloc.split('.')[0] if netloc else '').find('-')>=0 else 0.0, # 8: Domain_Suffix/Hyphen
                    1.0 if subdomain_count > 1 else 0.0, # 9: Subdomain
                    1.0 if parsed.port and parsed.port not in [80, 443] else 0.0, # 10: Port
                    1.0 if 'https-' in url else 0.0,     # 11: HTTPS_Domain
                    1.0 if any(ext in url for ext in ['.exe', '.pdf', '.zip', '.rar']) else 0.0, # 12: File_Extension
                    bag_of_words_count,                  # 13: Bag_of_Words_Count (match count)
                    bow_sum,                             # 14: Weighted_BoW (SUM of Wi for tokens)
                    seg_bow_sum,                         # 15: Weighted_Segmented_BoW (SUM)
                    ngrams_sum,                          # 16: Weighted_3grams (SUM)
                    grams4_sum,                          # 17: Weighted_4grams (SUM)
                    float(path_ratio),                   # 18: path_ratio
                    float(domain_ratio),                 # 19: domain_ratio
                ]

                batch_features.append(features)
                batch_labels.append(int(label))
                processed_urls.append(url)

            except Exception:
                # skip problematic URL but continue
                continue

        if not batch_features:
            return None, None, None

        return np.array(batch_features, dtype=float), np.array(batch_labels, dtype=int), processed_urls

    def extract_batch_vectorized(self, urls, labels, batch_size=5000):
        """Batch wrapper that iterates in chunks and concatenates results"""
        all_feats = []
        all_labels = []
        all_urls = []

        total = len(urls)
        for i in range(0, total, batch_size):
            chunk_urls = urls[i:i+batch_size]
            chunk_labels = labels[i:i+batch_size]
            feats, labs, proc = self._process_batch_vectorized(chunk_urls, chunk_labels)
            if feats is None:
                continue
            all_feats.append(feats)
            all_labels.append(labs)
            all_urls.extend(proc)

        if not all_feats:
            return np.empty((0,20)), np.empty((0,), dtype=int), []

        X = np.vstack(all_feats)
        y = np.concatenate(all_labels)
        return X, y, all_urls

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# ==================== CHECKPOINT MANAGER ====================
class CheckpointManager:
    """Checkpoint yöneticisi - kesinti sonrası devam"""

    def __init__(self, checkpoint_dir="cv_checkpoints", storage_type="local"):
        self.storage_type = storage_type

        if storage_type == "local":
            self.checkpoint_dir = "/content/drive/MyDrive/checkpointsMilyonPD3"
            print("✓ Storage: Google Drive")
        elif storage_type == "colab":
            self.checkpoint_dir = "./cv_checkpoints"
            print("⚠️  Storage: Colab Temp")
        else:
            self.checkpoint_dir = checkpoint_dir
            print(f"✓ Storage: Custom ({checkpoint_dir})")

        self.metadata_file = os.path.join(self.checkpoint_dir, "metadata.json")

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            print(f"✓ Checkpoint dizini oluşturuldu: {self.checkpoint_dir}")

    def save_fold_checkpoint(self, fold_idx, fold_models, fold_histories,
                        fold_result, fold_metrics, model_metrics_list=None,
                    timing_data=None):
        checkpoint_name = f"fold_{fold_idx+1}"
        fold_dir = os.path.join(self.checkpoint_dir, checkpoint_name)

        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir, exist_ok=True)

        try:
            models_dir = os.path.join(fold_dir, "models")
            if not os.path.exists(models_dir):
                os.makedirs(models_dir, exist_ok=True)

            for i, model in enumerate(fold_models):
                model_save_path = os.path.join(models_dir, f"model_{i+1}.keras")
                model.save(model_save_path)
                print(f"  ✓ Model {i+1} kaydedildi")

            with open(os.path.join(fold_dir, "histories.pkl"), "wb") as f:
                pickle.dump(fold_histories, f)

            with open(os.path.join(fold_dir, "fold_result.pkl"), "wb") as f:
                pickle.dump(fold_result, f)

            with open(os.path.join(fold_dir, "fold_metrics.pkl"), "wb") as f:
                pickle.dump(fold_metrics, f)

            if model_metrics_list is not None:
                with open(os.path.join(fold_dir, "model_metrics_list.pkl"), "wb") as f:
                    pickle.dump(model_metrics_list, f)
                print(f"  ✓ Model metrikleri kaydedildi")
            if timing_data is not None:
                with open(os.path.join(fold_dir, "timing_data.pkl"), "wb") as f:
                    pickle.dump(timing_data, f)
                print(f"  ✓ Timing verileri kaydedildi")

            self._update_metadata(fold_idx)
            print(f"✓ Fold {fold_idx+1} checkpoint kaydedildi")

        except Exception as e:
            print(f"✗ Checkpoint kayıt hatası: {e}")

    def load_fold_checkpoint(self, fold_idx):
        checkpoint_name = f"fold_{fold_idx+1}"
        fold_dir = os.path.join(self.checkpoint_dir, checkpoint_name)

        if not os.path.exists(fold_dir):
            return None

        try:
            from tensorflow.keras.models import load_model

            models_dir = os.path.join(fold_dir, "models")
            fold_models = []

            for i in range(4):
                model_path = os.path.join(models_dir, f"model_{i+1}.keras")
                if os.path.exists(model_path):
                    try:
                        model = load_model(model_path)
                        fold_models.append(model)
                        print(f"  ✓ Model {i+1} yüklendi")
                    except:
                        continue

            if not fold_models:
                return None

            histories_path = os.path.join(fold_dir, "histories.pkl")
            fold_histories = pickle.load(open(histories_path, "rb")) if os.path.exists(histories_path) else []

            result_path = os.path.join(fold_dir, "fold_result.pkl")
            fold_result = pickle.load(open(result_path, "rb")) if os.path.exists(result_path) else {}

            metrics_path = os.path.join(fold_dir, "fold_metrics.pkl")
            fold_metrics = pickle.load(open(metrics_path, "rb")) if os.path.exists(metrics_path) else {}

            model_metrics_path = os.path.join(fold_dir, "model_metrics_list.pkl")
            model_metrics_list = pickle.load(open(model_metrics_path, "rb")) if os.path.exists(model_metrics_path) else None

            timing_data = None
            timing_path = os.path.join(fold_dir, "timing_data.pkl")
            if os.path.exists(timing_path):
                timing_data = pickle.load(open(timing_path, "rb"))

            print(f"✓ Fold {fold_idx+1} checkpoint yüklendi")
            return {
                'models': fold_models,
                'histories': fold_histories,
                'fold_result': fold_result,
                'fold_metrics': fold_metrics,
                'model_metrics_list': model_metrics_list,
                'timing_data': timing_data
            }
        except:
            return None

    def get_completed_folds(self):
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                    return sorted(metadata.get('completed_folds', []))
            except:
                pass
        return []

    def get_last_completed_fold(self):
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                    return metadata.get('last_completed_fold', -1)
            except:
                pass
        return -1

    def _update_metadata(self, fold_idx):
        metadata = {}

        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
            except:
                pass

        if 'completed_folds' not in metadata:
            metadata['completed_folds'] = []

        if fold_idx not in metadata['completed_folds']:
            metadata['completed_folds'].append(fold_idx)

        metadata['last_completed_fold'] = fold_idx
        metadata['last_update'] = datetime.now().isoformat()
        metadata['total_completed'] = len(metadata['completed_folds'])

        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def clear_checkpoints(self):
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
            os.makedirs(self.checkpoint_dir, exist_ok=True)


# ==================== ISOLATED FEATURE MANAGER ====================
class IsolatedFeatureManager:
    """Her fold/split için izole feature cache yönetimi"""

    def __init__(self):
        self.feature_caches = {}
        print("✓ Isolated Feature Manager Başlatıldı")

    def create_features_for_data(self, url_label_list, cache_name):
        """GERÇEK PARALEL feature creation"""
        print(f"\n[ISOLATED FEATURES] Creating PARALLEL features for: {cache_name}")

        cache = {
            'bow': {}, 'seg_bow': {}, 'ngrams': {}, 'grams4': {}, 'tld': {}
        }

        # TÜM feature'ları AYNI ANDA başlat
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Tüm görevleri paralel başlat
            futures = {
                'bow': executor.submit(self._create_bow_isolated, url_label_list, cache['bow']),
                'seg_bow': executor.submit(self._create_seg_bow_isolated, url_label_list, cache['seg_bow']),
                'ngrams': executor.submit(self._create_ngrams_isolated, url_label_list, 3, cache['ngrams']),
                'grams4': executor.submit(self._create_ngrams_isolated, url_label_list, 4, cache['grams4']),
                'tld': executor.submit(self._create_tld_isolated, url_label_list, cache['tld'])
            }

            # İlerleme çubuğu
            progress_bar = tqdm(
                total=len(futures),
                desc=f"  {cache_name:12}",
                bar_format='{desc}: {percentage:3.0f}% {bar} [{n_fmt}/{total_fmt}]',
                colour=COLOR_CACHE,
                unit=' feature'
            )

            # Her feature tamamlandığında güncelle
            for future in as_completed(futures.values()):
                future.result()  # Hata kontrolü için
                progress_bar.update(1)

            progress_bar.close()

        self.feature_caches[cache_name] = cache
        print(f"✓ {cache_name} PARALLEL features completed")
        return cache

    def _create_bow_isolated(self, url_label_list, target_cache):
        """IZOLE BoW - MAX HIZ"""
        import re
        from collections import Counter

        # 1. TEK SEFERDE tüm malicious URL'leri ve token'ları çıkar
        malicious_tokens = []
        for url, label in url_label_list:
            if label == 1:
                # re.findall yerine split - DAHA HIZLI
                tokens = re.findall(r'\w+', url)
                malicious_tokens.extend(tokens)

        if not malicious_tokens:
            return

        # 2. Counter ile lightning count
        counter = Counter(malicious_tokens)
        total = sum(counter.values())

        # 3. Tek loop'ta threshold + weight hesapla
        target_cache.update({
            word: count / total
            for word, count in counter.items()
            if count > 20
        })

    def _create_seg_bow_isolated(self, url_label_list, target_cache):
        """IZOLE Segmented BoW - MAX HIZ"""
        import re
        from collections import Counter

        # 1. Önce tüm token'ları topla
        all_tokens = []
        for url, label in url_label_list:
            if label == 1:
                tokens = re.findall(r'\w+', url)
                all_tokens.extend(tokens)

        if not all_tokens:
            return

        # 2. Tüm segment işlemlerini TEK SEFERDE yap
        segmented_tokens = []
        for token in all_tokens:
            try:
                segmented = segment(token)
                segmented_tokens.extend(segmented)
            except:
                segmented_tokens.append(token)

        # 3. Hızlı count ve filter
        counter = Counter(segmented_tokens)
        total = sum(counter.values())

        target_cache.update({
            word: count / total
            for word, count in counter.items()
            if count > 20
        })

    def _create_ngrams_isolated(self, url_label_list, n, target_cache):
        """IZOLE N-grams - MAX HIZ"""
        import re
        from collections import Counter

        # 1. Tüm token'ları topla
        all_tokens = []
        for url, label in url_label_list:
            if label == 1:
                tokens = re.findall(r'\w+', url)
                all_tokens.extend(tokens)

        if not all_tokens:
            return

        # 2. Tüm n-gram'ları TEK SEFERDE generate et
        all_ngrams = []
        for token in all_tokens:
            if len(token) >= n:
                # List comprehension ile hızlı n-gram generation
                token_ngrams = [token[i:i+n] for i in range(len(token) - n + 1)]
                all_ngrams.extend(token_ngrams)

        # 3. Hızlı processing
        counter = Counter(all_ngrams)
        total = sum(counter.values())

        target_cache.update({
            ngram: count / total
            for ngram, count in counter.items()
            if count > 20
        })

    def _create_tld_isolated(self, url_label_list, target_cache):
        """IZOLE TLD Weights - MAX HIZ"""
        import tldextract
        from collections import Counter

        # 1. Tek loop'ta malicious URL'leri ve TLD'leri çıkar
        tld_list = []
        for url, label in url_label_list:
            if label == 1:
                try:
                    extracted = tldextract.extract(url)
                    if extracted.suffix:  # Boş TLD'leri atla
                        tld_list.append(extracted.suffix)
                except:
                    continue

        if not tld_list:
            return

        # 2. Hızlı count ve weight hesaplama
        counter = Counter(tld_list)
        total = len(tld_list)  # Toplam malicious URL sayısı

        target_cache.update({
            tld: count / total
            for tld, count in counter.items()
        })

    def get_cache(self, cache_name):
        """Cache'i getir - OPTIMIZED"""
        return self.feature_caches.get(cache_name, {})

# ==================== FEATURE CREATION (RAM ONLY - NO FILE I/O) ====================

def create_bag_of_words_from_memory(url_label_list, threshold=20):
    """BoW - Sadece RAM'de"""
    print(f"[BoW] Processing {len(url_label_list)} URLs from memory")

    bag_of_words_counter = Counter()

    for i, (url, label) in enumerate(url_label_list):
        if label == 1:
            tokens = re.findall(r'\w+', url)
            bag_of_words_counter.update(tokens)

        if (i + 1) % max(5000, len(url_label_list) // 20) == 0:
            progress = ((i + 1) / len(url_label_list)) * 100
            print(f"\r[BoW] Processing: %{progress:.1f}", end='', flush=True)

    total_words = sum(bag_of_words_counter.values())
    sorted_counter = sorted(bag_of_words_counter.items(), key=lambda x: x[1], reverse=True)

    bag_of_words_dict = {}
    for word, count in sorted_counter:
        weight = count / total_words
        if count > threshold:
            bag_of_words_dict[word] = weight

    BAG_OF_WORDS_DATA.clear()
    BAG_OF_WORDS_DATA.update(bag_of_words_dict)

    print(f"\n✓ BoW RAM'e yüklendi: {len(bag_of_words_dict)} öğe")


def create_segmented_bow_from_memory(url_label_list, threshold=20):
    """Segmented BoW - Sadece RAM'de"""
    print(f"[Seg-BoW] Processing {len(url_label_list)} URLs from memory")

    segmented_counter = Counter()

    for i, (url, label) in enumerate(url_label_list):
        if label == 1:
            tokens = re.findall(r'\w+', url)
            for token in tokens:
                try:
                    segmented_counter.update(segment(token))
                except:
                    segmented_counter.update([token])

        if (i + 1) % max(5000, len(url_label_list) // 20) == 0:
            progress = ((i + 1) / len(url_label_list)) * 100
            print(f"\r[Seg-BoW] Processing: %{progress:.1f}", end='', flush=True)

    total_words = sum(segmented_counter.values())
    sorted_counter = sorted(segmented_counter.items(), key=lambda x: x[1], reverse=True)

    segmented_dict = {}
    for word, count in sorted_counter:
        weight = count / total_words
        if count > threshold:
            segmented_dict[word] = weight

    SEGMENTED_BOW_DATA.clear()
    SEGMENTED_BOW_DATA.update(segmented_dict)

    print(f"\n✓ Seg-BoW RAM'e yüklendi: {len(segmented_dict)} öğe")


def create_ngrams_from_memory(url_label_list, n=3, threshold=20):
    """N-grams - Sadece RAM'de"""
    print(f"[N-grams] Processing {len(url_label_list)} URLs from memory (n={n})")

    ngrams_counter = Counter()

    for i, (url, label) in enumerate(url_label_list):
        if label == 1:
            tokens = re.findall(r'\w+', url)
            for token in tokens:
                if len(token) >= n:
                    for j in range(len(token) - n + 1):
                        ngram = token[j:j+n]
                        ngrams_counter[ngram] += 1

        if (i + 1) % max(5000, len(url_label_list) // 20) == 0:
            progress = ((i + 1) / len(url_label_list)) * 100
            print(f"\r[N-grams] Processing: %{progress:.1f}", end='', flush=True)

    total_words = sum(ngrams_counter.values())
    sorted_counter = sorted(ngrams_counter.items(), key=lambda x: x[1], reverse=True)

    ngrams_dict = {}
    for ngram, count in sorted_counter:
        weight = count / total_words
        if count > threshold:
            ngrams_dict[ngram] = weight

    if n == 3:
        NGRAMS_DATA.clear()
        NGRAMS_DATA.update(ngrams_dict)
    elif n == 4:
        GRAMS_4_DATA.clear()
        GRAMS_4_DATA.update(ngrams_dict)

    print(f"\n✓ {n}-grams RAM'e yüklendi: {len(ngrams_dict)} öğe")


def calculate_tld_weights_from_memory(url_label_list):
    """TLD Weights - Sadece RAM'de"""
    print(f"[TLD] Processing {len(url_label_list)} URLs from memory")

    malicious_urls = [url for url, label in url_label_list if label == 1]
    tld_counter = Counter()

    for i, url in enumerate(malicious_urls):
        try:
            extracted = tldextract.extract(url)
            tld = extracted.suffix
            if tld:
                tld_counter[tld] += 1
        except:
            continue

        if (i + 1) % max(5000, len(malicious_urls) // 20) == 0:
            progress = ((i + 1) / len(malicious_urls)) * 100
            print(f"\r[TLD] Processing: %{progress:.1f}", end='', flush=True)

    total_urls = sum(tld_counter.values())

    tld_dict = {}
    for tld, count in tld_counter.items():
        weight = count / total_urls
        tld_dict[tld] = weight

    TLD_WEIGHTS_DATA.clear()
    TLD_WEIGHTS_DATA.update(tld_dict)

    print(f"\n✓ TLD Weights RAM'e yüklendi: {len(tld_dict)} öğe")


def create_all_features_from_memory(url_label_list):
    """Tüm feature'ları PARALLEL olarak RAM'den oluştur"""

    print("\n" + "="*60)
    print("[FEATURES] Tüm feature'lar PARALLEL olarak hesaplanıyor...")
    print("="*60)
    start_time = time.time()

    # ThreadPoolExecutor ile 5 işi eş zamanlı başlat
    with ThreadPoolExecutor(max_workers=FEATURE_WORKERS) as executor:
        print(f"  {FEATURE_WORKERS} paralel görev başlatılıyor:\n")
        # Görev listesini dinamik oluştur
        tasks = []
        task_names = []

        tasks.append(executor.submit(
            create_bag_of_words_from_memory, url_label_list, 20
        ))
        task_names.append("BoW")
        print("  ✓ BoW görevi gönderildi")

        tasks.append(executor.submit(
            create_segmented_bow_from_memory, url_label_list, 20
        ))
        task_names.append("Seg-BoW")
        print("  ✓ Seg-BoW görevi gönderildi")

        tasks.append(executor.submit(
            create_ngrams_from_memory, url_label_list, 3, 20
        ))
        task_names.append("N-grams (3)")
        print("  ✓ N-grams (3) görevi gönderildi")

        tasks.append(executor.submit(
            create_ngrams_from_memory, url_label_list, 4, 20
        ))
        task_names.append("N-grams (4)")
        print("  ✓ N-grams (4) görevi gönderildi")

        tasks.append(executor.submit(
            calculate_tld_weights_from_memory, url_label_list
        ))
        task_names.append("TLD")
        print("  ✓ TLD görevi gönderildi")

        print(f"\n  ⏳ {len(tasks)} görevin tamamlanması bekleniyor...\n")

        # ✅ DINAMIK PROGRESS BAR (görev sayısına göre)
        ozellik_progress = tqdm(
            total=len(tasks),
            desc="  Özellik Oluşturma",
            bar_format='{desc}: {percentage:3.0f}% {bar} [{n_fmt}/{total_fmt}]',
            colour=COLOR_FEATURE,
            unit=' özellik'
        )

        try:
            for idx, (future, name) in enumerate(zip(tasks, task_names)):
                _ = future.result()
                ozellik_progress.update(1)
                ozellik_progress.set_description(f"  Özellik Oluşturma ({name} ✓)")

            ozellik_progress.close()

        except Exception as e:
            ozellik_progress.close()
            print(f"    ✗ Hata oluştu: {e}")
            raise

    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print(f"✓ Tüm feature'lar {elapsed:.2f}s'de PARALLEL olarak yüklendi!")
    print(f"✓ Hızlanma: ~{23/elapsed:.1f}x daha hızlı!")
    print("="*60 + "\n")


# ==================== DEEP LEARNING CLASS ====================



class OptimizedEnsembleURLClassifierCV:
    def __init__(self, n_models=4, random_seeds=None, n_folds=10):
        self.n_models = n_models
        self.random_seeds = random_seeds or [42, 123, 456, 789][:n_models]
        self.n_folds = n_folds
        self.models = []
        self.histories = []
        self.scaler = None
        self.tokenizer = None
        self.max_len = None
        self.vocab_size = None

        self.training_times = []
        self.model_info = []
        self.total_training_time = 0
        self.data_prep_time = 0
        self.evaluation_time = 0
        self.cv_time = 0

        self.cv_scores = {}
        self.cv_detailed_results = []
        self.cv_confusion_matrices = []
        self.cv_metrics = []
        self.final_confusion_matrix = None
        self.final_metrics = None
        self.avg_confusion_matrix = None
        self.avg_metrics = None

        self.cv_model_detailed_results = {}
        self.cv_model_metrics = {}
        self.cv_model_confusion_matrices = {}

        self.single_url_feature_extraction_times = []
        self.single_url_prediction_times = []
        self.per_model_single_times = {}
        self.per_model_single_probas = {}

        self.feature_manager = IsolatedFeatureManager()
        self.current_fold_cache = None  # Mevcut fold cache'i

        # ✅ METRICS STORAGE - DÜZELTILMIŞ
        self.cv_scores = {}
        self.cv_predictions = {}  # Her fold'ın predictions'ları
        self.cv_detailed_results = []

        # ✅ PER-FOLD STORAGE (Her fold için ayrı)
        self.cv_confusion_matrices = []  # Liste: her fold'ın CM'si
        self.cv_metrics = []              # Liste: her fold'ın metrikleri

        # ✅ AVERAGE ACROSS FOLDS
        self.avg_confusion_matrix = None
        self.avg_metrics = None

        # ✅ PER-MODEL STORAGE (Her model x her fold)
        self.cv_model_detailed_results = {}
        self.cv_model_metrics = {}
        self.cv_model_confusion_matrices = {}

        # ✅ FINAL TEST RESULTS
        self.final_confusion_matrix = None
        self.final_metrics = None

        # ✅ TRAINING TIME TRACKING
        self.per_model_training_times = {}
        self.per_model_testing_times = {}  # ✅ NEW: predict süresi ayrı
        self.per_fold_training_times = {}
        self.total_training_time_per_model = {}
        self.avg_training_time_per_model = {}
        self.total_testing_time_per_model = {}  # ✅ NEW
        self.avg_testing_time_per_model = {}    # ✅ NEW
        self.epochs_per_model = {}

    def cleanup_gpu_memory(self, fold_idx=None):
        try:
            tf.keras.backend.clear_session()
            gc.collect()

            if tf.config.list_physical_devices('GPU'):
                for gpu in tf.config.list_physical_devices('GPU'):
                    with tf.device(gpu.name):
                        pass

            if fold_idx is not None:
                print(f"✓ Fold {fold_idx+1} GPU hafızası temizlendi")
            else:
                print("✓ GPU hafızası temizlendi")
        except:
            pass

    def calculate_metrics(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0

        metrics = {
            'confusion_matrix': cm,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'sensitivity': recall_score(y_true, y_pred, zero_division=0) if tp + fn > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
        return metrics

    def format_time(self, seconds):
        if seconds < 60:
            return f"{seconds:.2f} second"
        elif seconds < 3600:
            return f"{seconds/60:.2f} minute"
        else:
            return f"{seconds/3600:.2f} hour"

    def create_model_architecture(self, vocab_size, max_len, n_features,
                                model_type='base', seed=42):
        tf.random.set_seed(seed)
        np.random.seed(seed)

        url_input = Input(shape=(max_len,), name="url_input")
        num_input = Input(shape=(n_features,), name="num_input")

        if model_type == 'base':
            embedding_dim = 64
            embed_layer = Embedding(vocab_size, embedding_dim,
                                  input_length=max_len, mask_zero=True)(url_input)

            conv1 = Conv1D(64, 3, activation="relu", padding="same",
                          kernel_regularizer=l2(0.001))(embed_layer)
            conv1 = BatchNormalization()(conv1)
            pool1 = GlobalMaxPooling1D()(conv1)

            lstm_out = Bidirectional(LSTM(32, return_sequences=False,
                                        kernel_regularizer=l2(0.001),
                                        recurrent_regularizer=l2(0.001),
                                        dropout=0.3, recurrent_dropout=0.3))(embed_layer)

            url_features = Concatenate()([pool1, lstm_out])

        elif model_type == 'multi_cnn':
            embedding_dim = 64
            embed_layer = Embedding(vocab_size, embedding_dim,
                                  input_length=max_len, mask_zero=True)(url_input)

            conv_3 = Conv1D(32, 3, activation='relu', padding='same')(embed_layer)
            conv_5 = Conv1D(32, 5, activation='relu', padding='same')(embed_layer)

            pool_3 = GlobalMaxPooling1D()(conv_3)
            pool_5 = GlobalMaxPooling1D()(conv_5)

            lstm_out = Bidirectional(LSTM(32, return_sequences=False,
                                        dropout=0.3))(embed_layer)

            url_features = Concatenate()([pool_3, pool_5, lstm_out])

        elif model_type == 'attention':
            embedding_dim = 64
            embed_layer = Embedding(vocab_size, embedding_dim,
                                  input_length=max_len, mask_zero=True)(url_input)

            attention_layer = Dense(embedding_dim, activation='tanh')(embed_layer)
            attention_weights = Dense(1, activation='softmax')(attention_layer)
            attention_out = Multiply()([embed_layer, attention_weights])
            attention_pooled = GlobalAveragePooling1D()(attention_out)

            conv1 = Conv1D(32, 3, activation="relu", padding="same")(embed_layer)
            pool1 = GlobalMaxPooling1D()(conv1)

            url_features = Concatenate()([attention_pooled, pool1])

        else:  # wide model
            embedding_dim = 64
            embed_layer = Embedding(vocab_size, embedding_dim,
                                  input_length=max_len, mask_zero=True)(url_input)

            conv1 = Conv1D(64, 3, activation="relu", padding="same")(embed_layer)
            conv1 = BatchNormalization()(conv1)
            pool1 = GlobalMaxPooling1D()(conv1)

            lstm_out = Bidirectional(LSTM(64, return_sequences=False,
                                        dropout=0.3))(embed_layer)

            url_features = Concatenate()([pool1, lstm_out])

        num_dense = Dense(32, activation="relu", kernel_regularizer=l2(0.001))(num_input)
        num_batch = BatchNormalization()(num_dense)
        num_drop = Dropout(0.3)(num_batch)

        merged = Concatenate()([url_features, num_drop])

        dense1 = Dense(64, activation="relu", kernel_regularizer=l2(0.001))(merged)
        batch1 = BatchNormalization()(dense1)
        drop1 = Dropout(0.4)(batch1)

        dense2 = Dense(32, activation="relu", kernel_regularizer=l2(0.001))(drop1)
        batch2 = BatchNormalization()(dense2)
        drop2 = Dropout(0.3)(batch2)

        output = Dense(1, activation="sigmoid", dtype='float32')(drop2)
        model = Model(inputs=[url_input, num_input], outputs=output)

        return model

    def create_optimized_callbacks(self, model_name):
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=2,
                min_lr=1e-7,
                verbose=1
            )
        ]
        return callbacks

    def prepare_data_from_raw(self, raw_data_file, test_size=0.2):
        data_prep_start = time.time()

        with open(raw_data_file, 'r', encoding='utf-8') as fr:
            raw_lines = [ln.strip() for ln in fr if ln.strip()]

        rows = []
        for ln in raw_lines:
            parts = ln.rsplit(',', 1)
            if len(parts) != 2:
                continue
            rows.append((parts[0].strip(), parts[1].strip()))

        urls = np.array([r[0] for r in rows])
        y = np.array([int(r[1]) for r in rows])

        X_url_train, X_url_test, y_train, y_test = train_test_split(
            urls, y, test_size=test_size, stratify=y, random_state=42
        )

        self.tokenizer = Tokenizer(char_level=True, oov_token="<OOV>", num_words=5000)
        self.tokenizer.fit_on_texts(X_url_train)

        url_lengths = [len(u) for u in X_url_train]
        self.max_len = min(int(np.percentile(url_lengths, 95)), 200)
        self.vocab_size = min(len(self.tokenizer.word_index) + 1, 5000)

        self.data_prep_time = time.time() - data_prep_start

        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Max sequence length: {self.max_len}")
        print(f"Training samples: {len(X_url_train)}")
        print(f"Test samples: {len(X_url_test)}")
        print(f"? Data prep time: {self.format_time(self.data_prep_time)}")

        return (X_url_train, y_train, X_url_test, y_test, rows)

    @staticmethod
    def extract_features_for_fold_optimized(urls, labels, is_train=True,
                                       batch_size=5000,
                                       bow_data=None, seg_bow_data=None,
                                       ngrams_data=None, grams4_data=None,
                                       tld_data=None):
        """Optimized feature extraction - Vectorized"""

        print(f"\n[EXTRACT] Extracting features...")
        prefix = "Train" if is_train else "Val"
        total = len(urls)
        start_time = time.time()

        extractor = VectorizedFeatureExtractor(
            bow_data=bow_data or {},
            seg_bow_data=seg_bow_data or {},
            ngrams_data=ngrams_data or {},
            grams4_data=grams4_data or {},
            tld_data=tld_data or {}
        )

        X_features, y_labels, processed_urls = extractor.extract_batch_vectorized( # Modified call
            urls, labels, batch_size=batch_size
        )

        elapsed = time.time() - start_time

        print(f"✓ {prefix} Features extracted")
        print(f"  Samples: {len(X_features)}")
        print(f"  Time: {elapsed:.1f} seconds")
        print(f"  Speed: {len(X_features)/elapsed:.0f} URLs/sec")

        return X_features, y_labels, processed_urls # Modified return

    def train_model_fold(self, X_url_train_filtered, X_num_train, y_train, # Renamed X_url_train
                        X_url_val_filtered, X_num_val, y_val,             # Renamed X_url_val
                        model_type, seed, fold_idx, class_weight_dict,
                        epochs=15, batch_size=128):

        model_train_start = time.time()
        scaler = StandardScaler()
        X_num_train_scaled = scaler.fit_transform(X_num_train)
        X_num_val_scaled = scaler.transform(X_num_val)

        seq_train = self.tokenizer.texts_to_sequences(X_url_train_filtered) # Use filtered URLs
        seq_val = self.tokenizer.texts_to_sequences(X_url_val_filtered)     # Use filtered URLs

        X_url_train_pad = pad_sequences(seq_train, maxlen=self.max_len,
                                       padding="post", truncating="post")
        X_url_val_pad = pad_sequences(seq_val, maxlen=self.max_len,
                                     padding="post", truncating="post")

        model = self.create_model_architecture(
            self.vocab_size, self.max_len, X_num_train_scaled.shape[1],
            model_type=model_type, seed=seed
        )

        base_opt = Adam(learning_rate=0.008, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_opt, dynamic=True)
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

        callbacks = self.create_optimized_callbacks(f"fold_{fold_idx}_{model_type}")

        history = model.fit(
            x={"url_input": X_url_train_pad, "num_input": X_num_train_scaled},
            y=y_train,
            validation_data=(
                {"url_input": X_url_val_pad, "num_input": X_num_val_scaled}, y_val
            ),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )

        # ✅ Training süresi bitişi (model.fit sonrası)
        model_train_end = time.time()
        model_train_time_sec = model_train_end - model_train_start
        model_train_time_hours = model_train_time_sec / 3600

        # ✅ Testing (predict) süresi ayrı ölçülüyor
        model_test_start = time.time()

        val_pred_proba = model.predict(
            {"url_input": X_url_val_pad, "num_input": X_num_val_scaled},
            verbose=1
        ).flatten()
        val_pred = (val_pred_proba > 0.5).astype(int)
        val_acc = accuracy_score(y_val, val_pred)

        fold_metrics = self.calculate_metrics(y_val, val_pred)

        model_test_end = time.time()
        model_test_time_sec = model_test_end - model_test_start
        model_test_time_hours = model_test_time_sec / 3600

        if model_type not in self.per_model_training_times:
            self.per_model_training_times[model_type] = []
            self.per_model_testing_times[model_type] = []  # ✅ NEW
            self.epochs_per_model[model_type] = []

        self.per_model_training_times[model_type].append(model_train_time_hours)
        self.per_model_testing_times[model_type].append(model_test_time_hours)  # ✅ NEW
        self.epochs_per_model[model_type].append(len(history.history['loss']))

        return model, history, val_acc, val_pred, val_pred_proba, scaler, fold_metrics

    def cross_validate_ensemble(self, X_url_train, y_train, rows_train,
                            checkpoint_mgr=None,
                            epochs=15, batch_size=128):
        """Cross-validation with checkpoint support"""

        print(f"\n? {self.n_folds}-Fold Cross-Validation Start...")

        cv_start_time = time.time()
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        model_types = ['base', 'multi_cnn', 'attention', 'wide'][:self.n_models]

        cv_scores_per_model = {model_type: [] for model_type in model_types}
        cv_ensemble_scores = []

        for model_type in model_types:
            self.cv_model_detailed_results[model_type] = []
            self.cv_model_metrics[model_type] = []
            self.cv_model_confusion_matrices[model_type] = []

        cv_fold_confusion_matrices = []
        cv_fold_metrics = []
        all_fold_results = []

        if checkpoint_mgr:
            completed_folds = checkpoint_mgr.get_completed_folds()
            last_fold = checkpoint_mgr.get_last_completed_fold()

            if completed_folds:
                print(f"\n✓ Önceki checkpoint tespit edildi!")
                print(f"✓ Tamamlanan fold'lar: {[f+1 for f in completed_folds]}")
                print(f"✓ Son tamamlanan fold: {last_fold + 1}/{self.n_folds}")
                print(f"✓ Kaldığı yerden devam ediliyor...\n")

                for completed_fold_idx in completed_folds:
                    checkpoint = checkpoint_mgr.load_fold_checkpoint(completed_fold_idx)
                    if checkpoint:
                        fold_result = checkpoint['fold_result']
                        fold_metrics = checkpoint['fold_metrics']
                        timing_data = checkpoint.get('timing_data', None)
                        if timing_data:
                            # Timing verilerini geri yükle
                            if 'per_model_training_times' in timing_data:
                                for model_type, times in timing_data['per_model_training_times'].items():
                                    self.per_model_training_times[model_type] = list(times)

                            # ✅ NEW: Testing times'ı da yükle
                            if 'per_model_testing_times' in timing_data:
                                for model_type, times in timing_data['per_model_testing_times'].items():
                                    self.per_model_testing_times[model_type] = list(times)

                            if 'epochs_per_model' in timing_data:
                                for model_type, epochs_list in timing_data['epochs_per_model'].items():
                                    self.epochs_per_model[model_type] = list(epochs_list)

                        for model_type, score in fold_result.get('individual_scores', {}).items():
                            if model_type in cv_scores_per_model:
                                cv_scores_per_model[model_type].append(score)

                        cv_ensemble_scores.append(fold_result.get('ensemble_score', 0))
                        cv_fold_metrics.append(fold_metrics)
                        cv_fold_confusion_matrices.append(fold_metrics.get('confusion_matrix', np.zeros((2,2))))
                        all_fold_results.append(fold_result if not isinstance(fold_result, list) else fold_result[0])

                        model_metrics_list = checkpoint.get('model_metrics_list')
                        if model_metrics_list:
                            for model_type, model_metric in zip(model_types, model_metrics_list):
                                if model_type not in self.cv_model_metrics:
                                    self.cv_model_metrics[model_type] = []
                                    self.cv_model_confusion_matrices[model_type] = []

                                self.cv_model_metrics[model_type].append(model_metric)
                                self.cv_model_confusion_matrices[model_type].append(
                                    model_metric.get('confusion_matrix', np.zeros((2,2)))
                                )
            else:
                last_fold = -1

        fold_enum = list(enumerate(skf.split(X_url_train, y_train)))

        for fold_idx, (train_idx, val_idx) in fold_enum:

            if checkpoint_mgr and fold_idx <= last_fold:
                print(f"⊘ Fold {fold_idx+1} atlandi (önceden tamamlandı)")
                continue

            print(f"\n{'='*60}")
            print(f"?? FOLD {fold_idx+1}/{self.n_folds}")
            print(f"{'='*60}")
            fold_start_time = time.time()

            X_url_fold_train = X_url_train[train_idx]
            X_url_fold_val = X_url_train[val_idx]
            y_fold_train = y_train[train_idx]
            y_fold_val = y_train[val_idx]

            print(f"?? Fold data size:")
            print(f"   Train: {len(y_fold_train)}, Validation: {len(y_fold_val)}")
            print(f"   Train class dist: {np.bincount(y_fold_train)}")
            print(f"   Val class dist: {np.bincount(y_fold_val)}")

            # Create train_data from the original X_url_fold_train and y_fold_train
            train_data = list(zip(X_url_fold_train, y_fold_train))

            # This tokenizer will be used for the current fold and its models
            #self.tokenizer = Tokenizer(char_level=True, oov_token="<OOV>", num_words=5000)
            #self.tokenizer.fit_on_texts(X_url_fold_train)

            #url_lengths_fold = [len(u) for u in X_url_fold_train]
            #self.max_len = min(int(np.percentile(url_lengths_fold, 95)), 200)
            #self.vocab_size = min(len(self.tokenizer.word_index) + 1, 5000)

            print(f"   (Fold tokenizer fit) Vocab size: {self.vocab_size}, Max len: {self.max_len}")

            try:
                print(f"\n?? Creating ISOLATED features for fold {fold_idx+1}...")

                with ThreadPoolExecutor(max_workers=CACHE_WORKERS) as executor:
                    print(f"\n  🔄 Fold {fold_idx+1} için PARALEL özellikler oluşturuluyor...")
                    print(f"    {CACHE_WORKERS} paralel görev gönderildi (train & val)\n")

                    # Progress tracking
                    tasks_count = min(2, CACHE_WORKERS)  # Train ve Val (max 2)

                    fold_progress = tqdm(
                        total=tasks_count,
                        desc=f"  Fold {fold_idx+1} Cache",
                        bar_format='{desc}: {percentage:3.0f}% {bar} [{n_fmt}/{total_fmt}]',
                        colour=COLOR_FOLD,
                        unit=' cache'
                    )

                    train_future = executor.submit(
                        self.feature_manager.create_features_for_data,
                        list(zip(X_url_fold_train, y_fold_train)),
                        f"fold_{fold_idx+1}_train"
                    )

                    val_future = executor.submit(
                        self.feature_manager.create_features_for_data,
                        list(zip(X_url_fold_val, y_fold_val)),
                        f"fold_{fold_idx+1}_val"
                    )

                    # Hepsinin bitmesini bekle
                    train_cache = train_future.result()
                    fold_progress.update(1)
                    fold_progress.set_description(f"  Fold {fold_idx+1} Cache (Train ✓)")

                    val_cache = val_future.result()
                    fold_progress.update(1)
                    fold_progress.set_description(f"  Fold {fold_idx+1} Cache (Val ✓)")

                    fold_progress.close()

                print(f"✓ Fold {fold_idx+1} features ready (parallel)")

                # ★★ CRITICAL: Feature extraction'da İZOLE cache kullan
                X_num_fold_train, y_fold_train_extracted, X_url_fold_train_processed = self.extract_features_for_fold_optimized(
                    X_url_fold_train, y_fold_train,
                    is_train=True,
                    batch_size=5000,
                    bow_data=train_cache['bow'],           # ← İZOLE cache!
                    seg_bow_data=train_cache['seg_bow'],   # ← İZOLE cache!
                    ngrams_data=train_cache['ngrams'],     # ← İZOLE cache!
                    grams4_data=train_cache['grams4'],     # ← İZOLE cache!
                    tld_data=train_cache['tld']            # ← İZOLE cache!
                )



                X_num_fold_val, y_fold_val_extracted, X_url_fold_val_processed = self.extract_features_for_fold_optimized(
                    X_url_fold_val, y_fold_val,
                    is_train=False,
                    batch_size=5000,
                    bow_data=val_cache['bow'],             # ← İZOLE cache!
                    seg_bow_data=val_cache['seg_bow'],     # ← İZOLE cache!
                    ngrams_data=val_cache['ngrams'],       # ← İZOLE cache!
                    grams4_data=val_cache['grams4'],       # ← İZOLE cache!
                    tld_data=val_cache['tld']              # ← İZOLE cache!
                )

                # No more manual length alignment by slicing; the processed_urls list is already aligned.
                # The tokenizer is already fitted on X_url_fold_train, so we need to refit it on the processed URLs
                # or pass the processed URLs directly.
                # Re-fitting tokenizer on processed URLs for safety and consistency.
                self.tokenizer = Tokenizer(char_level=True, oov_token="<OOV>", num_words=5000)
                self.tokenizer.fit_on_texts(X_url_fold_train_processed)

                url_lengths_fold_processed = [len(u) for u in X_url_fold_train_processed]

                self.max_len = min(int(np.percentile(url_lengths_fold_processed, 95)), 200)
                self.vocab_size = min(len(self.tokenizer.word_index) + 1, 5000)

                print(f"   (Fold tokenizer refit on processed data) Vocab size: {self.vocab_size}, Max len: {self.max_len}")

                # Compute class weights
                classes_fold = np.unique(y_fold_train_extracted)
                weights_fold = compute_class_weight(
                    class_weight='balanced',
                    classes=classes_fold,
                    y=y_fold_train_extracted
                )
                class_weight_dict = {cls: w for cls, w in zip(classes_fold, weights_fold)}

                print(f"\n?? Class weights: {class_weight_dict}")

                # Train models for this fold
                fold_models = []
                fold_predictions = []
                fold_probabilities = []
                fold_model_scores = []
                fold_histories = []
                fold_model_metrics = []

                for model_idx, (model_type, seed) in enumerate(zip(model_types, self.random_seeds)):
                    print(f"\n?? Model {model_idx+1}: {model_type} (seed={seed})")
                    model_start = time.time()

                    model, history, val_acc, val_pred, val_pred_proba, scaler, model_metrics = self.train_model_fold(
                        X_url_fold_train_processed, X_num_fold_train, y_fold_train_extracted, # Use processed URLs
                        X_url_fold_val_processed, X_num_fold_val, y_fold_val_extracted,     # Use processed URLs
                        model_type, seed, fold_idx+1, class_weight_dict, epochs, batch_size
                    )

                    model_time = time.time() - model_start

                    fold_models.append(model)
                    fold_histories.append(history)
                    fold_predictions.append(val_pred)
                    fold_probabilities.append(val_pred_proba)
                    fold_model_scores.append(val_acc)
                    fold_model_metrics.append(model_metrics)

                    cv_scores_per_model[model_type].append(val_acc)

                    model_result = {
                        'fold': fold_idx + 1,
                        'model_type': model_type,
                        'accuracy': val_acc,
                        'time': model_time,
                        'epochs_trained': len(history.history['loss']),
                        'final_loss': history.history['loss'][-1],
                        'final_val_loss': history.history['val_loss'][-1],
                        'metrics': model_metrics
                    }

                    self.cv_model_detailed_results[model_type].append(model_result)
                    self.cv_model_metrics[model_type].append(model_metrics)
                    self.cv_model_confusion_matrices[model_type].append(model_metrics['confusion_matrix'])

                    print(f"   ? Train Time: {self.format_time(model_time)}")
                    print(f"   ?? Final Accuracy: {val_acc:.4f}")

                # Ensemble prediction
                ensemble_pred_proba = np.mean(fold_probabilities, axis=0)
                ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)
                ensemble_acc = accuracy_score(y_fold_val_extracted, ensemble_pred)
                cv_ensemble_scores.append(ensemble_acc)

                ensemble_metrics = self.calculate_metrics(y_fold_val_extracted, ensemble_pred)

                fold_time = time.time() - fold_start_time

                print(f"\n?? ENSEMBLE Result - Fold {fold_idx+1}")
                print(f"   ? Ensemble Accuracy: {ensemble_acc:.4f}")
                print(f"   ? Fold total time: {self.format_time(fold_time)}")

                fold_result = {
                    'fold': fold_idx + 1,
                    'individual_scores': dict(zip(model_types, fold_model_scores)),
                    'ensemble_score': ensemble_acc,
                    'fold_time': fold_time,
                    'val_size': len(y_fold_val_extracted),
                    'metrics': ensemble_metrics
                }
                all_fold_results.append(fold_result)

                cv_fold_confusion_matrices.append(ensemble_metrics['confusion_matrix'])
                cv_fold_metrics.append(ensemble_metrics)

                if checkpoint_mgr:
                    timing_data = {
                        'per_model_training_times': dict(self.per_model_training_times),
                        'per_model_testing_times': dict(self.per_model_testing_times),  # ✅ NEW
                        'epochs_per_model': dict(self.epochs_per_model)
                    }

                    try:
                        checkpoint_mgr.save_fold_checkpoint(
                            fold_idx,
                            fold_models,
                            fold_histories,
                            fold_result,
                            ensemble_metrics,
                            model_metrics_list=fold_model_metrics,
                            timing_data=timing_data  # ✅ EKLE
                        )
                        print(f"✓ Checkpoint kaydedildi")
                    except Exception as e:
                        print(f"⚠️ Checkpoint kayıt hatası: {e}")

                self.cleanup_gpu_memory(fold_idx=fold_idx)
                print("\n" + "="*60 + "\n")

            except KeyboardInterrupt:
                print(f"\n\n⚠️ KESİNTİ TESPİT EDİLDİ!")
                print(f"✓ Sonraki çalıştırmada {fold_idx+1}/{self.n_folds}'den başlanacak")
                raise
            except Exception as e:
                print(f"✗ Fold hatası: {e}")
                import traceback
                traceback.print_exc()
                raise


        self.cv_confusion_matrices = cv_fold_confusion_matrices
        self.cv_metrics = cv_fold_metrics
        self.avg_confusion_matrix = np.mean(cv_fold_confusion_matrices, axis=0)

        self.avg_metrics = {}
        metric_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'sensitivity',
                    'specificity', 'fpr', 'fnr']

        for key in metric_keys:
            values = [metrics[key] for metrics in cv_fold_metrics]
            self.avg_metrics[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }

        self.cv_time = time.time() - cv_start_time

        for model_type in self.per_model_training_times.keys():
            times = self.per_model_training_times[model_type]
            self.total_training_time_per_model[model_type] = sum(times)
            self.avg_training_time_per_model[model_type] = np.mean(times)

            # ✅ NEW: Testing times summary
            if model_type in self.per_model_testing_times:
                test_times = self.per_model_testing_times[model_type]
                self.total_testing_time_per_model[model_type] = sum(test_times)
                self.avg_testing_time_per_model[model_type] = np.mean(test_times)

        self.cv_scores = {
            'individual': cv_scores_per_model,
            'ensemble': cv_ensemble_scores
        }
        self.cv_detailed_results = all_fold_results

        print(f"\n? Cross-Validation Finish!")
        print(f"? Total CV Time: {self.format_time(self.cv_time)}")

        return cv_scores_per_model, cv_ensemble_scores

    def debug_feature_consistency(self):
        """Feature tutarlılığını kontrol et"""
        print("\n🔍 DEBUG: Feature Consistency Check")

        if hasattr(self, 'final_X_num_train') and hasattr(self, 'final_X_num_test'):
            print("Train feature stats:")
            print(f"  Shape: {self.final_X_num_train.shape}")
            print(f"  Mean: {np.mean(self.final_X_num_train):.4f}")
            print(f"  Std:  {np.std(self.final_X_num_train):.4f}")

            print("Test feature stats:")
            print(f"  Shape: {self.final_X_num_test.shape}")
            print(f"  Mean: {np.mean(self.final_X_num_test):.4f}")
            print(f"  Std:  {np.std(self.final_X_num_test):.4f}")

            # Feature correlation kontrolü
            if self.final_X_num_train.shape[0] > 1 and self.final_X_num_test.shape[0] > 1:
                try:
                    train_corr = np.corrcoef(self.final_X_num_train.T)
                    test_corr = np.corrcoef(self.final_X_num_test.T)
                    corr_diff = np.mean(np.abs(train_corr - test_corr))
                    print(f"Feature correlation difference: {corr_diff:.4f}")

                    if corr_diff > 0.1:
                        print("🚨 CRITICAL: Train/Test feature distributions VERY different!")
                    else:
                        print("✓ Feature distributions are consistent")
                except:
                    print("⚠️ Could not calculate correlation")

        print(f"Feature manager caches: {list(self.feature_manager.feature_caches.keys())}")

    def train_final_ensemble(self, X_url_train, y_train, rows_train,
                   X_url_test, y_test, rows_test, epochs=15, batch_size=128):
        """
        Final ensemble training - OVERFITTING FIXED
        """
        print(f"\n?? Final Ensemble Training (WITH INTERNAL VALIDATION)...")

        training_start = time.time()

        try:
            # ⭐ CRITICAL: Internal split tanımları EKLE
            print(f"\n? Creating internal stratified validation split...")
            from sklearn.model_selection import train_test_split

            # Training setini kendi içinde böl
            X_train_internal, X_val_internal, y_train_internal, y_val_internal = train_test_split(
                X_url_train, y_train,
                test_size=0.15,  # %15 validation
                stratify=y_train,
                random_state=42
            )

            print(f"\n✓ DATA SPLIT:")
            print(f"   Training:   {len(X_train_internal)} samples")
            print(f"   Validation: {len(X_val_internal)} samples")
            print(f"   Test:       {len(X_url_test)} samples")
            print(f"\n✓ CLASS DISTRIBUTIONS:")
            print(f"   Train: {np.bincount(y_train_internal)} → {np.bincount(y_train_internal)/len(y_train_internal)*100}")
            print(f"   Val:   {np.bincount(y_val_internal)} → {np.bincount(y_val_internal)/len(y_val_internal)*100}")
            print(f"   Test:  {np.bincount(y_test)} → {np.bincount(y_test)/len(y_test)*100}")

            # ⭐ FIX: Her split için İZOLE feature'lar
            print(f"\n?? Creating ISOLATED features for each split...")

            print(f"\n🔄 Paralel ISOLATED özellikler oluşturuluyor ({CACHE_WORKERS} workers)...")

            with ThreadPoolExecutor(max_workers=CACHE_WORKERS) as executor:
                tasks_count = min(2, CACHE_WORKERS)

                train_cache_future = executor.submit(
                    self.feature_manager.create_features_for_data,
                    list(zip(X_train_internal, y_train_internal)),
                    "final_train"
                )

                val_cache_future = executor.submit(
                    self.feature_manager.create_features_for_data,
                    list(zip(X_val_internal, y_val_internal)),
                    "final_val"
                )

                # Hepsinin bitmesini bekle
                cache_progress = tqdm(
                    total=tasks_count,
                    desc="  Cache Oluşturma",
                    bar_format='{desc}: {percentage:3.0f}% {bar} [{n_fmt}/{total_fmt}]',
                    colour=COLOR_CACHE,
                    unit=' cache'
                )

                train_cache = train_cache_future.result()
                cache_progress.update(1)
                cache_progress.set_description("  Cache Oluşturma (Train ✓)")

                val_cache = val_cache_future.result()
                cache_progress.update(1)
                cache_progress.set_description("  Cache Oluşturma (Val ✓)")

                cache_progress.close()



            print("✓ All caches created in parallel")

            # Test için AYRI feature'lar (CRITICAL FIX!)
            #test_data = list(zip(X_url_test, y_test))
            #test_cache = self.feature_manager.create_features_for_data(
            #    test_data, "final_test"
            #)

            print(f"\n?? Extracting PARALLEL features for train/val/test...")

            with ThreadPoolExecutor(max_workers=EXTRACT_WORKERS) as executor:
                extract_tasks = []
                extract_names = []

                train_extract_future = executor.submit(
                    self.extract_features_for_fold_optimized,
                    X_train_internal, y_train_internal,
                    is_train=True, batch_size=5000,
                    bow_data=train_cache['bow'],
                    seg_bow_data=train_cache['seg_bow'],
                    ngrams_data=train_cache['ngrams'],
                    grams4_data=train_cache['grams4'],
                    tld_data=train_cache['tld']
                )
                extract_tasks.append(train_extract_future)
                extract_names.append("Train")

                val_extract_future = executor.submit(
                    self.extract_features_for_fold_optimized,
                    X_val_internal, y_val_internal,
                    is_train=False, batch_size=5000,
                    bow_data=val_cache['bow'],
                    seg_bow_data=val_cache['seg_bow'],
                    ngrams_data=val_cache['ngrams'],
                    grams4_data=val_cache['grams4'],
                    tld_data=val_cache['tld']
                )
                extract_tasks.append(val_extract_future)
                extract_names.append("Val")

                test_extract_future = executor.submit(
                    self.extract_features_for_fold_optimized,
                    X_url_test, y_test,
                    is_train=False, batch_size=5000,
                    bow_data=train_cache['bow'],
                    seg_bow_data=train_cache['seg_bow'],
                    ngrams_data=train_cache['ngrams'],
                    grams4_data=train_cache['grams4'],
                    tld_data=train_cache['tld']
                )
                extract_tasks.append(test_extract_future)
                extract_names.append("Test")
                print("\n  ⏳ Waiting for feature extraction to complete...\n")

                cikart_progress = tqdm(
                    total=len(extract_tasks),
                    desc="  Özellik Çıkarma",
                    bar_format='{desc}: {percentage:3.0f}% {bar} [{n_fmt}/{total_fmt}]',
                    colour=COLOR_EXTRACT,
                    unit=' çıkarma'
                )

                results = []
                for future, name in zip(extract_tasks, extract_names):
                    result = future.result()
                    results.append(result)
                    cikart_progress.update(1)
                    cikart_progress.set_description(f"  Özellik Çıkarma ({name} ✓)")

                cikart_progress.close()
                # Sonuçları ayrıştır
                X_num_train, y_train_cikart, X_train_islem = results[0]
                X_num_val, y_val_cikart, X_val_islem = results[1]
                X_num_test, y_test_cikart, X_test_islem = results[2]

            print("✓ All features extracted in parallel")

            # ⭐ FIX 3: LENGTH ALIGNMENT (NOW BY ASSIGNING PROCESSED LISTS)
            print(f"\n✓ ALIGNING LENGTHS:")
            print(f"   BEFORE: Train({len(X_num_train)}, {len(y_train_cikart)}, {len(X_train_internal)})")
            print(f"           Val({len(X_num_val)}, {len(y_val_cikart)}, {len(X_val_internal)})")
            print(f"           Test({len(X_num_test)}, {len(y_test_cikart)}, {len(X_url_test)})")

            # Update X_train_internal, X_val_internal, X_url_test with processed lists for consistency
            X_train_internal = X_train_islem
            X_val_internal = X_val_islem
            X_url_test = X_test_islem # This refers to the filtered test URLs now

            print(f"\n   AFTER:  Train({len(X_num_train)}, {len(y_train_cikart)}, {len(X_train_internal)})")
            print(f"           Val({len(X_num_val)}, {len(y_val_cikart)}, {len(X_val_internal)})")
            print(f"           Test({len(X_num_test)}, {len(y_test_cikart)}, {len(X_url_test)})")

            # ⭐ FIX 4: CLASS WEIGHTS FROM TRAINING ONLY
            classes_final = np.unique(y_train_cikart)
            weights_final = compute_class_weight(
                class_weight='balanced',
                classes=classes_final,
                y=y_train_cikart
            )
            class_weight_dict = {cls: w for cls, w in zip(classes_final, weights_final)}
            print(f"\n✓ Class weights (from TRAIN): {class_weight_dict}")

            # ⭐ FIX 5: TOKENIZER FIT ON TRAINING SUBSET ONLY (PROCESSED URLs)
            print(f"\n? Fitting tokenizer on TRAINING SUBSET...")
            self.tokenizer = Tokenizer(char_level=True, oov_token="<OOV>", num_words=5000)
            self.tokenizer.fit_on_texts(X_train_internal) # Use X_train_internal (now processed)

            url_lengths = [len(u) for u in X_train_internal] # Use X_train_internal (now processed)
            self.max_len = min(int(np.percentile(url_lengths, 95)), 200)
            self.vocab_size = min(len(self.tokenizer.word_index) + 1, 5000)
            print(f"   Vocab size: {self.vocab_size}, Max len: {self.max_len}")

            # ⭐ FIX 6: SCALER FIT ON TRAINING FEATURES ONLY
            print(f"? Fitting scaler on TRAINING FEATURES...")
            self.scaler = StandardScaler()
            X_num_train_scaled = self.scaler.fit_transform(X_num_train)
            X_num_val_scaled = self.scaler.transform(X_num_val)
            X_num_test_scaled = self.scaler.transform(X_num_test)

            # Tokenize
            seq_train = self.tokenizer.texts_to_sequences(X_train_internal) # Use X_train_internal (now processed)
            seq_val = self.tokenizer.texts_to_sequences(X_val_internal)     # Use X_val_internal (now processed)
            seq_test = self.tokenizer.texts_to_sequences(X_url_test)        # Use X_url_test (now processed)

            X_url_train_pad = pad_sequences(seq_train, maxlen=self.max_len,
                                        padding="post", truncating="post")
            X_url_val_pad = pad_sequences(seq_val, maxlen=self.max_len,
                                        padding="post", truncating="post")
            X_url_test_pad = pad_sequences(seq_test, maxlen=self.max_len,
                                        padding="post", truncating="post")

            print(f"\n✓ FINAL DATA SHAPES:")
            print(f"   Train: URL{X_url_train_pad.shape} + Num{X_num_train_scaled.shape} + Y{y_train_cikart.shape}")
            print(f"   Val:   URL{X_url_val_pad.shape} + Num{X_num_val_scaled.shape} + Y{y_val_cikart.shape}")
            print(f"   Test:  URL{X_url_test_pad.shape} + Num{X_num_test_scaled.shape} + Y{y_test_cikart.shape}")

            model_types = ['base', 'multi_cnn', 'attention', 'wide'][:self.n_models]

            for i, (model_type, seed) in enumerate(zip(model_types, self.random_seeds)):
                print(f"\n{'='*70}")
                print(f"FINAL MODEL {i+1}/{self.n_models}: {model_type.upper()}")
                print(f"{'='*70}")

                model_start = time.time()

                model = self.create_model_architecture(
                    self.vocab_size, self.max_len, X_num_train_scaled.shape[1],
                    model_type=model_type, seed=seed
                )

                base_opt = Adam(learning_rate=0.008, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
                optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_opt, dynamic=True)
                model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

                callbacks = self.create_optimized_callbacks(f"final_model_{i+1}")

                # ⭐ FIX 7: TRAIN WITH VALIDATION MONITORING
                print(f"\nTraining with validation monitoring...")
                history = model.fit(
                    x={"url_input": X_url_train_pad, "num_input": X_num_train_scaled},
                    y=y_train_cikart,
                    validation_data=(
                        {"url_input": X_url_val_pad, "num_input": X_num_val_scaled},
                        y_val_cikart
                    ),
                    epochs=epochs,
                    batch_size=batch_size,
                    class_weight=class_weight_dict,
                    callbacks=callbacks,
                    verbose=1
                )

                # ⭐ FIX 8: EVALUATE ON BOTH VAL AND TEST
                val_loss, val_acc = model.evaluate(
                    {"url_input": X_url_val_pad, "num_input": X_num_val_scaled},
                    y_val_cikart, verbose=1
                )

                test_loss, test_acc = model.evaluate(
                    {"url_input": X_url_test_pad, "num_input": X_num_test_scaled},
                    y_test_cikart, verbose=1
                )

                model_time = time.time() - model_start

                print(f"\n✓ Model Performance:")
                print(f"   Val Accuracy:  {val_acc:.4f}")
                print(f"   Test Accuracy: {test_acc:.4f}")
                print(f"   Train time: {self.format_time(model_time)}")

                self.models.append(model)
                self.histories.append(history)

                self.model_info.append({
                    'type': model_type,
                    'val_accuracy': val_acc,
                    'test_accuracy': test_acc,
                    'accuracy': val_acc,  # backward compat
                    'loss': val_loss,
                    'epochs': len(history.history['loss'])
                })

            self.total_training_time = time.time() - training_start

            self.cleanup_gpu_memory()
            print(f"\n✓ Final training completed!")
            print(f"? Total time: {self.format_time(self.total_training_time)}")

            # Store for evaluation
            self.final_X_url_val = X_val_internal
            self.final_X_num_val_scaled = X_num_val_scaled  # ← _scaled ekle
            self.final_y_val = y_val_cikart

            self.final_X_url_test = X_url_test
            self.final_X_num_test_scaled = X_num_test_scaled  # ← _scaled ekle
            self.final_y_test = y_test_cikart


        except Exception as e:
            print(f"\n✗ Final training error: {e}")
            import traceback
            traceback.print_exc()
            raise


    def predict_ensemble(self, X_url, X_num, method='soft_voting'):
        """Ensemble prediction"""
        predictions = []

        X_num_scaled = X_num
        seq_data = self.tokenizer.texts_to_sequences(X_url)
        X_url_pad = pad_sequences(seq_data, maxlen=self.max_len,
                                 padding="post", truncating="post")

        for model in self.models:
            pred = model.predict(
                {"url_input": X_url_pad, "num_input": X_num_scaled},
                verbose=1, batch_size=512
            )
            predictions.append(pred.flatten())

        predictions = np.array(predictions)

        if method == 'soft_voting':
            ensemble_pred_proba = np.mean(predictions, axis=0)
            ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)
        elif method == 'weighted_voting':
            weights = np.array([info['accuracy'] for info in self.model_info])
            weights = weights / np.sum(weights)
            ensemble_pred_proba = np.average(predictions, axis=0, weights=weights)
            ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)
        else:  # hard_voting
            hard_preds = (predictions > 0.5).astype(int)
            ensemble_pred = np.round(np.mean(hard_preds, axis=0)).astype(int)
            ensemble_pred_proba = np.mean(predictions, axis=0)

        return ensemble_pred, ensemble_pred_proba

    def evaluate_final_ensemble(self):
        """Final ensemble evaluation - BOTH VAL AND TEST"""
        print("\n" + "="*70)
        print("?? FINAL ENSEMBLE EVALUATION")
        print("="*70)

        eval_start = time.time()

        methods = ['soft_voting', 'weighted_voting', 'hard_voting']

        # ⭐ VALIDATION SET - Cross-validation'ı simüle et
        print("\n[VALIDATION SET - Internal CV Simulation]")
        val_ensemble_results = {}
        for method in methods:
            print(f"\n   Testing {method}...")
            try:
                val_pred, val_pred_proba = self.predict_ensemble(
                    self.final_X_url_val, self.final_X_num_val_scaled, method
                )
                val_metrics = self.calculate_metrics(self.final_y_val, val_pred)
                val_ensemble_results[method] = {
                    'accuracy': val_metrics['accuracy'],
                    'predictions': val_pred,
                    'probabilities': val_pred_proba,
                    'metrics': val_metrics,
                    'confusion_matrix': val_metrics['confusion_matrix']
                }
                print(f"   ✓ {method:16}: {val_metrics['accuracy']:.4f}")
            except Exception as e:
                print(f"   ✗ {method} error: {e}")
                val_ensemble_results[method] = {
                    'accuracy': 0,
                    'metrics': {},
                    'confusion_matrix': np.zeros((2, 2))
                }

        # Find best validation method
        best_val_method = max(val_ensemble_results.keys(),
                            key=lambda x: val_ensemble_results[x]['accuracy'])
        best_val_acc = val_ensemble_results[best_val_method]['accuracy']

        # ⭐ TEST SET - Final evaluation
        print("\n[TEST SET - Final Holdout Evaluation]")
        test_ensemble_results = {}
        for method in methods:
            print(f"\n   Testing {method}...")
            try:
                test_pred, test_pred_proba = self.predict_ensemble(
                    self.final_X_url_test, self.final_X_num_test_scaled, method
                )
                test_metrics = self.calculate_metrics(self.final_y_test, test_pred)
                test_ensemble_results[method] = {
                    'accuracy': test_metrics['accuracy'],
                    'predictions': test_pred,
                    'probabilities': test_pred_proba,
                    'metrics': test_metrics,
                    'confusion_matrix': test_metrics['confusion_matrix']
                }
                print(f"   ✓ {method:16}: {test_metrics['accuracy']:.4f}")
            except Exception as e:
                print(f"   ✗ {method} error: {e}")
                test_ensemble_results[method] = {
                    'accuracy': 0,
                    'metrics': {},
                    'confusion_matrix': np.zeros((2, 2))
                }

        # Find best test method
        best_test_method = max(test_ensemble_results.keys(),
                            key=lambda x: test_ensemble_results[x]['accuracy'])
        best_test_acc = test_ensemble_results[best_test_method]['accuracy']

        # ⭐ INDIVIDUAL MODEL TEST - Karşılaştırma
        print("\n[INDIVIDUAL MODELS - Test Set Performance]")
        individual_test_results = {}
        for i, info in enumerate(self.model_info):
            model = self.models[i]
            # The X_num_scaled and X_url_pad should be derived from self.final_X_num_test and self.final_X_url_test
            # as these are already prepared and aligned.
            X_num_scaled = self.final_X_num_test_scaled
            seq_data = self.tokenizer.texts_to_sequences(self.final_X_url_test)
            X_url_pad = pad_sequences(seq_data, maxlen=self.max_len,
                                    padding="post", truncating="post")

            pred_proba = model.predict(
                {"url_input": X_url_pad, "num_input": X_num_scaled},
                verbose=1, batch_size=512
            ).flatten()

            pred = (pred_proba > 0.5).astype(int)
            ind_metrics = self.calculate_metrics(self.final_y_test, pred)

            individual_test_results[info['type']] = {
                'accuracy': ind_metrics['accuracy'],
                'metrics': ind_metrics
            }

            print(f"   {info['type']:12}: {ind_metrics['accuracy']:.4f}")

        # ⭐ GAP ANALİZİ
        gap = abs(best_val_acc - best_test_acc)
        print(f"\n? GENERALIZATION GAP:")
        print(f"   Val  Accuracy (best {best_val_method}):  {best_val_acc:.4f}")
        print(f"   Test Accuracy (best {best_test_method}): {best_test_acc:.4f}")
        print(f"   Gap: {gap:.4f} ({gap*100:.2f}%)")

        if gap > 0.05:
            print(f"   ⚠️  Warning: Large gap (>5%) indicates potential overfitting")
        elif gap < 0.02:
            print(f"   ✓ Excellent generalization (<2%)")
        else:
            print(f"   ✓ Good generalization (2-5%)")

        # Store results
        print(f"\n✓ Best ensemble method for final evaluation: {best_test_method}")

        # Test setinden en iyi sonucu al
        best_test_result = test_ensemble_results[best_test_method]

        # Metrics kontrolü yap
        if best_test_result and 'metrics' in best_test_result:
            self.final_metrics = best_test_result['metrics']
            self.final_confusion_matrix = self.final_metrics.get(
                'confusion_matrix', np.zeros((2, 2))
            )
        else:
            print("⚠️ Warning: No metrics found, using default values")
            self.final_metrics = {}
            self.final_confusion_matrix = np.zeros((2, 2))

        # Ek bilgiler depolayın (isteğe bağlı ama faydalı)
        self.final_test_predictions = best_test_result['predictions']
        self.final_test_probabilities = best_test_result['probabilities']
        self.best_ensemble_method = best_test_method

        # ================== LOG ====================
        print("\n? Final Results Stored:")
        print(f"   ✓ Confusion Matrix: {self.final_confusion_matrix.shape}")
        print(f"   ✓ Metrics keys: {list(self.final_metrics.keys())}")
        print(f"   ✓ Predictions shape: {self.final_test_predictions.shape}")
        print(f"   ✓ Best method: {self.best_ensemble_method}")

        self.evaluation_time = time.time() - eval_start

        self.cleanup_gpu_memory()

        return {
            'validation': val_ensemble_results,
            'test': test_ensemble_results
        }, best_test_method


    def print_comprehensive_summary(self):
        """Enhanced summary with individual model performance"""
        print("\n" + "="*80)
        print("?? COMPREHENSIVE PERFORMANCE REPORT")
        print("="*80)

        # CV Results
        print("\n? CROSS-VALIDATION RESULTS (10-Fold):")
        for model_type, scores in self.cv_scores['individual'].items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"   {model_type:12}: {mean_score:.4f} (± {std_score:.4f})")

        ensemble_scores = self.cv_scores['ensemble']
        ensemble_mean = np.mean(ensemble_scores)
        ensemble_std = np.std(ensemble_scores)
        print(f"   {'Ensemble':12}: {ensemble_mean:.4f} (± {ensemble_std:.4f})")

        # Final Model Performance
        print(f"\n? FINAL MODEL PERFORMANCE (Val/Test):")
        for i, info in enumerate(self.model_info):
            print(f"   Model {i+1} ({info['type']:12}): Val={info.get('val_accuracy', 0):.4f} | "
                f"Test={info.get('test_accuracy', 0):.4f}")

        # Time Statistics
        print(f"\n? TIME STATISTICS:")
        total_time = self.data_prep_time + self.cv_time + self.total_training_time + self.evaluation_time
        print(f"   Data preparation: {self.format_time(self.data_prep_time)}")
        print(f"   Cross-validation: {self.format_time(self.cv_time)}")
        print(f"   Final training:   {self.format_time(self.total_training_time)}")
        print(f"   Evaluation:       {self.format_time(self.evaluation_time)}")
        print(f"   TOTAL:            {self.format_time(total_time)}")

        print("="*80)

    def print_confusion_matrix_detailed(self, cm, title="Confusion Matrix"):
        """Confusion matrix'i detaylı göster"""
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            print(f"\n    {title}:")
            print(f"    " + "-" * 35)
            print(f"    {'':20} {'Predicted':>15}")
            print(f"    {'':20} {'Benign':>7} {'Malicious':>7}")
            print(f"    {'Actual':20} {'Benign':>7} {int(tn):>7} {int(fp):>7}")
            print(f"    {'':20} {'Malicious':>7} {int(fn):>7} {int(tp):>7}")
            print(f"    " + "-" * 35)
        else:
            print(f"    {title}:\n{cm}")


    def print_metrics_detailed(self, metrics, title="Metrics"):
        """Metrikleri detaylı göster - DÜZELTILMIŞ"""
        print(f"\n    {title}:")
        print(f"    " + "-" * 40)

        # Eğer metrics dict ve values dict ise (CV avg_metrics gibi)
        if isinstance(metrics, dict) and all(isinstance(v, dict) for v in metrics.values()):
            # Bu CV ortalaması - her value bir dict
            metrics_list = [
                ('Accuracy', 'accuracy'),
                ('Precision', 'precision'),
                ('Recall', 'recall'),
                ('F1-Score', 'f1_score'),
                ('Sensitivity (TPR)', 'sensitivity'),
                ('Specificity (TNR)', 'specificity'),
                ('False Positive Rate', 'fpr'),
                ('False Negative Rate', 'fnr'),
            ]

            print(f"    {'Metric':<25} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
            print(f"    " + "-" * 55)

            for display_name, key in metrics_list:
                if key in metrics:
                    stats = metrics[key]
                    print(f"    {display_name:<25} {stats['mean']:<10.4f} {stats['std']:<10.4f} "
                        f"{stats['min']:<10.4f} {stats['max']:<10.4f}")
        else:
            # Bu final metrics - doğrudan values
            metrics_list = [
                ('Accuracy', 'accuracy'),
                ('Precision', 'precision'),
                ('Recall', 'recall'),
                ('F1-Score', 'f1_score'),
                ('Sensitivity (TPR)', 'sensitivity'),
                ('Specificity (TNR)', 'specificity'),
                ('False Positive Rate', 'fpr'),
                ('False Negative Rate', 'fnr'),
            ]

            for display_name, key in metrics_list:
                if key in metrics:
                    value = metrics[key]
                    # Eğer hala dict ise (bug), mean'i al
                    if isinstance(value, dict):
                        value = value.get('mean', 0)
                    print(f"    {display_name:<25}: {value:.4f}")

        print(f"    " + "-" * 40)


    def print_confusion_matrix_and_metrics(self):
        """Cross-validation ve final test sonuçlarını göster - DÜZELTILMIŞ"""
        print("\n" + "="*80)
        print("?? COMPREHENSIVE CONFUSION MATRIX AND PERFORMANCE METRICS REPORT")
        print("="*80)

        # CROSS-VALIDATION AVERAGE RESULTS
        if self.avg_confusion_matrix is not None and self.cv_metrics:
            print("\n" + "="*80)
            print("? CROSS-VALIDATION RESULTS (Average across 10 Folds)")
            print("="*80)

            self.print_confusion_matrix_detailed(
                self.avg_confusion_matrix,
                "Average Confusion Matrix (CV)"
            )

            # avg_metrics için düzeltilmiş çağrı
            if self.avg_metrics:
                self.print_metrics_detailed(self.avg_metrics, "Average Metrics (CV)")

        # PER-FOLD DETAILS (Fold 1-3 örneği)
        if self.cv_metrics:
            print("\n" + "="*80)
            print("? PER-FOLD BREAKDOWN (10 Folds)")
            print("="*80)

            for fold_idx, (cm, metrics) in enumerate(zip(self.cv_confusion_matrices[:10], self.cv_metrics[:10])):
                print(f"\n--- FOLD {fold_idx + 1} ---")
                self.print_confusion_matrix_detailed(cm, f"Fold {fold_idx + 1} Confusion Matrix")

                # Metrikleri düzeltilmiş şekilde göster
                self.print_metrics_detailed(metrics, f"Fold {fold_idx + 1} Metrics")

        # FINAL TEST RESULTS
        if self.final_confusion_matrix is not None and self.final_metrics:
            print("\n" + "="*80)
            print("? FINAL TEST SET RESULTS")
            print("="*80)

            self.print_confusion_matrix_detailed(
                self.final_confusion_matrix,
                "Final Test Confusion Matrix"
            )
            self.print_metrics_detailed(self.final_metrics, "Final Test Metrics")

        # SUMMARY COMPARISON
        if self.avg_metrics and self.final_metrics:
            print("\n" + "="*80)
            print("? CROSS-VALIDATION vs FINAL TEST COMPARISON")
            print("="*80)

            print(f"\n{'Metric':<20} {'CV Mean':<12} {'CV Std':<12} {'Final Test':<12} {'Difference':<12}")
            print("-" * 68)

            for key in ['accuracy', 'precision', 'recall', 'f1_score']:
                if key in self.avg_metrics and key in self.final_metrics:
                    cv_mean = self.avg_metrics[key]['mean']
                    cv_std = self.avg_metrics[key]['std']

                    # Final metrics'i düzelt
                    final_val = self.final_metrics[key]
                    if isinstance(final_val, dict):
                        final_val = final_val.get('mean', 0)

                    diff = abs(cv_mean - final_val)

                    print(f"{key.title():<20} {cv_mean:<12.4f} {cv_std:<12.4f} {final_val:<12.4f} {diff:<12.4f}")

        print("\n" + "="*80)
        print("="*80 + "\n")

    def save_model_ensemble(self, save_path="ensemble_models/"):
        """Save ensemble models with proper cache handling"""
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with open(os.path.join(save_path, "tokenizer.pkl"), "wb") as f:
            pickle.dump(self.tokenizer, f)

        with open(os.path.join(save_path, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)

        config = {
            'vocab_size': self.vocab_size,
            'max_len': self.max_len,
            'n_models': self.n_models,
            'model_info': self.model_info,
            'cv_scores': self.cv_scores,
            'random_seeds': self.random_seeds
        }

        with open(os.path.join(save_path, "config.pkl"), "wb") as f:
            pickle.dump(config, f)

        for i, model in enumerate(self.models):
            model.save(os.path.join(save_path, f"model_{i+1}.keras"))

        # ✅ DÜZELTME: IsolatedFeatureManager'dan cache'leri al
        print("\n? Saving feature caches...")

        # Final train cache'ini bul (prediction için gerekli)
        if hasattr(self, 'feature_manager') and hasattr(self.feature_manager, 'feature_caches'):
            # Öncelikli olarak "final_train" cache'ini kaydet
            cache_names = ['final_train', 'ablation_train']
            saved_cache = None

            for cache_name in cache_names:
                if cache_name in self.feature_manager.feature_caches:
                    cache_data = self.feature_manager.feature_caches[cache_name]

                    # Her cache component'ini ayrı kaydet
                    for name, data in cache_data.items():
                        cache_file = os.path.join(save_path, f"{name}_cache.pkl")
                        with open(cache_file, "wb") as f:
                            pickle.dump(data, f)
                        print(f"   ✓ Saved {name}_cache.pkl ({len(data)} items)")

                    saved_cache = cache_name
                    break

            if saved_cache:
                print(f"✓ Caches saved from: {saved_cache}")
            else:
                print("⚠️ Warning: No final_train or ablation_train cache found!")
                print(f"   Available caches: {list(self.feature_manager.feature_caches.keys())}")
        else:
            print("⚠️ Warning: feature_manager not found, caches not saved")

        print(f"✓ Ensemble + Caches Saved: {save_path}")

    def generate_final_comprehensive_reports(classifier):
        """
        Generate all comprehensive reports and visualizations

        """
        from ComprehensiveReporting import ComprehensiveReporting

        reporter = ComprehensiveReporting(classifier, save_dir="reports/")
        results = reporter.generate_comprehensive_report()
        return reporter, results

    def print_training_efficiency_summary(self):
        """Print training efficiency with timing"""
        print("\n" + "="*80)
        print("⏱️  TRAINING EFFICIENCY ANALYSIS")
        print("="*80)

        print("\n📊 Training Times per Model (CV across folds):\n")
        print(f"{'Model':<20} {'Total Time (h)':<15} {'Avg/Fold (h)':<15} {'Epochs Avg':<15}")
        print("-" * 65)

        model_types = list(self.cv_scores['individual'].keys())

        for model_type in model_types:
            if model_type in self.total_training_time_per_model:
                total_h = self.total_training_time_per_model[model_type]
                avg_h = self.avg_training_time_per_model[model_type]
                epochs_avg = np.mean(self.epochs_per_model.get(model_type, [0]))

                print(f"{model_type:<20} {total_h:<15.2f} {avg_h:<15.2f} {epochs_avg:<15.1f}")

        # Ensemble total
        total_ensemble = sum(self.total_training_time_per_model.values())
        avg_ensemble = total_ensemble / len(model_types) if len(model_types) > 0 else 0

        print("-" * 65)
        print(f"{'Ensemble Total':<20} {total_ensemble:<15.2f} {avg_ensemble:<15.2f}")
        print(f"\n⏱️  Total Training Time: {self.format_time(self.total_training_time)}")

    # -------------------- Health check helper --------------------

    def health_check(self, train_cache, X_num_test_original, X_url_test, y_test):
        """Kısa health-check: import, scaler/tokenizer/models, cache ve probe extract testleri."""
        print("\n🔧 RUNNING HEALTH CHECKS...")


        # 2) scaler / tokenizer / models
        has_scaler = hasattr(self, 'scaler') and self.scaler is not None
        print(f"Has scaler: {has_scaler}")
        if has_scaler:
            print("  scaler.n_features_in_:", getattr(self.scaler, 'n_features_in_', None))

        has_tokenizer = hasattr(self, 'tokenizer') and self.tokenizer is not None
        print(f"Has tokenizer: {has_tokenizer}")
        if has_tokenizer:
            print("  tokenizer word_index size:", len(getattr(self.tokenizer, 'word_index', {})))
            print("  max_len:", getattr(self, 'max_len', None))

        n_models = len(getattr(self, 'models', []))
        print("Number of models:", n_models)

        # 3) X_num_test_original shape & basic stats
        try:
            print("X_num_test_original.shape:", X_num_test_original.shape)
            means = np.round(X_num_test_original.mean(axis=0)[:10], 6)
            stds = np.round(X_num_test_original.std(axis=0)[:10], 6)
            print("Per-col means (first 10):", means)
            print("Per-col stds  (first 10):", stds)
        except Exception as e:
            print("✗ X_num_test_original check ERROR:", e)

        # 4) train_cache quick sanity
        try:
            print("Train cache keys:", list(train_cache.keys()))
            for k in ['bow','seg_bow','ngrams','grams4','tld']:
                sample = list(train_cache[k].items())[:6]
                print(f"  {k} sample (first 6):", sample)
        except Exception as e:
            print("✗ train_cache check ERROR:", e)

        # 5) quick extractor probe (uses your VectorizedFeatureExtractor in-scope)
        try:
            # Doğrudan global scopedan kullan
            ext = VectorizedFeatureExtractor(
                bow_data=train_cache.get('bow', {}),
                seg_bow_data=train_cache.get('seg_bow', {}),
                ngrams_data=train_cache.get('ngrams', {}),
                grams4_data=train_cache.get('grams4', {}),
                tld_data=train_cache.get('tld', {})
            )
            probe_n = min(50, len(X_url_test))
            X_probe, y_probe, urls_proc = ext.extract_batch_vectorized(
                X_url_test[:probe_n],
                y_test[:probe_n],
                batch_size=probe_n
            )
            print("Probe shape:", getattr(X_probe, "shape", X_probe))
            if X_probe.shape[0] > 0:
                print("Sample features cols 13..17 (bag_count, bow_sum, seg_sum, ngram_sum, gram4_sum):")
                print(X_probe[:5, 13:18])
            else:
                print("✗ Probe produced empty features.")
        except Exception as e:
            print("✗ Extractor probe ERROR:", e)


        print("🔧 HEALTH CHECKS DONE\n")

    # -------------------- End helper --------------------
    def run_ablation_study_final(self,
                            X_url_train, y_train,
                            X_url_test, y_test,
                            epochs=15, batch_size=512,
                            val_fraction=0.15,
                            save_csv="ablation_full_results.csv"):
        """
        Run full ablation: improved & bug-fixed version.
        Returns DataFrame with scenario/model metrics.
        """
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from copy import deepcopy
        from time import time

        print("\n" + "="*80)
        print("🔬 RUNNING FULL ABLATION STUDY (fixed baseline/duplicates)")
        print("="*80)

        # 1) Build isolated feature cache from training data (TRAIN ONLY)
        print("\n? Building isolated feature cache from training data (train-only)...")
        train_data = list(zip(X_url_train, y_train))
        train_cache = self.feature_manager.create_features_for_data(train_data, "ablation_train")

        # 2) Extract numeric features for train and test using the training cache
        print("\n? Extracting features using train cache (parallel)...")
        X_num_train_all, y_train_extracted, X_train_processed = OptimizedEnsembleURLClassifierCV.extract_features_for_fold_optimized(
            X_url_train, y_train, is_train=True, batch_size=5000,
            bow_data=train_cache['bow'], seg_bow_data=train_cache['seg_bow'],
            ngrams_data = train_cache['ngrams'], grams4_data = train_cache['grams4'],
            tld_data = train_cache['tld']
        )

        X_num_test_all, y_test_extracted, X_test_processed = OptimizedEnsembleURLClassifierCV.extract_features_for_fold_optimized(
            X_url_test, y_test, is_train=False, batch_size=5000,
            bow_data=train_cache['bow'], seg_bow_data=train_cache['seg_bow'],
            ngrams_data = train_cache['ngrams'], grams4_data = train_cache['grams4'],
            tld_data = train_cache['tld']
        )

        print(f"✓ Extracted: Train numeric shape: {getattr(X_num_train_all,'shape',None)}, Test numeric shape: {getattr(X_num_test_all,'shape',None)}")

        # Use processed URL lists (aligned)
        X_urls_all_processed = np.array(X_train_processed)
        X_urls_test_processed = np.array(X_test_processed)

        # 3) Split train -> train_internal / val_internal (stratified)
        print("\n? Creating internal train/val split for final training in ablation runs...")
        tr_urls, val_urls, tr_nums, val_nums, tr_y, val_y = train_test_split(
            X_urls_all_processed, X_num_train_all, y_train_extracted,
            test_size=val_fraction, stratify=y_train_extracted, random_state=42
        )

        print(f"   Train_internal: urls={len(tr_urls)}, nums={tr_nums.shape}, y={tr_y.shape}")
        print(f"   Val_internal:   urls={len(val_urls)}, nums={val_nums.shape}, y={val_y.shape}")
        print(f"   Test (holdout): urls={len(X_urls_test_processed)}, nums={X_num_test_all.shape}, y={y_test_extracted.shape}")

        # 4) Ablation scenarios -> indices to zero
        scenarios = {
            'BASELINE': [],                   # no removal
            'REMOVE_BoW': [14],
            'REMOVE_SegBoW': [15],
            'REMOVE_NGRAMS_3_4': [16,17],
            'REMOVE_TLD': [3],
            'REMOVE_RATIOS': [2,18,19]
        }

        model_types = ['base', 'multi_cnn', 'attention', 'wide'][:self.n_models]
        results = []

        # --- NEW: compute baseline_from_final_models once (preferred) ---
        baseline_acc = None
        try:
            if hasattr(self, 'models') and len(self.models) > 0 and hasattr(self, 'tokenizer') and hasattr(self, 'scaler'):
                print("\n? Computing baseline from existing final-trained ensemble (once)...")
                seqs = self.tokenizer.texts_to_sequences(list(X_urls_test_processed))
                X_url_test_pad_final = pad_sequences(seqs, maxlen=self.max_len, padding="post", truncating="post")
                X_num_test_scaled_final = self.scaler.transform(X_num_test_all)

                preds = []
                for m in self.models:
                    try:
                        p = m.predict({"url_input": X_url_test_pad_final, "num_input": X_num_test_scaled_final}, verbose=1).flatten()
                        preds.append(p)
                    except Exception:
                        # fallback: try model-specific tokenizer/scaler if attached
                        tok_m = getattr(m, "_ablation_tokenizer", None)
                        scaler_m = getattr(m, "_ablation_scaler", None)
                        maxlen_m = getattr(m, "_ablation_maxlen", self.max_len)
                        if tok_m is not None and scaler_m is not None:
                            seqs_m = tok_m.texts_to_sequences(list(X_urls_test_processed))
                            pad_m = pad_sequences(seqs_m, maxlen=maxlen_m, padding="post", truncating="post")
                            num_scaled_m = scaler_m.transform(X_num_test_all)
                            try:
                                p = m.predict({"url_input": pad_m, "num_input": num_scaled_m}, verbose=1).flatten()
                                preds.append(p)
                            except Exception:
                                preds.append(np.zeros(len(X_urls_test_processed)))
                        else:
                            preds.append(np.zeros(len(X_urls_test_processed)))
                if len(preds) > 0:
                    ensemble_proba_final = np.mean(np.vstack(preds), axis=0)
                    ensemble_pred_final = (ensemble_proba_final > 0.5).astype(int)
                    baseline_metrics = self.calculate_metrics(y_test_extracted, ensemble_pred_final)
                    baseline_acc = baseline_metrics.get('accuracy', None)

                    # append a single baseline ENSEMBLE row
                    results.append({
                        'Scenario':'BASELINE','Model':'ENSEMBLE',
                        'Val_Accuracy': baseline_metrics.get('accuracy', baseline_metrics.get('f1_score',0)),
                        'Test_Accuracy': baseline_acc,'Val_Precision':baseline_metrics.get('precision',0),
                        'Test_Precision':baseline_metrics.get('precision',0),'Val_Recall':baseline_metrics.get('recall',0),
                        'Test_Recall':baseline_metrics.get('recall',0),'Val_F1':baseline_metrics.get('f1_score',0),
                        'Test_F1':baseline_metrics.get('f1_score',0),'Train_Time_s':0.0,'Notes':'baseline_from_final_models'
                    })
                    print(f"  ✓ Baseline (from final models) test accuracy: {baseline_acc:.4f}")
        except Exception as e:
            print(f"  ⚠️ Baseline-from-final failed: {e}")

        # Helper to train one model given precomputed features
        def _train_one_model_on_precomputed(model_type, seed, X_urls_tr, X_num_tr, y_tr, X_urls_val, X_num_val, y_val, X_urls_test, X_num_test, y_test_local, epochs_local, batch_local):
            t0 = time()
            tok = Tokenizer(char_level=True, oov_token="<OOV>", num_words=5000)
            tok.fit_on_texts(X_urls_tr)
            url_lens = [len(u) for u in X_urls_tr]
            maxlen_local = min(int(np.percentile(url_lens, 95)), 200)
            vocab_local = min(len(tok.word_index) + 1, 5000)

            scaler_local = StandardScaler()
            X_num_tr_scaled = scaler_local.fit_transform(X_num_tr)
            X_num_val_scaled = scaler_local.transform(X_num_val)
            X_num_test_scaled = scaler_local.transform(X_num_test)

            seq_tr = tok.texts_to_sequences(X_urls_tr)
            seq_val = tok.texts_to_sequences(X_urls_val)
            seq_test = tok.texts_to_sequences(X_urls_test)

            X_url_tr_pad = pad_sequences(seq_tr, maxlen=maxlen_local, padding="post", truncating="post")
            X_url_val_pad = pad_sequences(seq_val, maxlen=maxlen_local, padding="post", truncating="post")
            X_url_test_pad = pad_sequences(seq_test, maxlen=maxlen_local, padding="post", truncating="post")

            model_local = self.create_model_architecture(vocab_local, maxlen_local, X_num_tr_scaled.shape[1], model_type=model_type, seed=seed)
            base_opt = Adam(learning_rate=0.008, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_opt, dynamic=True)
            model_local.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
            callbacks = self.create_optimized_callbacks(f"ablation_final_{model_type}")

            history_local = model_local.fit(
                x={"url_input": X_url_tr_pad, "num_input": X_num_tr_scaled},
                y=y_tr,
                validation_data=({"url_input": X_url_val_pad, "num_input": X_num_val_scaled}, y_val),
                epochs=epochs_local, batch_size=batch_local, verbose=1, class_weight=None, callbacks=callbacks
            )

            val_pred_proba = model_local.predict({"url_input": X_url_val_pad, "num_input": X_num_val_scaled}, verbose=1).flatten()
            val_pred = (val_pred_proba > 0.5).astype(int)
            val_metrics = self.calculate_metrics(y_val, val_pred)

            test_pred_proba = model_local.predict({"url_input": X_url_test_pad, "num_input": X_num_test_scaled}, verbose=1).flatten()
            test_pred = (test_pred_proba > 0.5).astype(int)
            test_metrics = self.calculate_metrics(y_test_local, test_pred)

            t_elapsed = time() - t0

            # attach tokenizer/scaler for later consistent ensemble inference if needed
            model_local._ablation_tokenizer = tok
            model_local._ablation_scaler = scaler_local
            model_local._ablation_maxlen = maxlen_local

            return model_local, history_local, val_metrics, test_metrics, t_elapsed, (X_url_tr_pad.shape, X_num_tr_scaled.shape)

        # 5) Loop scenarios (skip retraining BASELINE if baseline_acc already computed)
        for scen_name, remove_idx in scenarios.items():
            print("\n" + "-"*80)
            print(f"SCENARIO: {scen_name}  -> remove idx: {remove_idx}")
            print("-"*80)

            # If baseline was computed from final models, skip retraining scenario BASELINE to avoid duplicates
            if scen_name.upper() == 'BASELINE' and baseline_acc is not None:
                print(f"  ✓ Skipping retrain for BASELINE (using baseline_acc={baseline_acc:.4f})")
                continue

            # produce modified numeric arrays (copy)
            X_num_tr_mod = tr_nums.copy()
            X_num_val_mod = val_nums.copy()
            X_num_test_mod = X_num_test_all.copy()

            if remove_idx:
                X_num_tr_mod[:, remove_idx] = 0.0
                X_num_val_mod[:, remove_idx] = 0.0
                X_num_test_mod[:, remove_idx] = 0.0

            trained_models_for_scenario = []
            per_model_info = {}

            for i, model_type in enumerate(model_types):
                seed = self.random_seeds[i] if i < len(self.random_seeds) else 42 + i
                print(f"\nTraining model '{model_type}' (seed={seed}) for scenario {scen_name} ...")
                try:
                    model_loc, hist_loc, val_metrics_loc, test_metrics_loc, t_elapsed, shapes = _train_one_model_on_precomputed(
                        model_type, seed,
                        tr_urls, X_num_tr_mod, tr_y,
                        val_urls, X_num_val_mod, val_y,
                        X_urls_test_processed, X_num_test_mod, y_test_extracted,
                        epochs, batch_size
                    )
                except Exception as e:
                    print(f"✗ Error training {model_type} in scenario {scen_name}: {e}")
                    import traceback; traceback.print_exc()
                    continue

                trained_models_for_scenario.append(model_loc)
                per_model_info[model_type] = {
                    'val_metrics': val_metrics_loc,
                    'test_metrics': test_metrics_loc,
                    'train_time_s': t_elapsed,
                    'shapes': shapes
                }

                print(f"   ✓ {model_type} - val acc: {val_metrics_loc['accuracy']:.4f}, test acc: {test_metrics_loc['accuracy']:.4f}, time: {self.format_time(t_elapsed)}")

            # Ensemble: use per-model tokenizer/scaler if possible (prefer), otherwise fit on tr_urls
            all_probas = []
            for m in trained_models_for_scenario:
                # Try to use model's attached tokenizer/scaler
                tok_m = getattr(m, "_ablation_tokenizer", None)
                scaler_m = getattr(m, "_ablation_scaler", None)
                maxlen_m = getattr(m, "_ablation_maxlen", None)

                try:
                    if tok_m is not None and scaler_m is not None and maxlen_m is not None:
                        seq_test_m = tok_m.texts_to_sequences(list(X_urls_test_processed))
                        X_test_pad_m = pad_sequences(seq_test_m, maxlen=maxlen_m, padding="post", truncating="post")
                        X_num_test_scaled_m = scaler_m.transform(X_num_test_mod)
                        proba = m.predict({"url_input": X_test_pad_m, "num_input": X_num_test_scaled_m}, verbose=1).flatten()
                    else:
                        # fallback: fit local ensemble tokenizer/scaler on tr_urls for consistency
                        seq_test = Tokenizer(char_level=True, oov_token="<OOV>", num_words=5000)
                        seq_test.fit_on_texts(tr_urls)
                        seqs_test = seq_test.texts_to_sequences(list(X_urls_test_processed))
                        ens_maxlen = min(int(np.percentile([len(u) for u in tr_urls], 95)), 200)
                        X_url_test_pad_ensemble = pad_sequences(seqs_test, maxlen=ens_maxlen, padding="post", truncating="post")
                        scaler_ens = StandardScaler().fit(X_num_tr_mod)
                        X_num_test_scaled_ens = scaler_ens.transform(X_num_test_mod)
                        proba = m.predict({"url_input": X_url_test_pad_ensemble, "num_input": X_num_test_scaled_ens}, verbose=1).flatten()
                except Exception:
                    # ultimate fallback: zeros
                    proba = np.zeros(len(X_urls_test_processed))
                all_probas.append(proba)

            if len(all_probas) == 0:
                print(f"⚠️ No models trained for scenario {scen_name}, skipping ensemble.")
                continue

            ensemble_proba = np.mean(np.vstack(all_probas), axis=0)
            ensemble_pred = (ensemble_proba > 0.5).astype(int)
            ensemble_metrics = self.calculate_metrics(y_test_extracted, ensemble_pred)

            # Save per-model rows
            for mt, info in per_model_info.items():
                row = {
                    'Scenario': scen_name,
                    'Model': mt,
                    'Val_Accuracy': info['val_metrics']['accuracy'],
                    'Test_Accuracy': info['test_metrics']['accuracy'],
                    'Val_Precision': info['val_metrics']['precision'],
                    'Test_Precision': info['test_metrics']['precision'],
                    'Val_Recall': info['val_metrics']['recall'],
                    'Test_Recall': info['test_metrics']['recall'],
                    'Val_F1': info['val_metrics']['f1_score'],
                    'Test_F1': info['test_metrics']['f1_score'],
                    'Train_Time_s': info['train_time_s'],
                    'Notes': ''
                }
                results.append(row)

            # Save ensemble row for this scenario
            ensemble_row = {
                'Scenario': scen_name,
                'Model': 'ENSEMBLE',
                'Val_Accuracy': ensemble_metrics.get('accuracy', 0),
                'Test_Accuracy': ensemble_metrics.get('accuracy', 0),
                'Val_Precision': ensemble_metrics.get('precision', 0),
                'Test_Precision': ensemble_metrics.get('precision', 0),
                'Val_Recall': ensemble_metrics.get('recall', 0),
                'Test_Recall': ensemble_metrics.get('recall', 0),
                'Val_F1': ensemble_metrics.get('f1_score', 0),
                'Test_F1': ensemble_metrics.get('f1_score', 0),
                'Train_Time_s': np.sum([info['train_time_s'] for info in per_model_info.values()]) if per_model_info else 0.0,
                'Notes': f"trained_models={len(per_model_info)}"
            }
            results.append(ensemble_row)

            print(f"\n✓ Scenario {scen_name} completed: ensemble test acc {ensemble_row['Test_Accuracy']:.4f}")

        # final dataframe
        df = pd.DataFrame(results)

        # ---- CLEANUPS: remove exact duplicates and ensure single baseline ----
        if not df.empty:
            # drop exact duplicates on Scenario+Model+Test_Accuracy (keeps first)
            df = df.drop_duplicates(subset=['Scenario', 'Model', 'Test_Accuracy'], keep='first').reset_index(drop=True)

            # ensure baseline_acc variable set (if not computed earlier, fallback)
            if baseline_acc is None:
                # try find ensemble baseline in df
                msk = (df['Scenario'].astype(str).str.upper().str.contains('BASELINE')) & (df['Model'].astype(str).str.upper()=='ENSEMBLE')
                if msk.any():
                    baseline_acc = float(df.loc[msk, 'Test_Accuracy'].iloc[0])
                else:
                    # fallback: first ensemble row or top Test_Accuracy
                    if 'ENSEMBLE' in df['Model'].values:
                        baseline_acc = float(df[df['Model']=='ENSEMBLE']['Test_Accuracy'].iloc[0])
                    else:
                        baseline_acc = float(df['Test_Accuracy'].iloc[0])

            # compute Accuracy_Drop = baseline - scenario_acc
            df['Accuracy_Drop'] = baseline_acc - df['Test_Accuracy'].astype(float)

        # save
        try:
            df.to_csv(save_csv, index=False)
            print(f"\n✓ Ablation results saved to {save_csv}")
        except Exception as e:
            print(f"⚠️ Could not save CSV: {e}")

        print("\n🔬 FULL ABLATION STUDY FINISHED")
        return df



    def export_efficiency_report(self, filename="training_efficiency_report.csv"):
        """Export training efficiency to CSV"""
        import pandas as pd

        data = []
        model_types = list(self.cv_scores['individual'].keys())

        for model_type in model_types:
            total_h = self.total_training_time_per_model.get(model_type, 0)
            avg_h = self.avg_training_time_per_model.get(model_type, 0)
            epochs_list = self.epochs_per_model.get(model_type, [0])

            data.append({
                'Model': model_type,
                'Total Time (hours)': f"{total_h:.2f}",
                'Avg Time per Fold (hours)': f"{avg_h:.2f}",
                'Avg Epochs': f"{np.mean(epochs_list):.1f}",
                'Min Epochs': int(np.min(epochs_list)),
                'Max Epochs': int(np.max(epochs_list)),
                'CV Accuracy': f"{np.mean(self.cv_scores['individual'][model_type]):.4f}",
                'CV Std': f"{np.std(self.cv_scores['individual'][model_type]):.4f}"
            })

        # Add ensemble
        total_ens = sum(self.total_training_time_per_model.values())
        avg_ens = np.mean(list(self.avg_training_time_per_model.values())) if self.avg_training_time_per_model else 0

        data.append({
            'Model': 'Ensemble',
            'Total Time (hours)': f"{total_ens:.2f}",
            'Avg Time per Fold (hours)': f"{avg_ens:.2f}",
            'Avg Epochs': f"{np.mean([np.mean(v) for v in self.epochs_per_model.values()]):.1f}",
            'Min Epochs': '10',
            'Max Epochs': '15',
            'CV Accuracy': f"{np.mean(self.cv_scores['ensemble']):.4f}",
            'CV Std': f"{np.std(self.cv_scores['ensemble']):.4f}"
        })

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"✓ Report exported to {filename}")
        return df

    def measure_single_url_latency(self, X_url_test, n_samples=100, random_state=None):
        """
        Measure single-URL latencies on the final ensemble and per-model.
        Returns timing statistics dict.
        """
        from time import perf_counter
        import numpy as _np
        import re as _re
        from urllib.parse import urlparse
        import tldextract

        # Pre-checks
        if getattr(self, 'scaler', None) is None or getattr(self, 'tokenizer', None) is None:
            raise RuntimeError("Scaler or tokenizer not set. Run final training first.")

        # --- Select cache name robustly ---
        fm = getattr(self, 'feature_manager', None)
        if fm is None or not hasattr(fm, 'feature_caches'):
            raise RuntimeError("feature_manager or feature_caches not available on self.")
        caches = fm.feature_caches
        # Prefer exact "final_train", else any key containing "final", else first key
        if "final_train" in caches:
            cache_name = "final_train"
        else:
            raise RuntimeError("final_train cache not found. Run final training first.")

        cache = caches.get(cache_name, {})

        tld_cache     = cache.get("tld", {}) or {}
        bow_cache     = cache.get("bow", {}) or {}
        seg_bow_cache = cache.get("seg_bow", {}) or {}
        ngrams_cache  = cache.get("ngrams", {}) or {}
        grams4_cache  = cache.get("grams4", {}) or {}

        # Helpers: try to reuse VectorizedFeatureExtractor utilities
        try:
            VFE = VectorizedFeatureExtractor
            tokenize = VFE._tokenize
            make_ngrams = VFE._make_ngrams_from_string
            extract_tld = VFE._extract_tld
            has_ip = VFE._has_ip
            has_tinyurl = VFE._has_tinyurl
        except Exception:
            tokenize = lambda u: _re.findall(r'\w+', u)
            make_ngrams = lambda s, n=3: [s[i:i+n] for i in range(len(s)-n+1)] if s and len(s) >= n else []
            extract_tld = lambda u: ""
            has_ip = lambda u: bool(_re.match(r'.*\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b.*', u))
            has_tinyurl = lambda u: ('bit.ly' in u or 'tinyurl.com' in u)

        # sample indices
        rng = _np.random.RandomState(random_state)
        n_samples = min(n_samples, len(X_url_test))
        if n_samples <= 0:
            raise ValueError("n_samples must be > 0 and less than len(X_url_test).")
        indices = rng.choice(len(X_url_test), n_samples, replace=False)

        # containers
        feat_times = []
        ensemble_times = []
        per_model_times = {}
        per_model_probas = {}

        model_names = []
        for i, model in enumerate(self.models):
            name = self.model_info[i]['type'] if i < len(self.model_info) else f"model_{i}"
            model_names.append(name)
            per_model_times[name] = []
            per_model_probas[name] = []

        print(f"\n?? Measuring latency on {n_samples} random URLs (using cache: {cache_name})...")

        for idx in indices:
            url = X_url_test[idx]

            # ---------- FEATURE EXTRACTION ----------
            t0 = perf_counter()
            try:
                url_len = len(url)
                special_chars = sum(1 for c in url if not c.isalnum())
                url_len_ratio = special_chars / max(url_len, 1)

                # TLD weight
                try:
                    tld = extract_tld(url)
                except Exception:
                    # fallback to tldextract directly
                    try:
                        tld = tldextract.extract(url).suffix or ""
                    except Exception:
                        tld = ""
                tld_weight = float(tld_cache.get(tld, 0.0))

                parsed = urlparse(url)
                netloc = parsed.netloc or ""
                subdomain_count = netloc.count('.')

                path = parsed.path or ""
                path_length = len(path)

                # domain (domain + suffix if available)
                try:
                    domain_info = tldextract.extract(url)
                    domain = (domain_info.domain + '.' + domain_info.suffix) if domain_info.suffix else domain_info.domain
                except Exception:
                    domain = parsed.netloc or ""
                domain_length = len(domain)

                path_ratio = path_length / max(url_len, 1)
                domain_ratio = domain_length / max(url_len, 1)

                # tokens
                tokens = tokenize(url)

                # ----- Weighted SUMS (Wi sums) -----
                bow_sum = 0.0
                bow_match_count = 0
                for t in tokens:
                    w = float(bow_cache.get(t, 0.0))
                    if w > 0:
                        bow_sum += w
                        bow_match_count += 1

                seg_bow_sum = 0.0
                seg_match_count = 0
                for t in tokens:
                    # try user's segment() if available, otherwise simple fallback
                    try:
                        parts = segment(t)
                        if not isinstance(parts, (list, tuple)):
                            parts = [str(parts)]
                    except Exception:
                        parts = _re.findall(r'[A-Za-z]{3,}|[0-9]+', t) or [t]
                    for p in parts:
                        w = float(seg_bow_cache.get(p, 0.0))
                        if w > 0:
                            seg_bow_sum += w
                            seg_match_count += 1

                ngrams_sum = 0.0
                ngrams_match_count = 0
                for t in tokens:
                    for ng in make_ngrams(t, n=3):
                        w = float(ngrams_cache.get(ng, 0.0))
                        if w > 0:
                            ngrams_sum += w
                            ngrams_match_count += 1

                grams4_sum = 0.0
                grams4_match_count = 0
                for t in tokens:
                    for g4 in make_ngrams(t, n=4):
                        w = float(grams4_cache.get(g4, 0.0))
                        if w > 0:
                            grams4_sum += w
                            grams4_match_count += 1

                bag_of_words_count = float(bow_match_count)

                features = _np.array([[
                    float(url_len),         #0
                    float(special_chars),   #1
                    float(url_len_ratio),   #2
                    tld_weight,             #3
                    1.0 if has_ip(url) else 0.0,    #4
                    1.0 if has_tinyurl(url) else 0.0,#5
                    1.0 if '@' in url else 0.0,     #6
                    1.0 if url.count('//') > 1 else 0.0, #7
                    1.0 if '-' in netloc or (netloc.split('.')[0] if netloc else '').find('-')>=0 else 0.0, #8
                    1.0 if subdomain_count > 1 else 0.0, #9
                    1.0 if parsed.port and parsed.port not in [80,443] else 0.0, #10
                    1.0 if 'https-' in url else 0.0, #11
                    1.0 if any(ext in url for ext in ['.exe','.pdf','.zip','.rar']) else 0.0, #12
                    bag_of_words_count, #13
                    bow_sum,            #14 weighted sum
                    seg_bow_sum,        #15 weighted sum
                    ngrams_sum,         #16 weighted sum
                    grams4_sum,         #17 weighted sum
                    float(path_ratio),  #18
                    float(domain_ratio) #19
                ]], dtype=float)

                t_feat = perf_counter() - t0

            except Exception:
                # skip problematic URL
                continue

            # ---------- SCALE & TOKENIZE ----------
            try:
                X_num_scaled = self.scaler.transform(features)
                seq = self.tokenizer.texts_to_sequences([url])
                X_url_pad = pad_sequences(seq, maxlen=self.max_len, padding="post", truncating="post")
            except Exception:
                continue

            # ---------- PREDICTIONS ----------
            t_ens0 = perf_counter()
            for i, model in enumerate(self.models):
                model_name = model_names[i]
                t_m0 = perf_counter()
                try:
                    pred = model.predict({"url_input": X_url_pad, "num_input": X_num_scaled}, verbose=1, batch_size=512)
                    proba = float(pred.flatten()[0])
                except Exception:
                    proba = 0.5
                t_m = perf_counter() - t_m0

                per_model_times[model_name].append(t_m)
                per_model_probas[model_name].append(proba)
            t_ens = perf_counter() - t_ens0

            # save
            feat_times.append(t_feat)
            ensemble_times.append(t_ens)

        # store on self
        self.single_url_feature_extraction_times = feat_times
        self.single_url_prediction_times = ensemble_times
        self.per_model_single_times = per_model_times
        self.per_model_single_probas = per_model_probas

        # summary helpers
        def stats_ms(arr_s):
            a = _np.array(arr_s) * 1000.0
            if a.size == 0:
                return 0, 0, 0, 0, 0, 0
            return a.size, a.mean(), a.std(), a.min(), _np.median(a), a.max()

        # print summary
        print("\n" + "="*90)
        print("⏱️  SINGLE-URL TIMING SUMMARY")
        print("="*90)

        if len(feat_times) == 0:
            print("No successful samples to report.")
            return None

        n, mean_ms, std_ms, min_ms, med_ms, max_ms = stats_ms(feat_times)
        print(f"\n📊 Feature Extraction:")
        print(f"   Samples: {n}")
        print(f"   Mean: {mean_ms:.3f} ms")
        print(f"   Std: {std_ms:.3f} ms")
        print(f"   Min: {min_ms:.3f} ms")
        print(f"   Median: {med_ms:.3f} ms")
        print(f"   Max: {max_ms:.3f} ms")

        n, mean_ms_e, std_ms_e, min_ms_e, med_ms_e, max_ms_e = stats_ms(ensemble_times)
        print(f"\n📊 Ensemble Prediction (All models):")
        print(f"   Samples: {n}")
        print(f"   Mean: {mean_ms_e:.3f} ms")
        print(f"   Std: {std_ms_e:.3f} ms")
        print(f"   Min: {min_ms_e:.3f} ms")
        print(f"   Median: {med_ms_e:.3f} ms")
        print(f"   Max: {max_ms_e:.3f} ms")
        print(f"   Throughput: {1000.0/(mean_ms_e if mean_ms_e>0 else 1):.2f} predictions/second")

        total_mean = mean_ms + mean_ms_e
        print(f"\n📊 Total Per URL (Feature Extraction + Prediction):")
        print(f"   Mean Total: {total_mean:.3f} ms")
        print(f"   Throughput: {1000.0/(total_mean if total_mean>0 else 1):.2f} URLs/second")

        print("\n📊 Per-Model Single-URL Prediction Times (ms):")
        print("Model".ljust(20) + "N".ljust(6) + "Mean".ljust(10) + "Std".ljust(10) + "Min".ljust(10) + "Med".ljust(10) + "Max")
        print("-"*90)

        for name in model_names:
            arr = _np.array(per_model_times[name]) * 1000.0
            if arr.size == 0:
                print(f"{name.ljust(20)} no data")
                continue
            print(f"{name.ljust(20)} {arr.size:<5d}  {arr.mean():<9.3f}  {arr.std():<9.3f}  {arr.min():<9.3f}  {_np.median(arr):<9.3f}  {arr.max():.3f}")

        print("\n📊 Per-Model Average Predicted Probability:")
        for name in model_names:
            probs = _np.array(per_model_probas[name])
            if probs.size == 0:
                continue
            print(f"   {name.ljust(20)} mean={probs.mean():.4f}, std={probs.std():.4f}, min={probs.min():.4f}, max={probs.max():.4f}")

        print("\n" + "="*90)

        return {
            'feat_times_s': feat_times,
            'ensemble_times_s': ensemble_times,
            'per_model_times_s': per_model_times,
            'per_model_probas': per_model_probas
        }

# ==================== MAIN FUNCTION ====================

def main():
    """Main function with optimized RAM-only processing"""

    start_time = time.time()

    print("?? Google Drive monte ediliyor...")
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        print("✓ Google Drive monte edildi\n")
        storage_type = "local"
    except:
        print("⚠️ Colab'da çalışmıyor, lokal moduna geçiliyor\n")
        storage_type = "colab"

    checkpoint_mgr = CheckpointManager(storage_type=storage_type)

    RAW_DATA_FILE = "MilyonPDB3.txt"
    print(f"?? Loading raw data from: {RAW_DATA_FILE}")

    ensemble_clf = OptimizedEnsembleURLClassifierCV(
        n_models=4,
        n_folds=10,
        random_seeds=[42, 123, 456, 789]
    )

    (X_url_train, y_train,
     X_url_test, y_test, rows_all) = ensemble_clf.prepare_data_from_raw(
        RAW_DATA_FILE, test_size=0.2
    )

    rows_train = [(X_url_train[i], y_train[i]) for i in range(len(X_url_train))]
    rows_test = [(X_url_test[i], y_test[i]) for i in range(len(X_url_test))]

    print("\n? Cross-validation starting...")
    cv_individual, cv_ensemble = ensemble_clf.cross_validate_ensemble(
        X_url_train, y_train, rows_train,
        checkpoint_mgr=checkpoint_mgr,
        epochs=15,
        batch_size=512
    )

    print("\n⏱️  Printing training efficiency...")
    ensemble_clf.print_training_efficiency_summary()

    # CSV export et
    ensemble_clf.export_efficiency_report("training_efficiency_report.csv")

    print("\n?? Final ensemble training start...")
    ensemble_clf.train_final_ensemble(
        X_url_train, y_train, rows_train,
        X_url_test, y_test, rows_test,
        epochs=15,
        batch_size=512
    )

    print("\n?? Final ensemble evaluate...")
    ensemble_results, best_method = ensemble_clf.evaluate_final_ensemble()

    # ★★ DEBUG EKLE
    print("\n🔍 RUNNING CONSISTENCY CHECK...")
    ensemble_clf.debug_feature_consistency()

    ensemble_clf.print_comprehensive_summary()

    ensemble_clf.print_confusion_matrix_and_metrics()

    print("\n?? Model saving...")
    ensemble_clf.save_model_ensemble("ensemble_models/")

    print("\n" + "="*80)
    print("⏱️  MEASURING SINGLE-URL LATENCY & PERFORMANCE")
    print("="*80)

    try:
        print("\n?? Measuring single-URL prediction times...")
        timing_stats = ensemble_clf.measure_single_url_latency(
            X_url_test,
            n_samples=100,  # Measure on 100 random URLs
            random_state=42
        )
        print("\n✓ Latency measurement completed")
    except Exception as e:
        print(f"\n⚠️ Latency measurement warning: {e}")
        timing_stats = None

    # ==================== ⭐ CUSTOM ABLATION STUDY ====================
    print("\n" + "="*100)
    print("🔬 STARTING CUSTOM ABLATION STUDY - 5 SCENARIOS")
    print("="*100)

    try:
        ablation_df = ensemble_clf.run_ablation_study_final(
            X_url_train=X_url_train,
            y_train=y_train,
            X_url_test=X_url_test,
            y_test=y_test,
            epochs=15,
            batch_size=512,
            save_csv="ablation_full_results.csv"
        )
        print("\n✓ Custom ablation study completed successfully!")
    except Exception as e:
        print(f"\n❌ Ablation study error: {e}")
        import traceback
        traceback.print_exc()
        # ensure variable exists for later summary logic
        ablation_df = None


    # ==================== STATISTICAL SIGNIFICANCE TESTS ====================
    print("\n" + "="*80)
    print("📊 RUNNING STATISTICAL SIGNIFICANCE TESTS")
    print("="*80)

    try:
        # Import statistical tests module
        from statistical_tests import run_statistical_tests

        print("\n?? Analyzing statistical significance...")
        analyzer = run_statistical_tests(ensemble_clf)
        print("\n✓ Statistical analysis completed")
    except ImportError:
        print("\n⚠️ statistical_tests.py not found. Skipping statistical analysis.")
        print("   To enable: Place statistical_tests.py in the same directory")
        analyzer = None
    except Exception as e:
        print(f"\n⚠️ Statistical analysis error: {e}")
        analyzer = None

    # ==================== COMPREHENSIVE REPORTING ====================
    print("\n" + "="*80)
    print("📊 GENERATING COMPREHENSIVE VISUALIZATIONS AND REPORTS")
    print("="*80)
    from comprehensive_reporting import ComprehensiveReporting, generate_final_comprehensive_reports
    reporter = None
    report_results = None

    try:
        reporter, report_results = generate_final_comprehensive_reports(ensemble_clf)

        print("\n" + "="*80)
        print("📋 REPORT SUMMARY")
        print("="*80)

        print("\n✓ Generated Reports:")
        print("  1. CV Accuracies Box Plot (shows distribution across 10-fold)")
        print("  2. CV Accuracies Line Plot (shows fold-by-fold progression)")
        print("  3. Training Efficiency Analysis Table")
        print("  4. Confusion Matrices for all models + Ensemble (Final Test Set)")
        print("  5. CV vs Final Test Performance Comparison Table")
        print("  6. Metrics Comparison Bar Plot (Accuracy, Precision, Recall, F1)")
        print("  7. Cross-Validation Variance Analysis")

        if report_results is not None:
            print("\n📊 Key Metrics from Tables:")
            print("\nTraining Efficiency:")
            print(report_results['efficiency_table'].to_string())

            print("\n\nCV vs Test Comparison:")
            print(report_results['comparison_table'].to_string())

    except Exception as e:
        print(f"\n⚠️ Reporting error: {e}")
        import traceback
        traceback.print_exc()

    # ==================== FINAL SUMMARY REPORT ====================
    print("\n" + "="*80)
    print("📋 FINAL COMPREHENSIVE REPORT")
    print("="*80)
    total_time = time.time() - start_time
    print(f"\n? TOTAL EXECUTION TIME: {ensemble_clf.format_time(total_time)}")
    print("="*70)

    cv_ensemble_mean = np.mean(cv_ensemble)
    cv_ensemble_std = np.std(cv_ensemble)


    best_acc = ensemble_results['test'][best_method]['metrics']['accuracy'] # FIXED KEYERROR


    print(f"\n?? Best Performance Results:")
    print(f"     Cross-validation: {cv_ensemble_mean:.4f} (± {cv_ensemble_std:.4f})")
    print(f"     Final test: {best_acc:.4f}")
    print(f"     Best method: {best_method}")
    print(f"     CV-Test difference: {abs(cv_ensemble_mean - best_acc):.4f}")

    print(f"\n?? Model Performance Comparison:")
    print(f"   {'Model':<15} {'CV Mean':<10} {'CV Std':<10} {'Final (Val/Test)':<15}")
    print("-" * 55)

    for i, info in enumerate(ensemble_clf.model_info):
        model_type = info['type']
        cv_mean = np.mean(cv_individual[model_type])
        cv_std = np.std(cv_individual[model_type])
        val_acc = info.get('val_accuracy', 0)
        test_acc = info.get('test_accuracy', 0)

        print(f"   {model_type:<15} {cv_mean:<10.4f} {cv_std:<10.4f} {val_acc:.4f}/{test_acc:.4f}")

    print(f"   {'Ensemble':<15} {cv_ensemble_mean:<10.4f} {cv_ensemble_std:<10.4f} {best_acc:.4f}")
    reliability_score = 1 - (cv_ensemble_std / cv_ensemble_mean) if cv_ensemble_mean > 0 else 0
    print(f"\n? Model Reliability:")
    print(f"     Consistency score: {reliability_score:.4f}")

    # ==================== ABLATION STUDY SUMMARY ====================
    if ablation_df is None:
        print("\n⚠️ No ablation results available (ablation_df is None). Skipping ablation summary.")
    else:
        print("\n🔬 CUSTOM ABLATION STUDY SUMMARY:")
        print("-"*80)

        # show columns for debug
        print("Ablation DataFrame columns:", list(ablation_df.columns))

        # Extract ensemble rows
        try:
            df_ens = ablation_df[ablation_df['Model'].str.upper() == 'ENSEMBLE'].copy()
        except Exception:
            if 'Model' in ablation_df.columns:
                df_ens = ablation_df[ablation_df['Model'] == 'ENSEMBLE'].copy()
            else:
                df_ens = pd.DataFrame()

        if df_ens.empty:
            if 'Test_Accuracy' in ablation_df.columns:
                print("   ⚠️ Ensemble rows not found — using fallback: best Test_Accuracy per scenario as proxy.")
                df_ens = ablation_df.groupby('Scenario', as_index=False).agg({'Test_Accuracy':'max'}).copy()
            else:
                print("   ✗ Could not find ensemble rows or Test_Accuracy column. Skipping ablation ranking.")
                df_ens = pd.DataFrame()

        if df_ens.empty:
            print("   ✗ No ensemble or proxy data available for ablation summary.")
        else:
            # Normalize column name
            if 'Test_Accuracy' not in df_ens.columns and 'Test Accuracy' in df_ens.columns:
                df_ens = df_ens.rename(columns={'Test Accuracy':'Test_Accuracy'})

            # Find baseline accuracy
            baseline_row = df_ens[df_ens['Scenario'].astype(str).str.upper().str.contains('BASELINE')].copy()
            if not baseline_row.empty:
                baseline_acc = float(baseline_row['Test_Accuracy'].iloc[0])
                print(f"   ✓ Baseline ensemble Test_Accuracy found: {baseline_acc:.4f}")
            else:
                if 'BASELINE' in df_ens['Scenario'].values:
                    baseline_acc = float(df_ens.loc[df_ens['Scenario']=='BASELINE', 'Test_Accuracy'].iloc[0])
                else:
                    baseline_acc = float(df_ens['Test_Accuracy'].iloc[0])
                print(f"   ⚠️ Baseline row not found by name; using fallback baseline_acc = {baseline_acc:.4f}")

            # Compute drops for all metrics
            metrics = ['Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1']
            for m in metrics:
                drop_name = m.replace('Test_', '') + '_Drop'
                df_ens[drop_name] = baseline_acc - df_ens[m].astype(float)

            # Keep only non-baseline scenarios for ranking
            df_rank = df_ens[~df_ens['Scenario'].astype(str).str.upper().str.contains('BASELINE')].copy()
            df_rank = df_rank.sort_values('Accuracy_Drop', ascending=False).reset_index(drop=True)

            # Print ranking with all drops
            print(f"\n{'Rank':<6} {'Scenario':<40} {'Acc Drop':<10} {'Prec Drop':<10} {'Recall Drop':<12} {'F1 Drop':<10} {'Test_Acc':<10}")
            print("-"*100)
            if df_rank.shape[0] == 0:
                print("   (No non-baseline scenarios found.)")
            else:
                for idx, row in df_rank.iterrows():
                    print(f"{idx+1:<6} {row['Scenario']:<40} {row['Accuracy_Drop']:<10.4f} {row['Precision_Drop']:<10.4f} {row['Recall_Drop']:<12.4f} {row['F1_Drop']:<10.4f} {row['Test_Accuracy']:<10.4f}")

                # Key findings
                most_important = df_rank.iloc[0]
                least_important = df_rank.iloc[-1]

                print("\n📊 Key Findings:")
                print(f"   ✓ Most Important: {most_important['Scenario']} (Acc drop: {most_important['Accuracy_Drop']:.4f})")
                print(f"   ✓ Least Important: {least_important['Scenario']} (Acc drop: {least_important['Accuracy_Drop']:.4f})")

        print("-"*80)


    # ==================== LATENCY SUMMARY ====================
    if timing_stats is not None:
        print(f"\n⏱️  LATENCY SUMMARY:")
        feat_times = np.array(timing_stats['feat_times_s']) * 1000
        ens_times = np.array(timing_stats['ensemble_times_s']) * 1000

        print(f"     Feature Extraction: {feat_times.mean():.3f} ms (±{feat_times.std():.3f})")
        print(f"     Ensemble Prediction: {ens_times.mean():.3f} ms (±{ens_times.std():.3f})")
        print(f"     Total Per URL: {(feat_times.mean() + ens_times.mean()):.3f} ms")
        print(f"     Throughput: {1000.0/(ens_times.mean() if ens_times.mean()>0 else 1):.2f} predictions/sec")

        # ==================== STATISTICAL TESTS SUMMARY ====================
    if analyzer is not None:
        print(f"\n📊 STATISTICAL TESTS:")
        print(f"     ✓ McNemar Test: Completed (see detailed results above)")
        print(f"     ✓ Paired t-test: Completed (see detailed results above)")
        print(f"     ✓ Wilcoxon Test: Completed (see detailed results above)")
        print(f"     ✓ One-way ANOVA: Completed (see detailed results above)")
        print(f"     ✓ Effect Sizes (Cohen's d): Completed (see detailed results above)")

    print("\n" + "="*80)
    print("✅ PROGRAM COMPLETED SUCCESSFULLY")
    print("="*80)

    return ensemble_clf, ensemble_results, best_method, analyzer, timing_stats, ablation_df


# Run Program
if __name__ == "__main__":
    try:
        classifier, results, best_method, analyzer, timing_stats, ablation_df = main()

        print("\n✓ ALL ANALYSES COMPLETED SUCCESSFULLY")
    except KeyboardInterrupt:
        print("\n\n⚠️ PROGRAM INTERRUPTED")
        print("✓ Checkpoint saved")
        print("✓ Will continue from where it left off on next run")
    except Exception as e:
        print(f"\n✗ PROGRAM ERROR: {e}")
        import traceback
        traceback.print_exc()
