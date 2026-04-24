# -*- coding: latin5 -*-
import os
import shutil
import time
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import Counter
import re
import sys
from urllib.parse import urlparse
import warnings
from google.colab import drive
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
import json

# =============================================================================
# 1. ADIM: GOOGLE DRIVE VE AYARLAR
# =============================================================================
print("="*80)
print("=== GOOGLE DRIVE BAĞLANTISI VE AYARLAR ===")
print("="*80)

if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

drive_folder = "/content/drive/MyDrive/clasicraporIlkDataset"
os.makedirs(drive_folder, exist_ok=True)
print(f"[BİLGİ] Kayıt Klasörü: {drive_folder}\n")

sys.setrecursionlimit(21000)
warnings.filterwarnings('ignore')

# Kütüphane Kontrolleri
try:
    import tldextract
except ImportError:
    tldextract = None

try:
    from wordsegment import load, segment
    load()
except ImportError:
    def segment(text): return [text]

# Güvenli / cached tldextract (eğer mevcutsa)
if tldextract:
    try:
        tld_extractor = tldextract.TLDExtract(cache_file=os.path.join(drive_folder, "tld_cache"), suffix_list_urls=None)
    except Exception:
        tld_extractor = None
else:
    tld_extractor = None

# Regex Derlemeleri (Hız için global)
RE_WORD = re.compile(r'\w+')
RE_IP = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')
RE_SPECIAL = re.compile(r'[^\w\d\s]')

# CPU Çekirdek Sayısı
n_jobs = max(1, multiprocessing.cpu_count() - 1)
print(f"[SİSTEM] Kullanılacak CPU Çekirdeği: {n_jobs}")

os.environ["OMP_NUM_THREADS"] = str(n_jobs)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_jobs)
os.environ["MKL_NUM_THREADS"] = str(n_jobs)
print(f"Using n_jobs={n_jobs} and limited BLAS threads accordingly")

joblib_backend = 'loky'
joblib_verbose = 10

# =============================================================================
# 2. ADIM: VERİ YÜKLEME VE PARALEL ÖN İŞLEME (CACHE İLE)
# =============================================================================
RAW_DATA_FILE = "Yeniyirmibin_dataset_V1.txt"
rows = []
try:
    with open(RAW_DATA_FILE, 'r', encoding='utf-8') as fr:
        for ln in fr:
            ln = ln.strip()
            if not ln or ',' not in ln: continue
            parts = ln.rsplit(',', 1)
            if len(parts) == 2:
                rows.append((parts[0].strip(), parts[1].strip()))
except FileNotFoundError:
    print(f"HATA: {RAW_DATA_FILE} bulunamadı.")
    sys.exit(1)

n_samples_orig = len(rows)
print(f"Total samples (raw): {n_samples_orig}")
y_all_orig = np.array([int(lbl) for _, lbl in rows])
all_urls = [r[0] for r in rows]

# *** DÜZELTİLDİ: V4 dataseti için ayrı cache dosyası ***
cache_path = os.path.join(drive_folder, "preprocessed_cache_v1.pkl")

def fast_special_char_count(s):
    cnt = 0
    for ch in s:
        if not (ch.isalnum() or ch.isspace()):
            cnt += 1
    return cnt

def preprocess_url(url):
    try:
        parsed = urlparse(url)
        tokens = RE_WORD.findall(url)
        segmented_tokens = []
        for t in tokens:
            if len(t) > 5:
                try:
                    segmented_tokens.extend(segment(t))
                except:
                    segmented_tokens.append(t)
            else:
                segmented_tokens.append(t)
        ngrams3 = [url[i:i+3] for i in range(len(url)-2)] if len(url) >= 3 else []
        ngrams4 = [url[i:i+4] for i in range(len(url)-3)] if len(url) >= 4 else []
        try:
            if tld_extractor:
                tld = tld_extractor(url).suffix
            elif tldextract:
                ext = tldextract.extract(url)
                tld = ext.suffix
            else:
                tld = url.split('.')[-1] if '.' in url else ''
        except:
            tld = ''
        return {
            'url': url,
            'netloc': parsed.netloc,
            'path_len': len(parsed.path),
            'url_len': len(url),
            'special_char_count': fast_special_char_count(url),
            'tokens': tokens,
            'segmented_tokens': segmented_tokens,
            'ngrams3': ngrams3,
            'ngrams4': ngrams4,
            'tld': tld,
            'has_ip': bool(RE_IP.search(url)),
            'has_at': ('@' in url),
            'tiny': ('bit.ly' in url or 'tinyurl.com' in url),
            'https_dash': ('https-' in url),
            'file_ext': any(e in url for e in ['.exe', '.pdf', '.zip', '.rar']),
            'netloc_count_dots': parsed.netloc.count('.'),
            'has_dash_netloc': ('-' in parsed.netloc),
            'port': parsed.port,
            'double_slash': (url.count('//') > 1)
        }
    except Exception:
        return None

# --- Preprocessing with batches + tqdm + autosave ---
if os.path.exists(cache_path):
    t0 = time.time()
    try:
        with open(cache_path, "rb") as f:
            preprocessed_data = pickle.load(f)
        print(f"[CACHE] Preprocessed cache yüklendi. Süre: {time.time()-t0:.2f}s")
    except Exception as e:
        print(f"[CACHE] Yükleme hatası, yeniden preprocess yapılacak: {e}")
        try:
            os.remove(cache_path)
        except:
            pass
        preprocessed_data = None
else:
    preprocessed_data = None

if preprocessed_data is None:
    print(f"-> {n_samples_orig} URL paralel olarak işleniyor (önbelleğe kaydedilecek)...")
    t_start_pre = time.time()
    batch_size = 5000
    preprocessed_data = []
    for i in tqdm(range(0, n_samples_orig, batch_size), desc="Preprocessing batches"):
        batch_urls = all_urls[i:i+batch_size]
        batch_data = Parallel(n_jobs=n_jobs, batch_size=1000)(
            delayed(preprocess_url)(u) for u in batch_urls
        )
        preprocessed_data.extend(batch_data)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(preprocessed_data, f, protocol=4)
        except Exception:
            pass
    dur = time.time() - t_start_pre
    print(f"-> Preprocessing tamamlandı! Süre: {dur:.2f} sn")

valid_indices = [i for i, p in enumerate(preprocessed_data) if p is not None]
if len(valid_indices) != n_samples_orig:
    print(f"[UYARI] {n_samples_orig - len(valid_indices)} kayıt preprocess sırasında atıldı.")
preprocessed_data = [preprocessed_data[i] for i in valid_indices]
y_all = y_all_orig[valid_indices]
rows = [rows[i] for i in valid_indices]
n_samples = len(preprocessed_data)
print(f"[BILGI] Geçerli örnek sayısı: {n_samples}")

from sklearn.model_selection import train_test_split

# HOLDOUT SPLIT (DL ile aynı protokol)
all_indices = np.arange(n_samples)

train_indices, holdout_indices = train_test_split(
    all_indices,
    test_size=0.2,
    stratify=y_all,
    random_state=42
)

print("\n[HOLDOUT SPLIT]")
print(f"Train size : {len(train_indices)}")
print(f"Test size  : {len(holdout_indices)}")
print(f"Train dist : {np.bincount(y_all[train_indices])}")
print(f"Test dist  : {np.bincount(y_all[holdout_indices])}")

# =============================================================================
# 3. ADIM: RAM STORE VE FEATURE OLUŞTURUCU
# =============================================================================
class RAMFeatureStore:
    def __init__(self): self.clear()
    def clear(self):
        self.bag_of_words = {}
        self.segmented_bag_of_words = {}
        self.bag_of_ngrams = {}
        self.bag_of_4grams = {}
        self.tld_weights = {}

ram_store = RAMFeatureStore()

def check_features(items, store_dict):
    if not items or not store_dict:
        return 0.0
    get = store_dict.get
    s = 0.0
    for it in items:
        s += get(it, 0.0)
    return s

def build_counters_from_train(train_idx, threshold=20):
    total_items = len(train_idx)
    print(f"\n[INFO] build_counters_from_train başlatılıyor... Train örnek sayısı: {total_items}")

    cnt_tokens = Counter()
    cnt_segmented = Counter()
    cnt_ng3 = Counter()
    cnt_ng4 = Counter()
    cnt_tld = Counter()

    pos_count = 0
    t0 = time.time()

    for n, i in enumerate(tqdm(list(train_idx), desc="Building counters", unit="it", leave=True), start=1):
        label = int(y_all[i])
        if label == 1:
            pos_count += 1
            p = preprocessed_data[i]
            if p is None:
                continue
            cnt_tokens.update(p.get('tokens', []))
            cnt_segmented.update(p.get('segmented_tokens', []))
            cnt_ng3.update(p.get('ngrams3', []))
            cnt_ng4.update(p.get('ngrams4', []))
            tld_val = p.get('tld', '')
            if tld_val:
                cnt_tld[tld_val] += 1

        if (n % 50000 == 0) or (n == total_items):
            elapsed = time.time() - t0
            per_sec = n / elapsed if elapsed > 0 else 0.0
            remaining = max(total_items - n, 0)
            eta = remaining / per_sec if per_sec > 0 else float('inf')
            print(f"   -> {n}/{total_items} ({n/total_items*100:.1f}%) - {per_sec:.1f} it/s - ETA: {eta:.1f}s")

    elapsed_total = time.time() - t0
    print(f"[INFO] Pozitif (label=1) örnek sayısı: {pos_count}")
    try:
        avg_per_pos = elapsed_total/pos_count if pos_count else 0
    except Exception:
        avg_per_pos = 0
    print(f"[INFO] Sayaç oluşturma tamamlandı. Süre: {elapsed_total:.2f} s — Pozitif örnek başına ~{avg_per_pos:.6f}s (ortalama)")

    total = sum(cnt_tokens.values()) or 1
    ram_store.bag_of_words = {k: v/total for k, v in cnt_tokens.items() if v > threshold}

    total_seg = sum(cnt_segmented.values()) or 1
    ram_store.segmented_bag_of_words = {k: v/total_seg for k, v in cnt_segmented.items() if v > threshold}

    total_3 = sum(cnt_ng3.values()) or 1
    ram_store.bag_of_ngrams = {k: v/total_3 for k, v in cnt_ng3.items() if v > threshold}

    total_4 = sum(cnt_ng4.values()) or 1
    ram_store.bag_of_4grams = {k: v/total_4 for k, v in cnt_ng4.items() if v > threshold}

    total_tld = sum(cnt_tld.values()) or 1
    ram_store.tld_weights = {k: v/total_tld for k, v in cnt_tld.items()}

    print("[INFO] Ağırlıklar RAM'e kaydedildi.")
    print("[INFO] build_counters_from_train tamamlandı.\n")


def build_feature_vector_from_index(i):
    p = preprocessed_data[i]
    bow = ram_store.bag_of_words
    segbow = ram_store.segmented_bag_of_words
    ng3 = ram_store.bag_of_ngrams
    ng4 = ram_store.bag_of_4grams
    tldw = ram_store.tld_weights
    b_val = s_val = n_val = g4_val = 0.0
    if bow:
        getb = bow.get
        for t in p['tokens']:
            b_val += getb(t, 0.0)
    if segbow:
        gets = segbow.get
        for t in p['segmented_tokens']:
            s_val += gets(t, 0.0)
    if ng3:
        getn = ng3.get
        for t in p['ngrams3']:
            n_val += getn(t, 0.0)
    if ng4:
        getg = ng4.get
        for t in p['ngrams4']:
            g4_val += getg(t, 0.0)
    tld_val = tldw.get(p['tld'], 0.0) if tldw else 0.0
    fvals = [
        p['url_len'], p['special_char_count'],
        (p['special_char_count'] / p['url_len']) if p['url_len']>0 else 0,
        tld_val,
        1 if p['has_ip'] else 0,
        1 if p['tiny'] else 0,
        1 if p['has_at'] else 0,
        1 if p['double_slash'] else 0,
        1 if p['has_dash_netloc'] else 0,
        1 if p['netloc_count_dots'] > 1 else 0,
        1 if p['port'] and p['port'] not in [80,443] else 0,
        1 if p['https_dash'] else 0,
        1 if p['file_ext'] else 0,
        len(set(p['tokens'])),
        b_val, s_val, n_val, g4_val,
        (p['path_len'] / p['url_len']) if p['url_len']>0 else 0,
        (len(p['netloc'])/p['url_len']) if p['url_len']>0 else 0
    ]
    return fvals

# =============================================================================
# 4. ADIM: MODELLER
# =============================================================================
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

models = {
    'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42),
    'Naive Bayes': GaussianNB(),
    'MLP': MLPClassifier(hidden_layer_sizes=(180, 280), max_iter=1000, learning_rate_init=0.003, random_state=42, early_stopping=True, n_iter_no_change=10)
}

def drop_columns_safe(X, cols_to_drop):
    if X is None or X.size == 0:
        return X
    if not cols_to_drop:
        return X
    valid = [c for c in cols_to_drop if 0 <= c < X.shape[1]]
    return np.delete(X, valid, axis=1) if valid else X

from tqdm import tqdm as _tqdm

def predict_with_progress(clf, X, batch_size=5000, model_name="Model"):
    n = len(X)
    preds = []
    for start in _tqdm(range(0, n, batch_size), desc=f"Predicting {model_name}", leave=False):
        end = min(start + batch_size, n)
        batch = X[start:end]
        batch_preds = clf.predict(batch)
        preds.extend(batch_preds)
    return np.array(preds)

def calculate_performance_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4 and cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
    elif cm.size == 1 and cm.shape == (1,1):
        if np.all(y_true == 1):
            tp = cm[0,0]; tn = fp = fn = 0
        else:
            tn = cm[0,0]; tp = fp = fn = 0
    else:
        tn = fp = fn = tp = 0
    sens = tp/(tp+fn) if (tp+fn) > 0 else 0
    spec = tn/(tn+fp) if (tn+fp) > 0 else 0
    fnr = fn/(tp+fn) if (tp+fn) > 0 else 0
    fpr = fp/(fp+tn) if (fp+tn) > 0 else 0
    return {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1,
        'sensitivity': sens, 'specificity': spec, 'fnr': fnr, 'fpr': fpr,
        'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)
    }

scenarios = {
    'BASELINE': [],
    'REMOVE_BoW': [14],
    'REMOVE_SegBoW': [15],
    'REMOVE_NGRAMS_3_4': [16,17],
    'REMOVE_TLD': [3],
    'REMOVE_RATIOS': [2,18,19]
}

total_steps_per_fold = len(scenarios) * len(models)

# =============================================================================
# 4.5 ROBUST CHECKPOINT / RESUME HANDLER
# =============================================================================
partial_csv_name = "Detailed_Performance_Report_Per_Fold_partial.csv"
partial_csv_local = os.path.join(".", partial_csv_name)
partial_csv_drive = os.path.join(drive_folder, partial_csv_name)
completed_meta_name = "completed_folds.json"
completed_meta_local = os.path.join(".", completed_meta_name)
completed_meta_drive = os.path.join(drive_folder, completed_meta_name)

def atomic_save_csv(df, path):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    try:
        os.replace(tmp, path)
    except Exception:
        import shutil
        shutil.move(tmp, path)

def robust_copy(src, dst, max_retries=3, delay=1.0):
    for attempt in range(1, max_retries + 1):
        try:
            shutil.copy2(src, dst)
            return True
        except Exception as e:
            if attempt == max_retries:
                print(f"[CHECKPOINT] copy failed after {attempt} attempts: {e}")
                return False
            time.sleep(delay)

expected_rows_per_fold = len(scenarios) * len(models)

def load_partial_and_determine_completed(path):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[CHECKPOINT] Partial load failed ({path}): {e}")
        return [], set(), set()

    recs = df.to_dict(orient='records')
    folds_counts = df.groupby('Fold').size().to_dict() if 'Fold' in df.columns else {}
    completed = set()
    incomplete = set()
    for f, cnt in folds_counts.items():
        try:
            f_int = int(f)
        except:
            continue
        if cnt >= expected_rows_per_fold:
            completed.add(f_int)
        else:
            incomplete.add(f_int)
    return recs, completed, incomplete

full_report_data = []
completed_folds = set()
incomplete_folds = set()
existing_keys = set()

if os.path.exists(partial_csv_local):
    recs, comp, incom = load_partial_and_determine_completed(partial_csv_local)
    full_report_data.extend(recs)
    completed_folds.update(comp)
    incomplete_folds.update(incom)
    print(f"[CHECKPOINT] Yerel partial yüklendi. Tamamlanan foldlar (tam): {sorted(completed_folds)}; Eksik foldlar: {sorted(incomplete_folds)}")
else:
    if os.path.exists(partial_csv_drive):
        try:
            shutil.copy2(partial_csv_drive, partial_csv_local)
            recs, comp, incom = load_partial_and_determine_completed(partial_csv_local)
            full_report_data.extend(recs)
            completed_folds.update(comp)
            incomplete_folds.update(incom)
            print(f"[CHECKPOINT] Drive'dan partial kopyalandı ve yüklendi. Tamamlanan foldlar (tam): {sorted(completed_folds)}; Eksik foldlar: {sorted(incomplete_folds)}")
        except Exception as e:
            print(f"[CHECKPOINT] Drive partial kopyalanamadı: {e}")

if os.path.exists(completed_meta_local):
    try:
        with open(completed_meta_local, "r", encoding="utf-8") as f:
            folds_list = json.load(f)
            completed_folds.update(map(int, folds_list))
            print(f"[CHECKPOINT] Yerel completed_folds.json yüklendi: {sorted(completed_folds)}")
    except Exception:
        pass
elif os.path.exists(completed_meta_drive):
    try:
        shutil.copy2(completed_meta_drive, completed_meta_local)
        with open(completed_meta_local, "r", encoding="utf-8") as f:
            folds_list = json.load(f)
            completed_folds.update(map(int, folds_list))
            print(f"[CHECKPOINT] Drive completed_folds.json yüklendi: {sorted(completed_folds)}")
    except Exception:
        pass

for r in full_report_data:
    try:
        key = (r.get('Scenario'), int(r.get('Fold')), r.get('Model'))
        existing_keys.add(key)
    except Exception:
        continue

print(f"[CHECKPOINT] Başlangıç: {len(full_report_data)} kayıt yüklendi, {len(existing_keys)} benzersiz anahtar bulundu.")

def save_partial_and_meta(full_report_data_list, completed_folds_set):
    try:
        df_partial = pd.DataFrame(full_report_data_list)
        atomic_save_csv(df_partial, partial_csv_local)
    except Exception as e:
        print(f"[CHECKPOINT] Local partial save failed: {e}")
        return False
    try:
        robust_copy(partial_csv_local, partial_csv_drive)
    except Exception as e:
        print(f"[CHECKPOINT] Drive partial copy failed: {e}")
    try:
        tmp_meta = completed_meta_local + ".tmp"
        with open(tmp_meta, "w", encoding="utf-8") as mf:
            json.dump(sorted(list(completed_folds_set)), mf)
        os.replace(tmp_meta, completed_meta_local)
        try:
            robust_copy(completed_meta_local, completed_meta_drive)
        except:
            pass
    except Exception as e:
        print(f"[CHECKPOINT] completed_folds meta save failed: {e}")
    return True

def upsert_record(full_list, rec, existing_keys_set):
    key = (rec.get('Scenario'), int(rec.get('Fold')), rec.get('Model'))
    if key in existing_keys_set:
        try:
            for idx, old in enumerate(full_list):
                try:
                    old_key = (old.get('Scenario'), int(old.get('Fold')), old.get('Model'))
                except Exception:
                    old_key = None
                if old_key == key:
                    full_list[idx] = rec
                    return
            full_list.append(rec)
        except Exception:
            full_list.append(rec)
    else:
        full_list.append(rec)
        existing_keys_set.add(key)

# =============================================================================
# 5. ADIM: ANA DÖNGÜ (FOLD + SCENARIOS + TQDM)
# =============================================================================
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for fold, (train_idx_rel, val_idx_rel) in enumerate(cv.split(train_indices, y_all[train_indices]), start=1):

    # *** DÜZELTİLDİ: checkpoint kontrolü index hesabından önce ***
    if fold in completed_folds:
        print(f"\n=== Fold {fold} already completed (checkpoint). Skipping. ===")
        continue

    train_idx = train_indices[train_idx_rel]
    test_idx = train_indices[val_idx_rel]

    print(f"\n{'='*60}\n=== Fold {fold} Başlıyor ===\n{'='*60}")
    ram_store.clear()
    build_counters_from_train(train_idx)

    X_train_base = Parallel(n_jobs=n_jobs, batch_size=1000, backend=joblib_backend, verbose=joblib_verbose)(
        delayed(build_feature_vector_from_index)(i) for i in list(train_idx)
    )
    X_test_base = Parallel(n_jobs=n_jobs, batch_size=1000, backend=joblib_backend, verbose=joblib_verbose)(
        delayed(build_feature_vector_from_index)(i) for i in list(test_idx)
    )
    X_train_base = np.array(X_train_base)
    X_test_base = np.array(X_test_base)
    y_train_fold = y_all[train_idx]
    y_test_fold = y_all[test_idx]

    c0 = np.sum(y_train_fold == 0)
    c1 = np.sum(y_train_fold == 1)
    imb_ratio = c0 / c1 if c1 > 0 else 0
    print(f"-> Matris hazır. Train: {X_train_base.shape}, Test: {X_test_base.shape}")

    step = 0
    with _tqdm(total=total_steps_per_fold, desc=f"Fold {fold} progress", unit="step") as fold_pbar:
        for scen_name, remove_cols in scenarios.items():
            print(f"   Senaryo: {scen_name} (Fold {fold})")
            X_train = drop_columns_safe(X_train_base, remove_cols)
            X_test = drop_columns_safe(X_test_base, remove_cols)

            for model_name, model in models.items():
                step += 1
                model_desc = f"{scen_name} | {model_name}"
                fold_pbar.set_description(f"Fold {fold}: {model_desc}")
                fold_pbar.refresh()

                clf = clone(model)
                t0 = time.time()
                try:
                    classes = np.unique(y_train_fold)
                    weights = compute_class_weight('balanced', classes=classes, y=y_train_fold)
                    sw = np.array([weights[int(y)] for y in y_train_fold])
                    clf.fit(X_train, y_train_fold, sample_weight=sw)
                    fit_mode = "sample_weight"
                except Exception:
                    ros = RandomOverSampler(random_state=42)
                    Xr, yr = ros.fit_resample(X_train, y_train_fold)
                    clf.fit(Xr, yr)
                    fit_mode = "resample"
                train_time = time.time() - t0

                times = []
                n_sample_timing = min(100, len(X_test))
                if n_sample_timing > 0:
                    subset = X_test[:n_sample_timing]
                    for row in subset:
                        ts = time.perf_counter()
                        clf.predict(row.reshape(1, -1))
                        times.append((time.perf_counter() - ts) * 1000.0)
                    avg_time = float(np.mean(times))
                    std_time = float(np.std(times))
                else:
                    avg_time = 0.0
                    std_time = 0.0

                y_pred = predict_with_progress(clf, X_test, model_name=model_name)
                m = calculate_performance_metrics(y_test_fold, y_pred)

                record = {
                    'Scenario': scen_name, 'Fold': fold, 'Model': model_name,
                    'Train_Time_Sec': round(train_time, 4),
                    'Avg_Single_Pred_Time_ms': round(avg_time, 6),
                    'Std_Single_Pred_Time_ms': round(std_time, 6),
                    'Imbalance_Ratio': round(imb_ratio, 2),
                    'Accuracy': m['accuracy'], 'Precision': m['precision'], 'Recall': m['recall'],
                    'F1_Score': m['f1_score'], 'Sensitivity': m['sensitivity'], 'Specificity': m['specificity'],
                    'FNR': m['fnr'], 'FPR': m['fpr'],
                    'TN': m['TN'], 'FP': m['FP'], 'FN': m['FN'], 'TP': m['TP'],
                    'Fit_Mode': fit_mode
                }

                upsert_record(full_report_data, record, existing_keys)
                print(f"    ✓ {model_name}: Acc={m['accuracy']:.4f}, F1={m['f1_score']:.4f} ({train_time:.1f}s)")

                fold_pbar.update(1)

                try:
                    save_partial_and_meta(full_report_data, completed_folds)
                except Exception as e:
                    print(f"[CHECKPOINT] save_partial hatası (model): {e}")

    try:
        completed_folds.add(fold)
        save_partial_and_meta(full_report_data, completed_folds)
        print(f"[CHECKPOINT] Fold {fold} completed and recorded in checkpoint meta.")
    except Exception as e:
        print(f"[CHECKPOINT] Error while finalizing checkpoint for fold {fold}: {e}")

# =============================================================================
# 6. ADIM: FINAL HOLDOUT DEĞERLENDİRMESİ
# =============================================================================
print("\n" + "="*80)
print("FINAL HOLDOUT EVALUATION")
print("="*80)

# Tüm train seti üzerinden counter ve feature oluştur
build_counters_from_train(train_indices)

print("Building X_train_final...")
X_train_final = Parallel(n_jobs=2, backend="threading")(
    delayed(build_feature_vector_from_index)(i) for i in train_indices
)

print("Building X_test_holdout...")
X_test_holdout = Parallel(n_jobs=2, backend="threading")(
    delayed(build_feature_vector_from_index)(i) for i in holdout_indices
)

X_train_final = np.array(X_train_final)
X_test_holdout = np.array(X_test_holdout)

y_train_final = y_all[train_indices]
y_test_holdout = y_all[holdout_indices]

print(f"Final train shape: {X_train_final.shape}")
print(f"Holdout test shape: {X_test_holdout.shape}")

# ======================== FINAL MODEL EĞİTİMİ ========================
final_results = []
# Holdout tahminlerini saklayalım (istatistiksel test için)
holdout_predictions = {}

for model_name, model in models.items():
    print(f"\nTraining final model: {model_name}")

    clf = clone(model)
    classes = np.unique(y_train_final)
    weights = compute_class_weight('balanced', classes=classes, y=y_train_final)
    sw = np.array([weights[int(y)] for y in y_train_final])

    try:
        clf.fit(X_train_final, y_train_final, sample_weight=sw)
        fit_mode = "sample_weight"
    except TypeError:
        ros = RandomOverSampler(random_state=42)
        Xr, yr = ros.fit_resample(X_train_final, y_train_final)
        clf.fit(Xr, yr)
        fit_mode = "resample"

    y_pred = clf.predict(X_test_holdout)
    # Tahminleri sakla
    holdout_predictions[model_name] = y_pred

    metrics = calculate_performance_metrics(y_test_holdout, y_pred)

    print(f"{model_name} HOLDOUT -> Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")

    # *** GENİŞLETİLDİ: Tüm metrikler kaydediliyor ***
    final_results.append({
        "Model": model_name,
        "Accuracy":    round(metrics['accuracy'],    6),
        "Precision":   round(metrics['precision'],   6),
        "Recall":      round(metrics['recall'],      6),
        "F1":          round(metrics['f1_score'],    6),
        "Sensitivity": round(metrics['sensitivity'], 6),
        "Specificity": round(metrics['specificity'], 6),
        "FNR":         round(metrics['fnr'],         6),
        "FPR":         round(metrics['fpr'],         6),
        "TN":          metrics['TN'],
        "FP":          metrics['FP'],
        "FN":          metrics['FN'],
        "TP":          metrics['TP'],
        "Fit_Mode":    fit_mode
    })

# Holdout sonuçlarını CSV'ye kaydet
df_holdout = pd.DataFrame(final_results)
holdout_file = "Final_Holdout_Test_Results.csv"
df_holdout.to_csv(holdout_file, index=False)
print(f"\n[OK] Holdout test sonuçları kaydedildi: {holdout_file}")

try:
    shutil.copy2(holdout_file, os.path.join(drive_folder, holdout_file))
    print("[OK] Holdout sonuçları Drive'a kopyalandı")
except:
    print("[UYARI] Holdout sonuçları Drive'a kopyalanamadı")

# =============================================================================
# 6.1 HOLDOUT CONFUSION MATRIX GRAFİKLERİ
# =============================================================================
print("\n" + "="*80)
print("=== HOLDOUT CONFUSION MATRIX GRAFİKLERİ ===")
print("="*80)

for rec in final_results:
    model_name = rec['Model']
    try:
        tn, fp, fn, tp = rec['TN'], rec['FP'], rec['FN'], rec['TP']
        cm_arr = np.array([[tn, fp], [fn, tp]])

        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm_arr, annot=True, fmt=".0f", cbar=False,
            xticklabels=['Pred: Normal', 'Pred: Phishing'],
            yticklabels=['True: Normal', 'True: Phishing']
        )
        plt.title(f"Holdout CM - {model_name}")
        plt.ylabel("True"); plt.xlabel("Pred")
        plt.tight_layout()

        iname = f"CM_Holdout_{model_name.replace(' ', '_')}.png"
        plt.savefig(iname, dpi=300)
        plt.close()

        try:
            shutil.copy2(iname, os.path.join(drive_folder, iname))
            print(f"[OK] {iname} kaydedildi ve Drive'a kopyalandı.")
        except Exception as e:
            print(f"[UYARI] {iname} Drive'a kopyalanamadı: {e}")
    except Exception as e:
        print(f"[HATA] CM oluşturulurken ({model_name}) hata: {e}")

# =============================================================================
# 6.2 HOLDOUT ABLATION ÇALIŞMASI
# =============================================================================
print("\n" + "="*80)
print("=== HOLDOUT ABLATION ÇALIŞMASI ===")
print("="*80)

holdout_ablation_data = []

for scen_name, remove_cols in scenarios.items():
    print(f"\n[ABL] Senaryo: {scen_name}")

    X_train_scen = drop_columns_safe(X_train_final.copy(), remove_cols)
    X_test_scen  = drop_columns_safe(X_test_holdout.copy(), remove_cols)

    for model_name, model in models.items():
        clf = clone(model)
        t0 = time.time()
        try:
            classes = np.unique(y_train_final)
            weights = compute_class_weight('balanced', classes=classes, y=y_train_final)
            sw = np.array([weights[int(y)] for y in y_train_final])
            clf.fit(X_train_scen, y_train_final, sample_weight=sw)
            fit_mode = "sample_weight"
        except Exception:
            ros = RandomOverSampler(random_state=42)
            Xr, yr = ros.fit_resample(X_train_scen, y_train_final)
            clf.fit(Xr, yr)
            fit_mode = "resample"
        train_time = time.time() - t0

        y_pred = predict_with_progress(clf, X_test_scen, model_name=model_name)
        m = calculate_performance_metrics(y_test_holdout, y_pred)

        holdout_ablation_data.append({
            'Scenario':    scen_name,
            'Model':       model_name,
            'Accuracy':    round(m['accuracy'],    6),
            'Precision':   round(m['precision'],   6),
            'Recall':      round(m['recall'],      6),
            'F1_Score':    round(m['f1_score'],    6),
            'Sensitivity': round(m['sensitivity'], 6),
            'Specificity': round(m['specificity'], 6),
            'FNR':         round(m['fnr'],         6),
            'FPR':         round(m['fpr'],         6),
            'TN':          m['TN'], 'FP': m['FP'], 'FN': m['FN'], 'TP': m['TP'],
            'Train_Time_Sec': round(train_time, 4),
            'Fit_Mode':    fit_mode
        })
        print(f"   ✓ {model_name}: Acc={m['accuracy']:.4f}, F1={m['f1_score']:.4f}")

df_holdout_ablation = pd.DataFrame(holdout_ablation_data)

# Accuracy Drop sütunu (BASELINE'a göre)
baseline_accs = df_holdout_ablation[df_holdout_ablation['Scenario'] == 'BASELINE'].set_index('Model')['Accuracy']
df_holdout_ablation['Accuracy_Drop'] = df_holdout_ablation.apply(
    lambda row: round(baseline_accs.get(row['Model'], row['Accuracy']) - row['Accuracy'], 6), axis=1
)

abl_file = "Holdout_Ablation_Study.csv"
df_holdout_ablation.to_csv(abl_file, index=False)
print(f"\n[OK] Holdout ablation kaydedildi: {abl_file}")
try:
    shutil.copy2(abl_file, os.path.join(drive_folder, abl_file))
    print("[OK] Drive'a kopyalandı.")
except:
    print("[UYARI] Drive'a kopyalanamadı.")

# Ablation Özeti: Ortalama (model başına)
abl_summary = df_holdout_ablation.groupby(['Scenario', 'Model'])[[
    'Accuracy','Precision','Recall','F1_Score','Sensitivity','Specificity',
    'FNR','FPR','Accuracy_Drop'
]].mean().round(6).reset_index()

abl_summary_file = "Holdout_Ablation_Summary.csv"
abl_summary.to_csv(abl_summary_file, index=False)
print(f"[OK] Ablation özeti kaydedildi: {abl_summary_file}")
try:
    shutil.copy2(abl_summary_file, os.path.join(drive_folder, abl_summary_file))
except:
    pass

# Ablation bar grafiği
try:
    fig, ax = plt.subplots(figsize=(12, 6))
    df_bar = df_holdout_ablation.groupby(['Scenario', 'Model'])['Accuracy'].mean().unstack()
    df_bar.plot(kind='bar', ax=ax)
    ax.set_title("Holdout Ablation Study - Accuracy per Scenario & Model")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Scenario")
    ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    abl_fig = "Holdout_Ablation_BarPlot.png"
    plt.savefig(abl_fig, dpi=300)
    plt.close()
    try:
        shutil.copy2(abl_fig, os.path.join(drive_folder, abl_fig))
        print(f"[OK] {abl_fig} kaydedildi.")
    except:
        pass
except Exception as e:
    print(f"[HATA] Ablation bar grafiği oluşturulamadı: {e}")

# =============================================================================
# 6.3 HOLDOUT İSTATİSTİKSEL ANALİZ
# =============================================================================
print("\n" + "="*80)
print("=== HOLDOUT İSTATİSTİKSEL ANALİZ ===")
print("="*80)

try:
    from scipy.stats import ttest_rel, wilcoxon, f_oneway

    model_names_list = list(holdout_predictions.keys())
    stat_lines = []

    def slog(text):
        print(text)
        stat_lines.append(text)

    # Holdout'ta her modelin doğruluk skoru tek bir sayı olduğu için
    # karşılaştırma tahmin vektörleri üzerinden yapılır (McNemar tarzı).
    # Aynı zamanda CV fold sonuçlarından holdout skoru için istatistik üretiyoruz.

    slog("\n" + "="*60)
    slog("HOLDOUT İSTATİSTİKSEL ANALİZ RAPORU")
    slog("="*60)

    # --- 1) Model başına Holdout Metrikleri ---
    slog("\n--- Holdout Metrik Özeti ---")
    slog(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Spec':<10} {'FPR':<10} {'FNR':<10}")
    slog("-"*90)
    for rec in final_results:
        slog(f"{rec['Model']:<20} {rec['Accuracy']:<10.4f} {rec['Precision']:<10.4f} "
             f"{rec['Recall']:<10.4f} {rec['F1']:<10.4f} {rec['Specificity']:<10.4f} "
             f"{rec['FPR']:<10.4f} {rec['FNR']:<10.4f}")

    # --- 2) Paired Accuracy Comparison (tahmin vektörleri temelinde) ---
    # Tahmin doğruluğunu örnek bazında karşılaştır (McNemar tarzı, n=holdout set büyüklüğü)
    slog("\n--- Örnek Bazlı Doğru/Yanlış Tahmini (McNemar-tarzı) ---")
    for i, m1 in enumerate(model_names_list):
        for m2 in model_names_list[i+1:]:
            correct_m1 = (holdout_predictions[m1] == y_test_holdout).astype(int)
            correct_m2 = (holdout_predictions[m2] == y_test_holdout).astype(int)
            # McNemar: b = m1 doğru m2 yanlış, c = m1 yanlış m2 doğru
            b = np.sum((correct_m1 == 1) & (correct_m2 == 0))
            c = np.sum((correct_m1 == 0) & (correct_m2 == 1))
            n_discordant = b + c
            if n_discordant > 0:
                chi2 = (abs(b - c) - 1)**2 / (b + c)
                import scipy.stats as sp_stats
                p_mcnemar = 1 - sp_stats.chi2.cdf(chi2, df=1)
                sig = "Significant" if p_mcnemar < 0.05 else "Not significant"
                slog(f"{m1} vs {m2}: b={b}, c={c}, chi2={chi2:.4f}, p={p_mcnemar:.6f} ({sig})")
            else:
                slog(f"{m1} vs {m2}: identical predictions (no discordant pairs)")

    # --- 3) CV fold accuracies ile Wilcoxon & Paired T-Test (10 fold) ---
    # full_report_data'dan BASELINE+holdout fold accuracy'leri çek
    df_cv_base = pd.DataFrame(full_report_data)
    if not df_cv_base.empty and 'Scenario' in df_cv_base.columns:
        df_cv_base = df_cv_base[df_cv_base['Scenario'] == 'BASELINE']

    if not df_cv_base.empty:
        cv_fold_accs = {
            m: df_cv_base[df_cv_base['Model'] == m]['Accuracy'].values
            for m in df_cv_base['Model'].unique()
        }

        slog("\n--- CV Fold Accuracy: Paired T-Test ---")
        model_names_cv = list(cv_fold_accs.keys())
        for i, m1 in enumerate(model_names_cv):
            for m2 in model_names_cv[i+1:]:
                a = cv_fold_accs[m1]
                b_arr = cv_fold_accs[m2]
                if len(a) == len(b_arr) and len(a) > 1:
                    t_stat, p_val = ttest_rel(a, b_arr)
                    sig = "Significant" if p_val < 0.05 else "Not significant"
                    slog(f"{m1} vs {m2}: t={t_stat:.4f}, p={p_val:.6f} ({sig})")
                else:
                    slog(f"{m1} vs {m2}: insufficient data")

        slog("\n--- CV Fold Accuracy: Wilcoxon Test ---")
        for i, m1 in enumerate(model_names_cv):
            for m2 in model_names_cv[i+1:]:
                a = cv_fold_accs[m1]
                b_arr = cv_fold_accs[m2]
                if len(a) == len(b_arr) and len(a) > 1:
                    try:
                        w_stat, p_val = wilcoxon(a, b_arr)
                        sig = "Significant" if p_val < 0.05 else "Not significant"
                        slog(f"{m1} vs {m2}: W={w_stat:.4f}, p={p_val:.6f} ({sig})")
                    except Exception as e:
                        slog(f"{m1} vs {m2}: wilcoxon failed ({e})")
                else:
                    slog(f"{m1} vs {m2}: insufficient data")

        slog("\n--- CV Fold Accuracy: ANOVA ---")
        all_data = [v for v in cv_fold_accs.values() if len(v) > 0]
        if len(all_data) > 1:
            f_stat, p_val = f_oneway(*all_data)
            slog(f"F={f_stat:.4f}, p={p_val:.6f}")
        else:
            slog("ANOVA: yetersiz grup/veri")

        slog("\n--- CV Fold Accuracy: Cohen's d ---")
        for i, m1 in enumerate(model_names_cv):
            for m2 in model_names_cv[i+1:]:
                a = np.array(cv_fold_accs[m1])
                b_arr = np.array(cv_fold_accs[m2])
                if len(a) == len(b_arr) and len(a) > 1:
                    std_a, std_b = a.std(ddof=1), b_arr.std(ddof=1)
                    pooled = np.sqrt(((len(a)-1)*std_a**2 + (len(b_arr)-1)*std_b**2) / (len(a)+len(b_arr)-2))
                    d = (a.mean() - b_arr.mean()) / pooled if pooled > 0 else 0
                    slog(f"{m1} vs {m2}: d={d:.4f}")
                else:
                    slog(f"{m1} vs {m2}: insufficient data")
    else:
        slog("[UYARI] CV BASELINE verisi bulunamadı, fold testleri atlandı.")

    # --- 4) CV vs Holdout Karşılaştırması ---
    slog("\n--- CV Ortalama vs Holdout Karşılaştırması ---")
    slog(f"{'Model':<20} {'CV Mean':<12} {'CV Std':<12} {'Holdout':<12} {'Fark':<10}")
    slog("-"*68)
    if not df_cv_base.empty:
        for rec in final_results:
            mn = rec['Model']
            if mn in cv_fold_accs and len(cv_fold_accs[mn]) > 0:
                cv_mean = np.mean(cv_fold_accs[mn])
                cv_std  = np.std(cv_fold_accs[mn])
                diff    = abs(cv_mean - rec['Accuracy'])
                slog(f"{mn:<20} {cv_mean:<12.4f} {cv_std:<12.4f} {rec['Accuracy']:<12.4f} {diff:<10.4f}")

    # Raporu kaydet
    stat_path = "Holdout_Statistical_Significance_Report.txt"
    with open(stat_path, "w", encoding="utf-8") as f:
        f.write("\n".join(stat_lines))
    print(f"[OK] İstatistiksel rapor kaydedildi: {stat_path}")
    try:
        shutil.copy2(stat_path, os.path.join(drive_folder, stat_path))
        print("[OK] Drive'a kopyalandı.")
    except:
        print("[UYARI] Drive'a kopyalanamadı.")

except Exception as e:
    print(f"[HATA] İstatistiksel analiz sırasında hata: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# 7. ADIM: RAPORLAMA VE KAYIT
# =============================================================================
print("\n" + "="*80)
print("=== CV SONUÇLAR İŞLENİYOR VE DRIVE'A KAYDEDİLİYOR ===")
print("="*80)

try:
    if not os.path.exists('/content/drive'): drive.mount('/content/drive')
    if not os.path.exists(drive_folder): os.makedirs(drive_folder, exist_ok=True)
except Exception as e:
    print(f"[UYARI] Drive hazırlanırken hata: {e}")

def robust_copy_with_retries(src, dst, max_retries=3, delay_s=1.0):
    last_err = None
    for attempt in range(1, max_retries+1):
        try:
            ddir = os.path.dirname(dst)
            if ddir and not os.path.exists(ddir):
                os.makedirs(ddir, exist_ok=True)
            shutil.copy2(src, dst)
            return True, None
        except Exception as e:
            last_err = e
            time.sleep(delay_s)
    return False, last_err

def safe_copy_to_drive(local_path, drive_folder, fname=None, max_retries=3):
    try:
        if not fname:
            fname = os.path.basename(local_path)
        dst = os.path.join(drive_folder, fname)
        if not os.path.exists(drive_folder):
            try:
                os.makedirs(drive_folder, exist_ok=True)
            except Exception as e:
                print(f"[UYARI] Drive hedef klasör oluşturulamadı: {e}")
                return False
        ok, err = robust_copy_with_retries(local_path, dst, max_retries=max_retries)
        if ok:
            print(f"[OK] {fname} Drive'a yedeklendi: {dst}")
            return True
        else:
            print(f"[UYARI] {fname} Drive'a kopyalanamadı. Hata: {err}")
            return False
    except Exception as e:
        print(f"[UYARI] safe_copy_to_drive beklenmeyen hata: {e}")
        return False

def save_safe(df_obj, fname, copy_to_drive=True):
    try:
        tmp = fname + ".tmp"
        df_obj.to_csv(tmp, index=False)
        try:
            os.replace(tmp, fname)
        except Exception:
            shutil.move(tmp, fname)
        print(f"[OK] Yerel kaydedildi: {fname}")
    except Exception as e:
        print(f"[HATA] {fname} yerel kaydedilemedi: {e}")
        return False

    if copy_to_drive:
        try:
            success = safe_copy_to_drive(fname, drive_folder, os.path.basename(fname))
            if not success:
                print(f"[UYARI] {fname} Drive kopyası başarısız oldu.")
            return success
        except Exception as e:
            print(f"[UYARI] save_safe sırasında kopyalama hatası: {e}")
            return False
    return True

# DataFrame oluşturma ve sütun kontrolü
df_report = pd.DataFrame(full_report_data)

if df_report.empty:
    print("[UYARI] full_report_data boş — kayıt/rapor işlemleri atlanıyor.")
else:
    time_cols = ['Avg_Single_Pred_Time_ms', 'Std_Single_Pred_Time_ms']
    for c in time_cols:
        if c not in df_report.columns:
            df_report[c] = 0.0
        df_report[c] = pd.to_numeric(df_report[c], errors='coerce').fillna(0.0)

    df_report['Avg_Single_Pred_Time_ms'] = df_report['Avg_Single_Pred_Time_ms'].round(6)
    df_report['Std_Single_Pred_Time_ms'] = df_report['Std_Single_Pred_Time_ms'].round(6)

    detailed_fname = "Detailed_Performance_Report_Per_Fold.csv"
    save_safe(df_report, detailed_fname)

    metrics_agg = [
        'Train_Time_Sec', 'Avg_Single_Pred_Time_ms', 'Std_Single_Pred_Time_ms',
        'Imbalance_Ratio', 'Accuracy', 'F1_Score', 'Precision', 'Recall',
        'Sensitivity', 'Specificity', 'FNR', 'FPR', 'TN', 'FP', 'FN', 'TP'
    ]
    valid_metrics = [c for c in metrics_agg if c in df_report.columns]

    # *** DÜZELTİLDİ: num_cols scope sorunu giderildi ***
    num_cols = []

    if valid_metrics:
        ablation = df_report.groupby(['Scenario', 'Model'])[valid_metrics].mean().reset_index()
        num_cols = ablation.select_dtypes(include=[np.number]).columns.tolist()
        ablation[num_cols] = ablation[num_cols].round(6)
        save_safe(ablation, "Ablation_Study_Summary.csv")
    else:
        print("[UYARI] Ablation özeti oluşturulamadı — geçerli metric yok.")

    df_base = df_report[df_report['Scenario'] == 'BASELINE']
    if not df_base.empty:
        # num_cols burada da güvenli şekilde kullanılıyor
        avg_base_cols = [c for c in valid_metrics if c in df_base.columns]
        avg_base = df_base.groupby('Model')[avg_base_cols].mean().reset_index()
        avg_base_num = avg_base.select_dtypes(include=[np.number]).columns.tolist()
        avg_base[avg_base_num] = avg_base[avg_base_num].round(6)
        save_safe(avg_base, "Baseline_Average_Performance.csv")

        cm_cols = ['TN', 'FP', 'FN', 'TP']
        if all(c in df_base.columns for c in cm_cols):
            avg_cm = df_base.groupby('Model')[cm_cols].mean()
            for model_name in avg_cm.index:
                try:
                    tn, fp, fn, tp = avg_cm.loc[model_name].astype(float).values
                    cm_arr = np.array([[tn, fp], [fn, tp]])
                    plt.figure(figsize=(5, 4))
                    sns.heatmap(cm_arr, annot=True, fmt=".1f", cbar=False,
                                xticklabels=['Pred: Normal', 'Pred: Phishing'],
                                yticklabels=['True: Normal', 'True: Phishing'])
                    plt.title(f"Avg CM - {model_name} (Baseline, CV)")
                    plt.ylabel("True"); plt.xlabel("Pred")
                    plt.tight_layout()
                    iname = f"CM_Baseline_{model_name.replace(' ', '_')}.png"
                    plt.savefig(iname, dpi=300)
                    plt.close()
                    try:
                        safe_copy_to_drive(iname, drive_folder, iname)
                    except: pass
                except Exception as e:
                    print(f"[HATA] CM oluşturulurken ({model_name}) hata: {e}")
        else:
            print("[UYARI] Confusion Matrix sütunları eksik, grafik çizilemedi.")
    else:
        print("[UYARI] Baseline (Scenario=='BASELINE') için veri yok; CM çizilmedi.")

# --- CV İstatistiksel Analiz ---
try:
    df_base_stat = df_report[df_report['Scenario'] == 'BASELINE'] if not df_report.empty else pd.DataFrame()
    if not df_base_stat.empty:
        res_rec = {
            m: {'fold_accuracies': df_base_stat[df_base_stat['Model'] == m]['Accuracy'].values.tolist()}
            for m in df_base_stat['Model'].unique()
        }

        class StatisticalSignificanceAnalyzer_Fast:
            def __init__(self, results_per_model):
                self.results_per_model = results_per_model
                self.model_names = list(results_per_model.keys())
                self.report_lines = []

            def log(self, text):
                print(text)
                self.report_lines.append(text)

            def run_paired_ttest_all_vs_all(self):
                from scipy.stats import ttest_rel
                self.log("\n--- Paired T-Test ---")
                for i, m1 in enumerate(self.model_names):
                    for m2 in self.model_names[i+1:]:
                        a = np.array(self.results_per_model[m1]['fold_accuracies'])
                        b = np.array(self.results_per_model[m2]['fold_accuracies'])
                        if len(a) == len(b) and len(a) > 1:
                            t, p = ttest_rel(a, b)
                            res = "Significant" if p < 0.05 else "Not significant"
                            self.log(f"{m1} vs {m2}: t={t:.4f}, p={p:.6f} ({res})")
                        else:
                            self.log(f"{m1} vs {m2}: insufficient data for paired t-test")

            def run_wilcoxon_all_vs_all(self):
                from scipy.stats import wilcoxon
                self.log("\n--- Wilcoxon Test ---")
                for i, m1 in enumerate(self.model_names):
                    for m2 in self.model_names[i+1:]:
                        a = np.array(self.results_per_model[m1]['fold_accuracies'])
                        b = np.array(self.results_per_model[m2]['fold_accuracies'])
                        if len(a) == len(b) and len(a) > 0:
                            try:
                                w, p = wilcoxon(a, b)
                                res = "Significant" if p < 0.05 else "Not significant"
                                self.log(f"{m1} vs {m2}: w={w:.4f}, p={p:.6f} ({res})")
                            except Exception as e:
                                self.log(f"{m1} vs {m2}: wilcoxon failed ({e})")
                        else:
                            self.log(f"{m1} vs {m2}: insufficient data for wilcoxon")

            def run_anova_all_models(self):
                from scipy.stats import f_oneway
                self.log("\n--- ANOVA ---")
                all_data = []
                for m in self.model_names:
                    arr = np.array(self.results_per_model[m]['fold_accuracies'])
                    if len(arr) > 0:
                        all_data.append(arr)
                if len(all_data) > 1:
                    f, p = f_oneway(*all_data)
                    self.log(f"F={f:.4f}, p={p:.6f}")
                else:
                    self.log("ANOVA: insufficient groups/data")

            def calculate_effect_sizes(self):
                self.log("\n--- Cohen's d ---")
                for i, m1 in enumerate(self.model_names):
                    for m2 in self.model_names[i+1:]:
                        a = np.array(self.results_per_model[m1]['fold_accuracies'])
                        b = np.array(self.results_per_model[m2]['fold_accuracies'])
                        if len(a) == len(b) and len(a) > 1:
                            mean_a, mean_b = a.mean(), b.mean()
                            std_a, std_b = a.std(ddof=1), b.std(ddof=1)
                            pooled_std = np.sqrt(((len(a)-1)*std_a**2 + (len(b)-1)*std_b**2) / (len(a)+len(b)-2))
                            d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0
                            self.log(f"{m1} vs {m2}: d={d:.4f}")
                        else:
                            self.log(f"{m1} vs {m2}: insufficient data for Cohen's d")

            def save_report(self, filepath):
                try:
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write("\n".join(self.report_lines))
                    print(f"[OK] Statistical report saved to: {filepath}")
                except Exception as e:
                    print(f"[HATA] Statistical report kaydedilemedi: {e}")

        analyzer = StatisticalSignificanceAnalyzer_Fast(res_rec)
        analyzer.run_paired_ttest_all_vs_all()
        analyzer.run_wilcoxon_all_vs_all()
        analyzer.run_anova_all_models()
        analyzer.calculate_effect_sizes()

        stat_path = "Statistical_Significance_Report.txt"
        analyzer.save_report(stat_path)
        try:
            if os.path.exists(drive_folder):
                shutil.copy(stat_path, os.path.join(drive_folder, stat_path))
        except Exception as e:
            print(f"[HATA] Stat rapor Drive'a kopyalanamadı: {e}")
    else:
        print("[UYARI] İstatistiksel analiz için baseline verisi bulunamadı.")
except Exception as e:
    print(f"[HATA] İstatistiksel analiz sırasında hata: {e}")

print(f"\n=== İŞLEM TAMAMLANDI. Dosyalar: {drive_folder} ===")

# =============================================================================
# ÜRETILEN DOSYALAR ÖZETİ
# =============================================================================
print("\n" + "="*80)
print("ÜRETILEN DOSYALAR")
print("="*80)
output_files = [
    ("Detailed_Performance_Report_Per_Fold.csv",   "CV - Her fold/senaryo/model detay"),
    ("Ablation_Study_Summary.csv",                 "CV - Senaryo ablation özeti"),
    ("Baseline_Average_Performance.csv",           "CV - Baseline model ortalamaları"),
    ("CM_Baseline_*.png",                          "CV - Model başına ortalama CM grafikleri"),
    ("Statistical_Significance_Report.txt",        "CV - İstatistiksel testler"),
    ("Final_Holdout_Test_Results.csv",             "Holdout - Tüm metrikler"),
    ("CM_Holdout_*.png",                           "Holdout - Model başına CM grafikleri"),
    ("Holdout_Ablation_Study.csv",                 "Holdout - Senaryo x model detay"),
    ("Holdout_Ablation_Summary.csv",               "Holdout - Ablation özeti"),
    ("Holdout_Ablation_BarPlot.png",               "Holdout - Ablation bar grafiği"),
    ("Holdout_Statistical_Significance_Report.txt","Holdout - McNemar + CV testler"),
]
for fname, desc in output_files:
    print(f"  {fname:<50} -> {desc}")
print("="*80)