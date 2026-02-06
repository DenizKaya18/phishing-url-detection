import re
import numpy as np
import tldextract
from urllib.parse import urlparse
from wordsegment import load, segment
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import Counter
from .config import CACHE_WORKERS, FEATURE_WORKERS, COLOR_CACHE, COLOR_FEATURE

# Initialize word segmentation
load()

class VectorizedFeatureExtractor:
    """
    High-speed vectorized feature extraction using 
    pre-computed token weights from training data.
    """

    def __init__(self, bow=None, seg_bow=None, ngrams=None, grams4=None, tld=None):
        self.bow_data = bow or {}
        self.seg_bow_data = seg_bow or {}
        self.ngrams_data = ngrams or {}
        self.grams4_data = grams4 or {}
        self.tld_data = tld or {}

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
        # Regular expression for IPv4 addresses
        return bool(re.match(r'.*\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b.*', url))

    @staticmethod
    def _has_tinyurl(url):
        # Check for common URL shorteners
        shorteners = ['bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'rebrand.ly']
        return any(s in url for s in shorteners)

    @staticmethod
    def _tokenize(url):
        # Primary tokenization on non-alphanumeric characters
        return re.findall(r'\w+', url)

    @staticmethod
    def _make_ngrams_from_string(s, n=3):
        if not s or len(s) < n:
            return []
        return [s[i:i+n] for i in range(len(s)-n+1)]

    def _segment_token_safe(self, token):
        try:
            segmented = segment(token)
            if segmented and isinstance(segmented, (list, tuple)):
                return segmented
        except Exception:
            pass
        # Fallback heuristic: split on alpha/numeric boundaries
        parts = re.findall(r'[A-Za-z]{3,}|[0-9]+', token)
        return parts if parts else [token]

    def _process_batch_vectorized(self, urls, labels):
        batch_features = []
        batch_labels = []
        processed_urls = []

        for url, label in zip(urls, labels):
            try:
                # Structural Features
                url_len = len(url)
                special_chars = sum(1 for c in url if not c.isalnum())
                url_len_ratio = special_chars / max(url_len, 1)

                tld = self._extract_tld(url)
                tld_weight = float(self.tld_data.get(tld, 0.0))

                parsed = urlparse(url)
                netloc = parsed.netloc or ""
                subdomain_count = netloc.count('.')

                path = parsed.path or ""
                path_length = len(path)

                domain_info = tldextract.extract(url)
                domain = (domain_info.domain + '.' + domain_info.suffix) if domain_info.suffix else domain_info.domain
                domain_length = len(domain)

                path_ratio = path_length / max(url_len, 1)
                domain_ratio = domain_length / max(url_len, 1)

                # Tokenization and Weighted Metrics
                tokens = self._tokenize(url)

                # 1. Bag of Words Weighted SUM
                bow_sum = 0.0
                bow_match_count = 0
                for t in tokens:
                    w = float(self.bow_data.get(t, 0.0))
                    if w > 0:
                        bow_sum += w
                        bow_match_count += 1

                # 2. Segmented BoW Weighted SUM
                seg_bow_sum = 0.0
                for t in tokens:
                    parts = self._segment_token_safe(t)
                    for p in parts:
                        w = float(self.seg_bow_data.get(p, 0.0))
                        if w > 0:
                            seg_bow_sum += w

                # 3. 3-grams Weighted SUM
                ngrams_sum = 0.0
                for t in tokens:
                    ngrams = self._make_ngrams_from_string(t, n=3)
                    for ng in ngrams:
                        w = float(self.ngrams_data.get(ng, 0.0))
                        if w > 0:
                            ngrams_sum += w

                # 4. 4-grams Weighted SUM
                grams4_sum = 0.0
                for t in tokens:
                    grams4 = self._make_ngrams_from_string(t, n=4)
                    for g4 in grams4:
                        w = float(self.grams4_data.get(g4, 0.0))
                        if w > 0:
                            grams4_sum += w

                # Construct feature vector (Total 20 features)
                features = [
                    float(url_len),                  # 0
                    float(special_chars),            # 1
                    float(url_len_ratio),            # 2
                    tld_weight,                      # 3
                    1.0 if self._has_ip(url) else 0.0, # 4
                    1.0 if self._has_tinyurl(url) else 0.0, # 5
                    1.0 if '@' in url else 0.0,      # 6
                    1.0 if url.count('//') > 1 else 0.0, # 7
                    1.0 if '-' in netloc else 0.0,   # 8
                    1.0 if subdomain_count > 1 else 0.0, # 9
                    1.0 if parsed.port and parsed.port not in [80, 443] else 0.0, # 10
                    1.0 if 'https-' in url else 0.0, # 11
                    1.0 if any(ext in url for ext in ['.exe', '.pdf', '.zip', '.rar']) else 0.0, # 12
                    float(bow_match_count),          # 13
                    bow_sum,                         # 14
                    seg_bow_sum,                     # 15
                    ngrams_sum,                      # 16
                    grams4_sum,                      # 17
                    float(path_ratio),               # 18
                    float(domain_ratio)              # 19
                ]

                batch_features.append(features)
                batch_labels.append(int(label))
                processed_urls.append(url)

            except Exception:
                continue

        if not batch_features:
            return None, None, None

        return np.array(batch_features, dtype=float), np.array(batch_labels, dtype=int), processed_urls

    def extract_batch_vectorized(self, urls, labels, batch_size=5000):
        """Wrapper to iterate in chunks and stack results."""
        all_feats, all_labels, all_urls = [], [], []

        for i in range(0, len(urls), batch_size):
            feats, labs, proc = self._process_batch_vectorized(urls[i:i+batch_size], labels[i:i+batch_size])
            if feats is not None:
                all_feats.append(feats)
                all_labels.append(labs)
                all_urls.extend(proc)

        if not all_feats:
            return np.empty((0, 20)), np.empty((0,), dtype=int), []

        return np.vstack(all_feats), np.concatenate(all_labels), all_urls


class IsolatedFeatureManager:
    """Manages isolated feature caches for each fold to prevent data leakage."""

    def __init__(self):
        self.feature_caches = {}
        print("✓ Isolated Feature Manager Initialized")

    def create_features_for_data(self, url_label_list, cache_name):
        """Parallel creation of BoW, Seg-BoW, N-grams, and TLD weights."""
        print(f"\n[ISOLATED FEATURES] Building parallel cache for: {cache_name}")
        
        cache = {'bow': {}, 'seg_bow': {}, 'ngrams': {}, 'grams4': {}, 'tld': {}}

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                'bow': executor.submit(self._create_bow_isolated, url_label_list, cache['bow']),
                'seg_bow': executor.submit(self._create_seg_bow_isolated, url_label_list, cache['seg_bow']),
                'ngrams': executor.submit(self._create_ngrams_isolated, url_label_list, 3, cache['ngrams']),
                'grams4': executor.submit(self._create_ngrams_isolated, url_label_list, 4, cache['grams4']),
                'tld': executor.submit(self._create_tld_isolated, url_label_list, cache['tld'])
            }

            pbar = tqdm(total=len(futures), desc=f"  {cache_name:12}", colour=COLOR_CACHE, unit='task')

            for future in as_completed(futures.values()):
                future.result()
                pbar.update(1)
            pbar.close()

        self.feature_caches[cache_name] = cache
        return cache

    def _create_bow_isolated(self, url_label_list, target_cache, threshold=20):
        malicious_tokens = []
        for url, label in url_label_list:
            if label == 1:
                malicious_tokens.extend(re.findall(r'\w+', url))
        
        if malicious_tokens:
            counter = Counter(malicious_tokens)
            total = sum(counter.values())
            target_cache.update({word: count / total for word, count in counter.items() if count > threshold})

    def _create_seg_bow_isolated(self, url_label_list, target_cache, threshold=20):
        segmented_tokens = []
        for url, label in url_label_list:
            if label == 1:
                tokens = re.findall(r'\w+', url)
                for t in tokens:
                    try:
                        segmented_tokens.extend(segment(t))
                    except:
                        segmented_tokens.append(t)
        
        if segmented_tokens:
            counter = Counter(segmented_tokens)
            total = sum(counter.values())
            target_cache.update({word: count / total for word, count in counter.items() if count > threshold})

    def _create_ngrams_isolated(self, url_label_list, n, target_cache, threshold=20):
        all_ngrams = []
        for url, label in url_label_list:
            if label == 1:
                tokens = re.findall(r'\w+', url)
                for t in tokens:
                    if len(t) >= n:
                        all_ngrams.extend([t[i:i+n] for i in range(len(t)-n+1)])
        
        if all_ngrams:
            counter = Counter(all_ngrams)
            total = sum(counter.values())
            target_cache.update({ngram: count / total for ngram, count in counter.items() if count > threshold})

    def _create_tld_isolated(self, url_label_list, target_cache):
        tlds = []
        for url, label in url_label_list:
            if label == 1:
                try:
                    ext = tldextract.extract(url)
                    if ext.suffix: tlds.append(ext.suffix)
                except: continue
        
        if tlds:
            counter = Counter(tlds)
            total = len(tlds)
            target_cache.update({tld: count / total for tld, count in counter.items()})