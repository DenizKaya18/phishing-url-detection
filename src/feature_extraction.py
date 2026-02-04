"""
Feature extraction module matching Colab optimization.
Uses RAM-based caching and parallel execution.
"""

import numpy as np
import re
import tldextract
from urllib.parse import urlparse
from collections import Counter
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from wordsegment import load, segment
import time

# Load word segmentation data
load()

class VectorizedFeatureExtractor:
    """
    Feature extraction (vectorized) - using SUM of token weights per URL
    Matches Colab implementation exactly.
    """

    def __init__(self, bow_data=None, seg_bow_data=None, ngrams_data=None,
                 grams4_data=None, tld_data=None):
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

                path = parsed.path or ""
                path_length = len(path)

                domain_info = tldextract.extract(url)
                domain = (domain_info.domain + '.' + domain_info.suffix) if domain_info.suffix else domain_info.domain
                domain_length = len(domain)

                path_ratio = path_length / max(url_len, 1)
                domain_ratio = domain_length / max(url_len, 1)

                tokens = self._tokenize(url)

                # BoW weighted SUM
                bow_sum = 0.0
                bow_match_count = 0
                for t in tokens:
                    w = float(self.bow_data.get(t, 0.0))
                    if w > 0:
                        bow_sum += w
                        bow_match_count += 1

                # Segmented BoW weighted SUM
                seg_bow_sum = 0.0
                for t in tokens:
                    parts = self._segment_token_safe(t)
                    for p in parts:
                        w = float(self.seg_bow_data.get(p, 0.0))
                        if w > 0:
                            seg_bow_sum += w

                # N-grams (3-grams) weighted SUM
                ngrams_sum = 0.0
                for t in tokens:
                    ngrams = self._make_ngrams_from_string(t, n=3)
                    for ng in ngrams:
                        w = float(self.ngrams_data.get(ng, 0.0))
                        if w > 0:
                            ngrams_sum += w

                # 4-grams weighted SUM
                grams4_sum = 0.0
                for t in tokens:
                    grams4 = self._make_ngrams_from_string(t, n=4)
                    for g4 in grams4:
                        w = float(self.grams4_data.get(g4, 0.0))
                        if w > 0:
                            grams4_sum += w

                bag_of_words_count = float(bow_match_count)

                features = [
                    float(url_len),                      # 0
                    float(special_chars),                # 1
                    float(url_len_ratio),                # 2
                    tld_weight,                          # 3
                    1.0 if self._has_ip(url) else 0.0,   # 4
                    1.0 if self._has_tinyurl(url) else 0.0, # 5
                    1.0 if '@' in url else 0.0,          # 6
                    1.0 if url.count('//') > 1 else 0.0, # 7
                    1.0 if '-' in netloc or (netloc.split('.')[0] if netloc else '').find('-')>=0 else 0.0, # 8
                    1.0 if subdomain_count > 1 else 0.0, # 9
                    1.0 if parsed.port and parsed.port not in [80, 443] else 0.0, # 10
                    1.0 if 'https-' in url else 0.0,     # 11
                    1.0 if any(ext in url for ext in ['.exe', '.pdf', '.zip', '.rar']) else 0.0, # 12
                    bag_of_words_count,                  # 13
                    bow_sum,                             # 14
                    seg_bow_sum,                         # 15
                    ngrams_sum,                          # 16
                    grams4_sum,                          # 17
                    float(path_ratio),                   # 18
                    float(domain_ratio),                 # 19
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
            return np.empty((0, 20)), np.empty((0,), dtype=int), []

        X = np.vstack(all_feats)
        y = np.concatenate(all_labels)
        return X, y, all_urls


class IsolatedFeatureManager:
    """
    Manages isolated feature caches using RAM-based parallel processing.
    Matches Colab's ThreadPoolExecutor implementation.
    """

    def __init__(self):
        self.feature_caches = {}
        # Adjust workers based on Colab script logic
        from multiprocessing import cpu_count
        self.workers = min(8, cpu_count())
        print(f"✓ Isolated Feature Manager initialized (Workers: {self.workers})")

    def create_features_for_data(self, url_label_list, cache_name):
        print(f"\n[ISOLATED FEATURES] Creating PARALLEL features for: {cache_name}")

        cache = {
            'bow': {}, 'seg_bow': {}, 'ngrams': {}, 'grams4': {}, 'tld': {}
        }

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                'bow': executor.submit(self._create_bow_isolated, url_label_list, cache['bow']),
                'seg_bow': executor.submit(self._create_seg_bow_isolated, url_label_list, cache['seg_bow']),
                'ngrams': executor.submit(self._create_ngrams_isolated, url_label_list, 3, cache['ngrams']),
                'grams4': executor.submit(self._create_ngrams_isolated, url_label_list, 4, cache['grams4']),
                'tld': executor.submit(self._create_tld_isolated, url_label_list, cache['tld'])
            }

            progress_bar = tqdm(
                total=len(futures),
                desc=f"  {cache_name:12}",
                bar_format='{desc}: {percentage:3.0f}% {bar} [{n_fmt}/{total_fmt}]',
                colour='green',
                unit=' feature'
            )

            for future in as_completed(futures.values()):
                future.result()
                progress_bar.update(1)

            progress_bar.close()

        self.feature_caches[cache_name] = cache
        print(f"✓ {cache_name} parallel features completed")
        return cache

    def _create_bow_isolated(self, url_label_list, target_cache):
        malicious_tokens = []
        for url, label in url_label_list:
            if label == 1:
                tokens = re.findall(r'\w+', url)
                malicious_tokens.extend(tokens)

        if not malicious_tokens:
            return

        counter = Counter(malicious_tokens)
        total = sum(counter.values())

        target_cache.update({
            word: count / total
            for word, count in counter.items()
            if count > 20
        })

    def _create_seg_bow_isolated(self, url_label_list, target_cache):
        all_tokens = []
        for url, label in url_label_list:
            if label == 1:
                tokens = re.findall(r'\w+', url)
                all_tokens.extend(tokens)

        if not all_tokens:
            return

        segmented_tokens = []
        for token in all_tokens:
            try:
                segmented = segment(token)
                segmented_tokens.extend(segmented)
            except:
                segmented_tokens.append(token)

        counter = Counter(segmented_tokens)
        total = sum(counter.values())

        target_cache.update({
            word: count / total
            for word, count in counter.items()
            if count > 20
        })

    def _create_ngrams_isolated(self, url_label_list, n, target_cache):
        all_tokens = []
        for url, label in url_label_list:
            if label == 1:
                tokens = re.findall(r'\w+', url)
                all_tokens.extend(tokens)

        if not all_tokens:
            return

        all_ngrams = []
        for token in all_tokens:
            if len(token) >= n:
                token_ngrams = [token[i:i+n] for i in range(len(token) - n + 1)]
                all_ngrams.extend(token_ngrams)

        counter = Counter(all_ngrams)
        total = sum(counter.values())

        target_cache.update({
            ngram: count / total
            for ngram, count in counter.items()
            if count > 20
        })

    def _create_tld_isolated(self, url_label_list, target_cache):
        tld_list = []
        for url, label in url_label_list:
            if label == 1:
                try:
                    extracted = tldextract.extract(url)
                    if extracted.suffix:
                        tld_list.append(extracted.suffix)
                except:
                    continue

        if not tld_list:
            return

        counter = Counter(tld_list)
        total = len(tld_list)

        target_cache.update({
            tld: count / total
            for tld, count in counter.items()
        })

    def get_cache(self, cache_name):
        return self.feature_caches.get(cache_name, {})

def extract_features_for_fold_optimized(urls, labels, is_train=True,
                                       batch_size=5000,
                                       bow_data=None, seg_bow_data=None,
                                       ngrams_data=None, grams4_data=None,
                                       tld_data=None):
    """
    Optimized feature extraction wrapper used by ensemble classifier.
    """
    print(f"\n[EXTRACT] Extracting features...")
    prefix = "Train" if is_train else "Val"
    start_time = time.time()

    extractor = VectorizedFeatureExtractor(
        bow_data=bow_data or {},
        seg_bow_data=seg_bow_data or {},
        ngrams_data=ngrams_data or {},
        grams4_data=grams4_data or {},
        tld_data=tld_data or {}
    )

    X_features, y_labels, processed_urls = extractor.extract_batch_vectorized(
        urls, labels, batch_size=batch_size
    )

    elapsed = time.time() - start_time
    print(f"✓ {prefix} Features extracted")
    print(f"  Samples: {len(X_features)}")
    print(f"  Time: {elapsed:.1f} seconds")
    
    return X_features, y_labels, processed_urls