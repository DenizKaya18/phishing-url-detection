# -*- coding: utf-8 -*-
"""
Feature Building Module
"""

import numpy as np
from collections import Counter
from joblib import Parallel, delayed
from tqdm import tqdm


class RAMFeatureStore:
    """In-memory feature store"""
    
    def __init__(self):
        self.clear()
    
    def clear(self):
        """Clear all stored features"""
        self.bag_of_words = {}
        self.segmented_bag_of_words = {}
        self.bag_of_ngrams = {}
        self.bag_of_4grams = {}
        self.tld_weights = {}


class FeatureBuilder:
    """Builds feature vectors from preprocessed URLs"""
    
    def __init__(self, config, preprocessed_data):
        self.config = config
        self.preprocessed_data = preprocessed_data
        self.store = RAMFeatureStore()
        
    def build_counters_from_train(self, train_idx, labels, threshold=20):
        """Build feature counters from training data (positive samples only)"""
        total_items = len(train_idx)
        print(f"\n[INFO] Building counters from {total_items} training samples...")
        
        cnt_tokens = Counter()
        cnt_segmented = Counter()
        cnt_ng3 = Counter()
        cnt_ng4 = Counter()
        cnt_tld = Counter()
        
        pos_count = 0
        
        for i in tqdm(list(train_idx), desc="Building counters", unit="it", leave=True):
            label = int(labels[i])
            if label == 1:  # Only positive samples
                pos_count += 1
                p = self.preprocessed_data[i]
                
                if p is None:
                    continue
                
                cnt_tokens.update(p.get('tokens', []))
                cnt_segmented.update(p.get('segmented_tokens', []))
                cnt_ng3.update(p.get('ngrams3', []))
                cnt_ng4.update(p.get('ngrams4', []))
                
                tld_val = p.get('tld', '')
                if tld_val:
                    cnt_tld[tld_val] += 1
        
        print(f"[INFO] Positive samples: {pos_count}")
        
        # Normalize and store
        total = sum(cnt_tokens.values()) or 1
        self.store.bag_of_words = {k: v/total for k, v in cnt_tokens.items() if v > threshold}
        
        total_seg = sum(cnt_segmented.values()) or 1
        self.store.segmented_bag_of_words = {k: v/total_seg for k, v in cnt_segmented.items() if v > threshold}
        
        total_3 = sum(cnt_ng3.values()) or 1
        self.store.bag_of_ngrams = {k: v/total_3 for k, v in cnt_ng3.items() if v > threshold}
        
        total_4 = sum(cnt_ng4.values()) or 1
        self.store.bag_of_4grams = {k: v/total_4 for k, v in cnt_ng4.items() if v > threshold}
        
        total_tld = sum(cnt_tld.values()) or 1
        self.store.tld_weights = {k: v/total_tld for k, v in cnt_tld.items()}
        
        print("[INFO] Counters built and normalized.\n")
    
    def build_feature_vector_from_index(self, i):
        """Build feature vector for a single sample"""
        p = self.preprocessed_data[i]
        
        # Calculate bag-of-words features
        bow = self.store.bag_of_words
        segbow = self.store.segmented_bag_of_words
        ng3 = self.store.bag_of_ngrams
        ng4 = self.store.bag_of_4grams
        tldw = self.store.tld_weights
        
        b_val = s_val = n_val = g4_val = 0.0
        
        if bow:
            for t in p['tokens']:
                b_val += bow.get(t, 0.0)
        
        if segbow:
            for t in p['segmented_tokens']:
                s_val += segbow.get(t, 0.0)
        
        if ng3:
            for t in p['ngrams3']:
                n_val += ng3.get(t, 0.0)
        
        if ng4:
            for t in p['ngrams4']:
                g4_val += ng4.get(t, 0.0)
        
        tld_val = tldw.get(p['tld'], 0.0) if tldw else 0.0
        
        # Build feature vector
        fvals = [
            p['url_len'],
            p['special_char_count'],
            (p['special_char_count'] / p['url_len']) if p['url_len'] > 0 else 0,
            tld_val,
            1 if p['has_ip'] else 0,
            1 if p['tiny'] else 0,
            1 if p['has_at'] else 0,
            1 if p['double_slash'] else 0,
            1 if p['has_dash_netloc'] else 0,
            1 if p['netloc_count_dots'] > 1 else 0,
            1 if p['port'] and p['port'] not in [80, 443] else 0,
            1 if p['https_dash'] else 0,
            1 if p['file_ext'] else 0,
            len(set(p['tokens'])),
            b_val,
            s_val,
            n_val,
            g4_val,
            (p['path_len'] / p['url_len']) if p['url_len'] > 0 else 0,
            (len(p['netloc']) / p['url_len']) if p['url_len'] > 0 else 0
        ]
        
        return fvals
    
    def build_features(self, indices):
        """Build features for multiple samples in parallel"""
        X = Parallel(
            n_jobs=self.config.n_jobs,
            batch_size=1000,
            backend=self.config.joblib_backend,
            verbose=self.config.joblib_verbose
        )(delayed(self.build_feature_vector_from_index)(i) for i in list(indices))
        
        return np.array(X)
