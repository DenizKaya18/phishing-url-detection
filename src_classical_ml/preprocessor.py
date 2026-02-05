# -*- coding: utf-8 -*-
"""
URL Preprocessing Module
"""

import os
import pickle
import time
import re
from urllib.parse import urlparse
from joblib import Parallel, delayed
from tqdm import tqdm


class URLPreprocessor:
    """Handles URL preprocessing and caching"""
    
    def __init__(self, config):
        self.config = config
        self.preprocessed_data = None
        
        # Regex patterns
        self.RE_WORD = re.compile(r'\w+')
        self.RE_IP = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')
        
        # Optional libraries
        self._init_optional_libs()
        
    def _init_optional_libs(self):
        """Initialize optional libraries (tldextract, wordsegment)"""
        try:
            import tldextract
            try:
                self.tld_extractor = tldextract.TLDExtract(
                    cache_file=os.path.join(self.config.output_folder, "tld_cache"),
                    suffix_list_urls=None
                )
            except Exception:
                self.tld_extractor = None
        except ImportError:
            self.tld_extractor = None
            
        try:
            from wordsegment import load, segment
            load()
            self.segment = segment
        except ImportError:
            self.segment = lambda text: [text]
    
    @staticmethod
    def fast_special_char_count(s):
        """Count special characters efficiently"""
        cnt = 0
        for ch in s:
            if not (ch.isalnum() or ch.isspace()):
                cnt += 1
        return cnt
    
    def preprocess_url(self, url):
        """Preprocess single URL"""
        try:
            parsed = urlparse(url)
            tokens = self.RE_WORD.findall(url)
            
            # Word segmentation
            segmented_tokens = []
            for t in tokens:
                if len(t) > 5:
                    try:
                        segmented_tokens.extend(self.segment(t))
                    except:
                        segmented_tokens.append(t)
                else:
                    segmented_tokens.append(t)
            
            # N-grams
            ngrams3 = [url[i:i+3] for i in range(len(url)-2)] if len(url) >= 3 else []
            ngrams4 = [url[i:i+4] for i in range(len(url)-3)] if len(url) >= 4 else []
            
            # TLD extraction
            try:
                if self.tld_extractor:
                    tld = self.tld_extractor(url).suffix
                else:
                    import tldextract
                    ext = tldextract.extract(url)
                    tld = ext.suffix
            except:
                tld = url.split('.')[-1] if '.' in url else ''
            
            return {
                'url': url,
                'netloc': parsed.netloc,
                'path_len': len(parsed.path),
                'url_len': len(url),
                'special_char_count': self.fast_special_char_count(url),
                'tokens': tokens,
                'segmented_tokens': segmented_tokens,
                'ngrams3': ngrams3,
                'ngrams4': ngrams4,
                'tld': tld,
                'has_ip': bool(self.RE_IP.search(url)),
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
    
    def preprocess_all(self, urls):
        """Preprocess all URLs with caching"""
        cache_path = self.config.cache_path
        
        # Try loading from cache
        if os.path.exists(cache_path):
            t0 = time.time()
            try:
                with open(cache_path, "rb") as f:
                    self.preprocessed_data = pickle.load(f)
                print(f"[CACHE] Loaded preprocessed cache in {time.time()-t0:.2f}s")
                return self.preprocessed_data
            except Exception as e:
                print(f"[CACHE] Load failed, reprocessing: {e}")
                try:
                    os.remove(cache_path)
                except:
                    pass
        
        # Preprocess with batching
        print(f"-> Preprocessing {len(urls)} URLs in parallel...")
        t_start = time.time()
        batch_size = self.config.preprocessing_batch_size
        self.preprocessed_data = []
        
        for i in tqdm(range(0, len(urls), batch_size), desc="Preprocessing batches"):
            batch_urls = urls[i:i+batch_size]
            batch_data = Parallel(n_jobs=self.config.n_jobs, batch_size=1000)(
                delayed(self.preprocess_url)(u) for u in batch_urls
            )
            self.preprocessed_data.extend(batch_data)
            
            # Save intermediate results
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(self.preprocessed_data, f, protocol=4)
            except Exception:
                pass
        
        duration = time.time() - t_start
        print(f"-> Preprocessing completed in {duration:.2f}s")
        
        return self.preprocessed_data
    
    def filter_valid_samples(self, labels):
        """Filter out None values and return valid indices"""
        valid_indices = [i for i, p in enumerate(self.preprocessed_data) if p is not None]
        
        if len(valid_indices) != len(self.preprocessed_data):
            print(f"[WARNING] {len(self.preprocessed_data) - len(valid_indices)} samples dropped during preprocessing")
        
        self.preprocessed_data = [self.preprocessed_data[i] for i in valid_indices]
        valid_labels = labels[valid_indices]
        
        print(f"[INFO] Valid samples: {len(self.preprocessed_data)}")
        
        return valid_indices, valid_labels
