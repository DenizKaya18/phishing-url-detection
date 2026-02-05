# -*- coding: utf-8 -*-
"""
Data Loading Module
"""

import numpy as np
import sys


class DataLoader:
    """Handles loading and initial parsing of dataset"""
    
    def __init__(self, data_file):
        self.data_file = data_file
        self.rows = []
        self.urls = []
        self.labels = []
        
    def load_data(self):
        """Load data from text file"""
        print("="*80)
        print("=== LOADING DATA ===")
        print("="*80)
        
        try:
            with open(self.data_file, 'r', encoding='utf-8') as fr:
                for ln in fr:
                    ln = ln.strip()
                    if not ln or ',' not in ln:
                        continue
                    parts = ln.rsplit(',', 1)
                    if len(parts) == 2:
                        self.rows.append((parts[0].strip(), parts[1].strip()))
        except FileNotFoundError:
            print(f"ERROR: {self.data_file} not found.")
            sys.exit(1)
        
        self.urls = [r[0] for r in self.rows]
        self.labels = np.array([int(lbl) for _, lbl in self.rows])
        
        print(f"Total samples loaded: {len(self.rows)}")
        print(f"Label distribution: {np.bincount(self.labels)}")
        
        return self.urls, self.labels
    
    def get_data(self):
        """Return loaded data"""
        if not self.rows:
            self.load_data()
        return self.urls, self.labels
