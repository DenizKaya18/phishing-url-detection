# -*- coding: utf-8 -*-
"""
Checkpoint Management Module
"""

import os
import json
import time
import shutil
import pandas as pd


class CheckpointManager:
    """Handles checkpointing and resume functionality"""
    
    def __init__(self, config):
        self.config = config
        self.partial_csv_local = os.path.join(".", config.partial_csv_name)
        self.partial_csv_backup = os.path.join(config.output_folder, config.partial_csv_name)
        self.completed_meta_local = os.path.join(".", config.completed_meta_name)
        self.completed_meta_backup = os.path.join(config.output_folder, config.completed_meta_name)
        
        # Expected rows per fold
        self.expected_rows_per_fold = len({'BASELINE': []}) * 5  # scenarios × models
    
    def atomic_save_csv(self, df, path):
        """Save CSV atomically using temp file"""
        tmp = path + ".tmp"
        df.to_csv(tmp, index=False)
        try:
            os.replace(tmp, path)
        except Exception:
            shutil.move(tmp, path)
    
    def robust_copy(self, src, dst, max_retries=3, delay=1.0):
        """Copy file with retries"""
        for attempt in range(1, max_retries + 1):
            try:
                shutil.copy2(src, dst)
                return True
            except Exception as e:
                if attempt == max_retries:
                    print(f"[CHECKPOINT] Copy failed after {attempt} attempts: {e}")
                    return False
                time.sleep(delay)
        return False
    
    def load_partial_csv(self, path):
        """Load partial CSV and determine completed/incomplete folds"""
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[CHECKPOINT] Failed to load {path}: {e}")
            return [], set(), set()
        
        records = df.to_dict(orient='records')
        folds_counts = df.groupby('Fold').size().to_dict() if 'Fold' in df.columns else {}
        
        completed = set()
        incomplete = set()
        
        for fold, count in folds_counts.items():
            try:
                fold_int = int(fold)
            except:
                continue
            
            if count >= self.expected_rows_per_fold:
                completed.add(fold_int)
            else:
                incomplete.add(fold_int)
        
        return records, completed, incomplete
    
    def load_checkpoint(self):
        """Load checkpoint from local or drive"""
        full_report_data = []
        completed_folds = set()
        existing_keys = set()
        
        # Try loading local partial CSV
        if os.path.exists(self.partial_csv_local):
            recs, comp, incom = self.load_partial_csv(self.partial_csv_local)
            full_report_data.extend(recs)
            completed_folds.update(comp)
            print(f"[CHECKPOINT] Loaded local partial: {len(comp)} complete folds, {len(incom)} incomplete")
        
        # Try loading from backup if local doesn't exist
        elif os.path.exists(self.partial_csv_backup):
            try:
                shutil.copy2(self.partial_csv_backup, self.partial_csv_local)
                recs, comp, incom = self.load_partial_csv(self.partial_csv_local)
                full_report_data.extend(recs)
                completed_folds.update(comp)
                print(f"[CHECKPOINT] Loaded backup partial: {len(comp)} complete folds, {len(incom)} incomplete")
            except Exception as e:
                print(f"[CHECKPOINT] Failed to copy from backup: {e}")
        
        # Load completed folds metadata
        if os.path.exists(self.completed_meta_local):
            try:
                with open(self.completed_meta_local, "r", encoding="utf-8") as f:
                    folds_list = json.load(f)
                    completed_folds.update(map(int, folds_list))
                print(f"[CHECKPOINT] Loaded local metadata: {sorted(completed_folds)}")
            except Exception:
                pass
        
        elif os.path.exists(self.completed_meta_backup):
            try:
                shutil.copy2(self.completed_meta_backup, self.completed_meta_local)
                with open(self.completed_meta_local, "r", encoding="utf-8") as f:
                    folds_list = json.load(f)
                    completed_folds.update(map(int, folds_list))
                print(f"[CHECKPOINT] Loaded backup metadata: {sorted(completed_folds)}")
            except Exception:
                pass
        
        # Build existing keys set
        for r in full_report_data:
            try:
                key = (r.get('Scenario'), int(r.get('Fold')), r.get('Model'))
                existing_keys.add(key)
            except Exception:
                continue
        
        return full_report_data, completed_folds, existing_keys
    
    def save_checkpoint(self, full_report_data, completed_folds):
        """Save checkpoint both locally and to backup"""
        try:
            # Save CSV locally
            df_partial = pd.DataFrame(full_report_data)
            self.atomic_save_csv(df_partial, self.partial_csv_local)
        except Exception as e:
            print(f"[CHECKPOINT] Failed to save local CSV: {e}")
            return False
        
        # Try copying to backup
        try:
            self.robust_copy(self.partial_csv_local, self.partial_csv_backup)
        except Exception as e:
            print(f"[CHECKPOINT] Failed to copy CSV to backup: {e}")
        
        # Save metadata
        try:
            tmp_meta = self.completed_meta_local + ".tmp"
            with open(tmp_meta, "w", encoding="utf-8") as mf:
                json.dump(sorted(list(completed_folds)), mf)
            os.replace(tmp_meta, self.completed_meta_local)
            
            # Try copying to backup
            try:
                self.robust_copy(self.completed_meta_local, self.completed_meta_backup)
            except:
                pass
        except Exception as e:
            print(f"[CHECKPOINT] Failed to save metadata: {e}")
        
        return True
