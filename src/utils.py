import sys
import os
from datetime import datetime
from .config import RESULTS_DIR

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        log_file = os.path.join(RESULTS_DIR, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M')}.txt")
        self.log = open(log_file, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def format_time(seconds):
    if seconds < 60: return f"{seconds:.2f} second"
    elif seconds < 3600: return f"{seconds/60:.2f} min"
    else: return f"{seconds/3600:.2f} hour"