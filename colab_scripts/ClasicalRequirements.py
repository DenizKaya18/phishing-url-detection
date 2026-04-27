# Check GPU and install lib
!pip install scikit-learn imbalanced-learn pandas numpy scipy wordsegment tldextract -q

import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
