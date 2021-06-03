# An automatic diagnosis aid system to first classify lung cytologic images and then annotate malignant cells

---
## Setup files
-  Please download our pretrained model weights from:
>   - ResNet101: https://drive.google.com/file/d/1bYp_MV0IGImYCdXxUEWt8yYJuwYC69gt/view?usp=sharing
>   - HRFPN: https://drive.google.com/file/d/14EC7iw7oFdJ29JPZ3FPYcXAyBfqz2NB_/view?usp=sharing
- Please allocate the pretrained model weights of ResNet101 into file "models"
- Please allocate the pretrained model weights of HRFPN into file "models/HRFPN_try3/"
- Put your input images into file "test_case_whole"
---
## Setup environmnet
```
pip install -r requirements.txt
```
---
## Execute the diagnosis aid system by running code:
```
python diagnose.py
```
