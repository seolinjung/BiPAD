# Problematic Ad Detection

Inspired by the task behind the paper "Sheep’s clothing, wolfish impact: Automated detection and evaluation of problematic ‘allowed’ advertisements" (Roongta et al., 2025), this project aims to solve the problem of binary classification of problematic online advertisements.

## Models

This repository contains 3 classifier: a decision tree, an SVM, and BiPAD. 

To run the decision tree, type
```
python main.py --dt
```

To run the SVM, type
```
python main.py --svm
```

To run BiPAD, you can specify arguments to train, evaluate, or do both.
```
python main.py --bipad train evaluation
```

Please note that in order to run BiPAD, you must have the model folder titled "bipad" under the root of this repository.

## Dataset

The dataset used in this project is a modified version of the original dataset publicly released by Roongta et al.

    Roongta, R., Jose, J., Habib, H., & Greenstadt, R. (2025). Sheep’s clothing, wolfish intent: Automated detection and evaluation of problematic “allowed” advertisements. Proceedings on Privacy Enhancing Technologies, 2025(4), 507–529. https://doi.org/10.56553/popets-2025-0142

    Roongta, R. (2025, June 30). Sheep’s clothing, wolfish impact. GitHub. https://github.com/Racro/AcceptableAds_PETS/ 