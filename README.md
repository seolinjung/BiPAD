# Problematic Ad Detection

Inspired by the task behind the paper "Sheep’s clothing, wolfish impact: Automated detection and evaluation of problematic ‘allowed’ advertisements" (Roongta et al., 2025), this project aims to solve the problem of binary classification of problematic online advertisements.

## Folder Structure

```
├───bipad
│       config.json
│       model.safetensors
│       tokenizer.json
│       tokenizer_config.json
│
├───data
│   │   cleaned_data.csv
│   │   merged_ground_truth.csv
│   │   ocr_adblock.json
│   │   ocr_control.json
│   │
│   ├───adblock
│   │   └───adshots
│   └───control
│       └───adshots
└───prior
        llm_annotation.csv
        llm_annotation_dict.json
        llm_annotation_explanations.json
```

`bipad` is required if you want to run `python main.py --bipad evaluate` without training first. `data` is required for all tasks. `prior` is required if you want to run inference on the LLM annotation results from Roongta et al. 

## Environment

This project was run in a conda virtual environment with Python 3.12. To enable, download Conda and run

```
conda create --name virtue python=3.12
conda activate virtue
```

NVIDIA GeForce RTX 5070 Ti was used to run all scripts, but it is not required. If you run into CUDA/Torch issues, swap out the versions in `requirements.txt`.

## Models

This repository contains 3 classifiers: a decision tree, an SVM, and BiPAD, the bilingual model fine tuned on XLM-RoBERTa. 

To run the decision tree,
```
python main.py --dt
```

To run the SVM,
```
python main.py --svm
```

To run BiPAD, you can specify arguments to train, evaluate, or do both.
```
python main.py --bipad train evaluation
```

Please note that in order to run BiPAD, you must have the model folder titled `bipad` under the root of this repository.

## Dataset

The dataset used in this project is a modified version of the original dataset publicly released by Roongta et al. The data structure is

    Roongta, R., Jose, J., Habib, H., & Greenstadt, R. (2025). Sheep’s clothing, wolfish intent: Automated detection and evaluation of problematic “allowed” advertisements. Proceedings on Privacy Enhancing Technologies, 2025(4), 507–529. https://doi.org/10.56553/popets-2025-0142

    Roongta, R. (2025, June 30). Sheep’s clothing, wolfish impact. GitHub. https://github.com/Racro/AcceptableAds_PETS/ 
 