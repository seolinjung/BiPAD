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

This project was run in a Conda virtual environment with Python 3.12. To enable, download Conda if you don't have it already and run

```
conda create --name virtue (or a name you desire) python=3.12
conda activate virtue
```

NVIDIA GeForce RTX 5070 Ti was used to run all scripts, but it is not required. If you run into CUDA/Torch issues, swap out the versions in `requirements.txt`.

It would be beneficial to be authenticated for Huggingface to avoid rate limits, as the project loads pretrained weights for XLM-RoBERTa. Find instructions [here](https://huggingface.co/docs/huggingface_hub/en/quick-start).

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

You can also run inference on Roongta et al.'s LLM annotation results, given that you have the required files.
```
python main.py --sota
```

All scripts rely on `test_indices.yaml` for loading test indices. The complete test set is fixed to 100 samples, with 50 German and English samples each selected for the purpose of experiments. The file expects keys `de` and `en`. Upon a missing `test_indices.yaml`, one will be created with a new randomly selected batch of samples. In order to re-evaluate any model with a different test set, either delete `test_indices.yaml` entirely or just its contents. Inference will be done 3 times, first on the complete test set, the German subset, and lastly the English subset. Results will be displayed on the terminal.

## Dataset

The dataset used in this project is a modified version of the original dataset publicly released by Roongta et al. The data structure is

    Roongta, R., Jose, J., Habib, H., & Greenstadt, R. (2025). Sheep’s clothing, wolfish intent: Automated detection and evaluation of problematic “allowed” advertisements. Proceedings on Privacy Enhancing Technologies, 2025(4), 507–529. https://doi.org/10.56553/popets-2025-0142

    Roongta, R. (2025, June 30). Sheep’s clothing, wolfish impact. GitHub. https://github.com/Racro/AcceptableAds_PETS/ 
 