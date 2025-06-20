# Simulating the Emergence of Dependency Length Minimization Preferences

![GitHub](https://img.shields.io/github/license/facebookresearch/EGG)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

## Introduction

Portions of this ongoing work have been published as [Endowing Neural Language Learners with Human-like Biases: A Case Study on Dependency Length Minimization](https://aclanthology.org/2024.lrec-main.516/) at LREC-COLING 2024 and presented as [Neural-agent Language Learning and Communication: Emergence of Dependency Length Minimization](https://escholarship.org/uc/item/7fr1b90k) at CogSci 2024.

The implementation is based on the [NeLLCom](https://github.com/Yuchen-Lian/NeLLCom) framework and [EGG](https://github.com/facebookresearch/EGG) toolkit.

Parts of the generated files are stored in this [folder](https://drive.google.com/drive/folders/1_eUHRisx2DP6s7hf6DXCUzxLnDvweNyB?usp=drive_link).

## Running experiments

1. Installing [EGG](https://github.com/facebookresearch/EGG.git.) toolkit;
2. Moving to the EGG game design folder:
   ```
   cd EGG/egg/zoo
   ```
3. Cloning the current repo into the EGG game design folder:
   ```
   git clone https://github.com/yuqing0304/DLM_exp.git
   cd DLM_exp
   ```
4. Then, we can run a game, for example, communicating with an impatient listener using the verb-final language:
    ```bash
   cd DLM_exp_impa/DLM_halffinalSMrnnImpa
   sbatch run.sh
    ```


## Citation
If you find this study useful in your research, please cite this paper:
```
@inproceedings{zhang-etal-2024-endowing,
    title = "Endowing Neural Language Learners with Human-like Biases: A Case Study on Dependency Length Minimization",
    author = "Zhang, Yuqing  and
      Verhoef, Tessa  and
      van Noord, Gertjan  and
      Bisazza, Arianna",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.516/",
    pages = "5819--5832"
}
```
