# Mapping Conditional Distributions for Domain Adaptation Under Generalized Target Shift
This repository contains the official code of OSTAR in ["Mapping Conditional Distributions for Domain Adaptation Under Generalized Target Shift"][1] (ICLR 2022).

### Quickstart
* Install the requirements `pip install -r requirements.txt`
* Run training. ex: `python run.py -t 000000000001 -d digits -i 1 -g 0 -s 10`
* Results are logged in `./results/run_id` where run_id is the id of the run.

### Options
```
python run.py [-h] [-t MODEL] [-d DATASET] [-i RUNS] [-g GPUID] [-s SETTING]
```
- Choose the model (see Section 5 of the paper for more details):
  - `-t 100000000000`: `Source`
  - `-t 010000000000`: `DANN`
  - `-t 001000000000`: `WD_beta` for beta = 0
  - `-t 000111100000`: `WD_beta` for beta in {1, 2, 3, 4}
  - `-t 000000011000`: `MARSg` / `MARSc`
  - `-t 000000000100`: `IW-WD`
  - `-t 000000000010`: `WD_gt` with true class-rations
  - `-t 000000000001`: `OSTAR`
- Choose the dataset:
  - `-d digits`: Digits
  - `-d office`: Office31 and OfficeHome. Requires downloading pre-computed features at https://github.com/jindongwang/transferlearning/blob/master/data/dataset.md
  - `-d visda`: VisDA12. Requires downloading pre-computed features at http://csr.bu.edu/ftp/visda17/clf/ and preprocessing downloaded file with `prepare_data_visda12.py`
- Choose the number of runs (e.g. 1 for a single run)
- Choose the gpu id (e.g. 0)
- Choose the label shift setting defined in `compare_digits_setting.py`, `compare_office_setting.py`, `compare_visda_setting.py`

## Citation
```
@inproceedings{Kirchmeyer2022,
title={Mapping conditional distributions for domain adaptation under generalized target shift},
author={Matthieu Kirchmeyer and Alain Rakotomamonjy and Emmanuel de Bezenac and patrick gallinari},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=sPfB2PI87BZ}
}
```

[1]: https://openreview.net/forum?id=sPfB2PI87BZ