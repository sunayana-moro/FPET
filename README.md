# FPET Prototype

This repo is a small research prototype for testing `FPET` ideas on CIFAR-10 and a synthetic frequency dataset.

It includes four training paths:
- `baseline`: a plain CNN on the raw image
- `cnn_ll`: a plain CNN on the DWT `LL` subband
- `deq_full`: a DEQ-style classifier on the raw image
- `fpet`: an LL-first DEQ pipeline with coreset selection and a refinement stage

## Main Files

- [experiment.py](/Users/sunayana/Documents/fpet/experiment.py:1): runs the experiments and writes a text report
- [fpet/data.py](/Users/sunayana/Documents/fpet/fpet/data.py:1): loads CIFAR-10 and builds DWT features
- [fpet/models.py](/Users/sunayana/Documents/fpet/fpet/models.py:1): defines the CNN, DEQ-style models, and FPET refiner
- [environment.yml](/Users/sunayana/Documents/fpet/environment.yml:1): Conda environment spec
- [setup.sh](/Users/sunayana/Documents/fpet/setup.sh:1): creates or updates the Conda env

## Run

```bash
./setup.sh
conda activate fpet
python experiment.py --dataset cifar10 --report-path artifacts/latest_report.txt
```

Keep manually downloaded CIFAR-10 data in `data/cifar-10-batches-py`.
