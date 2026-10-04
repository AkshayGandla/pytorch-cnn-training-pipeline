# PyTorch Image-Classification Pipeline: Cleaning, Dataset, Augmentation, CNN Training Loop

A from-the-basics deep-learning pipeline in **PyTorch** on a small labelled image dataset supplied by the course.

## What it does
1. **Data cleaning**: loads raw train/test label CSVs, trims whitespace, separates rows with unknown ("?") labels, encodes class labels, and writes cleaned labelled / unlabelled CSVs (117 train and 30 test labelled rows after cleaning).
2. **Custom `Dataset` + `DataLoader`s** reading images by filename, with a stratified train/validation split (93 train / 24 val), seeded for reproducibility.
3. **Augmentation**: random flip, rotation and colour jitter on 64x64 inputs, with a visual check of augmented views.
4. **Model**: a small CNN built from basic layers (two conv + pool blocks and an MLP head), no pre-trained weights, as the task required.
5. **Training loop**: forward pass, cross-entropy loss, backprop with Adam, and per-epoch train / val metrics plus a test evaluation.

## Result (honest summary)
One training epoch was run (the assessment checks the training loop's correctness, not accuracy):

| Split | Loss | Accuracy |
|---|---|---|
| Train | 1.455 | 0.409 |
| Validation | 1.241 | 0.458 |
| Test | 1.174 | 0.533 |

With only about 100 training images and one epoch, these figures only demonstrate that the pipeline runs end to end; they are not a performance claim.

## Skills demonstrated
PyTorch (Dataset/DataLoader, nn.Module, training loops), data cleaning with pandas, augmentation, stratified splitting, reproducibility (seeding).

## Run
`pip install -r requirements.txt`; place the course image folders and label CSVs where the notebook's path variables point; run top to bottom (developed on Colab).

## Context
Built as an individual assessment for *Deep Learning Fundamentals* in the MSc in Artificial Intelligence & Machine Learning at the University of Adelaide (Trimester 3, 2025). Task text embedded in the notebook is the course template; the implementation and write-up are my own. Course datasets and material zips are not redistributed.

## Licence
MIT. See [LICENSE](LICENSE).
