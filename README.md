# Network Inference Ensemble

### Overview
Stacking ensemble for gene regulatory network inference. Datasets from four species/scenarios are presented here: DREAM chanllenges, Arabidopsis, B.subtilis and human embryonic stem cells(hESC).
<br> 
Training/testing data for Arabidopsis and B.subtilis networks can be found in this Google Drive link and is needed to replicate corresponding experiments: 
<br> 
https://drive.google.com/drive/folders/18sG4BgfYxMyAYyB1T4UY1QZyGWi1DAdo?usp=sharing

### Dependencies
```
numpy >= 1.20
pandas >= 1.4
scikit-learn >= 1.0
scipy >= 1.6
```

### Demo
Prepare first level network inference results on the GRN of concern in the form of a training set, a validation set, and a test set(optional).
Training and validation data files should be formated as following:
```
edge_name, edge_exist, ALGO#1, ALGO#2, ALGO#3
G1_G2, 1, 0.3, 0.4, 0.5
G1_G3, 0, 0.1, 0.2, 0.3
G1_G4, 0, 0.6, 0.9, 0.7
... ...

```
Test data file can be optional input with optional ``edge_exist`` column data.
Once dependency libraries are installed and the input data files are prepared, excute ``python demo/ensemble.py -r train.csv -v val.csv -t test.csv`` to do an ensemble inference experiment using Naive Bayes ensemble model, with the optional ensemble inference confidence score output to ``test_output.csv`` file if a test data input is provided.

### Adding new base inference method
For the three species we demonstrated here, new base inference method can be aded. First run the new method of your choice to get the edge ranking score on all possible edges, then add edge scores as a new column to all the training/testing split tables accordingly. Finally, edited the `config.py` files to include your new method's name and column name.
