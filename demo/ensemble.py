import getopt
import sys
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB




#!/usr/bin/python


def main(argv):
    train_data_file = ''
    val_data_file = ''
    test_data_file = ''
    try:
        opts, args = getopt.getopt(
            argv, "hr:v:t:", ["train=", "val=", "test="])
    except getopt.GetoptError:
        print('test.py -r <train_data_file> -v <val_data_file> ')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -r <train_data_file> -v <val_data_file> -t <test_data_file>')
            sys.exit()
        elif opt in ("-r", "--train_data_file"):
            train_data_file = arg
        elif opt in ("-v", "--val_data_file"):
            val_data_file = arg
        elif opt in ("-t", "--c"):
            test_data_file = arg
    
    min_max_scaler = preprocessing.MinMaxScaler()

    train_df = pd.read_csv(train_data_file)
    val_df = pd.read_csv(val_data_file)
    algo_names = train_df.drop(columns=['edge_exist', 'edge_name']).columns

    clf = GaussianNB(var_smoothing=2.848035868435805e-09)

    # some global variables

    algos_selected = []
    for algo in algo_names:
        if kurtosis(train_df[algo]) >= 0:
            algos_selected.append(algo)

    X_train = train_df[algos_selected].values
    X_train = min_max_scaler.fit_transform(X_train)
    y_train = train_df['edge_exist']

    X_val = val_df[algos_selected].values
    X_val = min_max_scaler.fit_transform(X_val)
    y_val = val_df['edge_exist']

    for algo in algo_names:
        pr = precision_recall_curve(y_val, val_df[algo].values)
        print('AUPRC of {}\'s prediction on validation set: {:.4f}'.format(
            algo, auc(pr[1], pr[0])))

    print('====================')
    print('Input algorithms whose prediction score has a positive kurtosis: {}'.format(
        ', '.join(algos_selected)))
    print('====================')

    clf.fit(X_train, y_train)
    ensemble_pr = precision_recall_curve(y_val, clf.predict_proba(X_val)[:, 1])
    print('AUPRC of Naive Gaussian ensemble model\'s prediction on validation set: {:.4f}'.format(
        auc(ensemble_pr[1], ensemble_pr[0])))

    if (len(test_data_file) > 0):
        test_df = pd.read_csv(test_data_file)
        X_test = test_df[algos_selected].values
        X_test = min_max_scaler.fit_transform(X_test)
        a = clf.predict_proba(X_test)[:, 1]
        out_df = pd.DataFrame(columns=['edge_name', 'ensemble_score'], data=np.array(
            [test_df.edge_name, clf.predict_proba(X_test)[:, 1]]).T)
        out_df.to_csv('test_output.csv', index=False)
        print('====================')
        print('Ensemble prediction on test set was saved to \"test_output.csv\"')


if __name__ == "__main__":
    main(sys.argv[1:])
