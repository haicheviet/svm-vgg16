from sklearn import svm
from read_data import create_dataset
import numpy as np
from sklearn.externals import joblib
from evaluate_model import evaluate_mode
from save_model import save_model, classification_report_csv
import os
from pathlib import Path

def main():
    path_output = 'exp/svmlinear/'
    clf = svm.LinearSVC(verbose = True)

    train_X, y_test, train_Y, y_true = create_dataset('db/train.txt', 'db/test.txt')

    if len(os.listdir(path_output)) != 0:
        print("model was already saved")
        print("Loaded y_pred. evaluating...")
        y_pred = np.load(path_output + 'y_pred.npy')
        report = evaluate_mode(y_true, y_pred)
        classification_report_csv(report, path_output)
        return

    clf.fit(train_X, train_Y)

    y_pred = clf.predict(y_test)

    evaluate_mode(y_true, y_pred)

    save_model(clf, classification_report(y_true, y_pred), path_output)

if __name__ == '__main__':
    main()
