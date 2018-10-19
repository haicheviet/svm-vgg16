from sklearn.externals import joblib
import pandas as pd

def save_model(clf, report, path):
    joblib.dump(clf, path + 'svm.joblib')
    classification_report_csv(report, path)

def classification_report_csv(report, path):
    report_data = []
    lines = report.split('\n')
    for count, line in enumerate (lines[2:-3]):
        row = {}
        row_data = line.split('      ')
        row['class'] = float(row_data[1])
        row['precision'] = float(row_data[2])
        row['recall'] = float(row_data[3])
        row['f1_score'] = float(row_data[4]) 
        row['support'] = float(row_data[5])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(path + 'classification_report.csv', index = False)
    print("Done saving")