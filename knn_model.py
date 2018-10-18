from sklearn.neighbors import KNeighborsClassifier
from read_data import create_dataset
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, classification_report
from save_model import save_model
import os
from pathlib import Path

def main():
    path_output = 'exp/knn/'
    if len(os.listdir(path_output)) != 0:
        print("model's already saved")
        return
    
    clf = KNeighborsClassifier(n_neighbors = 5)

    train_X, y_test, train_Y, y_true = create_dataset('db/train.txt', 'db/test.txt')

    clf.fit(train_X, train_Y)

    y_pred = clf.predict(y_test)

    print("\n" + "-"*20)
    print ('Accuracy:', accuracy_score(y_true, y_pred))
    print ('F1 score:', f1_score(y_true, y_pred, average='weighted'))
    print ('Recall:', recall_score(y_true, y_pred, average='weighted'))
    print ('Precision:', precision_score(y_true, y_pred, average='weighted'))
    print ('\n clasification report:\n', classification_report(y_true,y_pred))

    save_model(clf, classification_report(y_true, y_pred), path_output)

if __name__ == '__main__':
    main()
