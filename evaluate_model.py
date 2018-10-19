from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, classification_report

def evaluate_mode(y_true, y_pred):
    print("\n" + "-"*20)
    print ('Accuracy:', accuracy_score(y_true, y_pred))
    print ('F1 score:', f1_score(y_true, y_pred, average='weighted'))
    print ('Recall:', recall_score(y_true, y_pred, average='weighted'))
    print ('Precision:', precision_score(y_true, y_pred, average='weighted'))
    print ('\n clasification report:\n', classification_report(y_true,y_pred))
    return classification_report(y_true,y_pred)