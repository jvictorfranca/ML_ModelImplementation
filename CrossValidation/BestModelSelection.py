import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from time import time

val_features = pd.read_csv('CrossValidation/sets_titanic/val_features.csv')
val_labels = pd.read_csv('CrossValidation/sets_titanic/val_labels.csv', header=None)

te_features = pd.read_csv('CrossValidation/sets_titanic/test_features.csv')
te_labels = pd.read_csv('CrossValidation/sets_titanic/test_labels.csv', header=None)

models = {}

for model in ['LR', 'SVM', 'MLP', 'RF', 'GB']:
    models[model] = joblib.load('CrossValidation/best_estimators/{}_model.pkl'.format(model))

def evaluate_model(name, model, features, labels):
    start = time()
    pred = model.predict(features)
    end = time()
    accuracy = round(accuracy_score(labels, pred), 3)
    precision = round(precision_score(labels, pred), 3)
    recall = round(recall_score(labels, pred), 3)
    print('{} -- Accuracy: {} / Precision: {} / Recall: {} / Latency: {}ms'.format(name,
                                                                                   accuracy,
                                                                                   precision,
                                                                                   recall,
                                                                                   round((end - start)*1000, 1)))

for name, mdl in models.items():
    evaluate_model(name, mdl, val_features, val_labels)

evaluate_model('Random Forest', models['RF'], te_features, te_labels)