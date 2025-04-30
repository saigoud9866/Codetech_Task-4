import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report,
    roc_curve, 
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
data = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

print("Dataset shape:", data.shape)
data.head()

data['label'].value_counts()

plt.figure(figsize=(8, 5))
sns.countplot(x='label', data=data)
plt.title('Distribution of Spam vs Ham Messages')
plt.xlabel('Message Type')
plt.ylabel('Count')
plt.show()

data.isnull().sum()

data['label'] = data['label'].map({'ham': 0, 'spam': 1})

X = data['message']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF transformed training set shape:", X_train_tfidf.shape)
print("TF-IDF transformed test set shape:", X_test_tfidf.shape)

models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Support Vector Machine': SVC(kernel='linear', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    y_prob = model.predict_proba(X_test_tfidf)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'y_prob': y_prob
    }
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)

accuracies = {name: res['accuracy'] for name, res in results.items()}
df_acc = pd.DataFrame.from_dict(accuracies, orient='index', columns=['Accuracy'])

plt.figure(figsize=(10, 6))
sns.barplot(x=df_acc.index, y=df_acc['Accuracy'])
plt.title('Model Accuracy Comparison')
plt.ylim(0.8, 1.0)
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
for i, v in enumerate(df_acc['Accuracy']):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
plt.show()

plt.figure(figsize=(10, 8))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

plt.figure(figsize=(10, 8))
for name, res in results.items():
    precision, recall, _ = precision_recall_curve(y_test, res['y_prob'])
    avg_precision = average_precision_score(y_test, res['y_prob'])
    plt.plot(recall, precision, label=f'{name} (AP = {avg_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])

param_grid = {
    'tfidf__max_features': [1000, 5000, 10000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__C': [0.1, 1, 10],
    'clf__penalty': ['l2']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation accuracy: {:.4f}".format(grid_search.best_score_))

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("\nTest set accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

def classify_message(model, message):
    prediction = model.predict([message])[0]
    proba = model.predict_proba([message])[0]
    result = {
        'message': message,
        'prediction': 'spam' if prediction == 1 else 'ham',
        'spam_probability': proba[1],
        'ham_probability': proba[0]
    }
    return result

examples = [
    "Congratulations! You've won a $1000 Walmart gift card. Click here to claim your prize now!",
    "Hey, are we still meeting for lunch tomorrow at 12?",
    "Your account has been compromised. Please click this link to verify your identity.",
    "The meeting is scheduled for 3pm in conference room B."
]

for example in examples:
    print(classify_message(best_model, example))
    print("-" * 50)
