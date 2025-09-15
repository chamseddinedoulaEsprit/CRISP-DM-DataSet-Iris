# Étape 1 : Business Understanding

# Étape 2 : Data Understanding
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Charger le dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]

# Statistiques descriptives
print(df.describe())

# Vérifier les valeurs manquantes
print(df.isnull().sum())  # Devrait être zéro

# Distribution des classes
print(df['species'].value_counts())

# Visualisations
sns.pairplot(df, hue='species')
plt.show()

# Corrélations
corr = df.drop('species', axis=1).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

# Boxplots pour outliers
for feature in iris.feature_names:
    sns.boxplot(x='species', y=feature, data=df)
    plt.show()

# Étape 3 : Data Preparation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Encoder la target (optionnel, mais utile)
le = LabelEncoder()
df['species_encoded'] = le.fit_transform(df['species'])

# Features et target
X = df.drop(['species', 'species_encoded'], axis=1)
y = df['species_encoded']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Taille train :", X_train.shape)
print("Taille test :", X_test.shape)
# Étape 4 : Modeling
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'KNN': KNeighborsClassifier()
}

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    results[name] = scores.mean()
    print(f"{name}: {scores.mean():.4f}")

# Entraîner le meilleur (supposons Random Forest)
best_model = RandomForestClassifier()
best_model.fit(X_train_scaled, y_train)



# Étape 5 : Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Prédictions
y_pred = best_model.predict(X_test_scaled)

# Métriques
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Visualisation confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.show()

# Étape 6 : Deployment
import joblib

# Sauvegarder
joblib.dump(best_model, 'iris_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')

# Exemple de prédiction
def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    scaled = scaler.transform(input_data)
    pred = best_model.predict(scaled)
    return iris.target_names[pred[0]]

# Test
print(predict_iris(5.1, 3.5, 1.4, 0.2))  # Devrait être 'setosa'

