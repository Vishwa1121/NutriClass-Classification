import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb

# -------------------------------
# Paths
# -------------------------------
data_path = r"C:\Guvi\NutriClass Food Classification Using Nutritional Data\data\processed\food_data_final.csv"
models_dir = r"C:\Guvi\NutriClass Food Classification Using Nutritional Data\models"
plots_dir = os.path.join(models_dir, "plots")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv(data_path)

feature_cols = ['Calories', 'Protein', 'Fat', 'Carbs', 'Sugar', 'Fiber', 
                'Sodium', 'Cholesterol', 'Glycemic_Index', 'Water_Content']
X = df[feature_cols]
y = df['Meal_Type_encoded']

# -------------------------------
# Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# Define models with pipelines
# -------------------------------
models = {
    "Logistic Regression": Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('classifier', LogisticRegression(max_iter=500, random_state=42))
    ]),
    "Decision Tree": Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ]),
    "Random Forest": Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
    "K-Nearest Neighbors": Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('classifier', KNeighborsClassifier(n_neighbors=5))
    ]),
    "Support Vector Machine": Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('classifier', SVC(probability=True, random_state=42))
    ]),
    "XGBoost": Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
    ]),
    "Gradient Boosting": Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('classifier', GradientBoostingClassifier(random_state=42))
    ])
}

# -------------------------------
# Train, Evaluate, Save Best Model
# -------------------------------
results = {}
best_acc = 0
best_model_name = None
best_model = None

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    
    results[name] = acc
    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_model_name = name

# Save best model
best_model_path = os.path.join(models_dir, "model.pkl")
joblib.dump(best_model, best_model_path)
print(f"\n✅ Best model saved ({best_model_name}) with accuracy: {best_acc:.4f}")

# -------------------------------
# Plot and save accuracy comparison
# -------------------------------
plt.figure(figsize=(10,6))
plt.bar(results.keys(), results.values(), color='skyblue')
plt.xticks(rotation=45)
plt.ylabel("Accuracy")
plt.title("Classifier Accuracy Comparison")

accuracy_plot_path = os.path.join(plots_dir, "classifier_accuracy_comparison.png")
plt.savefig(accuracy_plot_path, bbox_inches='tight')
plt.close()
print(f"✅ Accuracy comparison plot saved at: {accuracy_plot_path}")
