import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# Paths
models_dir = r"C:\Guvi\NutriClass Food Classification Using Nutritional Data\models"
data_path = r"C:\Guvi\NutriClass Food Classification Using Nutritional Data\data\processed\food_data_final.csv"

# Load model and label encoder
best_model = joblib.load(os.path.join(models_dir, "model.pkl"))
le = joblib.load(os.path.join(models_dir, "label_encoder.pkl"))

# Load data
df = pd.read_csv(data_path)
feature_cols = ['Calories', 'Protein', 'Fat', 'Carbs', 'Sugar', 'Fiber', 
                'Sodium', 'Cholesterol', 'Glycemic_Index', 'Water_Content']
X = df[feature_cols]
y = df['Meal_Type_encoded']

# Predictions
y_pred = best_model.predict(X)

# Accuracy and classification report
print("Accuracy:", accuracy_score(y, y_pred))
print(classification_report(y, y_pred, target_names=le.classes_))

# Confusion matrix
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Feature importance (if available)
if hasattr(best_model, "feature_importances_"):
    import pandas as pd
    importance = best_model.feature_importances_
    feat_imp = pd.Series(importance, index=feature_cols).sort_values(ascending=False)
    plt.figure(figsize=(10,6))
    feat_imp.plot(kind='bar')
    plt.title("Feature Importance")
    plt.show()
