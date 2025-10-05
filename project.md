# NutriClass: Food Classification Using Nutritional Data

## Project Overview
NutriClass is a machine learning project aimed at classifying food items into different meal types based on their nutritional content. Using tabular data such as calories, protein, fat, carbohydrates, sugar, and other nutritional attributes, the system can automatically label food items and assist in dietary recommendations, health monitoring, and meal planning.

---

## **Skills & Techniques Learned**
- Data Preprocessing (handling missing values, duplicates, outliers, normalization)
- Feature Engineering (label encoding, optional dimensionality reduction)
- Multi-class Classification
- Model Evaluation Metrics (accuracy, precision, recall, F1-score, confusion matrix)
- Visualization with Matplotlib & Seaborn
- Python Programming & Scikit-learn

---

## **Domain**
- Food and Nutrition
- Machine Learning

---

## **Problem Statement**
In the era of dietary awareness, classifying foods based on their nutritional content is invaluable. NutriClass provides a robust classification system to label food items automatically, aiding nutritionists, smart dietary apps, and educational platforms.

---

## **Business Use Cases**
- **Smart Dietary Applications:** Recommend foods based on nutritional balance.
- **Health Monitoring Tools:** Assist nutritionists in diet planning.
- **Food Logging Systems:** Automatically classify foods for tracking apps.
- **Educational Platforms:** Help learners understand nutrition and food categories.
- **Grocery/Meal Planning Apps:** Suggest replacements within the same category.

---

## **Project Approach**

### **Step 1: Data Understanding & Exploration**
- Load dataset and examine class distribution.
- Visualize sample entries to understand inter-class variation.
- Check dataset size, imbalance, and noise levels.

**Techniques Used:** Pandas, Matplotlib, Seaborn

---

### **Step 2: Data Preprocessing**
- Handle missing values (drop or impute)
- Detect and remove/cap outliers
- Remove duplicates
- Normalize or standardize numerical features

**Script:** `src/data_preprocessing.py`

---

### **Step 3: Feature Engineering**
- Encode target labels using Label Encoding
- Optional: PCA or feature selection for dimensionality reduction

**Script:** `src/feature_engineering.py`

---

### **Step 4: Model Selection & Training**
- Train and compare multiple classifiers:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbors
  - Support Vector Machine
  - XGBoost
  - Gradient Boosting Classifier
- Split dataset into training and testing
- Save best-performing model automatically

**Script:** `src/train_and_compare_models.py`

---

### **Step 5: Model Evaluation**
- Evaluate best model using:
  - Accuracy
  - Precision, Recall, F1-score
  - Confusion Matrix
  - Feature Importance (if applicable)

**Script:** `src/evaluate_model.py`

---


