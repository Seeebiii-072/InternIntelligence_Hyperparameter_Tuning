# # train_model.py
#
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# import joblib
#
# # Load data
# df = pd.read_csv(r'C:\Users\Haseeb Ishtiaq\PycharmProjects\Hyperparameter\data\datasetwithprac.csv')
#
# # Drop rows with any missing values
# df.dropna(inplace=True)
#
# # Features and target
# X = df[['Midterm_Percentage', 'Assignment_Percentage', 'Practical_Percentage']]
# y = df['Actual_Result']
#
# # Split into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Train model
# model = RandomForestClassifier()
# model.fit(X_train, y_train)
#
# # Evaluate
# y_pred = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
#
# # Save model
# joblib.dump(model, "model.pkl")
# print("Model saved as model.pkl")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv(r'C:\Users\Haseeb Ishtiaq\PycharmProjects\Hyperparameter\data\datasetwithprac.csv')

# Clean target column
df['Actual_Result'] = pd.to_numeric(df['Actual_Result'], errors='coerce')
print("Unique values in Actual_Result:", df['Actual_Result'].unique())

# Drop rows with missing target
df.dropna(subset=['Actual_Result'], inplace=True)

# Define features and target
X = df.drop(columns=['Actual_Result'])
y = df['Actual_Result']

# Drop rows with NaNs in features
X = X.dropna()
y = y.loc[X.index]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model and grid search
model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("✅ Best Parameters:", grid_search.best_params_)

# Evaluate
y_pred = best_model.predict(X_test)
print("\n🔍 Classification Report:")
print(classification_report(y_test, y_pred))

print("📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Save model
model_path = r'C:\Users\Haseeb Ishtiaq\PycharmProjects\Hyperparameter\model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)
print(f"✅ Model saved at: {model_path}")

# Cross-validation
cv_scores = cross_val_score(best_model, X_scaled, y, cv=5, scoring='accuracy')
print(f"\n✅ Cross-Validation Accuracy Scores: {cv_scores}")
print(f"✅ Mean CV Accuracy: {cv_scores.mean():.4f}")

# --- 📊 Visualization ---

# Confusion matrix heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Feature importance
importances = best_model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=feature_names[indices])
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.show()
