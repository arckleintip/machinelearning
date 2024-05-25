import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import pickle  # For saving the model

# Assuming you have your data loaded in a DataFrame called 'df'
# X contains your features, y contains your target variable (swi_level_change)

# 1. Data Preparation (If not already done):
# - One-hot encode categorical variables (marital_status, dwelling_type, etc.)
# - Handle missing values
# - Perform any other necessary preprocessing

# Separate features (X) and target variable (y)
X = df.drop('swi_level_change', axis=1)
y = df['swi_level_change']

# 2. Train-Test Split:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Hyperparameter Tuning (Optional but recommended):
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier()
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")

# 4. Model Training:
model = RandomForestClassifier(**best_params, random_state=42)
model.fit(X_train, y_train)

# 5. Model Evaluation:
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

# 6. Feature Importance Analysis:
importances = model.feature_importances_
feature_importances = pd.DataFrame(
    {'feature': X.columns, 'importance': importances}
).sort_values('importance', ascending=False)
print("Feature Importances:\n", feature_importances)

# 7. Save the Model:
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(model, file)
