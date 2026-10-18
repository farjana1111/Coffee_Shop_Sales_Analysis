import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ✅ Define base directory relative to scripts folder
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
file_path = os.path.join(BASE_DIR, "coffee_shop_engineered.csv")

# 1. Load engineered dataset
df = pd.read_csv(file_path)

# 2. Encode product category
le_category = LabelEncoder()
df['product_category_encoded'] = le_category.fit_transform(df['product_category'])

# 3. Define features and target
X = df[['unit_price', 'transaction_qty', 'hour', 'month']]
y = df['product_category_encoded']

# 4. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train Random Forest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 6. Predictions
y_pred = clf.predict(X_test)

# 7. Evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le_category.classes_))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 8. Feature importance
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': clf.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(feature_importances)
