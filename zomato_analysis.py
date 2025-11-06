import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --- Create sample dataset ---
data = {
    'Order_ID': range(1, 21),
    'Restaurant_Name': ['Burger Hub', 'Pizza Point', 'Café Delhi', 'Tandoor Treat', 'Burger Hub',
                        'Pizza Point', 'Rolls King', 'Subway', 'Tandoor Treat', 'Cake Walk',
                        'Domino’s', 'Taco Bell', 'Rolls King', 'Subway', 'Café Delhi',
                        'Domino’s', 'Pizza Point', 'Cake Walk', 'Burger Hub', 'Tandoor Treat'],
    'Distance_km': [2.5, 4.2, 3.1, 5.6, 6.3, 3.8, 2.9, 1.5, 4.8, 5.2, 3.0, 4.1, 2.3, 1.9, 3.7, 6.8, 4.4, 5.0, 2.2, 4.6],
    'Delivery_Time': [32, 48, 45, 60, 70, 42, 35, 28, 52, 58, 40, 46, 30, 25, 44, 68, 50, 55, 33, 53],
    'Weather': ['Clear', 'Rainy', 'Cloudy', 'Rainy', 'Rainy', 'Clear', 'Cloudy', 'Clear', 'Rainy', 'Rainy',
                'Clear', 'Cloudy', 'Clear', 'Clear', 'Cloudy', 'Rainy', 'Rainy', 'Cloudy', 'Clear', 'Rainy'],
    'Traffic': ['Low', 'High', 'Medium', 'High', 'High', 'Low', 'Medium', 'Low', 'High', 'Medium',
                'Low', 'High', 'Medium', 'Low', 'Medium', 'High', 'High', 'Medium', 'Low', 'High'],
    'Order_Type': ['Online', 'Online', 'Dine-In', 'Online', 'Online', 'Online', 'Dine-In', 'Online', 'Online', 'Online',
                   'Online', 'Online', 'Dine-In', 'Online', 'Dine-In', 'Online', 'Online', 'Dine-In', 'Online', 'Online'],
    'Restaurant_Rating': [4.5, 3.8, 4.2, 3.6, 3.5, 4.7, 4.0, 4.8, 3.9, 3.7, 4.3, 4.1, 4.5, 4.9, 4.0, 3.6, 3.8, 3.9, 4.6, 3.7]
}

df = pd.DataFrame(data)

# Create target column
df['Is_Late'] = df['Delivery_Time'] > 45

# --- Basic info ---
print(df.info())
print("\n--- Descriptive Statistics ---")
print(df.describe())
# Traffic vs Late Delivery
sns.countplot(x='Traffic', hue='Is_Late', data=df)
plt.title("Impact of Traffic on Late Deliveries")
plt.show()

# Distance vs Delivery Time
sns.scatterplot(x='Distance_km', y='Delivery_Time', hue='Is_Late', data=df, s=100)
plt.title("Distance vs Delivery Time")
plt.show()


# Restaurant Rating vs Delivery Time
sns.boxplot(x='Is_Late', y='Restaurant_Rating', data=df)
plt.title("Restaurant Ratings for Late vs On-Time Deliveries")
plt.show()

#heatmap
corr = df.select_dtypes(include=['float64', 'int64', 'bool']).corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=1)
plt.title("Correlation Heatmap")
plt.show()

#data preprocessing
# One-hot encode categorical columns
df_encoded = pd.get_dummies(df, columns=['Weather', 'Traffic', 'Order_Type'], drop_first=True)

# Separate features (X) and target (y)
X = df_encoded.drop(['Order_ID', 'Restaurant_Name', 'Is_Late'], axis=1)
y = df_encoded['Is_Late']

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize and train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
print("\n✅ Model Evaluation Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance.sort_values(ascending=False).plot(kind='bar', figsize=(10,5))
plt.title("Feature Importance — Factors Affecting Late Delivery")
plt.ylabel("Importance Score")
plt.show()
