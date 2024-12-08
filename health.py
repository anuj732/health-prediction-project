# Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load Dataset
file_path = "health_data.xlsx"  # Excel file path
df = pd.read_excel(file_path)

# Step 2: Data Preprocessing
print("Dataset Head:\n", df.head())
df['condition'] = df['condition'].map({'Healthy': 0, 'COVID': 1})

# Step 3: Features and Target
X = df.drop(columns=['condition'])  # Symptoms (features)
y = df['condition']  # Target variable (0: Healthy, 1: COVID)

# Step 4: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 8: Confusion Matrix Graph
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Healthy", "COVID"], yticklabels=["Healthy", "COVID"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 9: User Input for Health Prediction
print("\n--- Health Prediction System ---")
fever = int(input("Do you have fever? (1 for Yes, 0 for No): "))
cough = int(input("Do you have cough? (1 for Yes, 0 for No): "))
fatigue = int(input("Do you feel fatigue? (1 for Yes, 0 for No): "))
shortness_of_breath = int(input("Do you have shortness of breath? (1 for Yes, 0 for No): "))
chest_pain = int(input("Do you have chest pain? (1 for Yes, 0 for No): "))

# Predict Health Condition
user_input = np.array([[fever, cough, fatigue, shortness_of_breath, chest_pain]])
prediction = model.predict(user_input)

if prediction[0] == 1:
    print("Prediction: You might have COVID. Please consult a doctor.")
else:
    print("Prediction: You seem to be Healthy.")
