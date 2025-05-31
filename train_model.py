# Step 1: Import required libraries
import pandas as pd                             # Handles loading and manipulating data (like Excel or SQL tables)
from sklearn.model_selection import train_test_split  # Splits your dataset into training and test sets
from sklearn.ensemble import RandomForestClassifier   # A type of machine learning model
from sklearn.metrics import accuracy_score            # Measures how good your model is
import joblib                                          # Used to save and load trained models

# Step 2: Load the dataset from the CSV file
data = pd.read_csv("data.csv")  # This reads your data.csv into a table format

# Step 3: Separate features (input columns) and target (output)
X = data.drop(columns=["needs_doctor"])  # X = input data; drop the column we're trying to predict
y = data["needs_doctor"]                 # y = target column (like the expected result in Java tests)

# Step 4: Split the dataset into training and testing parts
# 80% of data will be used to train the model, 20% to test it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Create a machine learning model (Random Forest = collection of smart decision trees)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 6: Train the model using the training dataset
model.fit(X_train, y_train)  # Model “learns” patterns from the training data

# Step 7: Predict using the test dataset (these are new patients the model hasn’t seen before)
predictions = model.predict(X_test)

# Step 8: Measure how accurate the model was on the test data
accuracy = accuracy_score(y_test, predictions)
print(f"✅ Model Accuracy: {accuracy:.2f}")  # Example output: "Model Accuracy: 0.85" (85% correct)

# Step 9: Save the trained model to a file so we can use it later
joblib.dump(model, "doctor_model.pkl")
print("📦 Model saved as doctor_model.pkl")

