import joblib      # To load the saved model
import numpy as np # To handle numeric arrays

# Step 1: Load the trained model
model = joblib.load("doctor_model.pkl")

# Step 2: Create a sample input for a new patient
# Format: [age, sex, blurred_vision, floaters, dryness, diabetes, screen_time_hours]
sample_input = np.array([[48, 1, 1, 0, 1, 1, 7]])

# Step 3: Use the model to predict whether a doctor visit is needed
prediction = model.predict(sample_input)

# Step 4: Interpret the result
if prediction[0] == 1:
    print("🚨 Doctor appointment is recommended.")
else:
    print("✅ No appointment needed.")

