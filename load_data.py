
'''

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the train data and test data
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Check the original columns before encoding
print("Original Columns in Train Data:", train_data.columns)

# Apply One-Hot Encoding
train_data_encoded = pd.get_dummies(train_data, drop_first=True)
test_data_encoded = pd.get_dummies(test_data, drop_first=True)

# Check the columns after One-Hot Encoding
print("Encoded Columns in Train Data:", train_data_encoded.columns)

# In One-Hot Encoding, 'class' will be replaced by something like 'class_normal', 'class_anomaly'
# Identify the new column name that represents the target class
class_column_name = 'class_normal'  # Or 'class_anomaly', depending on how it was encoded

# Split the train data into features (X_train) and labels (y_train)
X_train = train_data_encoded.drop(columns=[class_column_name])  # Features (all columns except 'class_normal')
y_train = train_data_encoded[class_column_name]  # Labels (the 'class_normal' column)

# Train the model with the training data
model = RandomForestClassifier()
model.fit(X_train, y_train)  # Training the model

# Predict on the test data (which does not contain 'class' column)
X_test = test_data_encoded  # Test data features (no 'class' column)
y_pred = model.predict(X_test)  # Predict the labels for test data

# Print the predictions
print("Predictions on test data:", y_pred)



# Optionally: Save the predictions to a CSV file
predictions = pd.DataFrame({'Prediction': y_pred})
predictions.to_csv('predictions.csv', index=False)


'''
#----------------------------------Adding 'predictions' to test_data-------------------------------------------#

'''


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the train data and test data
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Check the original columns before encoding
print("Original Columns in Train Data:", train_data.columns)

# Apply One-Hot Encoding
train_data_encoded = pd.get_dummies(train_data, drop_first=True)
test_data_encoded = pd.get_dummies(test_data, drop_first=True)

# Check the columns after One-Hot Encoding
print("Encoded Columns in Train Data:", train_data_encoded.columns)
print("Encoded Columns in Test Data:", test_data_encoded.columns)

# In One-Hot Encoding, 'class' will be replaced by something like 'class_normal', 'class_anomaly'
# Identify the new column name that represents the target class
class_column_name = 'class_normal'  # Or 'class_anomaly', depending on how it was encoded

# Split the train data into features (X_train) and labels (y_train)
X_train = train_data_encoded.drop(columns=[class_column_name])  # Features (all columns except 'class_normal')
y_train = train_data_encoded[class_column_name]  # Labels (the 'class_normal' column)

# Train the model with the training data
model = RandomForestClassifier()
model.fit(X_train, y_train)  # Training the model

# Predict on the test data (which does not contain 'class' column)
X_test = test_data_encoded  # Test data features (no 'class' column)
y_pred = model.predict(X_test)  # Predict the labels for test data

# Print the predictions
print("Predictions on test data:", y_pred)

# Update the 'predictions' column in the test_data DataFrame
test_data['predictions'] = y_pred

# Save the updated test_data DataFrame back to the same CSV file
test_data.to_csv('test_data.csv', index=False)

# Optionally: Save the predictions to a separate CSV file (if needed)
predictions = pd.DataFrame({'Prediction': y_pred})
predictions.to_csv('predictions.csv', index=False)

print("Test data file updated with predictions successfully!")


'''
#------------------------------------addind lid to predictions.csv-----------------------------------------#



'''

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the train data and test data
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')


# Apply One-Hot Encoding
train_data_encoded = pd.get_dummies(train_data, drop_first=True)
test_data_encoded = pd.get_dummies(test_data, drop_first=True)

# Check the columns after One-Hot Encoding
print("Encoded Columns in Train Data:", train_data_encoded.columns)
print("Encoded Columns in Test Data:", test_data_encoded.columns)

# In One-Hot Encoding, 'class' will be replaced by something like 'class_normal', 'class_anomaly'
# Identify the new column name that represents the target class
class_column_name = 'class_normal'  # Or 'class_anomaly', depending on how it was encoded

# Split the train data into features (X_train) and labels (y_train)
X_train = train_data_encoded.drop(columns=[class_column_name])  # Features (all columns except 'class_normal')
y_train = train_data_encoded[class_column_name]  # Labels (the 'class_normal' column)

# Train the model with the training data
model = RandomForestClassifier()
model.fit(X_train, y_train)  # Training the model

# Predict on the test data (which does not contain 'class' column)
X_test = test_data_encoded  # Test data features (no 'class' column)
y_pred = model.predict(X_test)  # Predict the labels for test data

# Print the predictions
print("Predictions on test data:", y_pred)

# Create the predictions DataFrame with an incremental 'lid' column
predictions = pd.DataFrame({
    'lid': range(1, len(y_pred) + 1),  # Incremental numeric 'lid' column starting from 1
    'Prediction': y_pred
})

# Save the predictions to 'predictions.csv'
predictions.to_csv('predictions.csv', index=False)

# Optionally, you can print out the predictions DataFrame to verify
print("Predictions saved to 'predictions.csv' successfully!")
print(predictions.head())

'''
#--------------------------------update predictions_with_value.csv---------------------------------------------

'''


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the train data and test data
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Apply One-Hot Encoding
train_data_encoded = pd.get_dummies(train_data, drop_first=True)
test_data_encoded = pd.get_dummies(test_data, drop_first=True)

# Check the columns after One-Hot Encoding
print("Encoded Columns in Train Data:", train_data_encoded.columns)
print("Encoded Columns in Test Data:", test_data_encoded.columns)

# In One-Hot Encoding, 'class' will be replaced by something like 'class_normal', 'class_anomaly'
# Identify the new column name that represents the target class
class_column_name = 'class_normal'  # Or 'class_anomaly', depending on how it was encoded

# Split the train data into features (X_train) and labels (y_train)
X_train = train_data_encoded.drop(columns=[class_column_name])  # Features (all columns except 'class_normal')
y_train = train_data_encoded[class_column_name]  # Labels (the 'class_normal' column)

# Train the model with the training data
model = RandomForestClassifier()
model.fit(X_train, y_train)  # Training the model

# Predict on the test data (which does not contain 'class' column)
X_test = test_data_encoded  # Test data features (no 'class' column)
y_pred = model.predict(X_test)  # Predict the labels for test data

# Print the predictions
print("Predictions on test data:", y_pred)

# Create the predictions DataFrame with an incremental 'lid' column
predictions = pd.DataFrame({
    'lid': range(1, len(y_pred) + 1),  # Incremental numeric 'lid' column starting from 1
    'Prediction': y_pred
})

# Save the predictions to 'predictions.csv'
predictions.to_csv('predictions.csv', index=False)

# Optionally, you can print out the predictions DataFrame to verify
print("Predictions saved to 'predictions.csv' successfully!")
print(predictions.head())

# Load the 'main_data_match.csv' file to add the 'value' column
main_data_match = pd.read_csv('main_data_match.csv')

# Merge predictions with main_data_match based on 'lid' and 'id'
# Assuming the 'id' column in main_data_match corresponds to 'lid' in predictions
predictions = predictions.merge(main_data_match[['id', 'class']], left_on='lid', right_on='id', how='left')

# Add the 'value' column to predictions (it comes from the 'class' column of main_data_match)
predictions['value'] = predictions['class']

# Drop the extra 'id' column after the merge
predictions.drop(columns=['id', 'class'], inplace=True)

# Save the updated predictions DataFrame with 'value' column to a new CSV file
predictions.to_csv('predictions_with_value.csv', index=False)

print("Predictions updated with 'value' column and saved to 'predictions_with_value.csv' successfully!")
print(predictions.head())

'''
#-----------------------------------------------predictions_with_value_and_ML_value.csv---------------


'''

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the train data and test data
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Apply One-Hot Encoding
train_data_encoded = pd.get_dummies(train_data, drop_first=True)
test_data_encoded = pd.get_dummies(test_data, drop_first=True)

# Check the columns after One-Hot Encoding
print("Encoded Columns in Train Data:", train_data_encoded.columns)
print("Encoded Columns in Test Data:", test_data_encoded.columns)

# In One-Hot Encoding, 'class' will be replaced by something like 'class_normal', 'class_anomaly'
# Identify the new column name that represents the target class
class_column_name = 'class_normal'  # Or 'class_anomaly', depending on how it was encoded

# Split the train data into features (X_train) and labels (y_train)
X_train = train_data_encoded.drop(columns=[class_column_name])  # Features (all columns except 'class_normal')
y_train = train_data_encoded[class_column_name]  # Labels (the 'class_normal' column)

# Train the model with the training data
model = RandomForestClassifier()
model.fit(X_train, y_train)  # Training the model

# Predict on the test data (which does not contain 'class' column)
X_test = test_data_encoded  # Test data features (no 'class' column)
y_pred = model.predict(X_test)  # Predict the labels for test data

# Print the predictions
print("Predictions on test data:", y_pred)

# Create the predictions DataFrame with an incremental 'lid' column
predictions = pd.DataFrame({
    'lid': range(1, len(y_pred) + 1),  # Incremental numeric 'lid' column starting from 1
    'Prediction': y_pred
})

# Save the predictions to 'predictions.csv'
predictions.to_csv('predictions.csv', index=False)

# Optionally, you can print out the predictions DataFrame to verify
print("Predictions saved to 'predictions.csv' successfully!")
print(predictions.head())

# Load the 'main_data_match.csv' file to add the 'value' column
main_data_match = pd.read_csv('main_data_match.csv')

# Merge predictions with main_data_match based on 'lid' and 'id'
# Assuming the 'id' column in main_data_match corresponds to 'lid' in predictions
predictions = predictions.merge(main_data_match[['id', 'class']], left_on='lid', right_on='id', how='left')

# Add the 'value' column to predictions (it comes from the 'class' column of main_data_match)
predictions['value'] = predictions['class']

# Drop the extra 'id' column after the merge
predictions.drop(columns=['id', 'class'], inplace=True)

# Create 'ML_value' column based on the condition
predictions['ML_value'] = ((predictions['Prediction'] == True) & (predictions['value'] == 'normal')) | \
                           ((predictions['Prediction'] == False) & (predictions['value'] == 'anomaly'))

# Convert boolean to 1 (True) and 0 (False)
predictions['ML_value'] = predictions['ML_value'].astype(int)

# Save the updated predictions DataFrame with 'value' and 'ML_value' column to a new CSV file
predictions.to_csv('predictions_with_value_and_ML_value.csv', index=False)

print("Predictions updated with 'value' and 'ML_value' column saved to 'predictions_with_value_and_ML_value.csv' successfully!")
print(predictions.head())


'''
#------------------------------------------with output cmd Final-------------------





import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from colorama import Fore, Style, init  # Import colorama for colorful output

# Initialize colorama
init(autoreset=True)

# Load the train data and test data
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Apply One-Hot Encoding
train_data_encoded = pd.get_dummies(train_data, drop_first=True)
test_data_encoded = pd.get_dummies(test_data, drop_first=True)

# Check the columns after One-Hot Encoding
print(Fore.CYAN + Style.BRIGHT + "Encoded Columns in Train Data:")
print(Fore.GREEN + Style.BRIGHT + str(train_data_encoded.columns))
print(Fore.CYAN + Style.BRIGHT + "Encoded Columns in Test Data:")
print(Fore.GREEN + Style.BRIGHT + str(test_data_encoded.columns))

# Identify the new column name that represents the target class
class_column_name = 'class_normal'  # Or 'class_anomaly', depending on how it was encoded

# Split the train data into features (X_train) and labels (y_train)
X_train = train_data_encoded.drop(columns=[class_column_name])  # Features
y_train = train_data_encoded[class_column_name]  # Labels

# Train the model with the training data
model = RandomForestClassifier()
model.fit(X_train, y_train)  # Training the model

# Predict on the test data
X_test = test_data_encoded  # Test data features
y_pred = model.predict(X_test)  # Predict the labels for test data

# Create the predictions DataFrame with an incremental 'lid' column
predictions = pd.DataFrame({
    'lid': range(1, len(y_pred) + 1),  # Incremental 'lid' starting from 1
    'Prediction': y_pred
})

# Save the predictions to 'predictions.csv'
predictions.to_csv('predictions.csv', index=False)

# Print confirmation and first few predictions
print(Fore.YELLOW + "Predictions saved to 'predictions.csv' successfully!")
print(Fore.GREEN + "First few predictions:\n", predictions.head())

# Load the 'main_data_match.csv' file to add the 'value' column
main_data_match = pd.read_csv('main_data_match.csv')

# Merge predictions with main_data_match based on 'lid' and 'id'
predictions = predictions.merge(main_data_match[['id', 'class']], left_on='lid', right_on='id', how='left')

# Add the 'value' column to predictions
predictions['value'] = predictions['class']

# Drop the extra 'id' column after the merge
predictions.drop(columns=['id', 'class'], inplace=True)

# Create 'ML_value' column based on the condition
predictions['ML_value'] = ((predictions['Prediction'] == True) & (predictions['value'] == 'normal')) | \
                           ((predictions['Prediction'] == False) & (predictions['value'] == 'anomaly'))

# Convert boolean to 1 (True) and 0 (False)
predictions['ML_value'] = predictions['ML_value'].astype(int)

# Save the updated predictions DataFrame with 'value' and 'ML_value' column to a new CSV file
predictions.to_csv('predictions_with_value_and_ML_value.csv', index=False)

# Calculate the ratio of incorrect (0) predictions to total predictions
incorrect_count = predictions['ML_value'].value_counts().get(0, 0)  # Get the count of '0', default to 0
total_count = len(predictions)  # Total number of predictions

# Calculate the ratio of incorrect predictions (0) to total predictions
incorrect_ratio = incorrect_count / total_count

# Print the ratio of incorrect predictions (0) as a percentage with colorful output
print(Fore.RED + Style.BRIGHT + f"\nRatio of incorrect predictions (0) to total predictions: {incorrect_ratio * 100:.2f}%")

print(Fore.YELLOW + Style.BRIGHT + "Predictions updated with 'value' and 'ML_value' column saved to 'predictions_with_value_and_ML_value.csv' successfully!")

# Optionally, print out the predictions DataFrame to verify
print(Fore.GREEN + "First few updated predictions:")
print(predictions.head())





#-------------------------------------save the model ----------------------------------

'''

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from colorama import Fore, Style, init  # Import colorama for colorful output
import joblib  # Import joblib for saving the model

# Initialize colorama
init(autoreset=True)

# Load the train data and test data
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Apply One-Hot Encoding
train_data_encoded = pd.get_dummies(train_data, drop_first=True)
test_data_encoded = pd.get_dummies(test_data, drop_first=True)

# Check the columns after One-Hot Encoding
print(Fore.CYAN + Style.BRIGHT + "Encoded Columns in Train Data:")
print(Fore.GREEN + Style.BRIGHT + str(train_data_encoded.columns))
print(Fore.CYAN + Style.BRIGHT + "Encoded Columns in Test Data:")
print(Fore.GREEN + Style.BRIGHT + str(test_data_encoded.columns))

# Identify the new column name that represents the target class
class_column_name = 'class_normal'  # Or 'class_anomaly', depending on how it was encoded

# Split the train data into features (X_train) and labels (y_train)
X_train = train_data_encoded.drop(columns=[class_column_name])  # Features
y_train = train_data_encoded[class_column_name]  # Labels

# Train the model with the training data
model = RandomForestClassifier()
model.fit(X_train, y_train)  # Training the model

# Save the trained model using joblib
joblib.dump(model, 'loganalysis_ML_model.pkl')

# Print confirmation message
print(Fore.YELLOW + Style.BRIGHT + "Model saved successfully as 'loganalysis_ML_model.pkl'")

# Predict on the test data
X_test = test_data_encoded  # Test data features
y_pred = model.predict(X_test)  # Predict the labels for test data

# Create the predictions DataFrame with an incremental 'lid' column
predictions = pd.DataFrame({
    'lid': range(1, len(y_pred) + 1),  # Incremental 'lid' starting from 1
    'Prediction': y_pred
})

# Save the predictions to 'predictions.csv'
predictions.to_csv('predictions.csv', index=False)

# Optionally, you can print out the predictions DataFrame to verify
print(Fore.YELLOW + "Predictions saved to 'predictions.csv' successfully!")
print(Fore.GREEN + "First few predictions:\n", predictions.head())

# Load the 'main_data_match.csv' file to add the 'value' column
main_data_match = pd.read_csv('main_data_match.csv')

# Merge predictions with main_data_match based on 'lid' and 'id'
predictions = predictions.merge(main_data_match[['id', 'class']], left_on='lid', right_on='id', how='left')

# Add the 'value' column to predictions
predictions['value'] = predictions['class']

# Drop the extra 'id' column after the merge
predictions.drop(columns=['id', 'class'], inplace=True)

# Create 'ML_value' column based on the condition
predictions['ML_value'] = ((predictions['Prediction'] == True) & (predictions['value'] == 'normal')) | \
                           ((predictions['Prediction'] == False) & (predictions['value'] == 'anomaly'))

# Convert boolean to 1 (True) and 0 (False)
predictions['ML_value'] = predictions['ML_value'].astype(int)

# Save the updated predictions DataFrame with 'value' and 'ML_value' column to a new CSV file
predictions.to_csv('predictions_with_value_and_ML_value.csv', index=False)

# Calculate the ratio of incorrect (0) predictions to total predictions
incorrect_count = predictions['ML_value'].value_counts().get(0, 0)  # Get the count of '0', default to 0
total_count = len(predictions)  # Total number of predictions

# Calculate the ratio of incorrect predictions (0) to total predictions
incorrect_ratio = incorrect_count / total_count

# Print the ratio of incorrect predictions (0) as a percentage with colorful output
print(Fore.RED + Style.BRIGHT + "\n" + "="*50)  # Add a border for visual impact
print(Fore.YELLOW + Style.BRIGHT + f"  Ratio of Incorrect Predictions (0) to Total Predictions: " + 
      Fore.GREEN + Style.BRIGHT + f"{incorrect_ratio * 100:.2f}%" + Fore.RED + Style.BRIGHT)
print(Fore.RED + Style.BRIGHT + "="*50)  # Add a border for visual impact

print(Fore.YELLOW + Style.BRIGHT + "Predictions updated with 'value' and 'ML_value' column saved to 'predictions_with_value_and_ML_value.csv' successfully!")

#--------------------------------------------------------------------------------------
'''
#------------------------------------with deep learning ---------------------------------

'''

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from colorama import Fore, Style, init  # For colorful output
import joblib  # For saving the model

# Initialize colorama
init(autoreset=True)

# Load the train and test data
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Apply One-Hot Encoding
train_data_encoded = pd.get_dummies(train_data, drop_first=True)
test_data_encoded = pd.get_dummies(test_data, drop_first=True)

# Check the columns after One-Hot Encoding
print(Fore.CYAN + Style.BRIGHT + "Encoded Columns in Train Data:")
print(Fore.GREEN + Style.BRIGHT + str(train_data_encoded.columns))
print(Fore.CYAN + Style.BRIGHT + "Encoded Columns in Test Data:")
print(Fore.GREEN + Style.BRIGHT + str(test_data_encoded.columns))

# Identify the new column name that represents the target class
class_column_name = 'class_normal'  # Or 'class_anomaly', depending on how it was encoded

# Split the train data into features (X_train) and labels (y_train)
X_train = train_data_encoded.drop(columns=[class_column_name])  # Features
y_train = train_data_encoded[class_column_name]  # Labels

# Preprocess the data (Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(test_data_encoded)  # Apply the same scaling to test data

# Train-Test Split (for better evaluation)
X_train, X_val, y_train, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

# Build the Deep Learning Model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))  # Input layer
model.add(Dense(32, activation='relu'))  # Hidden layer
model.add(Dense(1, activation='sigmoid'))  # Output layer (binary classification)

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Save the trained model
model.save('loganalysis_deep_learning_model.h5')
print(Fore.YELLOW + Style.BRIGHT + "Model saved successfully as 'loganalysis_deep_learning_model.h5'")

# Load the model (for future use)
model = tf.keras.models.load_model('loganalysis_deep_learning_model.h5')

# Predict on the test data
y_pred = model.predict(X_test_scaled)
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to 0 or 1

# Create the predictions DataFrame with an incremental 'lid' column
predictions = pd.DataFrame({
    'lid': range(1, len(y_pred) + 1),  # Incremental 'lid' starting from 1
    'Prediction': y_pred.flatten()  # Flattening to convert to 1D array
})

# Save the predictions to 'predictions.csv'
predictions.to_csv('predictions.csv', index=False)
print(Fore.YELLOW + "Predictions saved to 'predictions.csv' successfully!")

# Load the 'main_data_match.csv' file to add the 'value' column
main_data_match = pd.read_csv('main_data_match.csv')

# Merge predictions with main_data_match based on 'lid' and 'id'
predictions = predictions.merge(main_data_match[['id', 'class']], left_on='lid', right_on='id', how='left')

# Add the 'value' column to predictions
predictions['value'] = predictions['class']

# Drop the extra 'id' column after the merge
predictions.drop(columns=['id', 'class'], inplace=True)

# Create 'ML_value' column based on the condition
predictions['ML_value'] = ((predictions['Prediction'] == 1) & (predictions['value'] == 'normal')) | \
                           ((predictions['Prediction'] == 0) & (predictions['value'] == 'anomaly'))

# Convert boolean to 1 (True) and 0 (False)
predictions['ML_value'] = predictions['ML_value'].astype(int)

# Save the updated predictions DataFrame with 'value' and 'ML_value' column to a new CSV file
predictions.to_csv('predictions_with_value_and_ML_value.csv', index=False)

# Calculate the ratio of incorrect (0) predictions to total predictions
incorrect_count = predictions['ML_value'].value_counts().get(0, 0)  # Get the count of '0', default to 0
total_count = len(predictions)  # Total number of predictions

# Calculate the ratio of incorrect predictions (0) to total predictions
incorrect_ratio = incorrect_count / total_count

# Print the ratio of incorrect predictions (0) as a percentage
print(Fore.RED + Style.BRIGHT + "\n" + "="*50)  # Add a border for visual impact
print(Fore.YELLOW + Style.BRIGHT + f"  Ratio of Incorrect Predictions (0) to Total Predictions: " + 
      Fore.GREEN + Style.BRIGHT + f"{incorrect_ratio * 100:.2f}%" + Fore.RED + Style.BRIGHT)
print(Fore.RED + Style.BRIGHT + "="*50)  # Add a border for visual impact

print(Fore.YELLOW + Style.BRIGHT + "Predictions updated with 'value' and 'ML_value' column saved to 'predictions_with_value_and_ML_value.csv' successfully!")

'''

#----------   *****   more Deep learning  ******  ------------------------------------------------

'''

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import AdamW
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from colorama import Fore, Style, init  # For colorful output
import joblib  # For saving the model
from tensorflow.keras.callbacks import ReduceLROnPlateau  # For learning rate reduction on plateau

# Initialize colorama
init(autoreset=True)

# Load the train and test data
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Apply One-Hot Encoding
train_data_encoded = pd.get_dummies(train_data, drop_first=True)
test_data_encoded = pd.get_dummies(test_data, drop_first=True)

# Check the columns after One-Hot Encoding
print(Fore.CYAN + Style.BRIGHT + "Encoded Columns in Train Data:")
print(Fore.GREEN + Style.BRIGHT + str(train_data_encoded.columns))
print(Fore.CYAN + Style.BRIGHT + "Encoded Columns in Test Data:")
print(Fore.GREEN + Style.BRIGHT + str(test_data_encoded.columns))

# Identify the new column name that represents the target class
class_column_name = 'class_normal'  # Or 'class_anomaly', depending on how it was encoded

# Split the train data into features (X_train) and labels (y_train)
X_train = train_data_encoded.drop(columns=[class_column_name])  # Features
y_train = train_data_encoded[class_column_name]  # Labels

# Preprocess the data (Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(test_data_encoded)  # Apply the same scaling to test data

# Train-Test Split (for better evaluation)
X_train, X_val, y_train, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

# Build the Deep Learning Model with Dropout for regularization
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))  # Input layer with more neurons
model.add(Dropout(0.5))  # Adding dropout to prevent overfitting
model.add(Dense(64, activation='relu'))  # Hidden layer with more neurons
model.add(Dropout(0.5))  # Adding dropout to prevent overfitting
model.add(Dense(32, activation='relu'))  # Another hidden layer
model.add(Dense(1, activation='sigmoid'))  # Output layer (binary classification)

# Compile the model using AdamW optimizer
model.compile(optimizer=AdamW(), loss='binary_crossentropy', metrics=['accuracy'])

# Reduce learning rate when the validation loss plateaus
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, factor=0.5, min_lr=1e-6)

# Train the model with updated parameters and learning rate reduction
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val), callbacks=[lr_reduction])

# Save the trained model
model.save('loganalysis_deep_learning_model.h5')
print(Fore.YELLOW + Style.BRIGHT + "Model saved successfully as 'loganalysis_deep_learning_model.h5'")

# Load the model (for future use)
model = tf.keras.models.load_model('loganalysis_deep_learning_model.h5')

# Predict on the test data
y_pred = model.predict(X_test_scaled)
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to 0 or 1

# Create the predictions DataFrame with an incremental 'lid' column
predictions = pd.DataFrame({
    'lid': range(1, len(y_pred) + 1),  # Incremental 'lid' starting from 1
    'Prediction': y_pred.flatten()  # Flattening to convert to 1D array
})

# Save the predictions to 'predictions.csv'
predictions.to_csv('predictions.csv', index=False)
print(Fore.YELLOW + "Predictions saved to 'predictions.csv' successfully!")

# Load the 'main_data_match.csv' file to add the 'value' column
main_data_match = pd.read_csv('main_data_match.csv')

# Merge predictions with main_data_match based on 'lid' and 'id'
predictions = predictions.merge(main_data_match[['id', 'class']], left_on='lid', right_on='id', how='left')

# Add the 'value' column to predictions
predictions['value'] = predictions['class']

# Drop the extra 'id' column after the merge
predictions.drop(columns=['id', 'class'], inplace=True)

# Create 'ML_value' column based on the condition
predictions['ML_value'] = ((predictions['Prediction'] == 1) & (predictions['value'] == 'normal')) | \
                           ((predictions['Prediction'] == 0) & (predictions['value'] == 'anomaly'))

# Convert boolean to 1 (True) and 0 (False)
predictions['ML_value'] = predictions['ML_value'].astype(int)

# Save the updated predictions DataFrame with 'value' and 'ML_value' column to a new CSV file
predictions.to_csv('predictions_with_value_and_ML_value.csv', index=False)

# Calculate the ratio of incorrect (0) predictions to total predictions
incorrect_count = predictions['ML_value'].value_counts().get(0, 0)  # Get the count of '0', default to 0
total_count = len(predictions)  # Total number of predictions

# Calculate the ratio of incorrect predictions (0) to total predictions
incorrect_ratio = incorrect_count / total_count

# Print the ratio of incorrect predictions (0) as a percentage
print(Fore.RED + Style.BRIGHT + "\n" + "="*50)  # Add a border for visual impact
print(Fore.YELLOW + Style.BRIGHT + f"  Ratio of Incorrect Predictions (0) to Total Predictions: " + 
      Fore.GREEN + Style.BRIGHT + f"{incorrect_ratio * 100:.2f}%" + Fore.RED + Style.BRIGHT)
print(Fore.RED + Style.BRIGHT + "="*50)  # Add a border for visual impact

print(Fore.YELLOW + Style.BRIGHT + "Predictions updated with 'value' and 'ML_value' column saved to 'predictions_with_value_and_ML_value.csv' successfully!")

'''


#------------------------------------- Test with Another Format -----------------------------------------

'''
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from colorama import Fore, Style, init
import joblib

# Initialize colorama
init(autoreset=True)

# Load the train and test data
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Apply One-Hot Encoding
train_data_encoded = pd.get_dummies(train_data, drop_first=True)
test_data_encoded = pd.get_dummies(test_data, drop_first=True)

# Check the columns after One-Hot Encoding
print(Fore.CYAN + Style.BRIGHT + "Encoded Columns in Train Data:")
print(Fore.GREEN + Style.BRIGHT + str(train_data_encoded.columns))
print(Fore.CYAN + Style.BRIGHT + "Encoded Columns in Test Data:")
print(Fore.GREEN + Style.BRIGHT + str(test_data_encoded.columns))

# Identify the new column name that represents the target class
class_column_name = 'class_normal'

# Split the train data into features (X_train) and labels (y_train)
X_train = train_data_encoded.drop(columns=[class_column_name])  # Features
y_train = train_data_encoded[class_column_name]  # Labels

# Preprocess the data (Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(test_data_encoded)

# Train-Test Split (for better evaluation)
X_train, X_val, y_train, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

# Build the Deep Learning Model with Dropout for regularization
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model using AdamW optimizer
model.compile(optimizer=AdamW(), loss='binary_crossentropy', metrics=['accuracy'])

# Reduce learning rate when the validation loss plateaus
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, factor=0.5, min_lr=1e-6)

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with updated parameters and learning rate reduction
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping, lr_reduction])

# Save the trained model
model.save('loganalysis_deep_learning_model.h5')
print(Fore.YELLOW + Style.BRIGHT + "Model saved successfully as 'loganalysis_deep_learning_model.h5'")

# Load the model (for future use)
model = tf.keras.models.load_model('loganalysis_deep_learning_model.h5')

# Predict on the test data
y_pred = model.predict(X_test_scaled)
y_pred = (y_pred > 0.5).astype(int)

# Create the predictions DataFrame with an incremental 'lid' column
predictions = pd.DataFrame({
    'lid': range(1, len(y_pred) + 1),
    'Prediction': y_pred.flatten()
})

# Save the predictions to 'predictions.csv'
predictions.to_csv('predictions.csv', index=False)
print(Fore.YELLOW + "Predictions saved to 'predictions.csv' successfully!")

# Load the 'main_data_match.csv' file to add the 'value' column
main_data_match = pd.read_csv('main_data_match.csv')

# Merge predictions with main_data_match based on 'lid' and 'id'
predictions = predictions.merge(main_data_match[['id', 'class']], left_on='lid', right_on='id', how='left')

# Add the 'value' column to predictions
predictions['value'] = predictions['class']

# Drop the extra 'id' column after the merge
predictions.drop(columns=['id', 'class'], inplace=True)

# Create 'ML_value' column based on the condition
predictions['ML_value'] = ((predictions['Prediction'] == 1) & (predictions['value'] == 'normal')) | \
                           ((predictions['Prediction'] == 0) & (predictions['value'] == 'anomaly'))

# Convert boolean to 1 (True) and 0 (False)
predictions['ML_value'] = predictions['ML_value'].astype(int)

# Save the updated predictions DataFrame with 'value' and 'ML_value' column to a new CSV file
predictions.to_csv('predictions_with_value_and_ML_value.csv', index=False)

# Calculate the ratio of incorrect (0) predictions to total predictions
incorrect_count = predictions['ML_value'].value_counts().get(0, 0)  # Get the count of '0', default to 0
total_count = len(predictions)  # Total number of predictions

# Calculate the ratio of incorrect predictions (0) to total predictions
incorrect_ratio = incorrect_count / total_count

# Print the ratio of incorrect predictions (0) as a percentage
print(Fore.RED + Style.BRIGHT + "\n" + "="*50)
print(Fore.YELLOW + Style.BRIGHT + f"  Ratio of Incorrect Predictions (0) to Total Predictions: " + 
      Fore.GREEN + Style.BRIGHT + f"{incorrect_ratio * 100:.2f}%" + Fore.RED + Style.BRIGHT)
print(Fore.RED + Style.BRIGHT + "="*50)

print(Fore.YELLOW + Style.BRIGHT + "Predictions updated with 'value' and 'ML_value' column saved to 'predictions_with_value_and_ML_value.csv' successfully!")



'''

#----------------------------------------  GUI output -------------------------

'''

import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from colorama import Fore, Style, init  # Import colorama for colorful output

# Initialize colorama
init(autoreset=True)

# Function to load data, train the model and show results
def run_model():
    try:
        # Clear the previous output
        output_text.delete(1.0, tk.END)  # Clear Text widget before new output

        # Load the train data and test data
        train_data = pd.read_csv('train_data.csv')
        test_data = pd.read_csv('test_data.csv')

        # Apply One-Hot Encoding
        train_data_encoded = pd.get_dummies(train_data, drop_first=True)
        test_data_encoded = pd.get_dummies(test_data, drop_first=True)

        # Identify the new column name that represents the target class
        class_column_name = 'class_normal'  # Or 'class_anomaly', depending on how it was encoded

        # Split the train data into features (X_train) and labels (y_train)
        X_train = train_data_encoded.drop(columns=[class_column_name])  # Features
        y_train = train_data_encoded[class_column_name]  # Labels

        # Training the model
        output_text.insert(tk.END, "Training the model...\n")
        model = RandomForestClassifier()
        model.fit(X_train, y_train)  # Training the model
        output_text.insert(tk.END, "Model training completed...\n")

        # Predict on the test data
        X_test = test_data_encoded  # Test data features
        output_text.insert(tk.END, "Predicting on test data...\n")
        y_pred = model.predict(X_test)  # Predict the labels for test data
        output_text.insert(tk.END, "Prediction completed...\n")

        # Create the predictions DataFrame with an incremental 'lid' column
        predictions = pd.DataFrame({
            'lid': range(1, len(y_pred) + 1),  # Incremental 'lid' starting from 1
            'Prediction': y_pred
        })

        # Save the predictions to 'predictions.csv'
        predictions.to_csv('predictions.csv', index=False)

        # Display success message
        messagebox.showinfo("Success", "Model executed and predictions saved to 'predictions.csv'")

        # Show first few predictions in the GUI
        result_text = predictions.head().to_string()
        output_text.insert(tk.END, f"First few predictions:\n{result_text}\n\n")

        # Calculate the accuracy
        main_data_match = pd.read_csv('main_data_match.csv')
        predictions = predictions.merge(main_data_match[['id', 'class']], left_on='lid', right_on='id', how='left')
        predictions['value'] = predictions['class']
        predictions.drop(columns=['id', 'class'], inplace=True)

        # **Fix**: Convert y_pred (boolean) to match the 'normal'/'anomaly' labels
        predictions['Prediction'] = predictions['Prediction'].map({1: 'normal', 0: 'anomaly'})

        # Calculate the ratio of incorrect (0) predictions to total predictions
        predictions['ML_value'] = ((predictions['Prediction'] == 'normal') & (predictions['value'] == 'normal')) | \
                                   ((predictions['Prediction'] == 'anomaly') & (predictions['value'] == 'anomaly'))
        predictions['ML_value'] = predictions['ML_value'].astype(int)

        incorrect_count = predictions['ML_value'].value_counts().get(0, 0)
        total_count = len(predictions)
        incorrect_ratio = incorrect_count / total_count

        # Show accuracy
        accuracy_text = f"Accuracy: {accuracy_score(predictions['value'], predictions['Prediction']):.2f}"
        accuracy_label.config(text=accuracy_text)

        incorrect_text = f"Ratio of Incorrect Predictions: {incorrect_ratio * 100:.2f}%"
        incorrect_label.config(text=incorrect_text)

        output_text.insert(tk.END, f"\n{accuracy_text}\n")
        output_text.insert(tk.END, f"{incorrect_text}\n")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Set up the Tkinter Window
root = tk.Tk()
root.title("Log Analysis - ML Model")
root.geometry("700x600")

# Add a label and button to run the model
run_button = tk.Button(root, text="Run Model", command=run_model, font=("Arial", 12))
run_button.pack(pady=20)

# Add a label to show accuracy
accuracy_label = tk.Label(root, text="Accuracy: Not yet calculated", font=("Arial", 12), fg="green")
accuracy_label.pack(pady=10)

# Add a label to show incorrect prediction ratio
incorrect_label = tk.Label(root, text="Incorrect Prediction Ratio: Not yet calculated", font=("Arial", 12), fg="red")
incorrect_label.pack(pady=10)

# Add a Scrollable Text widget for displaying the output
output_text = tk.Text(root, height=15, width=80, wrap=tk.WORD, font=("Arial", 10), bg="lightgray")
output_text.pack(pady=10)

# Run the Tkinter main loop
root.mainloop()

'''
#--------------------------------------------------  simpified Deep learning ------------------------

'''

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from colorama import Fore, Style, init
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Initialize colorama
init(autoreset=True)

# Load the train and test data
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Apply One-Hot Encoding
train_data_encoded = pd.get_dummies(train_data, drop_first=True)
test_data_encoded = pd.get_dummies(test_data, drop_first=True)

# Check the columns after One-Hot Encoding
print(Fore.CYAN + Style.BRIGHT + "Encoded Columns in Train Data:")
print(Fore.GREEN + Style.BRIGHT + str(train_data_encoded.columns))
print(Fore.CYAN + Style.BRIGHT + "Encoded Columns in Test Data:")
print(Fore.GREEN + Style.BRIGHT + str(test_data_encoded.columns))

# Identify the new column name that represents the target class
class_column_name = 'class_normal'

# Split the train data into features (X_train) and labels (y_train)
X_train = train_data_encoded.drop(columns=[class_column_name])  # Features
y_train = train_data_encoded[class_column_name]  # Labels

# Preprocess the data (Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(test_data_encoded)  # Apply the same scaling to test data

# Train-Test Split (for better evaluation)
X_train, X_val, y_train, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

# Build the Deep Learning Model with simpler architecture
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))  # Reduced neurons in input layer
model.add(Dense(32, activation='relu'))  # Single hidden layer with reduced neurons
model.add(Dense(1, activation='sigmoid'))  # Output layer (binary classification)

# Compile the model using Adam optimizer
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Reduce learning rate when the validation loss plateaus
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5, min_lr=1e-6)

# Early stopping to prevent overfitting and allow early termination if the model stops improving
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with updated parameters, learning rate reduction, and early stopping
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val), callbacks=[lr_reduction, early_stopping])

# Save the trained model
model.save('loganalysis_deep_learning_model_simplified.h5')
print(Fore.YELLOW + Style.BRIGHT + "Model saved successfully as 'loganalysis_deep_learning_model_simplified.h5'")

# Load the model (for future use)
model = tf.keras.models.load_model('loganalysis_deep_learning_model_simplified.h5')

# Predict on the test data
y_pred = model.predict(X_test_scaled)
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to 0 or 1

# Create the predictions DataFrame with an incremental 'lid' column
predictions = pd.DataFrame({
    'lid': range(1, len(y_pred) + 1),  # Incremental 'lid' starting from 1
    'Prediction': y_pred.flatten()  # Flattening to convert to 1D array
})

# Save the predictions to 'predictions.csv'
predictions.to_csv('predictions.csv', index=False)
print(Fore.YELLOW + "Predictions saved to 'predictions.csv' successfully!")

# Load the 'main_data_match.csv' file to add the 'value' column
main_data_match = pd.read_csv('main_data_match.csv')

# Merge predictions with main_data_match based on 'lid' and 'id'
predictions = predictions.merge(main_data_match[['id', 'class']], left_on='lid', right_on='id', how='left')

# Add the 'value' column to predictions
predictions['value'] = predictions['class']

# Drop the extra 'id' column after the merge
predictions.drop(columns=['id', 'class'], inplace=True)

# Create 'ML_value' column based on the condition
predictions['ML_value'] = ((predictions['Prediction'] == 1) & (predictions['value'] == 'normal')) | \
                           ((predictions['Prediction'] == 0) & (predictions['value'] == 'anomaly'))

# Convert boolean to 1 (True) and 0 (False)
predictions['ML_value'] = predictions['ML_value'].astype(int)

# Save the updated predictions DataFrame with 'value' and 'ML_value' column to a new CSV file
predictions.to_csv('predictions_with_value_and_ML_value_simplified.csv', index=False)

# Calculate the ratio of incorrect (0) predictions to total predictions
incorrect_count = predictions['ML_value'].value_counts().get(0, 0)  # Get the count of '0', default to 0
total_count = len(predictions)  # Total number of predictions

# Calculate the ratio of incorrect predictions (0) to total predictions
incorrect_ratio = incorrect_count / total_count

# Print the ratio of incorrect predictions (0) as a percentage
print(Fore.RED + Style.BRIGHT + "\n" + "="*50)
print(Fore.YELLOW + Style.BRIGHT + f"  Ratio of Incorrect Predictions (0) to Total Predictions: " + 
      Fore.GREEN + Style.BRIGHT + f"{incorrect_ratio * 100:.2f}%" + Fore.RED + Style.BRIGHT)
print(Fore.RED + Style.BRIGHT + "="*50)

print(Fore.YELLOW + Style.BRIGHT + "Predictions updated with 'value' and 'ML_value' column saved to 'predictions_with_value_and_ML_value_simplified.csv' successfully!")

'''

#-------------------------------------------------------------------------------------

