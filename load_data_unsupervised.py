
#-----------------------------  first testy.py -----------------------------

'''

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Step 1: Load the labeled data
df = pd.read_csv('classified_traffic_full.csv')

# Step 2: Clean column names (redundant if already done)
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('/', '_')

# Step 3: Encode 'class' column to numeric (normal=0, anomaly=1)
df['class'] = df['class'].map({'normal': 0, 'anomaly': 1})

# Step 4: Define features and target
features = [
    'Flow_Duration', 'Total_Fwd_Packets', 'Total_Backward_Packets',
    'Fwd_Packet_Length_Max', 'Bwd_Packet_Length_Max',
    'Flow_Bytes_s', 'Flow_Packets_s',
    'Fwd_IAT_Mean', 'Bwd_IAT_Mean', 'Packet_Length_Variance',
    'Average_Packet_Size'
]
X = df[features]
y = df['class']

# Step 5: Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = clf.predict(X_test)
print("📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['normal', 'anomaly']))

# Step 8: Show feature importance
importances = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
print("\n🌲 Feature Importances:")
print(importances)

# Step 9 (Optional): Save the model
joblib.dump(clf, 'rf_anomaly_model.pkl')
print("\n✅ Model saved as 'rf_anomaly_model.pkl'")


'''

#-----------------------------  second testy.py - code_GR_1 -----------------------------

'''

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, f1_score
)
from colorama import Fore, Style, init
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize colorama
init(autoreset=True)

# Load trained model
model = joblib.load('rf_anomaly_model.pkl')

# Load test data and true labels
df_test = pd.read_csv('sample.csv')
df_true = pd.read_csv('classified_traffic_full.csv')

# Clean column names
df_test.columns = df_test.columns.str.strip().str.replace(' ', '_').str.replace('/', '_')
df_true.columns = df_true.columns.str.strip().str.replace(' ', '_').str.replace('/', '_')

# Features used by the model
features = [
    'Flow_Duration', 'Total_Fwd_Packets', 'Total_Backward_Packets',
    'Fwd_Packet_Length_Max', 'Bwd_Packet_Length_Max',
    'Flow_Bytes_s', 'Flow_Packets_s',
    'Fwd_IAT_Mean', 'Bwd_IAT_Mean', 'Packet_Length_Variance',
    'Average_Packet_Size'
]

# Preprocess test data
X_test = df_test[features].replace([np.inf, -np.inf], np.nan).dropna()

# Make predictions
y_pred = model.predict(X_test)

# Create DataFrame with predictions
df_pred = df_test.loc[X_test.index].copy()
df_pred.insert(0, 'lid', range(1, len(df_pred) + 1))
df_pred['predicted_class'] = pd.Series(y_pred).map({0: 'normal', 1: 'anomaly'})
df_pred.to_csv('test_output.csv', index=False)

print(Fore.YELLOW + "✅ Predictions saved to 'test_output.csv'")

# Merge predictions with true classes on 'lid'
df_pred = df_pred[['lid', 'predicted_class']].dropna()
df_true = df_true[['lid', 'class']].dropna()

df_result = pd.merge(df_pred, df_true, on='lid', how='inner')

# Map class labels
df_result['true_class'] = df_result['class'].map({'normal': 0, 'anomaly': 1})
df_result['predicted_class'] = df_result['predicted_class'].map({'normal': 0, 'anomaly': 1})

# Remove any rows where mapping failed
df_result = df_result.dropna(subset=['true_class', 'predicted_class'])

# Convert to integers
y_true = df_result['true_class'].astype(int)
y_pred = df_result['predicted_class'].astype(int)

# Evaluation Metrics
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)

# Print results
print(Fore.CYAN + f"\n🎯 Accuracy: {accuracy * 100:.2f}%")
print(Fore.GREEN + f"F1 Score: {f1:.4f}")
print(Fore.MAGENTA + "\nClassification Report:\n" + classification_report(y_true, y_pred, target_names=['normal', 'anomaly']))

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Normal', 'Predicted Anomaly'],
            yticklabels=['Actual Normal', 'Actual Anomaly'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix_plot.png')
plt.show()


'''

#-------------------  test phase  (with correct and error description)-----------


import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, f1_score
)
from colorama import Fore, Style, init
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize colorama
init(autoreset=True)

# Load trained model
model = joblib.load('rf_anomaly_model.pkl')

# Load test data and true labels
df_test = pd.read_csv('sample.csv')
df_true = pd.read_csv('classified_traffic_full.csv')

# Clean column names
df_test.columns = df_test.columns.str.strip().str.replace(' ', '_').str.replace('/', '_')
df_true.columns = df_true.columns.str.strip().str.replace(' ', '_').str.replace('/', '_')

# Features used by the model
features = [
    'Flow_Duration', 'Total_Fwd_Packets', 'Total_Backward_Packets',
    'Fwd_Packet_Length_Max', 'Bwd_Packet_Length_Max',
    'Flow_Bytes_s', 'Flow_Packets_s',
    'Fwd_IAT_Mean', 'Bwd_IAT_Mean', 'Packet_Length_Variance',
    'Average_Packet_Size'
]

# Preprocess test data
X_test = df_test[features].replace([np.inf, -np.inf], np.nan).dropna()

# Make predictions
y_pred = model.predict(X_test)

# Create DataFrame with predictions
df_pred = df_test.loc[X_test.index].copy()
df_pred.insert(0, 'lid', range(1, len(df_pred) + 1))
df_pred['predicted_class'] = pd.Series(y_pred).map({0: 'normal', 1: 'anomaly'})
df_pred.to_csv('test_output.csv', index=False)

print(Fore.YELLOW + "✅ Predictions saved to 'test_output.csv'")

# Merge predictions with true classes on 'lid'
df_pred = df_pred[['lid', 'predicted_class']].dropna()
df_true = df_true[['lid', 'class']].dropna()
df_result = pd.merge(df_pred, df_true, on='lid', how='inner')

# Map class labels
df_result['true_class'] = df_result['class'].map({'normal': 0, 'anomaly': 1})
df_result['predicted_class'] = df_result['predicted_class'].map({'normal': 0, 'anomaly': 1})
df_result = df_result.dropna(subset=['true_class', 'predicted_class'])

# Convert to integers
y_true = df_result['true_class'].astype(int)
y_pred = df_result['predicted_class'].astype(int)

# Evaluation Metrics
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)

# Extract values
tn, fp, fn, tp = conf_matrix.ravel()
total = tn + fp + fn + tp
wrong_normal = fp  # Normal wrongly predicted as Anomaly
wrong_anomaly = fn  # Anomaly wrongly predicted as Normal
wrong_total = wrong_normal + wrong_anomaly
right_total = tn + tp

# Percentages
wrong_percent = (wrong_total / total) * 100
right_percent = (right_total / total) * 100

# Output stats
print(Fore.CYAN + f"\n🧾 Total samples processed for testing: {total}")
print(Fore.GREEN + f"✅ Correct predictions: {right_total} ({right_percent:.2f}%)")
print(Fore.RED + f"❌ Wrong predictions: {wrong_total} ({wrong_percent:.2f}%)")
print(Fore.RED + f"   └─ Normal wrongly predicted as Anomaly: {wrong_normal}")
print(Fore.RED + f"   └─ Anomaly wrongly predicted as Normal: {wrong_anomaly}")
print(Fore.CYAN + f"\n🎯 Accuracy: {accuracy * 100:.2f}%")
print(Fore.GREEN + f"F1 Score: {f1:.4f}")
print(Fore.MAGENTA + "\n📋 Classification Report:\n" +
      classification_report(y_true, y_pred, target_names=['normal', 'anomaly']))

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Normal', 'Predicted Anomaly'],
            yticklabels=['Actual Normal', 'Actual Anomaly'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix_plot.png')
plt.show()


#--------------------------------------------------------------------