
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
wrong_normal = fp
wrong_anomaly = fn
wrong_total = wrong_normal + wrong_anomaly
right_total = tn + tp

# Percentages
wrong_percent = (wrong_total / total) * 100
right_percent = (right_total / total) * 100

# Print summary
print(Fore.CYAN + f"\n🧾 Total samples processed for testing: {total}")
print(Fore.GREEN + f"✅ Correct predictions: {right_total} ({right_percent:.2f}%)")
print(Fore.RED + f"❌ Wrong predictions: {wrong_total} ({wrong_percent:.2f}%)")
print(Fore.RED + f"   └─ Normal wrongly predicted as Anomaly: {wrong_normal}")
print(Fore.RED + f"   └─ Anomaly wrongly predicted as Normal: {wrong_anomaly}")
print(Fore.CYAN + f"\n🎯 Accuracy: {accuracy * 100:.2f}%")
print(Fore.GREEN + f"F1 Score: {f1:.4f}")
print(Fore.MAGENTA + "\n📋 Classification Report:\n" +
      classification_report(y_true, y_pred, target_names=['normal', 'anomaly']))

# Ask user if they want to see the confusion matrix
user_input = input(Fore.YELLOW + "\n📈 Do you want to view the confusion matrix plot? (yes/no): ").strip().lower()
if user_input in ['yes', 'y']:
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
else:
    print(Fore.CYAN + "🛑 Plot display skipped.")

    '''
#--------------------------------with detailed graph------------------------------------


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
wrong_normal = fp
wrong_anomaly = fn
wrong_total = wrong_normal + wrong_anomaly
right_total = tn + tp
wrong_percent = (wrong_total / total) * 100
right_percent = (right_total / total) * 100

# Print summary
print(Fore.CYAN + f"\n🧾 Total samples processed for testing: {total}")
print(Fore.GREEN + f"✅ Correct predictions: {right_total} ({right_percent:.2f}%)")
print(Fore.RED + f"❌ Wrong predictions: {wrong_total} ({wrong_percent:.2f}%)")
print(Fore.RED + f"   └─ Normal wrongly predicted as Anomaly: {wrong_normal}")
print(Fore.RED + f"   └─ Anomaly wrongly predicted as Normal: {wrong_anomaly}")
print(Fore.CYAN + f"\n🎯 Accuracy: {accuracy * 100:.2f}%")
print(Fore.GREEN + f"F1 Score: {f1:.4f}")
print(Fore.MAGENTA + "\n📋 Classification Report:\n" +
      classification_report(y_true, y_pred, target_names=['normal', 'anomaly']))

# Ask user if they want to see the confusion matrix
user_input = input(Fore.YELLOW + "\n📈 Do you want to view the confusion matrix plot? (yes/no): ").strip().lower()
if user_input in ['yes', 'y']:
    # Set figure size wide enough
    plt.figure(figsize=(10, 7))
    ax = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                     xticklabels=['Predicted Normal', 'Predicted Anomaly'],
                     yticklabels=['Actual Normal', 'Actual Anomaly'])
    
    # Clean readable text
    stats_text = (
        f"Total: {total}    Correct: {right_total} ({right_percent:.2f}%)\n"
        f"Wrong: {wrong_total} ({wrong_percent:.2f}%)\n"
        f"Normal → Anomaly (FP): {wrong_normal}   Anomaly → Normal (FN): {wrong_anomaly}\n"
        f"True Positives (TP): {tp}   True Negatives (TN): {tn}\n"
        f"Accuracy: {accuracy * 100:.2f}%    F1 Score: {f1:.4f}"
    )
    plt.title("Confusion Matrix with Explanation", fontsize=14)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Position text neatly
    plt.text(2.1, -0.6, stats_text, fontsize=10, va='top', ha='left', linespacing=1.5)

    plt.tight_layout()
    plt.savefig("confusion_matrix_plot_with_explanation.png")
    plt.show()
else:
    print(Fore.CYAN + "🛑 Plot display skipped.")


'''

#-----------------------with model check --------------------

'''

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, f1_score
)
from colorama import Fore, Style, init
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize colorama
init(autoreset=True)

# Feature list
features = [
    'Flow_Duration', 'Total_Fwd_Packets', 'Total_Backward_Packets',
    'Fwd_Packet_Length_Max', 'Bwd_Packet_Length_Max',
    'Flow_Bytes_s', 'Flow_Packets_s',
    'Fwd_IAT_Mean', 'Bwd_IAT_Mean', 'Packet_Length_Variance',
    'Average_Packet_Size'
]

# Check for model file
model_file = 'rf_anomaly_model.pkl'
if not os.path.exists(model_file):
    print(Fore.CYAN + "🔄 Model not found. Training a new model...")

    # Load training data
    df_train = pd.read_csv('classified_traffic_full.csv')
    df_train.columns = df_train.columns.str.strip().str.replace(' ', '_').str.replace('/', '_')
    df_train['class'] = df_train['class'].map({'normal': 0, 'anomaly': 1})

    # Train model
    X_train = df_train[features].replace([np.inf, -np.inf], np.nan).dropna()
    y_train = df_train.loc[X_train.index, 'class']

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    joblib.dump(model, model_file)

    print(Fore.GREEN + f"✅ Model trained and saved as '{model_file}'")
else:
    model = joblib.load(model_file)
    print(Fore.GREEN + f"✅ Model loaded from '{model_file}'")

# Load test and true data
df_test = pd.read_csv('sample.csv')
df_true = pd.read_csv('classified_traffic_full.csv')

# Clean columns
df_test.columns = df_test.columns.str.strip().str.replace(' ', '_').str.replace('/', '_')
df_true.columns = df_true.columns.str.strip().str.replace(' ', '_').str.replace('/', '_')

# Preprocess test features
X_test = df_test[features].replace([np.inf, -np.inf], np.nan).dropna()

# Predict
y_pred = model.predict(X_test)

# Create predictions DataFrame
df_pred = df_test.loc[X_test.index].copy()
df_pred.insert(0, 'lid', range(1, len(df_pred) + 1))
df_pred['predicted_class'] = pd.Series(y_pred).map({0: 'normal', 1: 'anomaly'})
df_pred.to_csv('test_output.csv', index=False)

print(Fore.YELLOW + "✅ Predictions saved to 'test_output.csv'")

# Merge predictions with ground truth
df_pred = df_pred[['lid', 'predicted_class']].dropna()
df_true = df_true[['lid', 'class']].dropna()
df_result = pd.merge(df_pred, df_true, on='lid', how='inner')

# Map class labels
df_result['true_class'] = df_result['class'].map({'normal': 0, 'anomaly': 1})
df_result['predicted_class'] = df_result['predicted_class'].map({'normal': 0, 'anomaly': 1})
df_result = df_result.dropna(subset=['true_class', 'predicted_class'])

# Final predictions
y_true = df_result['true_class'].astype(int)
y_pred = df_result['predicted_class'].astype(int)

# Metrics
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()

# Summary
total = tn + tp + fp + fn
right_total = tn + tp
wrong_total = fp + fn
wrong_percent = (wrong_total / total) * 100
right_percent = (right_total / total) * 100

print(Fore.CYAN + f"\n🧾 Total samples processed for testing: {total}")
print(Fore.GREEN + f"✅ Correct predictions: {right_total} ({right_percent:.2f}%)")
print(Fore.RED + f"❌ Wrong predictions: {wrong_total} ({wrong_percent:.2f}%)")
print(Fore.RED + f"   └─ Normal wrongly predicted as Anomaly (FP): {fp}")
print(Fore.RED + f"   └─ Anomaly wrongly predicted as Normal (FN): {fn}")
print(Fore.CYAN + f"\n🎯 Accuracy: {accuracy * 100:.2f}%")
print(Fore.GREEN + f"F1 Score: {f1:.4f}")
print(Fore.MAGENTA + "\n📋 Classification Report:\n" +
      classification_report(y_true, y_pred, target_names=['normal', 'anomaly']))

# Ask to show plot
user_input = input(Fore.YELLOW + "\n📈 Do you want to view the confusion matrix plot? (yes/no): ").strip().lower()
if user_input in ['yes', 'y']:
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Normal', 'Predicted Anomaly'],
                yticklabels=['Actual Normal', 'Actual Anomaly'])

    stats_text = (
        f"Total: {total}    Correct: {right_total} ({right_percent:.2f}%)\n"
        f"Wrong: {wrong_total} ({wrong_percent:.2f}%)\n"
        f"Normal → Anomaly (FP): {fp}   Anomaly → Normal (FN): {fn}\n"
        f"True Positives (TP): {tp}   True Negatives (TN): {tn}\n"
        f"Accuracy: {accuracy * 100:.2f}%    F1 Score: {f1:.4f}"
    )
    plt.title("Confusion Matrix with Explanation", fontsize=14)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.text(2.1, -0.6, stats_text, fontsize=10, va='top', ha='left', linespacing=1.5)
    plt.tight_layout()
    plt.savefig("confusion_matrix_plot_with_explanation.png")
    plt.show()
else:
    print(Fore.CYAN + "🛑 Plot display skipped.")


'''
#-----------------------total code --------------------

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from colorama import Fore, Style, init
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize colorama
init(autoreset=True)

# Feature list
features = [
    'Flow_Duration', 'Total_Fwd_Packets', 'Total_Backward_Packets',
    'Fwd_Packet_Length_Max', 'Bwd_Packet_Length_Max',
    'Flow_Bytes_s', 'Flow_Packets_s',
    'Fwd_IAT_Mean', 'Bwd_IAT_Mean', 'Packet_Length_Variance',
    'Average_Packet_Size'
]

# Filenames
sample_file = 'sample.csv'
labeled_file = 'classified_traffic_full.csv'
model_file = 'rf_anomaly_model.pkl'

# Step 1: Check if labeled data exists; if not, create it using Isolation Forest
if not os.path.exists(labeled_file):
    print(Fore.CYAN + "🔄 Labeled data not found. Creating labeled data using Isolation Forest...")

    df_sample = pd.read_csv(sample_file)
    df_sample.columns = df_sample.columns.str.strip().str.replace(' ', '_').str.replace('/', '_')

    # Select features and clean
    df_features = df_sample[features].replace([np.inf, -np.inf], np.nan).dropna()
    idx_valid = df_features.index

    # Train Isolation Forest on features
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    labels = iso_forest.fit_predict(df_features)

    # Map labels: 1 (normal), -1 (anomaly) -> normal= 'normal', anomaly= 'anomaly'
    label_map = {1: 'normal', -1: 'anomaly'}
    df_sample.loc[idx_valid, 'class'] = pd.Series(labels, index=idx_valid).map(label_map)

    # Save labeled data (including unlabeled rows without class)
    df_sample.to_csv(labeled_file, index=False)
    print(Fore.GREEN + f"✅ Labeled data saved to '{labeled_file}'")
else:
    print(Fore.GREEN + f"✅ Labeled data '{labeled_file}' found, skipping labeling step.")

# Step 2: Train model if not exists
if not os.path.exists(model_file):
    print(Fore.CYAN + "🔄 Model not found. Training a new Random Forest model...")

    df_labeled = pd.read_csv(labeled_file)
    df_labeled.columns = df_labeled.columns.str.strip().str.replace(' ', '_').str.replace('/', '_')

    # Filter rows where class is not null
    df_labeled = df_labeled.dropna(subset=['class'])
    df_labeled['class'] = df_labeled['class'].map({'normal': 0, 'anomaly': 1})

    X_train = df_labeled[features].replace([np.inf, -np.inf], np.nan).dropna()
    y_train = df_labeled.loc[X_train.index, 'class']

    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)

    joblib.dump(clf, model_file)
    print(Fore.GREEN + f"✅ Model trained and saved as '{model_file}'")
else:
    clf = joblib.load(model_file)
    print(Fore.GREEN + f"✅ Model loaded from '{model_file}'")

# Step 3: Predict on sample.csv
df_test = pd.read_csv(sample_file)
df_test.columns = df_test.columns.str.strip().str.replace(' ', '_').str.replace('/', '_')
X_test = df_test[features].replace([np.inf, -np.inf], np.nan).dropna()

y_pred = clf.predict(X_test)
df_pred = df_test.loc[X_test.index].copy()
df_pred.insert(0, 'lid', range(1, len(df_pred) + 1))
df_pred['predicted_class'] = pd.Series(y_pred).map({0: 'normal', 1: 'anomaly'})
df_pred.to_csv('test_output.csv', index=False)
print(Fore.YELLOW + "✅ Predictions saved to 'test_output.csv'")

# Step 4: Evaluate
df_labeled = pd.read_csv(labeled_file)
df_labeled.columns = df_labeled.columns.str.strip().str.replace(' ', '_').str.replace('/', '_')
df_true = df_labeled[['lid', 'class']] if 'lid' in df_labeled.columns else df_labeled.copy()
if 'lid' not in df_true.columns:
    df_true = df_true.reset_index().rename(columns={'index': 'lid'})

df_pred = df_pred[['lid', 'predicted_class']].dropna()
df_true = df_true.dropna(subset=['class'])

df_result = pd.merge(df_pred, df_true, on='lid', how='inner')

df_result['true_class'] = df_result['class'].map({'normal': 0, 'anomaly': 1})
df_result['predicted_class'] = df_result['predicted_class'].map({'normal': 0, 'anomaly': 1})

df_result = df_result.dropna(subset=['true_class', 'predicted_class'])

y_true = df_result['true_class'].astype(int)
y_pred = df_result['predicted_class'].astype(int)

accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()

total = tn + tp + fp + fn
right_total = tn + tp
wrong_total = fp + fn
wrong_percent = (wrong_total / total) * 100
right_percent = (right_total / total) * 100

print(Fore.CYAN + f"\n🧾 Total samples processed for testing: {total}")
print(Fore.GREEN + f"✅ Correct predictions: {right_total} ({right_percent:.2f}%)")
print(Fore.RED + f"❌ Wrong predictions: {wrong_total} ({wrong_percent:.2f}%)")
print(Fore.RED + f"   └─ Normal wrongly predicted as Anomaly (FP): {fp}")
print(Fore.RED + f"   └─ Anomaly wrongly predicted as Normal (FN): {fn}")
print(Fore.CYAN + f"\n🎯 Accuracy: {accuracy * 100:.2f}%")
print(Fore.GREEN + f"F1 Score: {f1:.4f}")
print(Fore.MAGENTA + "\n📋 Classification Report:\n" +
      classification_report(y_true, y_pred, target_names=['normal', 'anomaly']))

# Step 5: Show confusion matrix plot optionally
user_input = input(Fore.YELLOW + "\n📈 Do you want to view the confusion matrix plot? (yes/no): ").strip().lower()
if user_input in ['yes', 'y']:
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Normal', 'Predicted Anomaly'],
                yticklabels=['Actual Normal', 'Actual Anomaly'])
    stats_text = (
        f"Total: {total}    Correct: {right_total} ({right_percent:.2f}%)\n"
        f"Wrong: {wrong_total} ({wrong_percent:.2f}%)\n"
        f"Normal → Anomaly (FP): {fp}   Anomaly → Normal (FN): {fn}\n"
        f"True Positives (TP): {tp}   True Negatives (TN): {tn}\n"
        f"Accuracy: {accuracy * 100:.2f}%    F1 Score: {f1:.4f}"
    )
    plt.title("Confusion Matrix with Explanation", fontsize=14)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.text(2.1, -0.6, stats_text, fontsize=10, va='top', ha='left', linespacing=1.5)
    plt.tight_layout()
    plt.savefig("confusion_matrix_plot_with_explanation.png")
    plt.show()
else:
    print(Fore.CYAN + "🛑 Plot display skipped.")
