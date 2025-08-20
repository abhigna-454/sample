# ==========================================
# Network Anomaly Detection - Multi-class
# Classes: Normal, DoS, Probe, R2L, U2R
# ==========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load dataset (NSL-KDD)
column_names = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent",
    "hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
    "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","extra"
]

url_train = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
url_test = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"

train_df = pd.read_csv(url_train, names=column_names)
test_df = pd.read_csv(url_test, names=column_names)

print("Training set shape:", train_df.shape)
print("Testing set shape:", test_df.shape)

# 2. Map attack types into 5 categories
attack_map = {
    'normal': 'Normal',
    # DoS
    'neptune': 'DoS','back': 'DoS','land': 'DoS','pod': 'DoS','smurf': 'DoS','teardrop': 'DoS',
    'mailbomb':'DoS','apache2':'DoS','processtable':'DoS','udpstorm':'DoS',
    # Probe
    'satan':'Probe','ipsweep':'Probe','nmap':'Probe','portsweep':'Probe',
    'mscan':'Probe','saint':'Probe',
    # R2L
    'guess_passwd':'R2L','ftp_write':'R2L','imap':'R2L','phf':'R2L','multihop':'R2L',
    'warezmaster':'R2L','warezclient':'R2L','spy':'R2L','xlock':'R2L','xsnoop':'R2L',
    'snmpguess':'R2L','snmpgetattack':'R2L','httptunnel':'R2L','sendmail':'R2L','named':'R2L',
    # U2R
    'buffer_overflow':'U2R','loadmodule':'U2R','rootkit':'U2R','perl':'U2R',
    'sqlattack':'U2R','xterm':'U2R','ps':'U2R'
}

train_df['label'] = train_df['label'].map(attack_map)
test_df['label'] = test_df['label'].map(attack_map)

# 3. Encode categorical features
cat_cols = ["protocol_type", "service", "flag"]
encoder = LabelEncoder()
for col in cat_cols:
    train_df[col] = encoder.fit_transform(train_df[col])
    test_df[col] = encoder.transform(test_df[col])

# 4. Split features & labels
X_train = train_df.drop(["label","extra"], axis=1)
y_train = train_df["label"]

X_test = test_df.drop(["label","extra"], axis=1)
y_test = test_df["label"]

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Predictions
y_pred = model.predict(X_test)

# 7. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=["Normal","DoS","Probe","R2L","U2R"])
sns.heatmap(cm, annot=True, fmt="d", xticklabels=["Normal","DoS","Probe","R2L","U2R"], 
            yticklabels=["Normal","DoS","Probe","R2L","U2R"], cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# 8. Show some predictions
sample_results = pd.DataFrame({"Actual": y_test[:20].values, "Predicted": y_pred[:20]})
print("\nSample Predictions:\n", sample_results)
