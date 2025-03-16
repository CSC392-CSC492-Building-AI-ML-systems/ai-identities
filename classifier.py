import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data and skip the first 2 rows (empty rows and header row)
df = pd.read_csv("Benchmark_Performance.csv", skiprows=2, header=None, names=["Model", "Temperature", "MMLU-Pro CS", "MMLU-Pro Philosophy"])

# Reset the index to remove NaN in the index column
df.reset_index(drop=True, inplace=True)

# Replace '#VALUE!' with NaN
df.replace("#VALUE!", float("nan"), inplace=True)

# Convert columns to numeric (in case they are read as strings)
df["MMLU-Pro CS"] = pd.to_numeric(df["MMLU-Pro CS"], errors="coerce")
df["MMLU-Pro Philosophy"] = pd.to_numeric(df["MMLU-Pro Philosophy"], errors="coerce")

# Option 1: Drop rows with missing values in feature columns
# df.dropna(subset=["MMLU-Pro CS", "MMLU-Pro Philosophy"], inplace=True)

# Option 2: Fill missing values with 0 (or another value)
df.fillna(0, inplace=True)

# Print the first few rows to verify
print("First few rows of the dataset:")
print(df.head())

# Encode Model
le = LabelEncoder()
df["Model"] = le.fit_transform(df["Model"])

# Features (X) and Target (y)
X = df[["Temperature", "MMLU-Pro CS", "MMLU-Pro Philosophy"]]
y = df["Model"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))
