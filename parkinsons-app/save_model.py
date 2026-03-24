import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import pickle
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, '..', 'parkinsons.data')

print("Loading dataset...")
df = pd.read_csv(data_path)

# Prepare Features (X) and Target (Y)
X = df.drop(columns=['name', 'status'], axis=1)
Y = df['status']

print("Training Scaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Training SVM Model...")
# The standard tutorial uses linear kernel for this dataset usually
model = svm.SVC(kernel='linear', random_state=42)
model.fit(X_scaled, Y)

# Accuracy check just to confirm
acc = model.score(X_scaled, Y)
print(f"Model trained! Accuracy on full dataset: {acc*100:.2f}%")

print("Saving model and scaler to parkinsons-app/ directory...")
pickle.dump(model, open(os.path.join(base_dir, 'model.pkl'), 'wb'))
pickle.dump(scaler, open(os.path.join(base_dir, 'scaler.pkl'), 'wb'))

print("✅ Success! You can now run the Flask app.")
