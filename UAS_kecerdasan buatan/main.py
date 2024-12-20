# Import library
import pandas as pd
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
dataset_path = "UAS_kecerdasan buatan/dataset.json"  # Ganti sesuai lokasi file
with open(dataset_path, 'r') as file:
    data = json.load(file)

# Konversi data ke DataFrame
df = pd.DataFrame(data)

# Encode kategori ke numerik
df['Pekerjaan_Encoded'] = df['Pekerjaan'].astype('category').cat.codes
df['Label_Encoded'] = df['Label'].map({'Tinggi': 1, 'Rendah': 0})

# Definisi fitur (X) dan label (y)
X = df[['Pekerjaan_Encoded', 'Penghasilan']]
y = df['Label_Encoded']

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Output ukuran dataset
print("Training set size:", X_train.shape[0])
print("Validation set size:", X_val.shape[0])
print("Test set size:", X_test.shape[0])

# Inisialisasi dan pelatihan model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluasi Cross-Validation
cross_val_scores = cross_val_score(knn, X, y, cv=5)
print("Cross-Validation Scores:", cross_val_scores)
print("Average Cross-Validation Score:", cross_val_scores.mean())

# Validasi
y_val_pred = knn.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)

# Tes\uji
y_test_pred = knn.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print laporan evaluasi
print("\nValidation Accuracy:", val_accuracy)
print("Test Accuracy:", test_accuracy)
print("\nClassification Report (Test):\n", classification_report(y_test, y_test_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Rendah', 'Tinggi'], yticklabels=['Rendah', 'Tinggi'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plot Akurasi dengan Anotasi
accuracies = {'Validation': val_accuracy, 'Test': test_accuracy}
plt.figure(figsize=(6, 4))
ax = sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette='viridis')
plt.ylim(0, 1)
plt.title('Akurasi Model')

# Anotasi (nilai akurasi)
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=12, color='white', xytext=(0, 10), textcoords='offset points')

plt.ylabel('Akurasi')
plt.show()

# Evaluasi Model
print("\nEvaluasi Model:")
report = classification_report(y_test, y_test_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print(report_df)

# Visualisasi Precision, Recall, F1-Score
metrics = ['precision', 'recall', 'f1-score']
metrics_data = report_df[metrics].iloc[:-1]
metrics_data.plot(kind='bar', figsize=(8, 6), colormap='coolwarm')

plt.title('Precision, Recall, F1-Score for Each Class')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.show()
