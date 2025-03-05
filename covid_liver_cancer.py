import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("covid.csv", encoding="latin1")

# Select relevant columns
data = df[["Alive_Dead", "Age", "Gender", "Survival_fromMDM", "Prev_known_cirrhosis", "Treatment_grps"]]

# Make a copy to avoid CopyWarning
data = data.copy()

# Handling missing values using mode
data["Prev_known_cirrhosis"] = data["Prev_known_cirrhosis"].fillna(data["Prev_known_cirrhosis"].mode()[0])
data["Treatment_grps"] = data["Treatment_grps"].fillna(data["Treatment_grps"].mode()[0])

# Convert categorical variables to numerical (One-Hot Encoding)
data = pd.get_dummies(data, columns=['Alive_Dead', 'Gender', 'Prev_known_cirrhosis', 'Treatment_grps'], drop_first=True)

# Convert boolean columns to integers
for col in data.select_dtypes(include=["bool"]).columns:
    data[col] = data[col].astype(int)

# Convert all data to numeric to avoid dtype errors
data = data.apply(pd.to_numeric)

# Display updated column names
print("Updated Column Names:", data.columns)

# Visualizations 
# Boxplot for Age
plt.figure(figsize=(6, 4))
sns.boxplot(x=data['Age'])
plt.title("Boxplot for Age")
plt.show()

# Boxplot for Survival from MDM
plt.figure(figsize=(6, 4))
sns.boxplot(x=data["Survival_fromMDM"])
plt.title("Boxplot for Survival from MDM")
plt.show()

# Countplot for Alive_Dead
plt.figure(figsize=(6, 4))
sns.countplot(x=data["Alive_Dead_Dead"])
plt.title("Countplot for Alive or Dead")
plt.show()

# Countplot for Gender
plt.figure(figsize=(6, 4))
sns.countplot(x=data["Gender_M"])
plt.title("Countplot for Gender")
plt.show()

# Countplot for Prev_known_cirrhosis
plt.figure(figsize=(6, 4))
sns.countplot(x=data["Prev_known_cirrhosis_Y"])
plt.title("Countplot for Prev_known_cirrhosis")
plt.show()

# Pairplot to see the relationships between variables
sns.pairplot(data)
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 3), dpi=200)
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Define independent (X) and dependent (Y) variables
x = data.drop('Alive_Dead_Dead', axis=1)
y = data['Alive_Dead_Dead']

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=40)

# Check if Y is binary
print("Unique values in Y_train:", y_train.unique())

# Models and Hyperparameter Tuning

# 1. Logistic Regression
log_reg_model = LogisticRegression(max_iter=1000)
log_reg_model.fit(x_train, y_train)
log_reg_preds = log_reg_model.predict(x_test)
log_reg_accuracy = accuracy_score(y_test, log_reg_preds)
print(f"Logistic Regression Accuracy: {log_reg_accuracy}")

# 2. KNN with GridSearchCV
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'metric': ['euclidean', 'manhattan']
}
knn = KNeighborsClassifier()
grid_search_knn = GridSearchCV(estimator=knn, param_grid=param_grid_knn, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_knn.fit(x_train, y_train)
best_knn = grid_search_knn.best_estimator_
knn_preds = best_knn.predict(x_test)
knn_accuracy = accuracy_score(y_test, knn_preds)
print(f"KNN Accuracy: {knn_accuracy}")
print(f"Best KNN Hyperparameters: {grid_search_knn.best_params_}")

# 3. Decision Tree with GridSearchCV
param_grid_dt = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}
dt = DecisionTreeClassifier()
grid_search_dt = GridSearchCV(estimator=dt, param_grid=param_grid_dt, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_dt.fit(x_train, y_train)
best_dt = grid_search_dt.best_estimator_
dt_preds = best_dt.predict(x_test)
dt_accuracy = accuracy_score(y_test, dt_preds)
print(f"Decision Tree Accuracy: {dt_accuracy}")
print(f"Best Decision Tree Hyperparameters: {grid_search_dt.best_params_}")

# 4. Random Forest with GridSearchCV
param_grid_rf = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5],
    'criterion': ['gini', 'entropy']
}
rf = RandomForestClassifier()
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_rf.fit(x_train, y_train)
best_rf = grid_search_rf.best_estimator_
rf_preds = best_rf.predict(x_test)
rf_accuracy = accuracy_score(y_test, rf_preds)
print(f"Random Forest Accuracy: {rf_accuracy}")
print(f"Best Random Forest Hyperparameters: {grid_search_rf.best_params_}")

# 5. Support Vector Machine (SVM) with GridSearchCV
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
svm = SVC()
grid_search_svm = GridSearchCV(estimator=svm, param_grid=param_grid_svm, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_svm.fit(x_train, y_train)
best_svm = grid_search_svm.best_estimator_
svm_preds = best_svm.predict(x_test)
svm_accuracy = accuracy_score(y_test, svm_preds)
print(f"SVM Accuracy: {svm_accuracy}")
print(f"Best SVM Hyperparameters: {grid_search_svm.best_params_}")

# Summary of all models
print("\nModel Comparison:")
print(f"Logistic Regression Accuracy: {log_reg_accuracy}")
print(f"KNN Accuracy: {knn_accuracy}")
print(f"Decision Tree Accuracy: {dt_accuracy}")
print(f"Random Forest Accuracy: {rf_accuracy}")
print(f"SVM Accuracy: {svm_accuracy}")

# Print Classification Report
print("Decision Tree Classification Report:")
print(classification_report(y_test, dt_preds))
