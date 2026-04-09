import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler



# DATA
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

train_len = len(train)
y_train_full = train['Survived']

dataset = pd.concat([train.drop(columns=['Survived']), test], axis=0, ignore_index=True)



# FEATURE ENGINEERING
dataset['Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].mode()[0])
dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())

# Fill age based on Pclass and Sex
dataset['Age'] = dataset.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))

dataset['Title'] = dataset['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
dataset['Title'] = dataset['Title'].replace(rare_titles, 'Rare')
dataset['Title'] = dataset['Title'].replace(['Mlle', 'Ms'], 'Miss')
dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

dataset['Cabin'] = dataset['Cabin'].fillna('U') # Unknown
dataset['Deck'] = dataset['Cabin'].str[0]

dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# Binning
dataset['AgeGroup'] = pd.cut(dataset['Age'], 
                             bins=[0, 12, 20, 40, 60, 120], 
                             labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
dataset['FareGroup'] = pd.qcut(dataset['Fare'], 4, labels=['Low', 'Mid', 'High', 'Very_High'])



# CLEANING AND ENCODING
cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Age', 'Fare']
dataset = dataset.drop(columns=cols_to_drop)

categorical_cols = ['Pclass', 'Sex', 'Embarked', 'Title', 'Deck', 'AgeGroup', 'FareGroup']
dataset = pd.get_dummies(dataset, columns=categorical_cols, drop_first=True)

X_train_full = dataset[:train_len].copy()
X_test_real = dataset[train_len:].copy()



# SCALING
scaler = StandardScaler()
X_train_full_scaled = scaler.fit_transform(X_train_full)
X_test_real_scaled = scaler.transform(X_test_real)



# CROSS-VALIDATION STRATIFIEDKFOLD
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr_cv = cross_val_score(lr, X_train_full_scaled, y_train_full, cv=skf, scoring='accuracy')

print(f"LR Accuracy: {lr_cv.mean():.4f} (+/- {lr_cv.std():.4f})")

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf_cv = cross_val_score(rf, X_train_full, y_train_full, cv=skf, scoring='accuracy')

print(f"RF Accuracy: {rf_cv.mean():.4f} (+/- {rf_cv.std():.4f})")



# FINAL MODEL CHOOSING
if rf_cv.mean() > lr_cv.mean():
    print("Using Random Forest")
    rf.fit(X_train_full, y_train_full)
    test_predictions = rf.predict(X_test_real)
else:
    print("Using Logistic Regression")
    lr.fit(X_train_full_scaled, y_train_full)
    test_predictions = lr.predict(X_test_real_scaled)
    # Fit RF for feature importance
    rf.fit(X_train_full, y_train_full)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_predictions.astype(int)
})
submission.to_csv("submission.csv", index=False)



# FEATURE IMPORTANCE
importances = rf.feature_importances_
features = X_train_full.columns

feature_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_df = feature_df.sort_values(by='Importance', ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(feature_df['Feature'], feature_df['Importance'])
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
