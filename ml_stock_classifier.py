import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
df = pd.read_csv(r'C:\Users\HP\Desktop\bi\ml\AAPL.csv')
print("Columns in your CSV:")
print(df.columns.tolist())
df.columns=df.columns.str.strip()
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.sort_index()
df['Vol.'] = df['Vol.'].str.replace('M', 'e6').str.replace('K', 'e3')
df['Vol.'] = df['Vol.'].str.replace(',', '').astype(float)
df['Return'] = df['Price'].pct_change()
df['Sma_5'] = df['Price'].rolling(window=5).mean()
df['Sma_10'] = df['Price'].rolling(window=10).mean()
df['Vol_10'] = df['Return'].rolling(window=10).std()
df['Signal'] = (df['Return'].shift(-1) > 0).astype(int)
df = df.dropna()
X = df[['Sma_5', 'Sma_10', 'Vol_10']]
y = df['Signal']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
dt.fit(x_train, y_train)
rf.fit(x_train, y_train)
y_pred_dt = dt.predict(x_test)
y_pred_rf = rf.predict(x_test)
print(" Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print(" DT Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print(" Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(" RF Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

import matplotlib.pyplot as plt
feature_names = ['Sma_5', 'Sma_10', 'Vol_10']
importances = rf.feature_importances_
plt.figure(figsize=(6, 4))
plt.bar(feature_names, importances)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.grid(True)
plt.tight_layout()
plt.savefig("importance_plot.png")
plt.show()