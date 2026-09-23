import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.feature_selection import RFE

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
model = LinearRegression()
model.fit(X, y)
print('diabetes_r2', r2_score(y, model.predict(X)))

# auto CSV workflow
auto = pd.read_csv('auto-mpg.csv')
print('auto_shape', auto.shape)
print('columns', auto.columns.tolist())
auto['horse_power'] = pd.to_numeric(auto['horse_power'], errors='coerce')
auto = auto.dropna(axis=0, how='any').copy()
print('after_drop_shape', auto.shape)
print('min_max_model_year', auto['model_year'].min(), auto['model_year'].max())
print('cylinders', auto['cylinders'].value_counts().sort_index().to_dict())

X_auto = auto.drop(columns=['car_name', 'mpg'])
y_auto = auto['mpg']
X_train, X_test, y_train, y_test = train_test_split(X_auto, y_auto, test_size=0.2, random_state=42)
auto_model = LinearRegression(); auto_model.fit(X_train, y_train)
print('auto_train_r2', r2_score(y_train, auto_model.predict(X_train)))
print('auto_test_r2', r2_score(y_test, auto_model.predict(X_test)))

X_train09, X_test09, y_train09, y_test09 = train_test_split(X_auto, y_auto, test_size=0.1, random_state=42)
auto_model09 = LinearRegression(); auto_model09.fit(X_train09, y_train09)
print('auto09_test_r2', r2_score(y_test09, auto_model09.predict(X_test09)))

rfe = RFE(estimator=auto_model, n_features_to_select=3)
rfe.fit(X_train, y_train)
print('ranking', rfe.ranking_.tolist())
reduced_features = X_train.columns[rfe.support_].tolist()
print('reduced_features', reduced_features)
X_reduced = X_auto[reduced_features]
X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(X_reduced, y_auto, test_size=0.2, random_state=42)
auto_model_reduced = LinearRegression(); auto_model_reduced.fit(X_train_reduced, y_train_reduced)
print('reduced_test_r2', r2_score(y_test_reduced, auto_model_reduced.predict(X_test_reduced)))
print('DONE')
