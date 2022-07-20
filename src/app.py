## Pipeline para modelo del ejercicio (sólo código indispensable)

# importo librerias
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
import statsmodels.formula.api as smf
from sklearn.feature_selection import RFECV
import pickle

# datos
df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv')

# separo en muestra de entrenamiento y evaluación para aplicar EDA solamente a entrenamiento
X = df_raw.drop(columns = 'charges', axis = 1)
y = df_raw['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1807)

# defino datset de entrenamiento (a aplicar EDA)
df_train = pd.concat([X_train, y_train], axis = 1)

# creo nueva variable
df_train['children_gr'] = pd.cut(df_train['children'], [-1,0,1,5], labels = ['0', '1', '2_o_más'])

# vuelvo a definir X e y para actualizar variables creadas
X_train = df_train.drop(columns = ['children', 'charges'], axis = 1) # saco target y numérica de hijos, dejo la categórica
y_train = df_train['charges']

# nuevos data frames
X_train2 = X_train.drop(columns = ['sex', 'region'])
X_test2 = X_test.drop(columns = ['sex', 'region'])

# pipeline para preprocesamiento
cat_cols2 = X_train2.select_dtypes(include='category').columns
num_cols2 = X_train2.select_dtypes(include='number').columns
preprocessor2 = ColumnTransformer(transformers=[('num', num_transformer, num_cols2), ('cat', cat_transformer, cat_cols2)])
encode_data2 = Pipeline(steps=[('preprocessor', preprocessor2)]) # categorias en n-1 columnas

# ajusto nuevo modelo de regresión lineal sin variables sexo ni region
lreg2 = Pipeline(steps=[('preprocessor', preprocessor2), ('regressor', LinearRegression())])
lreg2.fit(X_train2, y_train)
print(f'R^2 score:{lreg2.score(X_train2, y_train)}')

# métricas
y_pred2 = lreg2.predict(X_test2)
print(f'R^2 score:{r2_score(y_test, y_pred2)}')
print(f'MSE score:{mean_squared_error(y_test, y_pred2)}')
print(f'RMSE score:{np.sqrt(mean_squared_error(y_test, y_pred2))}')

# se guarda modelo elegido
filename = '../models/finalized_model.sav'
pickle.dump(lreg2, open(filename, 'wb'))