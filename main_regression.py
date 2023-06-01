import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def calculate_rmse(predicted, true):
    return np.sqrt(mean_squared_error(predicted, true))

data = pd.read_csv('data/train.csv')

label_encoder_zvanje = LabelEncoder()
label_encoder_pol = LabelEncoder()
label_encoder_oblast = LabelEncoder()

data['zvanje_encoded'] = label_encoder_zvanje.fit_transform(data['zvanje'])
data['pol_encoded'] = label_encoder_pol.fit_transform(data['pol'])
data['oblast_encoded'] = label_encoder_oblast.fit_transform(data['oblast'])

X = data[['zvanje_encoded', 'oblast_encoded', 'godina_doktor', 'godina_iskustva', 'pol_encoded']]
y = data['plata']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=55)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

RMSE = calculate_rmse(y_pred, y_test)
print(f'RMSE: {RMSE}')
