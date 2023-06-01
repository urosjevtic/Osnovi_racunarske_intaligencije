

#Lak zadatak: Izvršiti aproksimaciju verovatnoće da zaposleni napuste kompaniju (atribut churn) na osnovu ukupne količine poziva koje su imali iz inostranstva (atribut total intl minutes). Ispisati verovatnoću odlaska iz kompanije za zaposlenog sa 5 minuta međunarodnih poziva, kao i za zaposlenog sa 60 minuta međunarodnih poziva.

#Srednji zadatak: Izvršiti klasterovanje zaposlenih na osnovu količine telefonskih razgovora sa inostranstvom (atribut total intl minutes) i na osnovu količine telefonskih razgovora u toku dana (atribut total day minutes), tako što ćete napraviti dva klastera. Nakon klasterovanja proveriti koliko zaposlenih iz prvog klastera je napustilo kompaniju, a koliko iz drugog (ispisati procenat ili odnos). Atribut churn pokazuje da li je zaposleni napustio kompaniju ili ne.

#Težak zadatak: Izvršiti predikciju da li će zaposleni napustiti kompaniju na osnovu istorije telefonskih poziva. Predikciju izvršiti na osnovu činjenice da li zaposleni ima pravo na međunarodne pozive (international plan), da li ima govorne poruke (voice mail plan), broja govornih poruka (number vmail messages), broja međunarodnih poziva (total intl calls), broja noćnih poziva (total night calls) i broja dnevnih poziva (total day calls). Atribut churn pokazuje da li je zaposleni napustio kompaniju ili ne. Skup podataka podeliti u odnosu od 70:30 i ispisati procenat tačnosti nad test skupom.






import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('customer_churn.csv')

X = df[['total intl minutes']]
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlp = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', random_state=42)

mlp.fit(X_train, y_train)

# Predict the probabilities for new values
new_X = [[5], [60]]  # New input values
predictions = mlp.predict(new_X)

for i, x in enumerate(new_X):
    print(f"{x[0]} minuta: {predictions[i]}")
    
    
#########################################################################################################3


# Učitavanje podataka iz CSV fajla
df = pd.read_csv('customer_churn.csv')

X = df[['total intl minutes', 'total day minutes']]

# Kreiranje instance KMeans modela sa 2 klastera
kmeans = KMeans(n_clusters=2, random_state=42)

# Klasterovanje podataka
kmeans.fit(X)

# Dodavanje informacija o klasterima u DataFrame
df['cluster'] = kmeans.labels_

# Brojanje zaposlenih koji su napustili kompaniju po klasterima
churn_counts = df.groupby('cluster')['churn'].sum()

# Ukupan broj zaposlenih po klasterima
total_counts = df['cluster'].value_counts()

# Izračunavanje procenata napuštanja kompanije po klasterima
churn_percentages = churn_counts / total_counts * 100

# Ispisivanje rezultata
for cluster, percentage in churn_percentages.items():
    print(f"Procenat napuštanja kompanije: {percentage}%")

# Plotiranje klastera
plt.scatter(X['total intl minutes'], X['total day minutes'], c=df['cluster'])
plt.show()

##############################################################################################################

data = pd.read_csv('customer_churn.csv')

lenc = LabelEncoder()
data['international plan'] = lenc.fit_transform(df['international plan'])
data['voice mail plan'] = lenc.fit_transform(df['voice mail plan'])

# Select the relevant features and target variable
X = data[['international plan', 'voice mail plan', 'number vmail messages', 'total intl calls', 'total night calls', 'total day calls']]
y = data['churn']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create an instance of the MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', random_state=42)

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, predictions)

# Print the accuracy
print("Accuracy on the test set:", accuracy)

