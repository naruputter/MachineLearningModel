import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/train.csv')

data = df.drop('price_range', axis=1)
result = df.price_range

# print(data)
# print(result)

data_train, data_test, result_train, result_test = train_test_split( data, result, test_size=0.3, random_state=1, stratify=result )

# print(data_test)
# print(result_test)


### PreProcess ###############

sc = StandardScaler()
sc.fit(data_train)
data_train_std = sc.transform(data_train)
data_test_std = sc.transform(data_test)

### Model ###################

ppn = Perceptron(eta0=0.1, random_state=1) # eta0 is learning rate
ppn.fit(data_train_std, result_train)

### Estimate ################

result_predict = ppn.predict(data_test_std)
# print( (result_test != result_predict).sum())

accuracy_score = accuracy_score(result_test, result_predict)
print(f'accuracy_score is {accuracy_score}')


### Implement ##############

df = pd.read_csv('data/test.csv') 

data = df

sc = StandardScaler()
sc.fit(data)
data_std = sc.transform(data)

result_predict = ppn.predict(data_test_std)

print(result_predict)






