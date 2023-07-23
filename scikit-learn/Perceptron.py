import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

def perceptron_model(filePath, target_column_name): # file csv

	df = pd.read_csv(filePath)

	data = df.drop( target_column_name, axis=1 )
	result = df.loc[ : , [target_column_name]]


	data_train, data_test, result_train, result_test = train_test_split( data, result, test_size=0.3, random_state=1, stratify=result )


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

	acc = accuracy_score(result_test, result_predict)
	print(f'accuracy_score is {acc}')


	### Implement ##############

	# df = pd.read_csv('data/test.csv') 

	# data = df

	# sc = StandardScaler()
	# sc.fit(data)
	# data_std = sc.transform(data)

	# result_predict = ppn.predict(data_test_std)

	# print(result_predict)

if __name__ == '__main__':
	
	perceptron_model('data/train.csv', target_column_name='price_range')






