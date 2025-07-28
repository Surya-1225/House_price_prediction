# Setting up the libraries

# import numpy as np                                             # for handle numerical values
import pandas as pd                                              # for data manipulation or we can say data analysis
import matplotlib.pyplot as plt                                  # graph
import seaborn as sb                                             # top of matplotlib and use to provide more visualization
from sklearn.model_selection import train_test_split             # To split the dataset into the training and testing sets
from sklearn.linear_model import LinearRegression                # its an model for pridiction
from sklearn.metrics import mean_squared_error                   # to evaluate the performance of the model
from sklearn.datasets import fetch_california_housing            # dataset




# Loading dataset
california=fetch_california_housing()
data=pd.DataFrame(california.data, columns=california.feature_names)
data['PRICE']=california.target


# sb.pairplot(data)
# plt.show()




#Prepairing the Data
X = data.drop('PRICE', axis=1)
y = data['PRICE']
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Training model 
model=LinearRegression() # create an instance based on the definition in linear regression 
model.fit(X_train, Y_train)

# pridicting on test data
Y_pred=model.predict(X_test)





# evaluating the model
mse=mean_squared_error(Y_test,Y_pred)
print(f'the mean squared error is {mse}')




# print(data)



# Calculating correlation

correlation_matrix=data.corr().round(2)
sb.heatmap(data=correlation_matrix, annot=True)
plt.show()

#Graph between actual value and predicition value 
plt.scatter(Y_test, Y_pred, color='red', label='Predicted Prices')
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], color='blue', linestyle='--', label='Ideal Fit')
plt.xlabel("actual value")
plt.ylabel("predicted value",)
plt.title("actual vs predicted")
plt.legend()
plt.show()
print()
comparison_df = pd.DataFrame({'Actual Value': Y_test, 'Predicted Value': Y_pred})
print(comparison_df.head(10)) 
