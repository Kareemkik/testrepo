import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures 
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split

filepath = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/medical_insurance_dataset.csv"
df = pd.read_csv(filepath, header=None)
df.head(10)
headers= ["age", "gender","bmi","no_of_children","smoker","region","charges"]
df.columns= headers # assign of headers 
print(df.head(10))
df.replace('?',np.nan,inplace=True) # cleaning data 
print(df.info()) # getting info to apply data wrangling 
idsmoker = df['smoker'].value_counts().idxmax()
df["smoker"].replace(np.nan,idsmoker,inplace=True) # replace of missing values for smoker with the most repeated value 
mean_age=df['age'].astype(float).mean(axis=0)
df["age"].replace(np.nan, mean_age, inplace=True) # replace of missing value of age with the mean of ages 
df[["age","smoker"]]=df[["age","smoker"]].astype("int") #changes data type to integer 
print(df.info())
df["charges"]=np.round(df[["charges"]],2) #rounding charges coulmn to be 2 decimals only
print(df.head())
sns.regplot(x="bmi",y="charges",data=df, line_kws={"color": "red"}) #reg plot for charges referenced by BMI
plt.ylim(0,)
sns.boxplot(x="smoker",y="charges",data=df) #box plot for charges by smoker 
print(df.corr()) # getting corr between all input values or all data set
#Linear regression using one input 
lr=LinearRegression()
lr.fit(df[['smoker']],df['charges'])
print(lr.score(df[['smoker']],df['charges'])) #calculating R2
#Linear regression using all valid inputs except our predicted data which is charges 
lre=LinearRegression()
x_values=df[['smoker','age','bmi','no_of_children','gender','region']] #variable for all values 
y_data=df['charges'] #variable for charges which is compared with predicted data from each model 
lre.fit(x_values,y_data) # fitting model using both variables 
rsq=lre.score(x_values,y_data) #calculating R2
print(rsq)
# creating a pipline with standardscaler, polynomialfeature and linear regression 
Input=[('scale',StandardScaler()),('polynomial',PolynomialFeatures(include_bias=False)),('model',LinearRegression())] #creating input for pipeline
pipe=Pipeline(Input)
x_values= x_values.astype(float)
pipe.fit(x_values,y_data) #fiiting our variables using pipeline
yhat_pipe=pipe.predict(x_values) # saving predicted values to a variable
print(r2_score(y_data,yhat_pipe)) # calculating R2 of Pipeline model

x_train,x_test,y_train,y_test=train_test_split(x_values,y_data,test_size=0.2,random_state=1) # splitting data to train and test data with ratio 20% for test data 
Rmodel1=Ridge(alpha=0.1) #creating ridge regression with alpha value = 0.1 
Rmodel1.fit(x_train,y_train) #fitting the model
yhat_Rmodel1= Rmodel1.predict(x_test) #getting the predicted values ith respect to test_data 
print(r2_score(y_test,yhat_Rmodel1)) #calculating R2 

pr=PolynomialFeatures(degree=2) #adding polynomial degree to our Ridge regssion model for enhancement 
x_train_pr=pr.fit_transform(x_train) #transform of the train data to polynomial degree 2
x_test_pr=pr.fit_transform(x_test) #transform of the test data to polynomial degree 2
Rmodel1.fit(x_train_pr,y_train) #fitting data after pr
yhat_Rmodel1_pr=Rmodel1.predict(x_test_pr) #prediction with respect to test_data after pr
print(r2_score(y_test,yhat_Rmodel1_pr)) #calculating R2 


#we could increase eff of the last model by getting the best value at both alpha and polynomial regression degree, Also by increasing the test data % may get better results
