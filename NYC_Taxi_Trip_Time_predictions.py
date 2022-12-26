#!/usr/bin/env python
# coding: utf-8

# # Project  : NYC Taxi trip duration Prediction :: Predicting total ride duration of taxi trips in NYC using the given Dataset

# # Mounting of Drive, Loading Data and Importing of the required libraries
# 

# In[1]:


# Importing necessary Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split,GridSearchCV
import statsmodels.formula.api as sm
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import datetime as dt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings; warnings.simplefilter('ignore')


# In[13]:


#Fetching the given dataet using pandas
taxi_data = pd.read_csv(r'C:\Users\bysub\Desktop\NYC Taxi Data.csv')


# #Details of The Data

# In[14]:


#Importing first 10 indexes of the taxi_data
taxi_data.head(10)


# In[15]:


#Disply last 10 indexes of the dataset
taxi_data.tail(10)


# In[16]:


#Total number of Rows and Columns in the given dataset
#shape of given dataframe
taxi_data.shape


# ###Number of rows is:  **1458644** 
# ###Number of columns is:  **11**

# In[17]:


# Variables present in the given dataset
taxi_data.columns


# In[18]:


#Datatypes of Variables present in data
taxi_data.dtypes


# The variables such as pickup_datetime, dropoff_datetime of the type 'object'.
# 
# Converting them into type 'datetime'.

# In[19]:


#  NULL/NAN values in given dataset
taxi_data.isnull().sum()


# NULL/NAN values are not present in given dataset.

# In[20]:


#  Converting them into 'datetime'
taxi_data['pickup_datetime'] = pd.to_datetime(taxi_data['pickup_datetime'])
taxi_data['dropoff_datetime'] = pd.to_datetime(taxi_data['dropoff_datetime'])

taxi_data.head()


# # Creation of Variables

# In[21]:


#  Creation of new Variables
taxi_data['pickup_day'] = taxi_data['pickup_datetime'].dt.day_name()
taxi_data['dropoff_day'] = taxi_data['dropoff_datetime'].dt.day_name()
taxi_data['Pickday_No'] = taxi_data['pickup_datetime'].dt.weekday
taxi_data['Hour_of_Pickup'] = taxi_data['pickup_datetime'].dt.hour
taxi_data['pickup_month'] = taxi_data['pickup_datetime'].dt.month


# The subsequent variables have been made.
# *   pickup_day : It includes the day when the ride was taken, as well as its name.
# *   Pickday_No : it contains the day number rather than characters with Monday = 0 and Sunday = 6.
# 
# *   Hour_of_Pickup : It includes the day in 24-hour format..
# *   pickup_month : Contains the month number, that is, January = 1 and December = 12.

# In[ ]:


# Check out the locations with the most bookings using our maps.
import folium
fol = folium.Figure(width = 1500, height = 500)
mapa = folium.Map(location= (40.7679, -73.9822), zoom_start=11).add_to(fol)

for index, row in taxi_data.sample(1000).iterrows():
  folium.Marker([row['pickup_latitude'],row['pickup_longitude']], icon = folium.Icon(color='purple')).add_to(mapa)
  folium.Marker([row['dropoff_latitude'],row['dropoff_longitude']],icon = folium.Icon(color='red')).add_to(mapa)
display(mapa)


# But we are unable to make any inferences or conclusions from that. Therefore, we shall extract the variable "distance" from this.

# In[43]:


get_ipython().system('pip install geopy')


# In[44]:


#  We will estimate the distance using geographic coordinates and the library.
from geopy.distance import great_circle
def cal_distance(pickup_lat,pickup_long,dropoff_lat,dropoff_long):
  start_coordinates = (pickup_lat,pickup_long)
  stop_coordinates = (dropoff_lat,dropoff_long)
  return great_circle(start_coordinates,stop_coordinates).km

# Applying the above details and creating the feature 'distance'
taxi_data['distance'] = taxi_data.apply(lambda x: cal_distance
                              (x['pickup_latitude'],x['pickup_longitude'],
                              x['dropoff_latitude'],x['dropoff_longitude']),
                              axis = 1)

# Calculation of SPEED in km per Hour
taxi_data['speed'] = (taxi_data.distance/(taxi_data.trip_duration/3600))


# Create a slots of time in a day, To identify what time of day the ride was taken.
# 
# Creating the four slots of time:-
# 
# 1.  Morning ( 6:00 am to 11:59 pm),
# 
# 2.  Afternoon ( 12 noon to 3:59 pm),
# 
# 3.  Evening ( 4:00 pm to 9:59 pm), and
# 
# 4.  Late Night ( 10:00 pm to 5:59 am)

# In[45]:


#Creating the time slots for the day
def time_in_day(x):
  if x in range(6,12):
    return 'Morning'
  elif x in range(12,16):
    return 'Afternoon'
  elif x in range(16,22):
    return 'Evening'
  else:
    return 'Late Night'

# Now using above function create new columns in the dataset
taxi_data['Pickup_Time'] = taxi_data['Hour_of_Pickup'].apply(time_in_day)

# Dataset description after creating a new Variable
taxi_data.head()


# # Univariate Analysis

# Let's look at the target variable, trip_duration.
# We will plot the graph of it because it might contain some outliers.

# In[25]:


# 'trip_duration' is a dependent variable
plt.figure(figsize=(10,6))
sns.distplot((taxi_data['trip_duration']), color = 'green')
plt.xlabel('Trip duration')
plt.show()


# In[26]:


# This shows right skewness, hence apply the log10 to transform it to the normal distribution
plt.figure(figsize=(10,6))
sns.distplot(np.log10(taxi_data['trip_duration']), color = 'orange')
plt.xlabel('Trip Duration')
plt.show()


# The distribution of trip duration has been observed to be normal.

# Plot the boxplot and check for Outliers

# In[27]:


plt.figure(figsize=(10,6))
sns.boxplot(taxi_data['trip_duration'])
plt.xlabel('Trip duration')
plt.show()


# There are outliers and they should be eliminated to ensure uniformity in the data.

# In[28]:


# checking for outliers and elinimating them
# Calculating 0-100 percentile to detect a correct percentile value for the removal of outlier
for i in range (0,100,10):
  duration = taxi_data['trip_duration'].values
  duration = np.sort(duration, axis = None)
  print("{} percentile value is {}".format(i, duration[int(len(duration)*(float(i)/100))]))
print("100 percentile value is ",duration[-1])  


# In[29]:


# some inconsistancy has been obseved in 90-100,
# we willfurther dig deep into 90-100 percentile to analyse the data and check for the outliers in order to overcome

for i in range(90,100):
  duration = taxi_data['trip_duration'].values
  duration = np.sort(duration, axis = None)
  print("{} percentile value is {} ".format(i, duration[int(len(duration)*(float(i)/100))]))
print("100 percentile value is ",duration[-1])


# Visualization of Number of trips taken with respect to trip duration

# In[30]:


# Visualisation of number of trips taken with respect to trip duration
plt.figure(figsize=(10,6))
taxi_data.trip_duration.groupby(pd.cut(taxi_data.trip_duration, np.arange(1,7200,600))).count().plot(kind='line',color ='red')
plt.xlabel("Trip duration (seconds)")
plt.ylabel("Trip counts")
plt.show()


# As per the above observation, most of the trip duration is completed in 1 hour(3600 seconds).
# 
# As per the above observation a very few trips have duration more than 5000 seconds and some are with as low as 1 second(0 km distance)

# In[31]:


# to mentain the data consistancy we will remove these outliers
# (Trips with duration more than 5000 seconds and less than 60 seconds)
taxi_data = taxi_data[taxi_data.trip_duration <=5000]
taxi_data = taxi_data[taxi_data.trip_duration >=60]


# In[32]:


# Plotting for insights
plt.figure(figsize = (10,6))
sns.violinplot(taxi_data.trip_duration, color = 'cyan')
plt.xlabel('Trip Duration (seconds)')
plt.show()


# The majority of the trips took 10 to 20 minutes to complete. As seen, the majority of the trips finished in 0 to 30 minutes (1800 seconds)
# Examine other variables as well

# ##Passenger count
# 
# Before analysing the passenger count, we are aware that a booked cab cannot have zero passengers or more than six passengers in a cab, thus it is time to define or eliminate the rows with these values.

# In[33]:


# Assuming the passenger count . If any, removing the rows which have Zero(0) or more than 6 passengers count
taxi_data = taxi_data[taxi_data['passenger_count'] !=0]
taxi_data = taxi_data[taxi_data['passenger_count'] <=6]

#passenger count
plt.figure(figsize=(10,6))
sns.kdeplot(x = 'passenger_count', data = taxi_data, color ='purple', shade =True)
plt.xlabel('Passenger count')
plt.ylabel('Count')
plt.show()


# According to the above observations, it should be highlighted that the majority of trips were taken by a single passenger, and that large groups of people travelling together is less common than single passengers.

# ## Slotwise trips per Day
# 
# 

# In[34]:


# Trips per time slot
plt.figure(figsize=(10,6))
sns.countplot(x='Pickup_Time', data = taxi_data,  palette='dark')
plt.title('Pickup time of the day')
plt.xlabel('Parts of the day')
plt.ylabel('count')
plt.show()


# Evenings are the busiest and top among  the all.

# ## Trips per week day

# In[35]:


# ANalysing the trips per week day
plt.figure(figsize=(10,6))
sns.countplot(taxi_data.pickup_day, palette='bright')
plt.xlabel("Day of pickup")
plt.ylabel("Pickup counts")
plt.show()


# As shown above, Fridays are busiest, followed by Saturday, and this may be due to the weekend.

# ##  Trips per month

# In[36]:


# Analysing the trips per month
plt.figure(figsize=(10,6))
sns.countplot(taxi_data.pickup_month, palette='Set2', saturation = 1.8)
plt.xlabel('Months (Jan=1 to June=6) ')
plt.ylabel('count')
plt.show()


# Not much of a change or no variation between months.

# ## Trips per hour

# In[37]:


# Analyzing the trips per hour
plt.figure(figsize = (10,6))
sns.countplot(taxi_data.Hour_of_Pickup, palette='colorblind')
plt.xlabel('Time of pickup (24hr format)')
plt.show()


# 6:00 pm to 7:00 pm were the busiest hours, which makes sense as this is the time for people to return home from school/work.
# 
# 

# ##Store and Forward Flag

# In[38]:


# Analysing the Store and Forward flag
taxi_data['store_and_fwd_flag'].value_counts(normalize = True)


# According to the aforementioned observation, just 1% of the trip information was saved in the vehicle's memory prior to sending it to the server. This might have happened because of GPS or mobile device issues, a dead battery, or other factors.

# ##Distance

# In[46]:


# Analysing the Distance
taxi_data['distance'].value_counts()


# Let's have a look at the boxplot

# In[47]:


plt.figure(figsize=(10,6))
sns.boxenplot(taxi_data.distance)
plt.xlabel("Distance Travelled")
plt.show()


# *   There are some trips with over 100 km distance and some trips with 0 km distance.
# 
# The possible reasons for zero km trips can be:
# *   The dropoff location couldn’t be tracked.
# *   The passengers or driver cancelled the trip due to some issue or technical issue in software, etc.

# In[48]:


# removing the outliers and updating the values
taxi_data = taxi_data[~(taxi_data.distance > 100)]
taxi_data = taxi_data[~(taxi_data.distance< 1)]

# Plotting the boxen plot
plt.figure(figsize=(10,6))
sns.boxenplot(taxi_data.distance, color= 'yellow')
plt.xlabel("Distance Travelled")
plt.show()


# ## Speed

# In[49]:


# Speed value counts
taxi_data['speed'].value_counts()


# In[50]:


# Largest value of speed
taxi_data['speed'].nlargest(10)


# plotting the boxplot for better understanding

# In[51]:


plt.figure(figsize=(10,6))
sns.boxplot(taxi_data.speed,color = 'orange')
plt.xlabel(" Average Speed")
plt.show()


# *   Some trips were done at a speed of over 100 km/h.
# 
# 

# As per the rule, the speed limit approx. 40km/h in New York City.
# So having average speed of over 60km/h is quite unreasonable.

# In[52]:


# Average speed less than 60
taxi_data= taxi_data[~(taxi_data.speed > 60)]


# In[53]:


#Look at the smallest speed as well
print(taxi_data['speed'].nsmallest(10))


# Some observations showing that speeds which are less than 1 km/hr for a trip which is quite unreasonable.

# In[54]:


#removing the data with less than 1km avg
taxi_data= taxi_data[~(taxi_data.speed < 1)]

# Plotting for boxplot
plt.figure(figsize=(10,6))
sns.boxplot(taxi_data.speed, color ='orange')
plt.xlabel("Average speed")
plt.show()


# ## Speed range distribution with the help of graph

# In[55]:


# Speed range per trip count
plt.figure(figsize=(10,6))
taxi_data.speed.groupby(pd.cut(taxi_data.speed, np.arange(0,104,10))).count().plot(kind= 'bar',color='c')
plt.xlabel("Average Speed")
plt.ylabel("count")
plt.show()


# *   Most of the trips are completed at a speed range of 10-20 km/h.
# 
# 

# ## Vendor Identifier

# In[56]:


# Analyse the Vendor_id variable
plt.figure(figsize=(10,6))
sns.countplot(taxi_data.vendor_id)
plt.xlabel('vendor ID')
plt.ylabel("count")
plt.show()


# As shown above, there are not many differences between the trips taken by the two vendors.

# # Bivariate Analysis

# ## Trip duration per month

# In[57]:


# Analysing trip duration month
plt.figure(figsize=(10,6))
sns.lineplot(x='pickup_month', y ='trip_duration', data = taxi_data, color = 'purple')
plt.xlabel("month of Trip")
plt.ylabel("Duration (seconds)")
plt.show()


# *   From February, we can see trip duration rising every month.
# *   There might be some seasonal parameters like wind/rain which can be a factor of this gradual increase in trip duration over a period. 
# 
# 

# ## Trip duration per weekday

# In[58]:


plt.figure(figsize = (10,6))
sns.lineplot(x='Pickday_No',y='trip_duration',data=taxi_data, color = 'violet')
plt.xlabel('')
plt.ylabel('Duration (seconds)')
plt.show()


# *   Trip duration on thursday is longest among all days.

# ## Trip Duration per hour

# In[59]:


# Plotting for hour of pickup, trip duration
plt.figure(figsize = (10,6))
sns.lineplot(x='Hour_of_Pickup',y='trip_duration',data=taxi_data)
plt.xlabel('Time of Pickup (24hr format)')
plt.ylabel('Duration (seconds)')
plt.show()


# *   As per observation, trip duration is the maximum around 3 pm,traffic might be the reason.
# *   Around 6 am trip duration is the lowest as the streets may not be busy.

# ## Trip Duration per Vendor

# In[60]:


#  Analysing trip duration per Vendor
plt.figure(figsize=(10,6))
sns.barplot(y='trip_duration', x = 'vendor_id', data = taxi_data, estimator=np.mean, palette='Set2')
plt.xlabel("Vendor ID")
plt.ylabel("Trip Duration")
plt.show()


# Compared to vendor id 1, vendor id 2 takes  longer trips

# ## Trip Duration per Store and Forward Flag

# In[61]:


# Analysing trip duration per Store and Forward Flag
plt.figure(figsize=(10,6))
sns.catplot(x='store_and_fwd_flag', y ='trip_duration', data = taxi_data, kind = 'strip')
plt.xlabel("store_and_fwd_flag")
plt.ylabel("Duration (seconds)")
plt.show()


# There is not much difference between N and Y

# ## Distance and Month

# In[62]:


# Distance per month
plt.figure(figsize=(10,6))
sns.lineplot(x= 'pickup_month', y = 'distance', data= taxi_data, color='lime')
plt.xlabel("Month")
plt.ylabel("Distance")
plt.show()


# Trip distance is lowest in 2nd month and maximum in 5th month.
# 
# 

# ## Distance per Weekday

# In[64]:


# Analysing Distance per week day
plt.figure(figsize=(10,6))
sns.lineplot(x = 'Pickday_No', y = "distance", data = taxi_data, color = 'Orange')
plt.xlabel("Pickup Day")
plt.ylabel("Distance")
plt.show()


# Fairly distributed with avg distance of 3.5km hour and sunday being the top, outstation trips or weekend trips may be the reason.

# ## Distance and Hour

# In[65]:


# Plotting for distance to Hour
plt.figure(figsize=(10,6))
sns.lineplot(x ='Hour_of_Pickup', y = 'distance', data=taxi_data, color = 'lime')
plt.xlabel("Pickup Hour")
plt.ylabel("Distance")
plt.show()


# It is fairly equal from morning till the evening varying around 3 - 3.5 kms.
# Trip distance is highest during early morning hours.
# It starts increasing gradually towards the late night hours starting from evening till 5 AM and decrease steeply towards morning.
# 
# 

# ## Distance and Trip Duration

# In[66]:


# We should remove those trips which covered 0 km
taxi_data= taxi_data[~(taxi_data.distance==0)]

#Plotting graph for trip duration and distance
plt.figure(figsize=(10,6))
sns.regplot(taxi_data.distance, taxi_data.trip_duration, color='orange')
plt.xlabel('Trip duration')
plt.ylabel('Distance')
plt.show()


# 
# The straight line shows some linear relation between the two.

# ## Distance and Store Forward Flag

# In[67]:


# Analyse Distance to store forward flag
plt.figure(figsize=(10,6))
sns.catplot(y = 'distance', x= 'store_and_fwd_flag', data=taxi_data, kind= 'strip')
plt.xlabel('Store and Forward Flag')
plt.ylabel('Distance')
plt.show()


# As observed for longer distances the trip is not stored.

# ## Distance and Vendor

# In[68]:


# Comparing distance and vendor
plt.figure(figsize = (10,6))
sns.barplot(y='distance',x='vendor_id',data=taxi_data,estimator=np.mean, palette='pastel')
plt.ylabel('Distance Travelled')
plt.xlabel('Vendor ID')
plt.show()


# As shown in above, Similar distribution has been observed btw both the vendors
# 
# 

# # Feature Engineering

# ## One Hot Encoding

# Features like 'store_and_fwd_flag', and 'pickup_day' are dummified.

# In[69]:


# Dummifying the features
dummy = pd.get_dummies(taxi_data.store_and_fwd_flag, prefix= 'store_and_fwd_flag')
taxi_data = pd.concat([taxi_data,dummy] , axis = 1)

dummy = pd.get_dummies(taxi_data.pickup_day, prefix='pickup_day', drop_first=True)
taxi_data = pd.concat([taxi_data,dummy], axis = 1,)


# In[70]:


# Trip durtion in hours
taxi_data['trip_duration_hour'] = taxi_data['trip_duration']/3600


# In[71]:


# Removing the Variables which are not necessary for further analysis
taxi_data= taxi_data.drop(['id', 'pickup_datetime','dropoff_datetime','store_and_fwd_flag','pickup_day','dropoff_day','Pickday_No','Pickup_Time','trip_duration','speed'], axis =1)
taxi_data.head()


# In[72]:


# Information of data
taxi_data.info()


# In[73]:


taxi_data.shape


# In[74]:


# Checking for NULL /NAN values
taxi_data.isnull().sum()


# NAN/NULL values are not present

# # Now, Splitting the data into train and test dataset, before fitting data into our models.

# In[75]:


# checking numerical features
variables = taxi_data.describe().columns
variables = list(variables)
variables = variables[:-1]
variables


# In[76]:


# Length of features
len(variables)


# In[77]:


get_ipython().system('pip install scipy')


# In[78]:


# Since the dataset having more rows, lets select some of it for training purpose
from scipy.stats import zscore

# Train test split
X = taxi_data[variables].apply (zscore)[:]
y = taxi_data['trip_duration_hour'] [:]


# In[79]:


#train_test_split imported using sklearn library
from sklearn.model_selection import train_test_split

# Data set splitted into 75-25 for training and testing respectively,
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state = 0)


# In[80]:


# Checking of the data splitted into 75-25
print(X_train.shape,y_train.shape)
print(X_test.shape, y_test.shape) 


# # Linear regression

# ##Linear regression

# In[81]:


# Correlation analysis using heatmap
plt.figure(figsize=(16,10))
corelation =taxi_data.corr()
sns.heatmap(abs(corelation), annot = True, center=1)


# In[82]:


# Fit into regression for train and test
reg = LinearRegression().fit(X_train,y_train)

reg.score(X_train,y_train)


# In[83]:


# training and testing prediction
y_pred_train = reg.predict(X_train)

y_pred_test = reg.predict(X_test)


# ## Evaluation

# In[84]:


# Importing required metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# In[85]:


# for train set metric
lr_trn_mse = mean_squared_error((y_train),(y_pred_train))
print("Train MSE: ", lr_trn_mse)

lr_trn_rmse = np.sqrt(lr_trn_mse)
print("Train RMSE: ", lr_trn_rmse)

lr_trn_r2= r2_score((y_train),(y_pred_train))
print("R2 Score: ",lr_trn_r2)

lr_trn_r2_ = 1-(1-r2_score((y_train), (y_pred_train)))*((X_train.shape[0]-1)/(X_train.shape[0]-X_train.shape[1]-1))
print("Train Adjusted R2 : ",lr_trn_r2_)


# In[86]:


# for test set metric
lr_tst_mse = mean_squared_error((y_test),(y_pred_test))
print("Train MSE: ", lr_tst_mse)

lr_tst_rmse = np.sqrt(lr_tst_mse)
print("Train RMSE: ", lr_trn_rmse)

lr_tst_r2= r2_score((y_test),(y_pred_test))
print("R2 Score: ",lr_tst_r2)

lr_tst_r2_ = 1-(1-r2_score((y_test), (y_pred_test)))*((X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1))
print("Train Adjusted R2 : ",lr_tst_r2_)


# In[87]:


# Actual vs Prediction
plt.figure(figsize=(10,6))
c = [i for i in range(0,len(y_test))]
plt.plot(c,y_test, color= 'cyan', linewidth = 2.5, linestyle='-')
plt.plot(c,y_pred_test, color= 'blue', linewidth = 2.5, linestyle='-')
plt.title("Actual Vs Predicted for test data")
plt.legend(["Actual","Predicted"])
plt.show()


# It has high prediction error on the metrics we tested.
# Linear regression model does not provide us with high accuracy.

# # Lasso Regression

# In[88]:


# Importing required models
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

# Cross Validation
lasso = Lasso()
parameters = {'alpha': [1e-15,1e-13,1e-10,1e-8,1e-5,1e-4,1e-3,1e-2,1e-1,1,5,10,20,30,40,45,50,55,60,100]}
lasso_regressor = GridSearchCV(lasso, parameters, scoring='r2', cv=5)
lasso_regressor.fit(X_train, y_train)


# In[89]:


# Best fit Alpha value
print("The best fit alpha value is found to be :" , lasso_regressor.best_params_)
print("The R2 score using the same alpha is :", lasso_regressor.best_score_)


# In[90]:


lasso_regressor.score(X_train, y_train)


# In[91]:


# Making Prediction
y_pred_lasso_train = lasso_regressor.predict(X_train)

y_pred_lasso_test = lasso_regressor.predict(X_test)


# Evaluation of model

# In[92]:


# Training metrics
lasso_trn_mse = mean_squared_error(y_train, y_pred_lasso_train)
print("Train MSE :", lasso_trn_mse)

lasso_trn_rmse = np.sqrt(lasso_trn_mse)
print("Train RMSE :", lasso_trn_rmse)

lasso_trn_r2 = r2_score(y_train, y_pred_lasso_train)
print("R2 score:", lasso_trn_r2)

lasso_trn_r2_ = 1-(1-r2_score(y_train, y_pred_lasso_train))*((X_train.shape[0]-1)/(X_train.shape[0]- X_train.shape[1]-1))
print("Train Adjusted R2:", lasso_trn_r2_)


# In[93]:


# Testing metrics
lasso_tst_mse  = mean_squared_error(y_test, y_pred_lasso_test)
print("Test MSE :" , lasso_tst_mse)

lasso_tst_rmse = np.sqrt(lasso_tst_mse)
print("Test RMSE :" ,lasso_tst_rmse)

lasso_tst_r2 = r2_score(y_test, y_pred_lasso_test)
print("Test R2 :" ,lasso_tst_r2)

lasso_tst_r2_= 1-(1-r2_score(y_test, y_pred_lasso_test))*((X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1))
print("Test Adjusted R2 : ", lasso_tst_r2_)


# In[94]:


# Actual vs Prediction
plt.figure(figsize=(10,6))
c = [i for i in range(0, len(y_test))]
plt.plot(c, y_test, color = 'cyan', linewidth = 2.5, linestyle = '-')
plt.plot(c,y_pred_lasso_test, color = 'blue', linewidth = 2.5, linestyle ='-')
plt.title("Actual Vs Predicted for test data", fontsize = 20)
plt.legend(["Actual","Predicted"])
plt.show()


# As per observation, Lasso regresion model doesn't improve on the Linear model either.

# # Decision Tree Regressor

# In[95]:


# importing decision tree
from sklearn.tree import DecisionTreeRegressor


# In[96]:


# Maximum depth of trees
max_depth = [4,6,8,10]

# Minimum number of samples rquired to split a node
min_samples_split = [10,20,30]

# Minimum number of samples required at each leaf node
min_samples_leaf = [8,16,22]

# Hyperparameter Grid
param_dict_dt = {'max_depth' : max_depth,
                 'min_samples_split' : min_samples_split,
                 'min_samples_leaf' : min_samples_leaf}

# The cache variable contains the best parameters for the Decision Tree to save the time when running again
cache= {'max_depth':[10],
        'min_samples_split':[10],
        'min_samples_leaf': [22]}


# In[97]:


dt = DecisionTreeRegressor()

# Grid Search 
dt_grid = GridSearchCV(estimator= dt,
                       param_grid = cache,
                       cv = 5, verbose = 2, scoring = 'r2')

dt_grid.fit(X_train, y_train)


# In[99]:


dt_grid.best_score_


# In[100]:


dt_grid.best_estimator_


# In[101]:


# Making prediction
y_pred_dt_train=dt_grid.predict(X_train)

y_pred_dt_test=dt_grid.predict(X_test)


# In[102]:


# training metrics
dt_trn_mse  = mean_squared_error(y_train, y_pred_dt_train)
print("Train MSE :" , dt_trn_mse)

dt_trn_rmse = np.sqrt(dt_trn_mse)
print("Train RMSE :" ,dt_trn_rmse)

dt_trn_r2 = r2_score(y_train, y_pred_dt_train)
print("Train R2 :" ,dt_trn_r2)

dt_trn_r2_= 1-(1-r2_score(y_train, y_pred_dt_train))*((X_train.shape[0]-1)/(X_train.shape[0]-X_train.shape[1]-1))
print("Train Adjusted R2 : ", dt_trn_r2_)


# In[103]:


# Testing metrics
dt_tst_mse  = mean_squared_error(y_test, y_pred_dt_test)
print("Test MSE :" , dt_tst_mse)
dt_tst_rmse = np.sqrt(dt_tst_mse)
print("Test RMSE :" ,dt_tst_rmse)

dt_tst_r2 = r2_score(y_test, y_pred_dt_test)
print("Test R2 :" ,dt_tst_r2)

dt_tst_r2_= 1-(1-r2_score(y_test, y_pred_dt_test))*((X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1))
print("Test Adjusted R2 : ", dt_tst_r2_)


# In[104]:


# Actual vs Prediction
plt.figure(figsize= (10,6))
c= [i for i in range(0, len(y_test))]
plt.plot(c, y_test, color='cyan', linewidth=2.5, linestyle='-')
plt.plot(c, y_pred_dt_test, color='blue', linewidth=2.5, linestyle='-')
plt.title('Actual vs Predicted for Test Data', fontsize=20)
plt.legend(["Actual", "Predicted"])
plt.show()


# In[105]:


plt.figure(figsize= (10,6))
c= [i for i in range(0, len(y_test))]
plt.plot(c, y_test-y_pred_dt_test, color='purple', linewidth=2.5, linestyle='-')
plt.title('Error Term', fontsize=20)
plt.show()


# The decision tree improved the predictions much better than Linear models.

# # Ridge Regression

# Ridge Regression

# In[106]:


# Importing Ridge
from sklearn.linear_model import Ridge

# Cross validation
ridge = Ridge()
parameters = {'alpha': [1e-15,1e-13,1e-10,1e-8,1e-5,1e-4,1e-3,1e-2,1e-1,1,5,10,20,30,40,45,50,55,60,100]}
ridge_regressor = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5)
ridge_regressor.fit(X_train, y_train)


# In[107]:


# best fit alpha value
print('The best fit alpha value is found out to be :' ,ridge_regressor.best_params_)
print('The R2 score using the same alpha is :', lasso_regressor.best_score_)


# In[108]:


# Estimator
ridge_regressor.best_estimator_


# In[109]:


ridge_regressor.score(X_train, y_train)


# In[110]:


# Making prediction
y_pred_ridge_train=ridge_regressor.predict(X_train)
y_pred_ridge_test = ridge_regressor.predict(X_test)


# In[111]:


# Training metrics
ridge_trn_mse  = mean_squared_error(y_train, y_pred_ridge_train)
print("Train MSE :" , ridge_trn_mse)

ridge_trn_rmse = np.sqrt(ridge_trn_mse)
print("Train RMSE :" ,ridge_trn_rmse)

ridge_trn_r2 = r2_score(y_train, y_pred_ridge_train)
print("Train R2 :" ,ridge_trn_r2)

ridge_trn_r2_= 1-(1-r2_score(y_train, y_pred_ridge_train))*((X_train.shape[0]-1)/(X_train.shape[0]-X_train.shape[1]-1))
print("Train Adjusted R2 : ", ridge_trn_r2)


# In[112]:


# Testing metrics
ridge_tst_mse  = mean_squared_error(y_test, y_pred_ridge_test)
print("Test MSE :" , ridge_tst_mse)

ridge_tst_rmse = np.sqrt(ridge_tst_mse)
print("Test RMSE :" ,ridge_tst_rmse)

ridge_tst_r2 = r2_score(y_test, y_pred_ridge_test)
print("Test R2 :" ,ridge_tst_r2)

ridge_tst_r2_= 1-(1-r2_score(y_test, y_pred_ridge_test))*((X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1))
print("Test Adjusted R2 : ", ridge_tst_r2_)


# In[113]:


# Actual vs Prediction
plt.figure(figsize= (10,6))
c= [i for i in range(0, len(y_test))]
plt.plot(c, y_test, color='purple', linewidth=2.5, linestyle='-')
plt.plot(c, y_pred_ridge_test, color='orange', linewidth=2.5, linestyle='-')
plt.title('Actual vs Predicted for Test Data', fontsize=18)
plt.legend(["Actual", "Predicted"])
plt.show()


# In[114]:


plt.figure(figsize= (10,6))
c= [i for i in range(0, len(y_test))]
plt.plot(c, y_test-y_pred_ridge_test, color='purple', linewidth=2.5, linestyle='-')
plt.title('Error Term', fontsize=20)
plt.show()


# As per above "error term" observation their is no improvement on linear model by ridge regression.

# ## XGBoost_Regressor

# In[115]:


n_estimators = [80,150,200]

#maximum depth of trees
max_depth = [5,8,10]
min_samples_split = [40,50]
learning_rate = [0.2,0.4,0.6]

# Hyperparameter Grid
param_xgb = {'n_estimators' : n_estimators,
             'max_depth' : max_depth,
             'min_samples_':min_samples_split,
             'learning_rate' : learning_rate
             }

# The cache  variables contains the best parameters for the XGBoost which we already before to save time when running it again
cache = {'n_estimators' : [200],
             'max_depth' : [8],
             'min_samples_':[40],
             'learning_rate' : [0.2]
             }


# In[116]:


param_xgb


# In[117]:


# Importing XGbooster
import xgboost as xgb
xgb_model = xgb.XGBRegressor()

# Grid Search
xgb_grid = GridSearchCV(estimator = xgb_model,
                        param_grid = cache,
                        cv = 3, verbose = 1,
                        scoring='r2')

xgb_grid.fit(X_train, y_train)


# In[118]:


xgb_grid.best_score_


# In[119]:


xgb_grid.best_params_


# In[120]:


# Making predictions
y_pred_xgb_train = xgb_grid.predict(X_train)
y_pred_xgb_test = xgb_grid.predict(X_test)


# In[121]:


from sklearn.metrics import r2_score
# Training metrics
xgb_trn_mse  = mean_squared_error(y_train, y_pred_xgb_train)
print("Train MSE :" , xgb_trn_mse)

xgb_trn_rmse = np.sqrt(xgb_trn_mse)
print("Train RMSE :" ,xgb_trn_rmse)

xgb_trn_r2 = r2_score(y_train, y_pred_xgb_train)
print("Train R2 :" ,xgb_trn_r2)

xgb_trn_r2_= 1-(1-r2_score((y_train), (y_pred_xgb_train)))*((X_train.shape[0]-1)/(X_train.shape[0]-X_train.shape[1]-1))
print("Train Adjusted R2 : ", xgb_trn_r2_)


# In[122]:


# Testing metrics
xgb_tst_mse  = mean_squared_error(y_test, y_pred_xgb_test)
print("Test MSE :" , xgb_tst_mse)

xgb_tst_rmse = np.sqrt(xgb_tst_mse)
print("Test RMSE :" ,xgb_tst_rmse)

xgb_tst_r2 = r2_score(y_test, y_pred_xgb_test)
print("Test R2 :" ,xgb_tst_r2)

xgb_tst_r2_= 1-(1-r2_score((y_test), (y_pred_xgb_test)))*((X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1))
print("Test Adjusted R2 : ", xgb_tst_r2_)


# In[123]:


# Actual vs Prediction
plt.figure(figsize= (10,6))
c= [i for i in range(0, len(y_test))]
plt.plot(c, y_test, color='purple', linewidth=2.5, linestyle='-')
plt.plot(c, y_pred_xgb_test, color='lime', linewidth=2.5, linestyle='-')
plt.title('Actual vs Predicted for Test Data', fontsize=20)
plt.legend(["Actual", "Predicted"])
plt.show()


# In[124]:


# Error term
plt.figure(figsize= (10,6))
c= [i for i in range(0, len(y_test))]
plt.plot(c, y_test-y_pred_xgb_test, color='purple', linewidth=2.5, linestyle='-')
plt.title('Error Term', fontsize=20)
plt.show()


# Now, look at the feature importance.

# In[125]:


importance_df= pd.DataFrame({'Features': variables, 'Feature_importance': list(xgb_grid.best_estimator_.feature_importances_)})
importance_df


# In[126]:


importance_df.sort_values(by=['Feature_importance'],ascending=False,inplace=True)


# Let's look it by using bar grabh.

# In[127]:


plt.figure(figsize=(15,6))
plt.title('Feature Importance', fontsize=20)
sns.barplot(x='Features',y="Feature_importance", data=importance_df[:6], palette='colorblind')
plt.show()


# As per above data observation,we can say that distance is the top contributor to trip duration followed by different days of the weeks.

# Error terms

# In[128]:


# Plotting the error terms to understand
plt.figure(figsize=(10,6))
sns.distplot(y_test - y_pred_xgb_test )
plt.title('Error Term', fontsize=20)
plt.show()


# # Analysis of Models 

# ## Analyzing or Evaluation of the models

# ## For the Train data.

# In[129]:


# Models summary for the Train data
models= ['Linear Regression', 'Lasso Regression', 'Ridge Regression','DecisionTree Regressor','XGBoost Regressor']
trn_mse= [lr_trn_mse, lasso_trn_mse, ridge_trn_mse, dt_trn_mse, xgb_trn_mse]
trn_rmse= [lr_trn_rmse, lasso_trn_rmse, ridge_trn_rmse, dt_trn_rmse, xgb_trn_rmse]
trn_r2= [lr_trn_r2, lasso_trn_r2, ridge_trn_r2, dt_trn_r2, xgb_trn_r2]
train_adjusted_r2= [lr_trn_r2_, lasso_trn_r2_, ridge_trn_r2_, dt_trn_r2_, xgb_trn_r2_]


# ## For the test data.
# 

# In[130]:


# Models Summary for the test data
models= ['Linear Regression', 'Lasso Regression', 'Ridge Regression','DecisionTree Regressor','XGBoost Regressor']
tst_mse= [lr_tst_mse, lasso_tst_mse, ridge_tst_mse, dt_tst_mse, xgb_tst_mse]
tst_rmse= [lr_tst_rmse, lasso_tst_rmse, ridge_tst_rmse, dt_tst_rmse, xgb_tst_rmse]
tst_r2= [lr_tst_r2, lasso_tst_r2, ridge_tst_r2, dt_tst_r2, xgb_tst_r2]
test_adjusted_r2= [lr_tst_r2_, lasso_tst_r2_, ridge_tst_r2_, dt_tst_r2_, xgb_tst_r2_]


# In[131]:


train_data_df = pd.DataFrame({'Model Name': models, 'Train MSE': trn_mse, 'Train RMSE': trn_rmse, 'Train R^2': trn_r2,
                              'Train Adjusted R^2' :train_adjusted_r2})
train_data_df


# In[132]:


test_data_df = pd.DataFrame({'Model Name': models, 'Test MSE': tst_mse, 'Test RMSE': tst_rmse, 'Test R^2': tst_r2,
                              'Test Adjusted R^2' :test_adjusted_r2})
test_data_df


# # **Conclusion**

# *   As we can see from the fact that MSE and RMSE, the metrics used to evaluate the performance of regression models, and R^2 are almost the same between training and testing time, there is not much of a difference between **Decision Tree** and **XGBoost Regressor** during training and testing.
#  
#  
# *  Performance of linear model is low or not good compare to Xgbooster models.
# 
#  
#  
#  
#  
# *   To predict the trip duration for a particular taxi, from above table we can conclude  that **XGBoost Regressor** is the best models as compare to the other models.

# 
