import pandas as pd
import datetime as dt
from matplotlib import pyplot as plt
import matplotlib

CVD = pd.read_csv('covid_19_india.csv')
print(CVD.tail())
print(CVD.dtypes)

CVD = CVD.drop(['Time','ConfirmedIndianNational','ConfirmedForeignNational'], axis = 1)

CVD.columns = ['Sr.No','Date', 'location','Cured','Deaths','Confirmed']
#cvd_states_data = pd.DataFrame(data = CVD, columns = new_cols)

CVD['Date'] = [dt.datetime.strptime(x,'%d/%m/%y') for x in CVD['Date']] 
print(CVD.dtypes)

#Let's look at multiple states
#states=['Chhattisgarh','Odisha']
states = ['Gujarat','Tamil Nadu','Delhi']
CVD_states = CVD[CVD.location.isin(states)]   #Create subset data frame for select states
print(CVD_states)

CVD_states.set_index('Date', inplace=True)  #Make date the index for easy plotting

#To create subset range based on dates
CVD_states = CVD_states.loc['2020-01-30':'2020-05-13']

print(CVD_states.tail())  #Check the last date 

#To calculate mortality rate
CVD_states['mortality_rate'] = CVD_states['Deaths']/CVD_states['Confirmed']

#To calculate recovery rate
CVD_states['recovery_rate'] = CVD_states['Cured']/CVD_states['Confirmed']

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14,14))

CVD_states.groupby('location')['Confirmed'].plot(ax=axes[0,0], legend=True) #for log scale add logy=True
CVD_states.groupby('location')['Deaths'].plot(ax=axes[0,1], legend=True)
CVD_states.groupby('location')['Cured'].plot(ax=axes[1,0], legend=True)
CVD_states.groupby('location')['mortality_rate'].plot(ax=axes[1,1], legend=True)
CVD_states.groupby('location')['recovery_rate'].plot(ax=axes[2,0], legend=True)
#CVD_country.groupby('location')['mortality_rate'].plot(ax=axes[1,1], legend=True)
#CVD_country.to_csv('data/output.csv')

axes[0, 0].set_title("Confirmed")
axes[0,0].grid(color='black', linestyle='dashed', linewidth=1)
axes[0, 1].set_title("Deaths")
axes[0,1].grid(color='black', linestyle='dashed', linewidth=1)
axes[1, 0].set_title("Cured")
axes[1,0].grid(color='black', linestyle='dashed', linewidth=1)
axes[1, 1].set_title("mortality_rate")
axes[1,1].grid(color='black', linestyle='dashed', linewidth=1)
axes[2, 0].set_title("recovery_rate")
axes[2,0].grid(color='black', linestyle='dashed', linewidth=1)
fig.tight_layout()  # adjust subplot parameters to give specified padding.


CVD_no_maharashtra = CVD.loc[~(CVD['location'].isin(["Maharashtra"]))]
CVD_no_maharashtra = pd.DataFrame(CVD_no_maharashtra.groupby(['location', 'Date'])['Confirmed', 'Deaths','Cured'].sum()).reset_index()
print(CVD_no_maharashtra)

#Sort values by each state and by date - descending. Easy to interpret plots
CVD_no_maharashtra = CVD_no_maharashtra.sort_values(by = ['location','Date'], ascending=False)
print(CVD_no_maharashtra)

import seaborn as sns
def plot_bar(feature, value, title, df, size):
    f, ax = plt.subplots(1,1, figsize=(4*size,4))
    df = df.sort_values([value], ascending=False).reset_index(drop=True)
    g = sns.barplot(df[feature][0:10], df[value][0:10], palette='Set3')
    g.set_title("Number of {} - highest 10 values".format(title))
#    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    plt.show()    

filtered_CVD_no_maharashtra = CVD_no_maharashtra.drop_duplicates(subset = ['location'], keep='first')
plot_bar('location', 'Confirmed', 'Total cases in the India except Maharashtra', filtered_CVD_no_maharashtra, size=4)
plot_bar('location', 'Deaths', 'Total deaths in the India except Maharashtra', filtered_CVD_no_maharashtra, size=4)
plot_bar('location', 'Cured', 'Total recovered in the India except Maharashtra', filtered_CVD_no_maharashtra, size=4)


#Plot India aggregate numbers for total cases and deaths. 
def plot_state_aggregate(df, title='Aggregate plot', size=1):
    f, ax = plt.subplots(1,1, figsize=(4*size,2*size))
    g = sns.lineplot(x="Date", y='Confirmed', data=df, color='blue', label='Total Cases')
    g = sns.lineplot(x="Date", y='Deaths', data=df, color='red', label='Total Deaths')
    plt.xlabel('Date')
    plt.ylabel(f'Total {title} cases')
    plt.xticks(rotation=90)
    plt.title(f'Total {title} cases')
    ax.grid(color='black', linestyle='dotted', linewidth=0.75)
    plt.show()  

#Group by dates. 
#Reset index because groupby by default makes grouped columns indices
#Sum values from all states per given date
CVD_no_maharashtra_aggregate = CVD_no_maharashtra.groupby(['Date']).sum().reset_index()
print(CVD_no_maharashtra_aggregate)

plot_state_aggregate(CVD_no_maharashtra_aggregate, 'Rest of the India except Maharshtra', size=4)

def plot_mortality(df, title='Maharshtra', size=1):
    f, ax = plt.subplots(1,1, figsize=(4*size,2*size))
    g = sns.lineplot(x="Date", y='mortality_rate', data=df, color='blue', label='Mortality (Deaths / Total Cases)')
    plt.xlabel('Date')
    plt.ylabel(f'Mortality {title} [%]')
    plt.xticks(rotation=90)
    plt.title(f'Mortality percent {title}\nCalculated as Deaths/Confirmed cases')
    ax.grid(color='black', linestyle='dashed', linewidth=1)
    plt.show()  

CVD_no_maharashtra_aggregate['mortality_rate'] = CVD_no_maharashtra_aggregate['Deaths'] / CVD_no_maharashtra_aggregate['Confirmed'] * 100
plot_mortality(CVD_no_maharashtra_aggregate, title = ' - Rest of the India except Maharshtra', size = 3)

def plot_recovery(df, title='Maharshtra', size=1):
    f, ax = plt.subplots(1,1, figsize=(4*size,2*size))
    g = sns.lineplot(x="Date", y='recovery_rate', data=df, color='green', label='Recovery (Cured / Total Cases)')
    plt.xlabel('Date')
    plt.ylabel(f'Recovery {title} [%]')
    plt.xticks(rotation=90)
    plt.title(f'Recovery percent {title}\nCalculated as Cured/Confirmed cases')
    ax.grid(color='black', linestyle='dashed', linewidth=1)
    plt.show()  

CVD_no_maharashtra_aggregate['recovery_rate'] = CVD_no_maharashtra_aggregate['Cured'] / CVD_no_maharashtra_aggregate['Confirmed'] * 100
plot_recovery(CVD_no_maharashtra_aggregate, title = ' - Rest of the India except Maharshtra', size = 3)

import scipy
import numpy as np
#PREDICTION

def plot_exponential_fit_data(d_df, title, delta):
    d_df = d_df.sort_values(by=['Date'], ascending=True)
    d_df['x'] = np.arange(len(d_df)) + 1  #Add column x to the dataframe 
    d_df['y'] = d_df['Confirmed']   #Add column y to the dataframe 

    x = d_df['x'][:-delta]  #Remove delta number of data points (so we can predict them)
    y = d_df['y'][:-delta]  #Remove delta number of data points (so we can predict them)

#Use non-linear least squares to fit a function, f, to data.
#Let us fit data to exponential function: #y = Ae^(Bt)
    
    c2 = scipy.optimize.curve_fit(lambda t, a, b: a*np.exp(b*t),  x,  y,  p0=(20, 0.2)) 
# Function: lambda t, a, b: a*np.exp(b*t)
# xm y and po for initial values. 
    
    A, B = c2[0]  #Coefficients
    print(f'(y = Ae^(Bx)) A: {A}, B: {B}\n')
    x = range(1,d_df.shape[0] + 1)
    y_fit = A * np.exp(B * x)
#    print(y_fit)
    f, ax = plt.subplots(1,1, figsize=(12,6))
    g = sns.scatterplot(x=d_df['x'][:-delta], y=d_df['y'][:-delta], label='Confirmed cases (used for model creation)', color='red')
    g = sns.scatterplot(x=d_df['x'][-delta:], y=d_df['y'][-delta:], label='Confirmed cases (not used for model, va;idation)', color='blue')
    g = sns.lineplot(x=x, y=y_fit, label='Predicted values', color='green')  #Predicted
    x_future=range(105,110) #As of 30 JAN 2020 we have 105 days of info. 
    y_future=A * np.exp(B * x_future)
    print("Expected cases for the next 5 days: \n", y_future)
    plt.xlabel('Days since first case')
    plt.ylabel(f'Total cases')
    plt.title(f'Confirmed cases & projected cases: {title}')
    plt.xticks(rotation=90)
    ax.grid(color='black', linestyle='dotted', linewidth=0.75)
    plt.show()
    
print(CVD_no_maharashtra)
CVD_Gujarat = CVD_no_maharashtra[CVD_no_maharashtra['location']=='Gujarat']
CVD_Madhyapradesh = CVD_no_maharashtra[CVD_no_maharashtra['location']=='Madhya Pradesh']
CVD_Chhattishgarh = CVD_no_maharashtra[CVD_no_maharashtra['location']=='Chhattishgarh']

d_df = CVD_Gujarat.copy()
print(CVD_Gujarat)
plot_exponential_fit_data(d_df, 'Gujarat', 5)

covid_confirmed = CVD.iloc[:,-1]
covid_deaths = CVD.iloc[:,4]
covid_recovered = CVD.iloc[:,3]
print(covid_confirmed)
print(covid_deaths)
print(covid_recovered)


covid_confirmed_count = covid_confirmed.max()
covid_deaths_count = covid_deaths.max()
covid_recovered_count = covid_recovered.max()

print('Total confirmed, dead, and recovered numbers in the India, respectively: ', 
     covid_confirmed_count, covid_deaths_count, covid_recovered_count)

#For easy plotting let us store all these numbers in a dataframe. 
#Let us also calculate active cases.
#Active=Confirmed−Deaths−Recovered

india_df = pd.DataFrame({
    'confirmed': [covid_confirmed_count],
    'deaths': [covid_deaths_count],
    'recovered': [covid_recovered_count],
    'active': [covid_confirmed_count - covid_deaths_count - covid_recovered_count]
})

print(india_df)

#!pip install plotly==4.5.2

import matplotlib.ticker as ticker
import seaborn as sns
import plotly.express as px
from plotly.offline import plot 

#Unpivot the DataFrame from wide to long format
india_long_df = india_df.melt(value_vars=['active', 'deaths', 'recovered'],
                              var_name="status",
                              value_name="count")

india_long_df['upper'] = 'confirmed'

print(india_long_df)

fig = px.sunburst(india_long_df , path = ['status'], values='count', color_discrete_sequence=['#3498db', '#2ecc71', '#e74c3c'])

fig.show()

state=['Maharashtra']
CVD_maharashtra = CVD[CVD.location.isin(state)]   #Create subset data frame for select countries
print(CVD_maharashtra)

CVD_maharashtra.set_index('Date', inplace=True)  #Make date the index for easy plotting

#To create subset range based on dates
CVD_maharashtra = CVD_maharashtra.loc['2020-01-30':'2020-05-13']

print(CVD_maharashtra.tail())  #Check the last date 

#To calculate mortality rate
CVD_maharashtra['mortality_rate'] = CVD_maharashtra['Deaths']/CVD_maharashtra['Confirmed']

#To calculate recovery rate
CVD_maharashtra['recovery_rate'] = CVD_maharashtra['Cured']/CVD_maharashtra['Confirmed']

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14,14))

CVD_maharashtra.groupby('location')['Confirmed'].plot(ax=axes[0,0], legend=True) #for log scale add logy=True
CVD_maharashtra.groupby('location')['Deaths'].plot(ax=axes[0,1], legend=True)
CVD_maharashtra.groupby('location')['Cured'].plot(ax=axes[1,0], legend=True)
CVD_maharashtra.groupby('location')['mortality_rate'].plot(ax=axes[1,1], legend=True)
CVD_maharashtra.groupby('location')['recovery_rate'].plot(ax=axes[2,0], legend=True)

#CVD_maharashtra.to_csv('Maharashtra.csv')

axes[0, 0].set_title("Confirmed")
axes[0,0].grid(color='black', linestyle='dashed', linewidth=1)
axes[0, 1].set_title("Deaths")
axes[0,1].grid(color='black', linestyle='dashed', linewidth=1)
axes[1, 0].set_title("Cured")
axes[1,0].grid(color='black', linestyle='dashed', linewidth=1)
axes[1, 1].set_title("mortality_rate")
axes[1,1].grid(color='black', linestyle='dashed', linewidth=1)
axes[2, 0].set_title("recovery_rate")
axes[2,0].grid(color='black', linestyle='dashed', linewidth=1)

fig.tight_layout()  # adjust subplot parameters to give specified padding.


#Plot India aggregate numbers for total cases and deaths. 
def plot_state_aggregate(df, title='Aggregate plot', size=1):
    f, ax = plt.subplots(1,1, figsize=(4*size,2*size))
    g = sns.lineplot(x="Date", y='Confirmed', data=df, color='blue', label='Total Cases')
    g = sns.lineplot(x="Date", y='Deaths', data=df, color='red', label='Total Deaths')
    plt.xlabel('Date')
    plt.ylabel(f'Total {title} cases')
    plt.xticks(rotation=90)
    plt.title(f'Total {title} cases')
    ax.grid(color='black', linestyle='dotted', linewidth=0.75)
    plt.show()  

#Group by dates. 
#Reset index because groupby by default makes grouped columns indices
#Sum values from all states per given date
covid_maharashtra = CVD_maharashtra.groupby(['Date']).sum().reset_index()
print(covid_maharashtra)

plot_state_aggregate(covid_maharashtra, 'Maharashtra', size=4)

#Plot India aggregate numbers for total cases and deaths. 
def plot_state_aggregate(df, title='Aggregate plot', size=1):
    f, ax = plt.subplots(1,1, figsize=(4*size,2*size))
    g = sns.lineplot(x="Date", y='Confirmed', data=df, color='blue', label='Total Cases')
    g = sns.lineplot(x="Date", y='Deaths', data=df, color='red', label='Total Deaths')
    plt.xlabel('Date')
    plt.ylabel(f'Total {title} cases')
    plt.xticks(rotation=90)
    plt.title(f'Total {title} cases')
    ax.grid(color='black', linestyle='dotted', linewidth=0.75)
    plt.show()  

#Group by dates. 
#Reset index because groupby by default makes grouped columns indices
#Sum values from all states per given date
CVD_maharashtra = CVD_maharashtra.groupby(['Date']).sum().reset_index()
print(CVD_maharashtra)

plot_state_aggregate(CVD_maharashtra, 'Maharashtra', size=4)


def plot_mortality(df, title='Maharshtra', size=1):
    f, ax = plt.subplots(1,1, figsize=(4*size,2*size))
    g = sns.lineplot(x="Date", y='mortality_rate', data=df, color='blue', label='Mortality (Deaths / Total Cases)')
    plt.xlabel('Date')
    plt.ylabel(f'Mortality {title} [%]')
    plt.xticks(rotation=90)
    plt.title(f'Mortality percent {title}\nCalculated as Deaths/Confirmed cases')
    ax.grid(color='black', linestyle='dashed', linewidth=1)
    plt.show()  

CVD_maharashtra['mortality_rate'] = CVD_maharashtra['Deaths'] / CVD_maharashtra['Confirmed'] * 100
plot_mortality(CVD_maharashtra, title = ' - Maharshtra', size = 3)

def plot_recovery(df, title='Maharshtra', size=1):
    f, ax = plt.subplots(1,1, figsize=(4*size,2*size))
    g = sns.lineplot(x="Date", y='recovery_rate', data=df, color='green', label='Recovery (Cured / Total Cases)')
    plt.xlabel('Date')
    plt.ylabel(f'Recovery {title} [%]')
    plt.xticks(rotation=90)
    plt.title(f'Recovery percent {title}\nCalculated as Deaths/Confirmed cases')
    ax.grid(color='black', linestyle='dashed', linewidth=1)
    plt.show()  

CVD_maharashtra['recovery_rate'] = CVD_maharashtra['Cured'] / CVD_maharashtra['Confirmed'] * 100
plot_recovery(CVD_maharashtra, title = ' - Maharshtra', size = 3)

import scipy
import numpy as np
#PREDICTION

def plot_exponential_fit_data(d_df, title, delta):
    d_df = d_df.sort_values(by=['Date'], ascending=True)
    d_df['x'] = np.arange(len(d_df)) + 1  #Add column x to the dataframe 
    d_df['y'] = d_df['Confirmed']   #Add column y to the dataframe 

    x = d_df['x'][:-delta]  #Remove delta number of data points (so we can predict them)
    y = d_df['y'][:-delta]  #Remove delta number of data points (so we can predict them)

#Use non-linear least squares to fit a function, f, to data.
#Let us fit data to exponential function: #y = Ae^(Bt)
    
    c2 = scipy.optimize.curve_fit(lambda t, a, b: a*np.exp(b*t),  x,  y,  p0=(20, 0.2)) 
# Function: lambda t, a, b: a*np.exp(b*t)
# xm y and po for initial values. 
    
    A, B = c2[0]  #Coefficients
    print(f'(y = Ae^(Bx)) A: {A}, B: {B}\n')
    x = range(1,d_df.shape[0] + 1)
    y_fit = A * np.exp(B * x)
#    print(y_fit)
    f, ax = plt.subplots(1,1, figsize=(12,6))
    g = sns.scatterplot(x=d_df['x'][:-delta], y=d_df['y'][:-delta], label='Confirmed cases (used for model creation)', color='red')
    g = sns.scatterplot(x=d_df['x'][-delta:], y=d_df['y'][-delta:], label='Confirmed cases (not used for model, va;idation)', color='blue')
    g = sns.lineplot(x=x, y=y_fit, label='Predicted values', color='green')  #Predicted
    x_future=range(95,100) #As of 09 FEB 2020 we have 95 days of info. 
    y_future=A * np.exp(B * x_future)
    print("Expected cases for the next 5 days: \n", y_future)
    plt.xlabel('Days since first case')
    plt.ylabel(f'Total cases')
    plt.title(f'Confirmed cases & projected cases: {title}')
    plt.xticks(rotation=90)
    ax.grid(color='black', linestyle='dotted', linewidth=0.75)
    plt.show()
    


d_df = CVD_maharashtra.copy()
print(CVD_maharashtra)
plot_exponential_fit_data(d_df, 'Maharashtra', 5)

covid_confirmed_maharashtra = CVD_maharashtra.iloc[:,4]
covid_deaths_maharashtra = CVD_maharashtra.iloc[:,3]
covid_recovered_maharashtra = CVD_maharashtra.iloc[:,2]
print(covid_confirmed_maharashtra)
print(covid_deaths_maharashtra)
print(covid_recovered_maharashtra)


covid_confirmed_count_maharashtra = covid_confirmed_maharashtra.max()
covid_deaths_count_maharashtra = covid_deaths_maharashtra.max()
covid_recovered_count_maharashtra = covid_recovered_maharashtra.max()

print('Total confirmed, dead, and recovered numbers in the Maharashtra, respectively: ', 
     covid_confirmed_count_maharashtra, covid_deaths_count_maharashtra, covid_recovered_count_maharashtra)

#For easy plotting let us store all these numbers in a dataframe. 
#Let us also calculate active cases.
#Active=Confirmed−Deaths−Recovered

maharashtra_df = pd.DataFrame({
    'confirmed': [covid_confirmed_count_maharashtra],
    'deaths': [covid_deaths_count_maharashtra],
    'recovered': [covid_recovered_count_maharashtra],
    'active': [covid_confirmed_count_maharashtra - covid_deaths_count_maharashtra - covid_recovered_count_maharashtra]
})

print(maharashtra_df)

 #Unpivot the DataFrame from wide to long format
maharashtra_long_df = maharashtra_df.melt(value_vars=['active', 'deaths', 'recovered'],
                              var_name="status",
                              value_name="count")

maharashtra_df['upper'] = 'confirmed'

print(maharashtra_df)

fig = px.sunburst(maharashtra_long_df , path = ['status'], values='count', color_discrete_sequence=['#3498db', '#2ecc71', '#e74c3c'])

fig.show()

#Multiple Regression

dataset = CVD_maharashtra.drop(['Sr.No','mortality_rate','recovery_rate'],axis = 1)
columns_titles = ["Confirmed","Cured","Deaths"]
dataset=dataset.reindex(columns=columns_titles)
print(dataset)


X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


# Taking care of Missing value
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan,strategy = 'mean')
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])


# Splitting dataset into test set and training set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y,test_size = 0.2 , random_state = 0)

# Fitting Simple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting to Test set result
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=0)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

