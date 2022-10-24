#!/usr/bin/env python
# coding: utf-8

# # Project Title
# Economic/Financial impact and adaptations from COVID-19 Pandemic.

# ## Topic
# *What problem are you (or your stakeholder) trying to address?*
# Using data analysis, machine learning, and data manipulation to investigate the econnomic consequences contributed by COVID-19 by searching the current case, deaths, and testing levels by country, and both financial and economic status of each country.
# ## Project Question
# *What specific question are you seeking to answer with this project?*
# *This is not the same as the questions you ask to limit the scope of the project.*
# What are the major economic and financial consequences from COVID-19 Pandemic? How people can adapt to the pandemic financially?
# ## What would an answer look like?
# *What is your hypothesized answer to your question?*
# ![](COVID_Data.png)
# ![](COVID_Vaccine_Data.png)
#  
# COVID-19 Pandemic contributes to the decline in performance of the global economy. As people continue to get infected, health and financial situations start to unravel. Major businesses start to either shut down or scale back on their operations. That in turn contributes to the economic decline because people had to take care of their sick family members. People can adapt to the situation by requiring masks, testing, and vaccinations so that they can slowly go back to normal operations over time. Graphs and visualization will be applied to show the changes in the economy during COVID-19 Pandemic.
# ## Data Sources
# *What 3 data sources have you identified for this project?*
# *How are you going to relate these datasets?*
# *How will you use this data to answer your project question?*
# 
# Data Sources:
# - https://vaccovid-coronavirus-vaccine-and-treatment-tracker.p.rapidapi.com/api/npm-covid-data/ (Note: This needs an API key. Go to https://rapidapi.com/axisbits-axisbits-default/api/covid-19-statistics/)
# - EO_14102022180121049.csv (URL: https://stats.oecd.org/Index.aspx?DatasetCode=STLABOUR#)
# - EO_14102022183231815.csv (URL: https://stats.oecd.org/Index.aspx?DatasetCode=STLABOUR#)
# - https://covid19.who.int/who-data/vaccination-data.csv
# 
# The 2 datasets will show the current COVID cases that includes their severity, deaths, testing, and vaccination levels by country. GDP growth and Unemployment data of each country will help explain the economic and financial consequences of COVID-19. The more people adapt to the COVID-19 Pandemic, the easier it is to get used to it and keep on moving. 

# In[1]:


# Start your code here

import requests
import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
plt.style.use("bmh")
# API

url = "https://vaccovid-coronavirus-vaccine-and-treatment-tracker.p.rapidapi.com/api/npm-covid-data/"

headers = {
	"X-RapidAPI-Key": "fbf638b489msh484afad00eb0877p19941bjsnfcab5b02a55a",
	"X-RapidAPI-Host": "vaccovid-coronavirus-vaccine-and-treatment-tracker.p.rapidapi.com"
}

response = requests.request("GET", url, headers=headers)


# In[2]:


# Database file
gdp_growth_df = pd.read_csv('EO_14102022183231815.csv')
pd.set_option('display.max_columns', None)
gdp_growth_df


# In[3]:


unemployment_df = pd.read_csv('EO_14102022180121049.csv')
pd.set_option('display.max_columns', None)
unemployment_df


# In[4]:


# Database or File URL
vaccination_df = pd.read_csv('https://covid19.who.int/who-data/vaccination-data.csv')
vaccination_df


# In[5]:


json = response.json()
covid_df = pd.DataFrame(json)
display(covid_df)


# In[6]:


display(covid_df.describe())
display(covid_df.corr())
display(covid_df.info())
display(covid_df.isnull().sum())


# Showing descriptive statistics, correlation, and information about the dataset that holds the current cases, deaths, and testing data on COVID-19 in each country. It also includes the first 2 rows that both show total amount of cases, deaths, and testing globally. Only TwoLetterSymbol and TwoLetterSymbol have missing data that is also found on those same rows after analyzing it. Those are not needed so it should be removed. Besides that, no duplicated date found in this dataset. I noticed that some test percentage values are above 100%. From what I could tell, people tend to get tested more than once for specific reasons that include going to events or travel to places.

# In[7]:


covid_df.dropna(inplace=True)
display(covid_df.isnull().sum())


# Removed rows that have missing data.

# In[8]:


px.bar(covid_df.sample(20), x='Country', y='Infection_Risk', labels={'Infection_Risk':'Infection Risk (%)'}, title='Infection Risk by Country')


# Presenting a bar chart using plotly.express to show infection risk levels from 20 random countries. These risk levels will show the chances of getting infected at these countries. Some governments take extreme measures like enforcing mandatory quarantine to contain the virus while others are struggling with their main industry like tourism. That in turn can cause decline in economic performance.

# In[9]:


display(gdp_growth_df.describe())
display(gdp_growth_df.corr())
display(gdp_growth_df.info())


# Showing descriptive statistics, correlation, and information about the dataset that holds quarterly gdp growth from each country, starting from first quarter of 2019 to the projected GDP growth in 2023. No correlations are found in this dataset due to seeing missing data on all integer columns except the value column itself. I can change either "Time" or "TIME" column to date/time data type if I'm going to use them for a graph or other visualization. "Flag Codes" and "Flags" columns are empty so they will be removed.

# In[10]:


if('Flag Codes', 'Flags') in gdp_growth_df.columns:
    gdp_growth_df.drop(columns=['Flag Codes', 'Flags'], inplace=True)
gdp_growth_df['TIME'] = pd.to_datetime(gdp_growth_df['TIME'])


# Removing empty "Flag Codes" and "Flags" columns and changed "TIME" column to datetime data type. The "TIME" column will be used to show the time progression of GDP growth in each country and understand its impact from COVID-19 pandemic.

# In[11]:


px.line(gdp_growth_df, x='TIME', y='Value', color='Country', labels={'Value':'GDP Growth (%)'}, title='GDP Growth Progression')


# This is a line graph from plotly express. This shows GDP growth of each country over time. As you can see, some countries didn't adapt as quickly as others at the height of the pandemic in 2020 which explains the moderate to sharp drop in GDP growth. 

# In[12]:


display(unemployment_df.describe())
display(unemployment_df.median())
display(unemployment_df.corr())
display(unemployment_df.info())


# Showing descriptive statistics, correlation, information, and boxplot on the dataset that holds quarterly unemployment rate from each country, starting from first quarter of 2019 to the projected GDP growth in 2023. No correlations are found in this dataset due to seeing missing data on all integer columns except the value column itself. I can change either "Time" or "TIME" column to date/time data type if I'm going to use them for a graph or other visualization. "Flag Codes", "Flags", "Reference Period Code",and "Reference Period" columns are empty so they will be removed.

# In[13]:


unemployment_box = sns.boxplot(data=unemployment_df, x='TIME', y='Value')
unemployment_box.set_xticklabels(labels=unemployment_box.get_xticklabels(), rotation=90)


# Used boxplot from seaborn visualization library. Rotated TIME labels by 90 degrees prevent them from being overlapped by each other. From the boxplot above, there are some outliers in the unemployment rate on each of the "TIME" columns.

# In[14]:


if('Flag Codes', 'Flags', 'Reference Period Code', 'Reference Period') in unemployment_df.columns:
    unemployment_df.drop(columns=['Flag Codes', 'Flags', 'Reference Period Code', 'Reference Period'],inplace = True)
unemployment_df['TIME'] = pd.to_datetime(unemployment_df['TIME'])
unemployment_df


# Removed "Flag Codes", "Flags", "Reference Period Code", "Reference Period" columns because they are not relevant to this project. Changing "TIME" column to datetime data type.

# In[15]:


px.line(unemployment_df, x='TIME', y='Value', color='Country', labels={'Value':'Unemployment Rate (%)'}, title='Unemployment by Country')


# Here's the line graph from plotly express. This shows the unemployment rates at each country over time during the COVID-19 pandemic. It gives the insight on how well each country has adapted during the COVID-19 Pandemic.

# In[16]:


display(vaccination_df.describe())
display(vaccination_df.median())
display(vaccination_df.corr())
display(vaccination_df.info())


# Almost all the columns from the dataset above have positive correlations with each other. I noticed that some columns have missing data including first vaccine date and amount of people who got the booster. The median and the 50th Percentile values on each column are the same in the vaccination dataset.

# In[17]:


sns.barplot(vaccination_df.sample(5), x='COUNTRY', y='PERSONS_FULLY_VACCINATED_PER100').set(title="PERSONS_FULLY_VACCINATED_PER100 by country")
sns.set(rc={"figure.figsize":(10,10)})


# This is a bar graph from seaborn library. It shows the number of people fully vaccinated per 100 by each country based on most recent data. This gives an insight on how far each country has progressed to get their citizens full vaccinated. Businesses evolve over time to get their employees back to work while at the same time, minimize infections so that they can smoothly and efficiently recover their losses from the pandemic.

# In[18]:


sns.scatterplot(vaccination_df, x='PERSONS_VACCINATED_1PLUS_DOSE_PER100',
                y='PERSONS_FULLY_VACCINATED_PER100').set(title="Vaccinated Per 100 vs. Fully Vaccinated Per 100")
sns.set(rc={"figure.figsize": (10, 10)})


# Using scatterplot from seaborn library to plot between number of people per 100 population that are vaccinated with at least 1 dose and fully vaccinated. It represents similarly like a comparison between the percentage of people that had at least one dose of a vaccine and are fully vaccinated. Countries are doing their best to adapt from the pandemic by getting people vaccinated so that they can loosen the covid restrictions on business and recover their losses from the pandemic. 

# Machine Learning Plan:
# - What type of machine learning model are you planning to use?
# I'm planning to use supervised machine learning model. Most of the data from the datasets are labelled.
# - What are the challenges have you identified/are you anticipating in building your machine learning model?
# There are some challenges that identified or anticipating in building my machine learning model. They include insufficient or partially irrelevant data and poor quality data.
# - How are you planning to address these challenges?
# Investigate the data being present to see if anything can be improved. More data will be researched and used.

# In[1]:


get_ipython().system('jupyter nbconvert --to python source.ipynb')


# In[ ]:




