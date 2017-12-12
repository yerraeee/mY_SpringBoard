#Importing various python libraries for Data-Analysis
import re
import pandas as pd
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sb
from io import StringIO
import csv
import scipy.stats as st
import zipfile
import glob
import geocoder as gc

import time
import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


#resetting max-columns to be displayed to None.
pd.set_option('display.max_columns', None)


#Storing File locations  multiple Excel-Workbooks with Sales-Data
sales_files = glob.glob("Annualized_20Rolling_20Sales_20Update"+"/*.xls") #list of paths matching pathname pattern for sales data files.

#
df_sales = pd.DataFrame()

for i in range(len(sales_files)):
    df = pd.read_excel(sales_files[i],skiprows=4,converters={'BOROUGH':str, 'NEIGHBORHOOD':str, 'BUILDING CLASS CATEGORY':str,
       'TAX CLASS AT PRESENT':str, 'BLOCK':str, 'LOT':str, 'EASE-MENT':str,
       'BUILDING CLASS AT PRESENT':str, 'ADDRESS':str, 'APARTMENT NUMBER':str, 'ZIP CODE':str,
       'RESIDENTIAL UNITS':np.int64, 'COMMERCIAL UNITS':np.int64, 'TOTAL UNITS':np.int64,
       'LAND SQUARE FEET':np.int64, 'GROSS SQUARE FEET':np.int64, 'YEAR BUILT':np.int64,
       'TAX CLASS AT TIME OF SALE':str, 'BUILDING CLASS AT TIME OF SALE':str,
       'SALE PRICE':np.float64, 'SALE DATE':pd.tslib.Timestamp})
    df_sales = df_sales.append(df)


print(df_sales.dtypes)
               
#Reviewing Sales-Data
df_sales.head()

#Reading crime data
df_crime = pd.read_csv("NYPD_Complaint_Data_Historic.csv",dtype = {'CMPLNT_NUM' : str, 'CMPLNT_FR_DT' : pd.tslib.Timestamp, 
                        'CMPLNT_FR_TM' : pd.tslib.Timestamp, 'CMPLNT_TO_DT' : pd.tslib.Timestamp,
                        'CMPLNT_TO_TM' : pd.tslib.Timestamp, 'RPT_DT' : pd.tslib.Timestamp, 'KY_CD' : str, 
                        'OFNS_DESC' : str, 'PD_CD' : str, 'PD_DESC' : str,'CRM_ATPT_CPTD_CD' : str, 
                        'LAW_CAT_CD' : str, 'JURIS_DESC' : str, 'BORO_NM' : str,'ADDR_PCT_CD' : str, 
                        'LOC_OF_OCCUR_DESC' : str, 'PREM_TYP_DESC' : str, 'PARKS_NM' : str,'HADEVELOPT' : str, 
                        'X_COORD_CD' : str, 'Y_COORD_CD' : str, 'Latitude' : str, 'Longitude' : str,'Lat_Lon' : str})

df_crime.head()

Data-Wrangling
#comparing data min max dates for available data
print(len(df_sales),'\n','min:',min(df_sales['SALE DATE']),'max:',max(df_sales['SALE DATE']))
print(len(df_crime),'\n','min:',min(df_crime['RPT_DT']),'max:',max(df_crime['RPT_DT']))

#looking for missing data values in data
print("SALE DATE: null values: \n",pd.isnull(df_sales["SALE DATE"]).value_counts())
print("CRIME DATA/RPT_DT: null values: \n",pd.isnull(df_crime["RPT_DT"]).value_counts())
print("CRIME DATA/CMPLNT_FR_DT: null values: \n",pd.isnull(df_crime["CMPLNT_FR_DT"]).value_counts())
print("CRIME DATA/CMPLNT_TO_DT: null values: \n",pd.isnull(df_crime["CMPLNT_TO_DT"]).value_counts())

#Filling missing data values
x = pd.isnull(df_crime["CMPLNT_FR_DT"])
y = pd.isnull(df_crime["CMPLNT_TO_DT"])

print("Before filling missing values:")
print(df_crime[["RPT_DT","CMPLNT_FR_DT","CMPLNT_TO_DT"]][x].head())
print(df_crime[["RPT_DT","CMPLNT_FR_DT","CMPLNT_TO_DT"]][y].head())

print("Filling missing values with corresponding row valus from RPT_DT")

df_crime["CMPLNT_FR_DT"] = df_crime["CMPLNT_FR_DT"].fillna(df_crime["RPT_DT"])
df_crime["CMPLNT_TO_DT"] = df_crime["CMPLNT_TO_DT"].fillna(df_crime["RPT_DT"])

print("After filling missing values:")
print(df_crime[["RPT_DT","CMPLNT_FR_DT","CMPLNT_TO_DT"]][x].head())
print(df_crime[["RPT_DT","CMPLNT_FR_DT","CMPLNT_TO_DT"]][y].head())

#correcting years 1015,1016 and 1026 in 'CMPLNT_FR_DT'. it is evident from RPT date that these years are typo errors in data.
pat1 = re.compile('(?P<one>\d\d/\d\d)(?P<two>/101)(?P<three>\d)')
pat2 = re.compile('(?P<one>\d\d/\d\d)(?P<two>/102)(?P<three>\d)')

repl1 = lambda x: pat1.sub(r'\g<1>/201\g<3>',x)
repl2 = lambda x: pat2.sub(r'\g<1>/201\g<3>',x)

index1 = df_crime[df_crime['CMPLNT_FR_DT'].str.contains(pat1) == True].index
index2 = df_crime[df_crime['CMPLNT_FR_DT'].str.contains(pat2) == True].index

for i in index1:
    df_crime['CMPLNT_FR_DT'][i] = repl1(df_crime['CMPLNT_FR_DT'][i])
    
    
for i in index2:
    df_crime['CMPLNT_FR_DT'][i] = repl2(df_crime['CMPLNT_FR_DT'][i])
    
#data type converstion to dates-time
df_crime['CMPLNT_FR_DT'] = pd.to_datetime(df_crime['CMPLNT_FR_DT'])
df_crime['CMPLNT_TO_DT'] = pd.to_datetime(df_crime['CMPLNT_TO_DT'])
df_crime['RPT_DT'] = pd.to_datetime(df_crime['RPT_DT'])

#Reviewing data after corrections:
df_crime.head()

#Since, crime_data does not have zip code information, 
#constructing data frame of lat_lon with crime count for each lat_lon coordinates
geo_codes = df_crime.Lat_Lon.value_counts()
geo_codes_zip = pd.DataFrame({'crime_count':geo_codes.values, 'lat_lon':geo_codes.index})
geo_codes_zip['zip'] = '0' #adding new column for zip-code

#Challenge:cannot extract reverse geocoding from google API for more than 100,000 records in a day
#work around: breaking geo_codes_zip dataframe into smaller data frames
#This is a very long process, as the open-source interfaces time-out every 10 minutes.

geo_codes_zip_1 = geo_codes_zip[0:90000]
geo_codes_zip_2 = geo_codes_zip[90000:]

#writing the data into local folders
geo_codes_zip_1.to_csv('geo_codes_1.csv')
geo_codes_zip_2.to_csv('geo_codes_2.csv')

geo_index = geo_codes_zip_1[(geo_codes_zip_1.zip == '0') | (pd.notnull(geo_codes_zip_1.zip) == False) ].index
#We used alternate APIs after every timeout to balance the load and to fasten the geo-coding process.
'''
#using google API for for extracting zip codes
for i in geo_index:
    zip_code = gc.google(geo_codes_zip_1.lat_lon[i],method='reverse').postal
    geo_codes_zip_1.zip[i] = zip_code
'''

#using OSM(Open Street Maps) API for for extracting zip codes
for i in geo_index:
    zip_code = gc.osm(geo_codes_zip_1.lat_lon[i],method='reverse').postal
    geo_codes_zip_1.zip[i] = zip_code

geo_codes_zip_1.to_csv('geo_codes_zip_1.csv')

geocoded_index = geo_codes_zip_1[(geo_codes_zip_1.zip != '0') & (pd.notnull(geo_codes_zip_1.zip)) ].index

#geo_codes_zip_1.iloc[geo_index,]
#Reviewing Geo-coded data after each timeout
geo_codes_zip_1.iloc[geocoded_index,]


#Reverse geo-coding 2nd set of lat-long data
geo_index = geo_codes_zip_2[(geo_codes_zip_2.zip == '0') | (pd.notnull(geo_codes_zip_2.zip) == False) ].index

#using google API for for extracting zip codes
for i in geo_index:
    zip_code = gc.google(geo_codes_zip_2.lat_lon[i],method='reverse').postal
    geo_codes_zip_2.zip[i] = zip_code

'''
#using OSM(Open Street Maps) API for for extracting zip codes
for i in geo_index:
    zip_code = gc.osm(geo_codes_zip_2.lat_lon[i],method='reverse').postal
    geo_codes_zip_1.zip[i] = zip_code
'''
geo_codes_zip_2.to_csv('geo_codes_zip_2.csv')


geocoded_index = geo_codes_zip_2[(geo_codes_zip_2.zip != '0') & (pd.notnull(geo_codes_zip_2.zip)) ].index

geo_codes_zip_2.iloc[geo_index,]
geo_codes_zip_2.iloc[geocoded_index,]

#reading saved zip_code files back from local drive
geo_codes_zip_1 = pd.read_csv('geo_codes_zip_1.csv',usecols=[0,1,2,3],index_col=0)

geo_codes_zip_2 = pd.read_csv('geo_codes_zip_2.csv',usecols=[0,1,2,3],index_col=0)

#appending 2 zip_code datasets
geo_codes_zip = geo_codes_zip_1.append(geo_codes_zip_2)

#truncating crime_count from zip_code file as this count is Lat_Lon level but not at zip_code level.
#zip_code level crime count will be calculated from crime data for different past-time intervals from sale-date.
geo_codes_zip.columns = ['crime_count','Lat_Lon','zip_code']
geo_codes_zip = geo_codes_zip[['Lat_Lon','zip_code']]
geo_codes_zip.head()


#merging zip_codes with crime data.
temp = pd.merge(df_crime,geo_codes_zip, on = 'Lat_Lon')
df_crime = temp

#writing dataframe to csv file in a local folder to save computational efforts in future
df_crime.to_csv('df_crime.csv',index = False)

#writing dataframe to csv file in a local folder to save computations in future
df_sales.to_csv('df_sales.csv',index = False)
#Next step: calculate number of crime-incidents reported for each zip-code in Sale-Data.
df_crime = pd.read_csv('df_crime.csv')
df_crime.head()
df_sales = pd.read_csv('df_sales.csv')
df_sales.head()

#Filtering invalid data records
temp_sales = df_sales.copy()
temp_sales = temp_sales[(temp_sales['SALE PRICE']!=0)&(temp_sales['YEAR BUILT']!=0)&(temp_sales['LAND SQUARE FEET']!=0)&(temp_sales['LAND SQUARE FEET'] <= temp_sales['GROSS SQUARE FEET'])]
temp_sales.shape

#sampling for Data Analysis
temp_sales_sample = temp_sales.sample(n=8000,random_state=9001)
temp_sales_sample = temp_sales_sample.reset_index(drop = True)
temp_sales_sample.head()

#Sorting Crime-data to ease query operations
df_crime = df_crime.sort_values(by = ["zip_code","RPT_DT"])
df_crime.head()

#Functions for calculating crime count for each zip code for different time-intervals in the past.

def past_crime_count(date,zip_code):
    '''
    This function calculates the number of past crimes from crime database
    reported at a given zip_code as of given date.
    '''
    x = len(df_crime[(df_crime['zip_code'] == zip_code) & (df_crime['RPT_DT']< date)])
    return x

def past6M_crime_count(date,zip_code):
    '''
    This function calculates the number of past crimes from crime database
    reported at a given zip_code in past 6 months from Sale_date of property.
    '''
    x = len(df_crime[(df_crime['zip_code'] == zip_code) & 
                     ((df_crime['RPT_DT']< date)&(pd.to_datetime(df_crime['RPT_DT'])> pd.to_datetime(date)-np.timedelta64(6,'M')))])
    return x

def past1Y_crime_count(date,zip_code):
    '''
    This function calculates the number of past crimes from crime database
    reported at a given zip_code in past 1 Year from Sale_date of property.
    '''
    x = len(df_crime[(df_crime['zip_code'] == zip_code) & 
                     ((df_crime['RPT_DT']< date)&(pd.to_datetime(df_crime['RPT_DT'])> pd.to_datetime(date)-np.timedelta64(1,'Y')))])
    return x

def past2Y_crime_count(date,zip_code):
    '''
    This function calculates the number of past crimes from crime database
    reported at a given zip_code in past 2 Years from Sale_date of property.
    '''
    x = len(df_crime[(df_crime['zip_code'] == zip_code) & 
                     ((df_crime['RPT_DT']< date)&(pd.to_datetime(df_crime['RPT_DT'])> pd.to_datetime(date)-np.timedelta64(2,'Y')))])
    return x

def past3Y_crime_count(date,zip_code):
    '''
    This function calculates the number of past crimes from crime database
    reported at a given zip_code in past 3 Years from Sale_date of property.
    '''
    x = len(df_crime[(df_crime['zip_code'] == zip_code) & 
                     ((df_crime['RPT_DT']< date)&(pd.to_datetime(df_crime['RPT_DT'])> pd.to_datetime(date)-np.timedelta64(3,'Y')))])
    return x

def past4Y_crime_count(date,zip_code):
    '''
    This function calculates the number of past crimes from crime database
    reported at a given zip_code in past 4 Years from Sale_date of property.
    '''
    x = len(df_crime[(df_crime['zip_code'] == zip_code) & 
                     ((df_crime['RPT_DT']< date)&(pd.to_datetime(df_crime['RPT_DT'])> pd.to_datetime(date)-np.timedelta64(4,'Y')))])
    return x


def past5Y_crime_count(date,zip_code):
    '''
    This function calculates the number of past crimes from crime database
    reported at a given zip_code in past 5 Years from Sale_date of property.
    '''
    x = len(df_crime[(df_crime['zip_code'] == zip_code) & 
                     ((df_crime['RPT_DT']< date)&(pd.to_datetime(df_crime['RPT_DT'])> pd.to_datetime(date)-np.timedelta64(5,'Y')))])
    return x

#Function calls to calculate crime data
print('\nStart Time: ',datetime.datetime.now())

temp_sales_sample['past_crime_count'] = temp_sales_sample.apply(lambda row: past_crime_count(row['SALE DATE'],row['ZIP CODE']),axis=1)
temp_sales_sample.to_csv('sales_data.csv',index = False)
print('\n: past_crime_count calculated : Time: ',datetime.datetime.now())

temp_sales_sample['past5Y_crime_count'] = temp_sales_sample.apply(lambda row: past5Y_crime_count(row['SALE DATE'],row['ZIP CODE']),axis=1)
temp_sales_sample.to_csv('sales_data.csv',index = False)
print('\n: past5Y_crime_count calculated : Time: ',datetime.datetime.now())

temp_sales_sample['past2Y_crime_count'] = temp_sales_sample.apply(lambda row: past2Y_crime_count(row['SALE DATE'],row['ZIP CODE']),axis=1)
temp_sales_sample.to_csv('sales_data.csv',index = False)
print('\n: past2Y_crime_count calculated : Time: ',datetime.datetime.now())

temp_sales_sample['past6M_crime_count'] = temp_sales_sample.apply(lambda row: past6M_crime_count(row['SALE DATE'],row['ZIP CODE']),axis=1)
temp_sales_sample.to_csv('sales_data.csv',index = False)
print('\n: past6M_crime_count calculated : Time: ',datetime.datetime.now())

temp_sales_sample['past1Y_crime_count'] = temp_sales_sample.apply(lambda row: past1Y_crime_count(row['SALE DATE'],row['ZIP CODE']),axis=1)
temp_sales_sample.to_csv('sales_data.csv',index = False)
print('\n: past1Y_crime_count calculated : Time: ',datetime.datetime.now())


temp_sales_sample['past3Y_crime_count'] = temp_sales_sample.apply(lambda row: past3Y_crime_count(row['SALE DATE'],row['ZIP CODE']),axis=1)
temp_sales_sample.to_csv('sales_data.csv',index = False)
print('\n: past3Y_crime_count calculated : Time: ',datetime.datetime.now())

temp_sales_sample['past4Y_crime_count'] sales_data= temp_sales_sample.apply(lambda row: past4Y_crime_count(row['SALE DATE'],row['ZIP CODE']),axis=1)
temp_sales_sample.to_csv('sales_data.csv',index = False)
print('\n: past4Y_crime_count calculated : Time: ',datetime.datetime.now())

#Functions for calculating crime types count for each zip code for different time-intervals in the past.

def past_MISDEMEANOR_count(date,zip_code):
    '''
    This function calculates the number of past MISDEMEANOR crimes from crime database
    reported at a given zip_code as of given date.
    '''
    x = len(df_crime[(df_crime['zip_code'] == zip_code) & (df_crime['RPT_DT']< date) & (df_crime['LAW_CAT_CD'] == 'MISDEMEANOR')])
    return x

def past_FELONY_count(date,zip_code):
    '''
    This function calculates the number of past FELONY crimes from crime database
    reported at a given zip_code as of given date.
    '''
    x = len(df_crime[(df_crime['zip_code'] == zip_code) & (df_crime['RPT_DT']< date) & (df_crime['LAW_CAT_CD'] == 'FELONY')])
    return x

def past_VIOLATION_count(date,zip_code):
    '''
    This function calculates the number of past VIOLATION crimes from crime database
    reported at a given zip_code as of given date.
    '''
    x = len(df_crime[(df_crime['zip_code'] == zip_code) & (df_crime['RPT_DT']< date) & (df_crime['LAW_CAT_CD'] == 'VIOLATION')])
    return x

def past6M_MISDEMEANOR_count(date,zip_code):
    '''
    This function calculates the number of past MISDEMEANOR crimes from crime database
    reported at a given zip_code in past 6 months from Sale_date of property.
    '''
    x = len(df_crime[(df_crime['zip_code'] == zip_code) & (df_crime['LAW_CAT_CD'] == 'MISDEMEANOR') & 
                     ((df_crime['RPT_DT']< date)&(pd.to_datetime(df_crime['RPT_DT'])> pd.to_datetime(date)-np.timedelta64(6,'M')))])
    return x

def past6M_FELONY_count(date,zip_code):
    '''
    This function calculates the number of past FELONY crimes from crime database
    reported at a given zip_code in past 6 months from Sale_date of property.
    '''
    x = len(df_crime[(df_crime['zip_code'] == zip_code) & (df_crime['LAW_CAT_CD'] == 'FELONY') & 
                     ((df_crime['RPT_DT']< date)&(pd.to_datetime(df_crime['RPT_DT'])> pd.to_datetime(date)-np.timedelta64(6,'M')))])
    return x

def past6M_VIOLATION_count(date,zip_code):
    '''
    This function calculates the number of past VIOLATION crimes from crime database
    reported at a given zip_code in past 6 months from Sale_date of property.
    '''
    x = len(df_crime[(df_crime['zip_code'] == zip_code) & (df_crime['LAW_CAT_CD'] == 'VIOLATION') & 
                     ((df_crime['RPT_DT']< date)&(pd.to_datetime(df_crime['RPT_DT'])> pd.to_datetime(date)-np.timedelta64(6,'M')))])
    return x

def past1Y_MISDEMEANOR_count(date,zip_code):
    '''
    This function calculates the number of past crimes from crime database
    reported at a given zip_code in past 1 Year from Sale_date of property.
    '''
    x = len(df_crime[(df_crime['zip_code'] == zip_code) & (df_crime['LAW_CAT_CD'] == 'MISDEMEANOR') & 
                     ((df_crime['RPT_DT']< date)&(pd.to_datetime(df_crime['RPT_DT'])> pd.to_datetime(date)-np.timedelta64(1,'Y')))])
    return x

def past1Y_FELONY_count(date,zip_code):
    '''
    This function calculates the number of past crimes from crime database
    reported at a given zip_code in past 1 Year from Sale_date of property.
    '''
    x = len(df_crime[(df_crime['zip_code'] == zip_code) & (df_crime['LAW_CAT_CD'] == 'FELONY') & 
                     ((df_crime['RPT_DT']< date)&(pd.to_datetime(df_crime['RPT_DT'])> pd.to_datetime(date)-np.timedelta64(1,'Y')))])
    return x

def past1Y_VIOLATION_count(date,zip_code):
    '''
    This function calculates the number of past crimes from crime database
    reported at a given zip_code in past 1 Year from Sale_date of property.
    '''
    x = len(df_crime[(df_crime['zip_code'] == zip_code) & (df_crime['LAW_CAT_CD'] == 'VIOLATION') & 
                     ((df_crime['RPT_DT']< date)&(pd.to_datetime(df_crime['RPT_DT'])> pd.to_datetime(date)-np.timedelta64(1,'Y')))])
    return x

def past2Y_MISDEMEANOR_count(date,zip_code):
    '''
    This function calculates the number of past crimes from crime database
    reported at a given zip_code in past 2 Years from Sale_date of property.
    '''
    x = len(df_crime[(df_crime['zip_code'] == zip_code) & (df_crime['LAW_CAT_CD'] == 'MISDEMEANOR') & 
                     ((df_crime['RPT_DT']< date)&(pd.to_datetime(df_crime['RPT_DT'])> pd.to_datetime(date)-np.timedelta64(2,'Y')))])
    return x

def past2Y_FELONY_count(date,zip_code):
    '''
    This function calculates the number of past crimes from crime database
    reported at a given zip_code in past 2 Years from Sale_date of property.
    '''
    x = len(df_crime[(df_crime['zip_code'] == zip_code) & (df_crime['LAW_CAT_CD'] == 'FELONY') & 
                     ((df_crime['RPT_DT']< date)&(pd.to_datetime(df_crime['RPT_DT'])> pd.to_datetime(date)-np.timedelta64(2,'Y')))])
    return x

def past2Y_VIOLATION_count(date,zip_code):
    '''
    This function calculates the number of past crimes from crime database
    reported at a given zip_code in past 2 Years from Sale_date of property.
    '''
    x = len(df_crime[(df_crime['zip_code'] == zip_code) & (df_crime['LAW_CAT_CD'] == 'VIOLATION') & 
                     ((df_crime['RPT_DT']< date)&(pd.to_datetime(df_crime['RPT_DT'])> pd.to_datetime(date)-np.timedelta64(2,'Y')))])
    return x

def past3Y_MISDEMEANOR_count(date,zip_code):
    '''
    This function calculates the number of past crimes from crime database
    reported at a given zip_code in past 3 Years from Sale_date of property.
    '''
    x = len(df_crime[(df_crime['zip_code'] == zip_code) & (df_crime['LAW_CAT_CD'] == 'MISDEMEANOR') & 
                     ((df_crime['RPT_DT']< date)&(pd.to_datetime(df_crime['RPT_DT'])> pd.to_datetime(date)-np.timedelta64(3,'Y')))])
    return x

def past3Y_FELONY_count(date,zip_code):
    '''
    This function calculates the number of past crimes from crime database
    reported at a given zip_code in past 3 Years from Sale_date of property.
    '''
    x = len(df_crime[(df_crime['zip_code'] == zip_code) & (df_crime['LAW_CAT_CD'] == 'FELONY') & 
                     ((df_crime['RPT_DT']< date)&(pd.to_datetime(df_crime['RPT_DT'])> pd.to_datetime(date)-np.timedelta64(3,'Y')))])
    return x

def past3Y_VIOLATION_count(date,zip_code):
    '''
    This function calculates the number of past crimes from crime database
    reported at a given zip_code in past 3 Years from Sale_date of property.
    '''
    x = len(df_crime[(df_crime['zip_code'] == zip_code) & (df_crime['LAW_CAT_CD'] == 'VIOLATION') & 
                     ((df_crime['RPT_DT']< date)&(pd.to_datetime(df_crime['RPT_DT'])> pd.to_datetime(date)-np.timedelta64(3,'Y')))])
    return x

def past4Y_MISDEMEANOR_count(date,zip_code):
    '''
    This function calculates the number of past crimes from crime database
    reported at a given zip_code in past 4 Years from Sale_date of property.
    '''
    x = len(df_crime[(df_crime['zip_code'] == zip_code) & (df_crime['LAW_CAT_CD'] == 'MISDEMEANOR') & 
                     ((df_crime['RPT_DT']< date)&(pd.to_datetime(df_crime['RPT_DT'])> pd.to_datetime(date)-np.timedelta64(4,'Y')))])
    return x

def past4Y_FELONY_count(date,zip_code):
    '''
    This function calculates the number of past crimes from crime database
    reported at a given zip_code in past 4 Years from Sale_date of property.
    '''
    x = len(df_crime[(df_crime['zip_code'] == zip_code) & (df_crime['LAW_CAT_CD'] == 'FELONY') & 
                     ((df_crime['RPT_DT']< date)&(pd.to_datetime(df_crime['RPT_DT'])> pd.to_datetime(date)-np.timedelta64(4,'Y')))])
    return x

def past4Y_VIOLATION_count(date,zip_code):
    '''
    This function calculates the number of past crimes from crime database
    reported at a given zip_code in past 4 Years from Sale_date of property.
    '''
    x = len(df_crime[(df_crime['zip_code'] == zip_code) & (df_crime['LAW_CAT_CD'] == 'VIOLATION') & 
                     ((df_crime['RPT_DT']< date)&(pd.to_datetime(df_crime['RPT_DT'])> pd.to_datetime(date)-np.timedelta64(4,'Y')))])
    return x

def past5Y_MISDEMEANOR_count(date,zip_code):
    '''
    This function calculates the number of past crimes from crime database
    reported at a given zip_code in past 5 Years from Sale_date of property.
    '''
    x = len(df_crime[(df_crime['zip_code'] == zip_code) & (df_crime['LAW_CAT_CD'] == 'MISDEMEANOR') & 
                     ((df_crime['RPT_DT']< date)&(pd.to_datetime(df_crime['RPT_DT'])> pd.to_datetime(date)-np.timedelta64(5,'Y')))])
    return x

def past5Y_FELONY_count(date,zip_code):
    '''
    This function calculates the number of past crimes from crime database
    reported at a given zip_code in past 5 Years from Sale_date of property.
    '''
    x = len(df_crime[(df_crime['zip_code'] == zip_code) & (df_crime['LAW_CAT_CD'] == 'FELONY') & 
                     ((df_crime['RPT_DT']< date)&(pd.to_datetime(df_crime['RPT_DT'])> pd.to_datetime(date)-np.timedelta64(5,'Y')))])
    return x

def past5Y_VIOLATION_count(date,zip_code):
    '''
    This function calculates the number of past crimes from crime database
    reported at a given zip_code in past 5 Years from Sale_date of property.
    '''
    x = len(df_crime[(df_crime['zip_code'] == zip_code) & (df_crime['LAW_CAT_CD'] == 'VIOLATION') & 
                     ((df_crime['RPT_DT']< date)&(pd.to_datetime(df_crime['RPT_DT'])> pd.to_datetime(date)-np.timedelta64(5,'Y')))])
    return x

#Function calls to calculate past MISDEMEANOR crime counts for each sale sample
print('\nStart Time: ',datetime.datetime.now())

df_sales_sample['past_MISDEMEANOR_count'] = df_sales_sample.apply(lambda row: past_MISDEMEANOR_count(row['SALE DATE'],row['ZIP CODE']),axis=1)
df_sales_sample.to_csv('df_sales_sample_2.csv',index = False)
print('\n: past_MISDEMEANOR_count calculated : Time: ',datetime.datetime.now())
df_sales_sample.head()

#Function calls to calculate past FELONY crime counts for each sale sample
print('\nStart Time: ',datetime.datetime.now())

df_sales_sample['past_FELONY_count'] = df_sales_sample.apply(lambda row: past_FELONY_count(row['SALE DATE'],row['ZIP CODE']),axis=1)
df_sales_sample.to_csv('df_sales_sample_2.csv',index = False)
print('\n: past_FELONY_count calculated : Time: ',datetime.datetime.now())
df_sales_sample.head()

#Function calls to calculate past VIOLATION crime counts for each sale sample
print('\nStart Time: ',datetime.datetime.now())

df_sales_sample['past_VIOLATION_count'] = df_sales_sample.apply(lambda row: past_VIOLATION_count(row['SALE DATE'],row['ZIP CODE']),axis=1)
df_sales_sample.to_csv('df_sales_sample_2.csv',index = False)
print('\n: past_VIOLATION_count calculated : Time: ',datetime.datetime.now())
df_sales_sample.head()

Exploratory Data Analysis:

#Reading cleaned sample file from local folder.
df_sales_sample = pd.read_csv('df_sales_sample_2.csv')
df_sales_sample.head()

#Reviewing Sales-Price
df_sales_sample["SALE PRICE"].describe()

#converting date values to epoch time to make it as continuous variable for Data-Analysis.
sample_data = df_sales_sample.loc[:,['BOROUGH', 'NEIGHBORHOOD', 'BUILDING CLASS CATEGORY','TAX CLASS AT PRESENT', 'BLOCK', 'LOT','BUILDING CLASS AT PRESENT', 'ZIP CODE','RESIDENTIAL UNITS', 'COMMERCIAL UNITS', 'TOTAL UNITS','LAND SQUARE FEET', 'GROSS SQUARE FEET', 'YEAR BUILT',
                                    'TAX CLASS AT TIME OF SALE', 'BUILDING CLASS AT TIME OF SALE','SALE PRICE', 'SALE DATE', 'past_crime_count', 'past5Y_crime_count','past2Y_crime_count', 'past6M_crime_count', 'past1Y_crime_count','past3Y_crime_count', 'past_MISDEMEANOR_count', 
                                    'past_FELONY_count', 'past_VIOLATION_count']].copy()
sample_data["SALE DATE"] = sample_data["SALE DATE"].apply(lambda x:(time.mktime(pd.datetime.strptime(x, "%Y-%m-%d").timetuple())))
sample_data = sample_data[(sample_data['SALE PRICE']!=0)&(sample_data['YEAR BUILT']!=0)&(sample_data['GROSS SQUARE FEET']!=0)]
sample_data.shape

#histogram of target variable
temp = sample_data["SALE PRICE"].copy()
fig, ax = plt.subplots(9,2,figsize=(15,40))

ax[0,0].hist(temp,bins = 50)
ax[0,0].set_title("SALE PRICE  - {} data points".format(temp.shape[0]))

ax[0,1].hist(np.log(temp),bins = 50)
ax[0,1].set_title("log(SALE PRICE)  - {} data points".format(temp.shape[0]))

ax[1,0].hist(temp[temp <= 1e8],bins = 50)
ax[1,0].set_title("SALE PRICE <= 100M USD  - {} data points".format(temp[temp <= 1e8].shape[0]))

ax[1,1].hist(np.log(temp[temp <= 1e8]),bins = 50)
ax[1,1].set_title("log(SALE PRICE <= 100M USD ) - {} data points".format(temp[temp <= 1e8].shape[0]))

ax[2,0].hist(temp[temp <= 1e7],bins = 50)
ax[2,0].set_title("SALE PRICE <= 10M USD  - {} data points".format(temp[temp <= 1e7].shape[0]))

ax[2,1].hist(np.log(temp[temp <= 1e7]),bins = 50)
ax[2,1].set_title("log(SALE PRICE <= 10M USD)  - {} data points".format(temp[temp <= 1e7].shape[0]))

ax[3,0].hist(temp[(temp >= 1e4) & (temp <= 1e7)],bins = 50)
ax[3,0].set_title("10K USD <= SALE PRICE <= 10M USD  - {} data points".format(temp[(temp >= 1e4) & (temp <= 1e7)].shape[0]))

ax[3,1].hist(np.log(temp[(temp >= 1e4) & (temp <= 1e7)]),bins = 50)
ax[3,1].set_title("log(10K USD < SALE PRICE < 10M USD)  - {} data points".format(temp[(temp >= 1e4) & (temp <= 1e7)].shape[0]))

ax[4,0].hist(temp[(temp >= 1e5) & (temp <= 1e7)],bins = 50)
ax[4,0].set_title("100K USD <= SALE PRICE <= 10M USD  - {} data points".format(temp[(temp >= 1e5) & (temp <= 1e7)].shape[0]))

ax[4,1].hist(np.log(temp[(temp >= 1e5) & (temp <= 1e7)]),bins = 50)
ax[4,1].set_title("log(100K USD <= SALE PRICE <= 10M USD)  - {} data points".format(temp[(temp >= 1e5) & (temp <= 1e7)].shape[0]))

ax[5,0].hist(temp[(temp >= 3e5) & (temp <= 1e7)],bins = 50)
ax[5,0].set_title("300K USD <= SALE PRICE <= 10M USD  - {} data points".format(temp[(temp >= 3e5) & (temp <= 1e7)].shape[0]))

ax[5,1].hist(np.log(temp[(temp >= 3e5) & (temp <= 1e7)]),bins = 50)
ax[5,1].set_title("log(300K USD <= SALE PRICE <= 10M USD)  - {} data points".format(temp[(temp >= 3e5) & (temp <= 1e7)].shape[0]))

ax[6,0].hist(temp[(temp >= 3e5) & (temp <= 1e6)],bins = 50)
ax[6,0].set_title("300K USD <= SALE PRICE <= 1M USD  - {} data points".format(temp[(temp >= 3e5) & (temp <= 1e6)].shape[0]))

ax[6,1].hist(np.log(temp[(temp >= 3e5) & (temp <= 1e6)]),bins = 50)
ax[6,1].set_title("log(300K USD <= SALE PRICE <= 1M USD)  - {} data points".format(temp[(temp >= 3e5) & (temp <= 1e6)].shape[0]))

ax[7,0].hist(temp[(temp >= 1e5) & (temp <= 1e6)],bins = 50)
ax[7,0].set_title("100K USD <= SALE PRICE <= 1M USD  - {} data points".format(temp[(temp >= 1e5) & (temp <= 1e6)].shape[0]))

ax[7,1].hist(np.log(temp[(temp >= 1e5) & (temp <= 1e6)]),bins = 50)
ax[7,1].set_title("log(100K USD <= SALE PRICE <= 1M USD)  - {} data points".format(temp[(temp >= 1e5) & (temp <= 1e6)].shape[0]))

ax[8,0].hist(temp[(temp >= 3.25e5) & (temp <= 8.5e5)],bins = 50)
ax[8,0].set_title("325K USD <= SALE PRICE <= 850K USD  - {} data points".format(temp[(temp >= 3.25e5) & (temp <= 8.5e5)].shape[0]))

ax[8,1].hist(np.log(temp[(temp >= 3.25e5) & (temp <= 8.5e5)]),bins = 50)
ax[8,1].set_title("log(325K USD <= SALE PRICE <= 850K USD)  - {} data points".format(temp[(temp >= 3.25e5) & (temp <= 8.5e5)].shape[0]))

plt.subplots_adjust(wspace= 0.25, hspace = 0.25)
plt.show()


#Finding Correlation between Features Vs. Target variable(SALES-PRICE)
model_data = pd.get_dummies(sample_data[(sample_data['SALE PRICE']>=1e5) & (sample_data['SALE PRICE']<=1e7)])
print("model_data  shape:",model_data.shape)

temp_x = model_data.drop("SALE PRICE",axis = 1).copy()
temp_y = model_data["SALE PRICE"].copy()
corr = []
corr_abs = []
p_value = []
for i in range(len(temp_x.columns)):
    corr.append(st.pearsonr(temp_x.iloc[:,i],temp_y)[0])
    p_value.append(st.pearsonr(temp_x.iloc[:,i],temp_y)[1])
    corr_abs.append(abs(st.pearsonr(temp_x.iloc[:,i],temp_y)[0]))

corr = pd.DataFrame({'feature':temp_x.columns,'corr_coef':corr,'p_value':p_value,'corr_abs':corr_abs})

corr = corr.sort_values(by = ["corr_abs","p_value"],ascending = False).reset_index(drop = True)
corr_tbl = corr.loc[:,["feature","corr_coef","p_value"]].copy()
corr_tbl.to_csv("correlation_table.csv")

corr[corr.p_value <=0.05]


#correlation of crime-data with sales-price
corr[corr.feature.str.contains('past')]

Modeling and Evaluation:

#Final
model_data = pd.get_dummies(sample_data[(sample_data['SALE PRICE']>=1e5) & (sample_data['SALE PRICE']<=1e7)])
print('model_data shape for 100K USD <= SALE PRICE <= 10M USD:',model_data.shape)

#Splitting Train & Test data
X = model_data.drop("SALE PRICE",axis = 1).values
Y = model_data["SALE PRICE"].values
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2,random_state = 21)

#Creating data frame for adding model performance metrics
model_perf = pd.DataFrame(columns = ['model_name','RMSE','Rsquare','Train_RMSE','Train_Rsquare'])

fig, ax = plt.subplots(3,2,figsize=(15,10))

#Running Linear model
print('\nLinear Regresion results:')
model = LinearRegression()
model.fit(x_train,y_train)
model_linear = model

y_pred_train = model.predict(x_train)
y_pred_train_mse = mean_squared_error(y_train,y_pred_train)
y_pred_train_rmse = np.sqrt(y_pred_train_mse)
train_Rsquare = model.score(x_train,y_train)

y_pred_test = model.predict(x_test)
y_pred_test_mse = mean_squared_error(y_test,y_pred_test)
y_pred_test_rmse = np.sqrt(y_pred_test_mse)
test_Rsquare = model.score(x_test,y_test)

df_pred_train = pd.DataFrame({'y_train':y_train,'y_pred_train':y_pred_train,'residuals':(y_train - y_pred_train),'deviation%':(y_train - y_pred_train)*100/y_train})
df_pred_test = pd.DataFrame({'y_test':y_test,'y_pred_test':y_pred_test,'residuals':(y_test - y_pred_test),'deviation%':(y_test - y_pred_test)*100/y_test})

print('\nModel Performance on Training data')
print("Mean-Squared Error: {} \nRoot-Mean-Squared Error: {} \nRsquare: {}".format(y_pred_train_mse,y_pred_train_rmse,train_Rsquare))
print("\nPredictions on Test Data:\n",df_pred_train.head(10))

print('\nModel Performance on Test data')
print("Mean-Squared Error: {} \nRoot-Mean-Squared Error: {} \nRsquare: {}".format(y_pred_test_mse,y_pred_test_rmse,test_Rsquare))
print("\nPredictions on Test Data:\n",df_pred_test.head(10))

#Appending Model performance metrics
model_perf = model_perf.append({'model_name':'Linear','RMSE':y_pred_test_rmse,'Rsquare':test_Rsquare,'Train_RMSE':y_pred_train_rmse,'Train_Rsquare':train_Rsquare},ignore_index = True)

#residual plot
residuals = df_pred_test['residuals'].values
fitted = y_pred_test
ax[0,0].scatter(fitted,residuals)
ax[0,0].set_title("Residual Plott for Linear Model")
ax[0,0].set_xlabel("predicted values")
ax[0,0].set_ylabel("residuals")

#running Ridge-Regression

print('\n\nRidge-Regression results:')
#Tuning paramerters
param_grid = {'alpha':[0.001,0.01,0.1,1,10,100],'normalize': ["True"]}
ridge = Ridge()
model = GridSearchCV(ridge,param_grid, cv = 5)
model.fit(x_train,y_train)
print("best parameters: {}".format(model.best_params_))
params  = model.best_params_

#Running model again with best parameters:
model= Ridge(params['alpha'],params['normalize'])
model.fit(x_train, y_train)
model_ridge = model

y_pred_train = model.predict(x_train)
y_pred_train_mse = mean_squared_error(y_train,y_pred_train)
y_pred_train_rmse = np.sqrt(y_pred_train_mse)
train_Rsquare = model.score(x_train,y_train)

y_pred_test = model.predict(x_test)
y_pred_test_mse = mean_squared_error(y_test,y_pred_test)
y_pred_test_rmse = np.sqrt(y_pred_test_mse)
test_Rsquare = model.score(x_test,y_test)

df_pred_train = pd.DataFrame({'y_train':y_train,'y_pred_train':y_pred_train,'residuals':(y_train - y_pred_train),'deviation%':(y_train - y_pred_train)*100/y_train})
df_pred_test = pd.DataFrame({'y_test':y_test,'y_pred_test':y_pred_test,'residuals':(y_test - y_pred_test),'deviation%':(y_test - y_pred_test)*100/y_test})

print('\nModel Performance on Training data:')
print("Mean-Squared Error: {} \nRoot-Mean-Squared Error: {} \nRsquare: {}".format(y_pred_train_mse,y_pred_train_rmse,train_Rsquare))
print("\nPredictions on Training Data:\n",df_pred_train.head(10))

print('\nModel Performance on Test data:')
print("Mean-Squared Error: {} \nRoot-Mean-Squared Error: {} \nRsquare: {}".format(y_pred_test_mse,y_pred_test_rmse,test_Rsquare))
print("\nPredictions on Test Data:\n",df_pred_test.head(10))

#Appending Model performance metrics
model_perf = model_perf.append({'model_name':'Ridge-Regression','RMSE':y_pred_test_rmse,'Rsquare':test_Rsquare,'Train_RMSE':y_pred_train_rmse,'Train_Rsquare':train_Rsquare},ignore_index = True)

#residual plot
residuals = df_pred_test['residuals'].values
fitted = y_pred_test
ax[0,1].scatter(fitted,residuals)
ax[0,1].set_title("Residual Plott for Ridge-Regression model")
ax[0,1].set_xlabel("predicted values")
ax[0,1].set_ylabel("residuals")

#running Lasso-Regression

print('\n\nLasso-Regression results:')
#Tuning paramerters
param_grid = {'alpha':[100,200,300,400,500],'normalize': ["True"]}
lasso= Lasso()
model = GridSearchCV(lasso,param_grid, cv = 5)
model.fit(x_train,y_train)
print("\nbest parameters: {}".format(model.best_params_))
params  = model.best_params_

#Running model again with best parameters:
model= Lasso(params['alpha'],params['normalize'])
model.fit(x_train, y_train)
model_lasso = model

y_pred_train = model.predict(x_train)
y_pred_train_mse = mean_squared_error(y_train,y_pred_train)
y_pred_train_rmse = np.sqrt(y_pred_train_mse)
train_Rsquare = model.score(x_train,y_train)

y_pred_test = model.predict(x_test)
y_pred_test_mse = mean_squared_error(y_test,y_pred_test)
y_pred_test_rmse = np.sqrt(y_pred_test_mse)
test_Rsquare = model.score(x_test,y_test)

df_pred_train = pd.DataFrame({'y_train':y_train,'y_pred_train':y_pred_train,'residuals':(y_train - y_pred_train),'deviation%':(y_train - y_pred_train)*100/y_train})
df_pred_test = pd.DataFrame({'y_test':y_test,'y_pred_test':y_pred_test,'residuals':(y_test - y_pred_test),'deviation%':(y_test - y_pred_test)*100/y_test})

print('\nModel Performance on Training data:')
print("Mean-Squared Error: {} \nRoot-Mean-Squared Error: {} \nRsquare: {}".format(y_pred_train_mse,y_pred_train_rmse,train_Rsquare))
print("\nPredictions on Training Data:\n",df_pred_train.head(10))

print('\nModel Performance on Test data:')
print("Mean-Squared Error: {} \nRoot-Mean-Squared Error: {} \nRsquare: {}".format(y_pred_test_mse,y_pred_test_rmse,test_Rsquare))
print("\nPredictions on Test Data:\n",df_pred_test.head(10))

#Appending Model performance metrics
model_perf = model_perf.append({'model_name':'Lasso-Regression','RMSE':y_pred_test_rmse,'Rsquare':test_Rsquare,'Train_RMSE':y_pred_train_rmse,'Train_Rsquare':train_Rsquare},ignore_index = True)

#residual plot
residuals = df_pred_test['residuals'].values
fitted = y_pred_test
ax[1,0].scatter(fitted,residuals)
ax[1,0].set_title("Residual Plott for Lasso-Regression model")
ax[1,0].set_xlabel("predicted values")
ax[1,0].set_ylabel("residuals")

#running RandomForest regressor

print('\n\nRandom-Forest results:')
model = RandomForestRegressor(n_estimators = 100)
model.fit(x_train,y_train)

model_rf = model

y_pred_train = model.predict(x_train)
y_pred_train_mse = mean_squared_error(y_train,y_pred_train)
y_pred_train_rmse = np.sqrt(y_pred_train_mse)
train_Rsquare = model.score(x_train,y_train)

y_pred_test = model.predict(x_test)
y_pred_test_mse = mean_squared_error(y_test,y_pred_test)
y_pred_test_rmse = np.sqrt(y_pred_test_mse)
test_Rsquare = model.score(x_test,y_test)

df_pred_train = pd.DataFrame({'y_train':y_train,'y_pred_train':y_pred_train,'residuals':(y_train - y_pred_train),'deviation%':(y_train - y_pred_train)*100/y_train})
df_pred_test = pd.DataFrame({'y_test':y_test,'y_pred_test':y_pred_test,'residuals':(y_test - y_pred_test),'deviation%':(y_test - y_pred_test)*100/y_test})

print('\nModel Performance on Training data:')
print("Mean-Squared Error: {} \nRoot-Mean-Squared Error: {} \nRsquare: {}".format(y_pred_train_mse,y_pred_train_rmse,train_Rsquare))
print("\nPredictions on Training Data:\n",df_pred_train.head(10))

print('\nModel Performance on Test data:')
print("Mean-Squared Error: {} \nRoot-Mean-Squared Error: {} \nRsquare: {}".format(y_pred_test_mse,y_pred_test_rmse,test_Rsquare))
print("\nPredictions on Test Data:\n",df_pred_test.head(10))

#Appending Model performance metrics
model_perf = model_perf.append({'model_name':'Random-Forest','RMSE':y_pred_test_rmse,'Rsquare':test_Rsquare,'Train_RMSE':y_pred_train_rmse,'Train_Rsquare':train_Rsquare},ignore_index = True)

#residual plot
residuals = df_pred_test['residuals'].values
fitted = y_pred_test
ax[1,1].scatter(fitted,residuals)
ax[1,1].set_title("Residual Plott for Random-Forest Model")
ax[1,1].set_xlabel("predicted values")
ax[1,1].set_ylabel("residuals")


#running Gradient-Boost regressor:

print('\n\nGradient-Boost  results:')
model = GradientBoostingRegressor()
model.fit(x_train,y_train)
model_gb = model

y_pred_train = model.predict(x_train)
y_pred_train_mse = mean_squared_error(y_train,y_pred_train)
y_pred_train_rmse = np.sqrt(y_pred_train_mse)
train_Rsquare = model.score(x_train,y_train)

y_pred_test = model.predict(x_test)
y_pred_test_mse = mean_squared_error(y_test,y_pred_test)
y_pred_test_rmse = np.sqrt(y_pred_test_mse)
test_Rsquare = model.score(x_test,y_test)

df_pred_train = pd.DataFrame({'y_train':y_train,'y_pred_train':y_pred_train,'residuals':(y_train - y_pred_train),'deviation%':(y_train - y_pred_train)*100/y_train})
df_pred_test = pd.DataFrame({'y_test':y_test,'y_pred_test':y_pred_test,'residuals':(y_test - y_pred_test),'deviation%':(y_test - y_pred_test)*100/y_test})


print('\nModel Performance on Training data:')
print("Mean-Squared Error: {} \nRoot-Mean-Squared Error: {} \nRsquare: {}".format(y_pred_train_mse,y_pred_train_rmse,train_Rsquare))
print("\nPredictions on Training Data:\n",df_pred_train.head(10))

print('\nModel Performance on Test data:')
print("Mean-Squared Error: {} \nRoot-Mean-Squared Error: {} \nRsquare: {}".format(y_pred_test_mse,y_pred_test_rmse,test_Rsquare))
print("\nPredictions on Test Data:\n",df_pred_test.head(10))

#Appending Model performance metrics
model_perf = model_perf.append({'model_name':'Gradient-Boost','RMSE':y_pred_test_rmse,'Rsquare':test_Rsquare,'Train_RMSE':y_pred_train_rmse,'Train_Rsquare':train_Rsquare},ignore_index = True)

#residual plot
residuals = df_pred_test['residuals'].values
fitted = y_pred_test
ax[2,0].scatter(fitted,residuals)
ax[2,0].set_title("Residual Plott for Gradeint-Boost Model")
ax[2,0].set_xlabel("predicted values")
ax[2,0].set_ylabel("residuals")

print('\n\nModel-Performance Table:\n',model_perf)
plt.subplots_adjust(wspace= 0.25, hspace = 0.5)
plt.show()
print('\n\nModel-Coefficients/Estimators:\n')
print('\nLinear-Regression Model Coefficients:\n',model_linear.coef_)
print('\nRidge-Regression Model Coefficients:\n',model_ridge.coef_)
print('\nLasso-Regression Model Coefficients:\n',model_lasso.coef_)
print('\nRandom Forest-Model estimators:\n',model_rf.estimators_)
print('\nRandom Forest-Model Feature-Importances:\n',model_rf.feature_importances_)
print('\nGradient Boost-Model estimators:\n',model_gb.estimators_)
print('\nGradient Boost-Model Feature-Importances:\n',model_gb.feature_importances_)

