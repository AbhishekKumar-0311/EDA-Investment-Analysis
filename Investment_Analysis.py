#!/usr/bin/env python
# coding: utf-8

# # Sparks Funds Investment Analysis

# ## Checkpoint 0: Setting up the system

# In[1]:


# To get multiple outputs in the same cell

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[2]:


# Import the EDA required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Import the generic utility libraries

import os
import random
import datetime as datetime


# In[3]:


# Set the required global options

# To display all the columns in dataframe
pd.set_option( "display.max_columns", None)
# Setting the display fromat
pd.set_option('display.float_format', lambda x: '%.2f' % x)
#pd.reset_option('display.float_format')


# ## Checkpoint 1: Data Extraction and Cleaning

# In[4]:


# Read the raw csv file 'companies.csv' - containing the basic data of the companies
# encoding - The type of encoding format needs to be used for data reading

companies = pd.read_csv('companies.csv', sep = ',', encoding = 'ISO-8859-1')


# In[5]:


# Read the raw csv file 'rounds2.csv' - containing the basic data of the companies
# encoding - The type of encoding format needs to be used for data reading

rounds2 = pd.read_csv('rounds2.csv', encoding = 'ISO-8859-1')


# In[6]:


# Displaying the random 6 rows of the entire dataframe
# This step is executed multiple times, to get a random view of rows present and its typical properties

companies.sample(6)
rounds2.sample(6)


# In[7]:


companies.shape


# In[8]:


# Creating a User-Defined function to get the additional info on describe()

def df_stats(df,dt_flg):
    '''This function replicates the df.describe() with few additional features
        1. total_count
        2. null_count
        3. duplicate_with_null_count
        4. Renaming the existing columns
        Parameter:
        1. df - DataFrame name
        2. dt_flg - Flag to indicate the datatype of the columns in dataframe
            Possible values are:
                0 - All numeric
                1 - All non-numeric or object
                2 - Both Numeric and Object columns
    '''
    
    if dt_flg == 0:
        contents = df.describe().T.reset_index()
        contents.rename(columns={'index':'col_name','count':'non_null_count'}, inplace=True)
        contents['total_count'] = len(df.index)
        contents['non_null_count'] = contents['non_null_count'].astype('int')
        contents['null_count'] = contents['total_count'] - contents['non_null_count']
        contents['null%'] = np.round(contents['null_count']/contents['total_count']*100,2)
        columns = ['col_name','total_count','non_null_count','null_count','mean','std','min','25%','50%','75%','max']
        contents = contents[columns].infer_objects()
    elif dt_flg == 1:
        contents = df.describe().T.reset_index()
        contents.rename(columns={'index':'col_name','count':'non_null_count','unique':'unique_wo_null_count'}, inplace=True)
        contents['total_count'] = len(df.index)
        contents['non_null_count'] = contents['non_null_count'].astype('int')
        contents['unique_wo_null_count'] = contents['unique_wo_null_count'].astype('float')
        contents['null_count'] = contents['total_count'] - contents['non_null_count']
        contents['null%'] = np.round(contents['null_count']/contents['total_count']*100,2)
        contents['unique%'] = np.round(contents['unique_wo_null_count']/contents['non_null_count']*100,2)
        contents['duplicate_wo_null_count'] = contents['non_null_count'] - contents['unique_wo_null_count']
        columns = ['col_name','total_count','non_null_count','null_count','null%','unique_wo_null_count','unique%','duplicate_wo_null_count','top','freq']
        contents = contents[columns].infer_objects()
    elif dt_flg == 2:
        contents = df.describe(include='all').T.reset_index()
        contents.rename(columns={'index':'col_name','count':'non_null_count','unique':'unique_wo_null_count'}, inplace=True)
        contents['total_count'] = len(df.index)
        contents['non_null_count'] = contents['non_null_count'].astype('int')
        contents['unique_wo_null_count'] = contents['unique_wo_null_count'].astype('float')
        contents['null_count'] = contents['total_count'] - contents['non_null_count']
        contents['null%'] = np.round(contents['null_count']/contents['total_count']*100,2)
        contents['unique%'] = np.round(contents['unique_wo_null_count']/contents['non_null_count']*100,2)
        contents['duplicate_wo_null_count'] = contents['non_null_count'] - contents['unique_wo_null_count']
        columns = ['col_name','total_count','non_null_count','null_count','null%','unique_wo_null_count','unique%','duplicate_wo_null_count','mean','std','min','25%','50%','75%','max','top','freq']
        contents = contents[columns].infer_objects()
    return contents


# In[9]:


# How many unique companies are present in companies?

# Using the user-defined function - df_stats() to get the descriptive stats
companies_stats = df_stats(companies,1)
companies_stats


# #### How many unique companies are present in companies?
# >There are 66368 unique companies present in companies dataframe.

# In[10]:


# How many unique companies are present in rounds2?

# Using the user-defined function - df_stats() to get the descriptive stats
rounds2_stats = df_stats(rounds2,2)
rounds2_stats


# #### How many unique companies are present in rounds2?
# >There are 90247 unique companies present in rounds2 dataframe.

# In[11]:


# Are there any companies in the rounds2 file which are not  present in companies ? 

# Converting the permalinks in both dataframes to uppercase
companies.permalink = companies.permalink.str.strip().str.upper()
rounds2.company_permalink = rounds2.company_permalink.str.strip().str.upper()

# To find the count of companies in rounds2 which are not available in companies df
rounds2.loc[~rounds2.company_permalink.isin(companies.permalink)]
len(rounds2.loc[~rounds2.company_permalink.isin(companies.permalink)])


# #### Are there any companies in the rounds2 file which are not  present in companies ?
# > Yes, 7 companies

# > Most of these companies have missing raised_amt_usd, which is our target variable. So, I will remove these while creating a master_frame

# In[12]:


# companies and rounds2 are merged to get a unified data.
# The join type is taken as INNER 

master_frame = pd.merge(companies, rounds2, how='inner', left_on=['permalink'], right_on=['company_permalink'])


# #### The join type is taken as INNER to create master_frame, to remove these companies for which all the information is not available and make sure the further analysis is not done for such companies.
# 

# In[13]:


# Displaying the random 6 rows of the master_frame
# This step is executed multiple times, to get a random view of rows present and its typical properties

master_frame.sample(3)


# In[14]:


# Checking the shape of the master_frame

master_frame.shape


# In[15]:


# Using the user-defined function - df_stats() to get the descriptive stats

master_frame_stats = df_stats(master_frame,2)
master_frame_stats


# In[16]:


# Cleaning the dataframe 

# Removing observations with missing raised_amount_usd & keeping only the ['venture', 'angel', 'seed', 'private_equity'] funding types.
master_frame = master_frame.loc[(~master_frame.raised_amount_usd.isna()) & (master_frame.funding_round_type.isin(['venture', 'angel', 'seed', 'private_equity']))]
master_frame.shape

# Removing the unnecessary columns
master_frame = master_frame.drop(['state_code','region','city','founded_at','company_permalink','funding_round_permalink','funding_round_code','funded_at'], axis=1)
master_frame.shape


# #### The records left are 80626 and columns are 8 after the process of initial data cleaning.

# # Checkpoint 2: Funding Type Analysis

# In[17]:


# Grouping the observations based on the funding type
# Each group is analysed for the statistical info such as the total sum of the raised amt, mean or max in each group.

df_by_funding_type = master_frame.groupby('funding_round_type')

funding_type = df_by_funding_type.agg({'raised_amount_usd':['count','sum','mean','median','max','min']})
funding_type.columns = funding_type.columns.get_level_values(1)
funding_type = funding_type.reset_index().sort_values('mean').rename(columns={'sum':'Total Investment'})
funding_type


# #### The funding type venture has the highest total investments. Moreover the mean raised amount as well the median is also high compared to seed and angel fund types except private_equity which is very few in number.
# > So the Venture Fund Type is the best among all the four funding types.

# ### We can observe that, the best investment type is *Venture Type* and so will remove the observations of other funding types. 

# In[18]:


df_venture_funding = master_frame.loc[master_frame['funding_round_type'] == 'venture']
df_venture_funding.shape


# # Checkpoint 3 : Country Analysis

# In[19]:


# Reading the wiki page to extract the list of countries where English is an official language

url = 'https://en.wikipedia.org/wiki/List_of_territorial_entities_where_English_is_an_official_language'
dfs = pd.read_html(url)

df = pd.DataFrame()
for table in dfs[:3]:
    df = df.append(table, sort=True).reset_index(drop=True)

df.dropna(subset=['Alpha-3 code'], inplace=True)
# Creating a list of the english speaking countries
country_code_eng = list(df['Alpha-3 code'].dropna())


# #### Merging the df_venture_funding and df to get the Country name

# In[20]:


# Merging with the dataframe extracted from wiki page of english speaking country 

df_vf = pd.merge(df_venture_funding,df[['Alpha-3 code','Country']],how='left',left_on='country_code',right_on='Alpha-3 code')


# In[21]:


# Checking the shape of df_vf_eng for correctness of above process

df_vf.shape


# In[22]:


# Cleaning the column Country as it contains square brackets at the end 

df_vf['Country'] = df_vf['Country'].str.split('[',1).str[0]

# Renaming the column category_list_x
df_vf = df_vf.drop('Alpha-3 code', axis=1).rename(columns={'category_list_x':'category_list'})

# Checking the shape of df_vf_eng for correctness of above process

df_vf.shape
df_vf.head()


# #### Creating the dataframe with the top 9 countries in terms of the total raised amount.

# In[23]:


top9 = df_vf.groupby('country_code')['raised_amount_usd'].agg(['sum']).sort_values('sum', ascending=False).rename(columns={'sum':'tot_raised_amt_usd'}).head(9).reset_index()
# top9


# #### Creating a language flag to indicate whether the country is English speaking or Non-English.

# In[24]:


top9.loc[top9.country_code.isin(country_code_eng), 'Language'] = 'English'
top9.loc[~top9.country_code.isin(country_code_eng), 'Language'] = 'Non-English'
top9


# #### We can observe that USA, GBR nad IND are the top 3 English speaking countries with the highest investments.
# #### The top 9 data could be now plotted with English and Non-English labels.

# In[25]:


# Removing the observations of Non-English speaking country

df_vf_eng = df_vf.loc[df_vf.country_code.isin(country_code_eng)].copy()
df_vf_eng.shape


# # Checkpoint 4: Sector Analysis 1

# In[26]:


mapping = pd.read_csv('mapping.csv', sep = ',', encoding = 'ISO-8859-1')


# In[27]:


mapping.info()


# In[28]:


# Transforming the dataframe from wide to long, to get the main sector corresponding to the category list

mapping_long = pd.melt(mapping , id_vars=['category_list'] , var_name='main_sector', value_name='ms_flg')


# In[29]:


# Removing the rows for which the main sector does not correspond category list i.e, with ms_flg == 0
# Dropping the flag column which is not required anymore

mapping_long = mapping_long.loc[mapping_long['ms_flg'] == 1, ['category_list','main_sector'] ]


# In[30]:


mapping_long.shape


# In[31]:


mapping_long


# #### Creating a Primary Sector from the Category list for the companies that have multiple values separated by '|'

# In[32]:


df_vf_eng['primary_sector'] = df_vf_eng['category_list'].str.split('|').str[0]


# #### Checking out the newly created column primary_sector

# In[33]:


df_vf_eng.sample(5)


# #### Checking out the shape to make sure that the record count is same. In this case, it is fine

# In[34]:


df_vf_eng.shape


# In[35]:


# Using the user-defined function - df_stats() to get the descriptive stats

df_vf_eng_stats = df_stats(df_vf_eng,2)
df_vf_eng_stats


# In[36]:


# Merging master_frame, aka now, df_vf_eng with the mapping_long frame to get the main_sector and naming it as master_frame
# The join = inner ensures that no observations exist with NULL main_sector.
# This is important as the analysis will be done on main_sector

master_frame = pd.merge(df_vf_eng, mapping_long, how='inner', left_on=['primary_sector'], right_on=['category_list'])


# In[37]:


# Checking out the shape of df for correctness of the above process

master_frame.shape


# In[38]:


# Removing rows with NULL primary_sector equivalent to removing rows with main_sector == 'Blanks'

master_frame = master_frame.loc[~master_frame.primary_sector.isna()].drop('category_list_y', axis=1)


# In[39]:


# master_frame.shape


# In[40]:


master_frame_stats = df_stats(master_frame,2)
master_frame_stats


# # Checkpoint 5: Sector Analysis 2

# #### Creating a new column 'tot_count_inv_by_sector' which holds count for each group by Country and Sector

# In[41]:


master_frame['tot_count_inv_by_sector'] = master_frame.groupby(['country_code','main_sector'])['raised_amount_usd'].transform(lambda x : x.count())


# #### Creating a new column 'tot_amount_inv_by_sector' which holds total raised amount for each group by Country and Sector

# In[42]:


master_frame['tot_amount_inv_by_sector'] = master_frame.groupby(['country_code','main_sector'])['raised_amount_usd'].transform(lambda x : x.sum())


# In[43]:


# Checking the shape to verify the correctness of the process. In this case, it is correct.

master_frame.shape


# In[44]:


# Filtering records and keeping only for USA, GBR and IND - The top 3 english speaking countries

master_frame = master_frame.loc[master_frame.country_code.isin(['USA','GBR','IND'])]

# Checking the shape to verify the correctness of the above filter process
master_frame.shape


# In[45]:


# Displaying 3 random observations

master_frame.sample(3)


# #### Creating D1, D2 and D3 dataframes for USA, GBR and IND respectively.

# In[46]:


D1 = master_frame.loc[master_frame.country_code == 'USA'].copy()
D2 = master_frame.loc[master_frame.country_code == 'GBR'].copy()
D3 = master_frame.loc[master_frame.country_code == 'IND'].copy()


# In[47]:


#D1
#D2
#D3


# In[48]:


# Total number of Investments (count) for USA

len(D1)


# In[49]:


# Total number of Investments (count) for GBR

len(D2)


# In[50]:


# Total number of Investments (count) for IND

len(D3)


# In[51]:


# Total amount of investment (USD) for USA

np.sum(D1.raised_amount_usd)


# In[52]:


# Total amount of investment (USD) for GBR

np.sum(D2.raised_amount_usd)


# In[53]:


# Total amount of investment (USD) for IND

np.sum(D3.raised_amount_usd)


# In[54]:


# List of Sector name based on the no. of investments in USA

D1_top_sectors = D1.groupby('main_sector').size().sort_values(ascending=False)
D1_top_sectors
D1_top_sector_list = D1_top_sectors.head(3).index
D1_top_sector_list


# #### The top 3 sectors in USA in terms of no of investments are 'Others', 'Cleantech / Semiconductors', 'Social, Finance, Analytics, Advertising'

# In[55]:


# List of Sector name based on the no. of investments in GBR

D2_top_sectors = D2.groupby('main_sector').size().sort_values(ascending=False)
D2_top_sectors
D2_top_sector_list = D2_top_sectors.head(3).index
D2_top_sector_list


# #### The top 3 sectors in GBR in terms of no of investments are 'Others', 'Cleantech / Semiconductors', 'Social, Finance, Analytics, Advertising'

# In[56]:


# List of Sector name based on the no. of investments in IND

D3_top_sectors = D3.groupby('main_sector').size().sort_values(ascending=False)
D3_top_sectors
D3_top_sector_list = D3_top_sectors.head(3).index
D3_top_sector_list


# #### The top 3 sectors in IND in terms of no of investments are 'Others', 'News, Search and Messaging', 'Social, Finance, Analytics, Advertising'

# In[57]:


D1['tot_raised_amount_usd'] = D1.groupby(['main_sector','permalink'])['raised_amount_usd'].transform(lambda x : x.sum())


# In[58]:


# To get the company receiving highest investment for the top 2 sectors in United States(USA)

idx = D1.groupby(['main_sector'])['tot_raised_amount_usd'].transform(max) == D1['tot_raised_amount_usd']
D1.loc[idx & (D1.main_sector.isin(D1_top_sector_list)), ['Country','main_sector','permalink','name','tot_raised_amount_usd','tot_count_inv_by_sector']].sort_values('tot_count_inv_by_sector', ascending=False).drop_duplicates()


# #### The company with the highest no of investments in top 2 sectors in USA are SoFi and Freescale Semiconductor respectively

# In[59]:


D2['tot_raised_amount_usd'] = D2.groupby(['main_sector','permalink'])['raised_amount_usd'].transform(lambda x : x.sum())


# In[60]:


# To get the company receiving highest investment for the top 2 sectors in United Kingdom(GBR)

idx = D2.groupby(['main_sector'])['tot_raised_amount_usd'].transform(max) == D2['tot_raised_amount_usd']
D2.loc[idx & (D2.main_sector.isin(D2_top_sector_list)), ['Country','main_sector','permalink','name','tot_raised_amount_usd','tot_count_inv_by_sector']].sort_values('tot_count_inv_by_sector', ascending=False).drop_duplicates()


# #### The company with the highest no of investments in top 2 sectors in GBR are OneWeb and Immunocore respectively

# In[61]:


D3['tot_raised_amount_usd'] = D3.groupby(['main_sector','permalink'])['raised_amount_usd'].transform(lambda x : x.sum())


# In[62]:


# To get the company receiving highest investment for the top 2 sectors in India(IND)

idx = D3.groupby(['main_sector'])['tot_raised_amount_usd'].transform(max) == D3['tot_raised_amount_usd']
D3.loc[idx & (D3.main_sector.isin(D3_top_sector_list)), ['Country','main_sector','permalink','name','tot_raised_amount_usd','tot_count_inv_by_sector']].sort_values('tot_count_inv_by_sector', ascending=False).drop_duplicates()


# #### The company with the highest no of investments in top 2 sectors in IND are Flipkart and Quikr respectively

# ## The below code is to find the Number of Investments in top 3 sectors in top 3 english speaking countries

# In[63]:


top3_sector = pd.DataFrame(master_frame.loc[master_frame.country_code.isin(['USA','GBR','IND'])].groupby(['Country','main_sector'])['permalink'].size())

top3_sector = top3_sector.reset_index().rename(columns={'permalink':'No of Investments'}).sort_values(['Country','No of Investments'], ascending=False).reset_index(drop='index')
top3_sector = top3_sector.groupby('Country').head(3)
top3_sector


# #### Creating a pivot from top3_sector to plot

# In[64]:


top3_sector_piv = top3_sector.pivot(index='Country', columns='main_sector', values='No of Investments')
top3_sector_piv


# # Checkpoint 6: Plots

# #### 1a. Plotting the average amount of investments in angel, venture, seed, and private equity

# In[65]:


ax = sns.barplot(data=funding_type, x='funding_round_type' , y='mean')
ax
plt.title("Average Amount of Investment");
plt.ylabel("Average Amount (in USD millions)");
plt.xlabel("Funding Type");

# fig = px.bar(funding_type, x='funding_round_type', y=funding_type['mean'], title = 'Average Amount of Investment', labels={'y':'Average Amount','funding_round_type':'Funding Type'})
# fig.show()


# #### 1b. Plotting the fraction of total investments (globally) in angel, venture, seed, and private equity

# In[66]:


# funding_type.plot.pie(x='funding_round_type', y='Total Investment');

fig = px.pie(funding_type, values=funding_type['Total Investment'], names='funding_round_type', title='Fraction of Investment across Funding types', labels={'value':'Total Investment','funding_round_type':'Funding Type'});
fig.show();


# #### 2. Plotting the top 9 Countries for Venture Funding

# In[67]:


ax = sns.barplot(x="country_code", y="tot_raised_amt_usd", hue="Language",
                 data=top9, dodge=False)
plt.title("Top 9 Countries for Venture Funding");
plt.ylabel("Total Raised Amount (in USD millions)");
plt.xlabel("Country Code");


# The above result shows the top 9 country based on total raised amount.
# We can conclude that the top 3 english speaking countries are USA, GBR and IND.
# 

# #### 3. Plotting the number of investments in the top 3 sectors of the top 3 countries

# In[68]:


fig = px.bar(top3_sector_piv, title='Top 3 sectors in the top 3 countries', labels={'value':'Number of Investment'})
fig.show()


# In[69]:


top3_sector_piv.plot.bar(stacked=True);

