# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import re
from scipy.stats import ttest_ind
import pandas as pd
import seaborn as sb
import numpy as np
import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist as fdist
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt


behaviour = pd.read_csv("D:/Data Analysis/Jobs Seeking/Quantium/dataset/task1/QVI_purchase_behaviour.csv")
behaviour
transaction = pd.read_excel("D:/Data Analysis/Jobs Seeking/Quantium/dataset/task1/QVI_transaction_data.xlsx")
transaction.head()

# data info for behaviour
behaviour.info()
behaviour.LYLTY_CARD_NBR.unique()
behaviour.LIFESTAGE.unique()
behaviour.PREMIUM_CUSTOMER.unique()

#
transaction.info()
    # DATE column is int64 --> change it to datetype
#copy
transaction_1 = transaction.copy()
transaction_1.DATE = pd.to_datetime(transaction_1['DATE'], origin = "1899-12-30", unit='D')

transaction_1.info()
transaction_1.PROD_NAME.unique()

len(transaction_1) # 264836
transaction_1['TXN_ID'].nunique() #263127
# find duplicated TXN_ID
transaction_1[transaction_1.duplicated(['TXN_ID'])].head()
transaction_1.loc[transaction_1['TXN_ID']==7739,:]
# Duplicated TXN_ID occured because different brands existed in one transaction

#Missing values in transactions
transaction_1.isna().sum()   # no missing values in the dataset

# Get weight from product name

transaction_1['PACK_SIZE'] = transaction_1['PROD_NAME'].str.extract("(\d+)")
transaction_1['PACK_SIZE'] = pd.to_numeric(transaction_1['PACK_SIZE'])
transaction_1
'''
So, when put together, (\d+) means that we’re capturing any groups of digit characters in the string. Without the +, we’d still be capturing all the digits, but since \d represents just a single character, we’d be pulling a bunch of single character groups instead.

For example, if I had a string ‘123 456 789’:

(\d+) would match 3 groups: 123, 456, and 789.
(\d) would match 9 groups: 1,2,3,4,5,6,7,8,9

'''

#create text clean function : remove &/,

import re
def clean_text(text):
    text = re.sub('[&/]', ' ', text)
    text = re.sub('\d\w*', ' ', text)   #\d - Matches any decimal digit. Equivalent to [0-9],The star symbol * matches zero or more occurrences of the pattern left to it. \wMatches any alphanumeric character

    return text
transaction_1['PROD_NAME'] = transaction_1['PROD_NAME'].apply(clean_text)

cleanProdName = transaction_1['PROD_NAME']
string = ''.join(cleanProdName)
prodWord = word_tokenize(string)
# Apply 'fdist' function which computes the frequency of each token and put it into a dataframe

wordFrequency = fdist(prodWord)
freq_df = pd.DataFrame(list(wordFrequency.items()), columns = ["Word", "Frequency"]).sort_values(by = 'Frequency', ascending = False)


freq_df.head()




#drop rows with 'salsa' in prod_name
transaction_1['PROD_NAME'] = transaction_1['PROD_NAME'].apply(lambda x: x.lower())
transaction_1 = transaction_1[~transaction_1['PROD_NAME'].str.contains('salsa')]
#transaction_1['PROD_NAME'] = transaction_1['PROD_NAME'].apply(lambda x: x.title())
transaction_1['PROD_NAME'] = transaction_1['PROD_NAME'].apply(lambda x: x.strip())
transaction_1.PROD_NAME

# Value counts of PROD_QTY

transaction_1['PROD_QTY'].value_counts()
#We have two occurences of 200 in the dataset. This seems odd so let's explore further.

transaction_1.loc[transaction_1['PROD_QTY'] == 200, :]

#Both these transactions have been made by the same person at the same store. Let's see all the transactions this person has made by tracking his loyalty card number.
transaction_1.loc[transaction_1['LYLTY_CARD_NBR'] == 226000, :]


#This customer has only made two transactions over the entire year so unlikely to be a retail customer. He/she is most likely purchasing for commercial purposes so it is safe for us to drop these this customer from both 'transactionData' and 'customerData' dataset.

transaction_1.drop(transaction_1.index[transaction_1['LYLTY_CARD_NBR'] == 226000], inplace = True)
behaviour.drop(behaviour.index[behaviour['LYLTY_CARD_NBR'] == 226000], inplace = True)

# Now let's examine the number of transactions over time to see if there are any obvious data issues e.g. missing data

transaction_1['DATE'].nunique()
pd.date_range(start = '2018-07-01', end = '2019-06-30').difference(transaction_1['DATE'])

#We have a missing date on Christmas Day. This makes sense because most retail stores are closed that day.

# Create a new dataframe which contains the total sale for each date

a = pd.pivot_table(transaction_1, values = 'TOT_SALES', index = 'DATE', aggfunc = 'sum')
a.head()
b = pd.DataFrame(index = pd.date_range(start = '2018-07-01', end = '2019-06-30'))
b['TOT_SALES'] = 0
len(b)

c = a + b
c.fillna(0, inplace = True)

c.head()
c.index.name = 'Date'
c.rename(columns = {'TOT_SALES': 'Total Sales'}, inplace = True)
c.head()

timeline = c.index
graph = c['Total Sales']

fig, ax = plt.subplots(figsize = (10, 5))
ax.plot(timeline, graph)

date_form = DateFormatter("%Y-%m")
ax.xaxis.set_major_formatter(date_form)
plt.title('Total Sales from July 2018 to June 2019')
plt.xlabel('Time')
plt.ylabel('Total Sales')

plt.show()

# Confirm the date where sales count equals to zero

c[c['Total Sales'] == 0]
# Let's look at the December month only

c_december = c[(c.index < "2019-01-01") & (c.index > "2018-11-30")]
c_december.head()

plt.figure(figsize = (15, 5))
plt.plot(c_december)
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.title('Total Sales in December')
plt.show()


# Reset index

c_december.reset_index(drop = True, inplace = True)
c_december.head()


# Relabel Date

c_december['Date'] = c_december.index + 1
c_december.head()

plt.figure(figsize = (15,5))
sb.barplot(x = 'Date', y ='Total Sales', data = c_december)
plt.show()

#Now, we move on to PACK_SIZE that we created at the beginning by extracting the weight from the PROD_NAME column.

transaction_1['PACK_SIZE'].head()
transaction_1['PACK_SIZE'].unique()

# Check the distribution of PACK_SIZE
plt.figure(figsize = (10, 5))
plt.hist(transaction_1['PACK_SIZE'])
plt.xlabel('Pack Size')
plt.ylabel('Frequency')
plt.title('Pack Size Histogram')
plt.show()


part = transaction_1['PROD_NAME'].str.partition()
transaction_1['BRAND'] = part[0]
transaction_1.head()

transaction_1['BRAND'].unique()
# Rename brand names for consistency

transaction_1['BRAND'].replace('Ncc', 'Natural', inplace = True)
transaction_1['BRAND'].replace('Ccs', 'CCS', inplace = True)
transaction_1['BRAND'].replace('Smith', 'Smiths', inplace = True)
transaction_1['BRAND'].replace(['Grain', 'Grnwves'], 'Grainwaves', inplace = True)
transaction_1['BRAND'].replace('Dorito', 'Doritos', inplace = True)
transaction_1['BRAND'].replace('Ww', 'Woolworths', inplace = True)
transaction_1['BRAND'].replace('Infzns', 'Infuzions', inplace = True)
transaction_1['BRAND'].replace(['Red', 'Rrd'], 'Red Rock Deli', inplace = True)
transaction_1['BRAND'].replace('Snbts', 'Sunbites', inplace = True)

transaction_1['BRAND'].unique()

# Which brand had the most sales?

transaction_1.groupby('BRAND').TOT_SALES.sum().sort_values(ascending = False)




# Customer data :

customer = behaviour
customer.head()
customer.isna().sum()  # No missing values shown

len(customer)
customer['LYLTY_CARD_NBR'].nunique()
#Since the number of rows in customerData is equal to number of unique loyalty card number, we conclude that loyalty card numbers are unique to each row.

# How many unique lifestages?
customer['LIFESTAGE'].nunique()

# What are those lifestages?
customer['LIFESTAGE'].unique()

customer['LIFESTAGE'].value_counts().sort_values(ascending = False)
sb.countplot(y = customer['LIFESTAGE'], order = customer['LIFESTAGE'].value_counts().index)


# How many unique premium customer categories?

customer['PREMIUM_CUSTOMER'].nunique()

# Value counts for each premium customer category

customer['PREMIUM_CUSTOMER'].value_counts().sort_values(ascending = False)

plt.figure(figsize = (12, 7))
sb.countplot(y = customer['PREMIUM_CUSTOMER'], order = customer['PREMIUM_CUSTOMER'].value_counts().index)
plt.xlabel('Number of Customers')
plt.ylabel('Premium Customer')
plt.show()

# Merge transactionData and customerData together

combineData = pd.merge(transaction_1, customer)

print("Transaction data shape: ", transaction_1.shape)
print("Customer data shape: ", customer.shape)
print("Combined data shape: ", combineData.shape)

combineData.isnull().sum()


#Data analysis on customer segments
#Who spends the most on chips, describing customers by lifestage and how premium their general purchasing behaviour is
#How many customers are in each segment
#How many chips are bought per customer by segment
#What is the average chip price by customer segment

# Total sales by PREMIUM_CUSTOMER and LIFESTAGE

sales = pd.DataFrame(combineData.groupby(['PREMIUM_CUSTOMER', 'LIFESTAGE']).TOT_SALES.sum())
sales.rename(columns = {'TOT_SALES': 'Total Sales'}, inplace = True)
sales.sort_values(by = 'Total Sales', ascending = False, inplace = True)
sales


# Visualise

salesPlot = pd.DataFrame(combineData.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).TOT_SALES.sum())
salesPlot.unstack().plot(kind = 'bar', stacked = True, figsize = (12, 7), title = 'Total Sales by Customer Segment')
plt.ylabel('Total Sales')
plt.legend(['Budget', 'Mainstream', 'Premium'], loc = 2)
plt.show()

#Top 3 sales come from budget older families, mainstream young singles/couples and mainstream retirees.


# Number of customers by PREMIUM_CUSTOMER and LIFESTAGE

customers = pd.DataFrame(combineData.groupby(['PREMIUM_CUSTOMER', 'LIFESTAGE']).LYLTY_CARD_NBR.nunique())
customers.rename(columns = {'LYLTY_CARD_NBR': 'Number of Customers'}, inplace = True)
customers.sort_values(by = 'Number of Customers', ascending = False).head(10)

# Visualise

customersPlot = pd.DataFrame(combineData.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).LYLTY_CARD_NBR.nunique())
customersPlot.unstack().plot(kind = 'bar', stacked = True, figsize = (12, 7), title = 'Number of Customers by Customer Segment')
plt.ylabel('Number of Customers')
plt.legend(['Budget', 'Mainstream', 'Premium'], loc = 2)
plt.show()

#There are more mainstream young singles/couples and retirees. This contributes to to more chips sales in these segments however this is not the major driver for the budget older families segment.

# Average units per customer by PREMIUM_CUSTOMER and LIFESTAGE

avg_units = combineData.groupby(['PREMIUM_CUSTOMER', 'LIFESTAGE']).PROD_QTY.sum() / combineData.groupby(['PREMIUM_CUSTOMER', 'LIFESTAGE']).LYLTY_CARD_NBR.nunique()
avg_units = pd.DataFrame(avg_units, columns = {'Average Unit per Customer'})
avg_units.sort_values(by = 'Average Unit per Customer', ascending = False).head()

# Visualise

avgUnitsPlot = pd.DataFrame(combineData.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).PROD_QTY.sum() / combineData.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).LYLTY_CARD_NBR.nunique())
avgUnitsPlot.unstack().plot(kind = 'bar', figsize = (12, 7), title = 'Average Unit by Customer Segment')
plt.ylabel('Average Number of Units')
plt.legend(['Budget', 'Mainstream', 'Premium'], loc = 2)
plt.show()
#Older families and young families buy more chips per customer.




# Average price per unit by PREMIUM_CUSTOMER and LIFESTAGE

avg_price = combineData.groupby(['PREMIUM_CUSTOMER', 'LIFESTAGE']).TOT_SALES.sum() / combineData.groupby(['PREMIUM_CUSTOMER', 'LIFESTAGE']).PROD_QTY.sum()
avg_price = pd.DataFrame(avg_price, columns = {'Price per Unit'})
avg_price.sort_values(by = 'Price per Unit', ascending = False).head()


# Visualise

avgPricePlot = pd.DataFrame(combineData.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).TOT_SALES.sum() / combineData.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).PROD_QTY.sum())
avgPricePlot.unstack().plot(kind = 'bar', figsize = (12, 7), title = 'Average Price by Customer Segment', ylim = (0, 6))
plt.ylabel('Average Price')
plt.legend(['Budget', 'Mainstream', 'Premium'], loc = 2)
plt.show()

#Mainstream midage and young singles and couples are more willing to pay more per packet of chips compared to their budget and premium counterparts. This may be due to premium shoppers being more likely to buy healthy snacks and when they do buy chips, it is mainly for entertainment purposes rather than their own consumption. This is also supported by there being fewer premium midage and young singles and couples buying chips compared to their mainstream counterparts.

# Perform an independent t-test between mainstream vs non-mainstream midage and young singles/couples to test this difference

# Create a new dataframe pricePerUnit
pricePerUnit = combineData

# Create a new column under pricePerUnit called PRICE
pricePerUnit['PRICE'] = pricePerUnit['TOT_SALES'] / pricePerUnit['PROD_QTY']

# Let's have a look
pricePerUnit.head()



# Let's group our data into mainstream and non-mainstream

mainstream = pricePerUnit.loc[(pricePerUnit['PREMIUM_CUSTOMER'] == 'Mainstream') & ( (pricePerUnit['LIFESTAGE'] == 'YOUNG SINGLES/COUPLES') | (pricePerUnit['LIFESTAGE'] == 'MIDAGE SINGLES/COUPLES') ), 'PRICE']
nonMainstream = pricePerUnit.loc[(pricePerUnit['PREMIUM_CUSTOMER'] != 'Mainstream') & ( (pricePerUnit['LIFESTAGE'] == 'YOUNG SINGLES/COUPLES') | (pricePerUnit['LIFESTAGE'] == 'MIDAGE SINGLES/COUPLES') ), 'PRICE']

## Compare histograms of mainstream and non-mainstream customers

plt.figure(figsize = (10, 5))
plt.hist(mainstream, label = 'Mainstream')
plt.hist(nonMainstream, label = 'Premium & Budget')
plt.legend()
plt.xlabel('Price per Unit')
plt.show()

print("Mainstream average price per unit: ${:.2f}".format(np.mean(mainstream)))
print("Non-mainstream average price per unit: ${:.2f}".format(np.mean(nonMainstream)))
if np.mean(mainstream) > np.mean(nonMainstream):
    print("Mainstream customers have higher average price per unit. ")
else:
    print("Non-mainstream customers have a higher average price per unit. ")

# Perform t-test

ttest_ind(mainstream, nonMainstream)

#Mainstream customers have higher average price per unit than that of non-mainstream customers.

#We have found quite a few interesting insights that we can dive deeper into. For example,
# we might want to target customers segments that contribute the most to sales to retain them to further increase sales.
# Let's examine mainstream young singles/couples against the rest of the cutomer segments to see if they prefer any particular brand of chips.

target = combineData.loc[(combineData['LIFESTAGE'] == 'YOUNG SINGLES/COUPLES') & (combineData['PREMIUM_CUSTOMER'] == 'Mainstream'), :]
nonTarget = combineData.loc[(combineData['LIFESTAGE'] != 'YOUNG SINGLES/COUPLES' ) & (combineData['PREMIUM_CUSTOMER'] != 'Mainstream'), :]
target.head()


#Affinity to brand

# Target Segment
targetBrand = target.loc[:, ['BRAND', 'PROD_QTY']]
targetSum = targetBrand['PROD_QTY'].sum()
targetBrand['Target Brand Affinity'] = targetBrand['PROD_QTY'] / targetSum
targetBrand = pd.DataFrame(targetBrand.groupby('BRAND')['Target Brand Affinity'].sum())

# Non-target segment
nonTargetBrand = nonTarget.loc[:, ['BRAND', 'PROD_QTY']]
nonTargetSum = nonTargetBrand['PROD_QTY'].sum()
nonTargetBrand['Non-Target Brand Affinity'] = nonTargetBrand['PROD_QTY'] / nonTargetSum
nonTargetBrand = pd.DataFrame(nonTargetBrand.groupby('BRAND')['Non-Target Brand Affinity'].sum())

# Merge the two dataframes together

brand_proportions = pd.merge(targetBrand, nonTargetBrand, left_index = True, right_index = True)
brand_proportions.head()

brand_proportions['Affinity to Brand'] = brand_proportions['Target Brand Affinity'] / brand_proportions['Non-Target Brand Affinity']
brand_proportions.sort_values(by = 'Affinity to Brand', ascending = False)

#Mainstream young singles/couples are more likely to purchase Tyrrells chips compared to other brands.

#Affinity to pack size


# Target segment
targetSize = target.loc[:, ['PACK_SIZE', 'PROD_QTY']]
targetSum = targetSize['PROD_QTY'].sum()
targetSize['Target Pack Affinity'] = targetSize['PROD_QTY'] / targetSum
targetSize = pd.DataFrame(targetSize.groupby('PACK_SIZE')['Target Pack Affinity'].sum())

# Non-target segment
nonTargetSize = nonTarget.loc[:, ['PACK_SIZE', 'PROD_QTY']]
nonTargetSum = nonTargetSize['PROD_QTY'].sum()
nonTargetSize['Non-Target Pack Affinity'] = nonTargetSize['PROD_QTY'] / nonTargetSum
nonTargetSize = pd.DataFrame(nonTargetSize.groupby('PACK_SIZE')['Non-Target Pack Affinity'].sum())

## Merge the two dataframes together

pack_proportions = pd.merge(targetSize, nonTargetSize, left_index = True, right_index = True)
pack_proportions.head()

pack_proportions['Affinity to Pack'] = pack_proportions['Target Pack Affinity'] / pack_proportions['Non-Target Pack Affinity']
pack_proportions.sort_values(by = 'Affinity to Pack', ascending = False)
#It looks like mainstream singles/couples are more likely to purchase a 270g pack size compared to other pack sizes.

# Which brand offers 270g pack size?

combineData.loc[combineData['PACK_SIZE'] == 270, :].head(10)
# Is Twisties the only brand who sells 270g pack size?

combineData.loc[combineData['PACK_SIZE'] == 270, 'BRAND'].unique()


#Twisties is the only brand that offers 270g pack size.

#Conclusion
#Sales are highest for (Budget, OLDER FAMILIES), (Mainstream, YOUNG SINGLES/COUPLES) and (Mainstream, RETIREES)
#We found that (Mainstream, YOUNG SINGLES/COUPLES) and (Mainstream, RETIREES) are mainly due to the fact that there are more customers in these segments
#(Mainstream, YOUNG SINGLES/COUPLES) are more likely to pay more per packet of chips than their premium and budget counterparts
#They are also more likely to purchase 'Tyrrells' and '270g' pack sizes than the rest of the population


'''
import chardet
with open("D:/Data Analysis/Jobs Seeking/Quantium/dataset/task1/QVI_transaction_data.xlsx", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result
'''









# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
