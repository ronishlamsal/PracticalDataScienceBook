# test_str = 'a test string'
# a =2
# b =2
# import pdb; pdb.set_trace()
# c= a+b 


#Opening a file 

# file = open(file='textfile.txt', mode='r') 
# text = file.readlines()
# print(text)



# #Opening/writing a file with write test

# with open(file='writetest.txt', mode='w') as f:
#     f.write('testing writing out')



# data_dictionary = {
#     'books' : 12,
#     'articles' : 100,
#     'subjects' : ['math' ,
#                     'programming' ,
#                     'data science']}

# import json
# json_string = json.dumps(data_dictionary)
# print(json_string)

#json.dumps = converts string/dictionary to json format
#json.loads() = converts json format to string/dictionary

# with open('reading.json', 'w') as f:
#     json.dump(data_dictionary, f)




# import credentials as creds         #importing credentials.py file here 
# print(f'username: {creds.username}\npassword: {creds.password}')        #\n is the line break here


#Saving python objects with pickle

# import pickle as pk

# data_dictionary = {
#     'books' : 12,
#     'articles' : 100,
#     'subjects' : ['math' ,
#                     'programming' ,
#                     'data science']}

# with open('readings.pk', 'wb') as f:            #wb = write binary
#     pk.dump(data_dictionary, f)

# #once we have data in a pickle file, we can load it into Python so:

# with open('readings.pk', 'rb') as f:
#     data = pk.load(f)
#     print(data)

#Connecting sql lite 3 to python
# import sqlite3      #Importing sqlite3
# connection = sqlite3.connect('chinook.db')      #establishes a connection to the database chinook.db
# cursor = connection.cursor()                    #creates a cursor to execute SQL command


# # cursor.execute('SELECT * FROM artists LIMIT 5;')    #excutes a SQL query to select first 5 rows from the artists table
# # #this returns curosr object not data itself

# # cursor.fetchall()   #fetches all rows from the result of the query

# # query = """           #Another way of writing the above code. Broken down into different lines for clarity
# # SELECT *
# # FROM artists
# # LIMIT 5;
# # """

# # cursor.execute(query)
# # cursor.fetachall()

# cursor.execute(
#     """SELECT Total, InvoiceDate
#     FROM invoices
#     ORDER BY Total DESC
#     LIMIT 5; """
#     )
# print(cursor.fetchall())

# connection.close()

# #Storing a SQLite database and storing data
# import sqlite3
# book_data = [
#     ('12-1-2020', 'Practical Data Science With Python', 19.99, 1),
#     ('12-15-2020', 'Python Machine Learning', 27.99, 1),
#     ('12-17-2020', 'Machine Learning For Algorithmic Trading', 34.99, 1)
# ]

# connection = sqlite3.connect('book_sales.db')
# cursor = connection.cursor()

# cursor.execute('''CREATE TABLE IF NOT EXISTS book_sales
#                 (date text, book_title text, price real, quantity real)''')








#Chapter 4 - Loading and Wrangling Data with Pandas and Numpy

#importing data with pandas

# import pandas as pd 
# from tabulate import tabulate
# import os

# csv_df = pd.read_csv('/Users/ronishlamsal/Desktop/Python/Knoxville/itunes_data.csv')
# csv_df.head()

# #Print the Dataframe as a nicely formatted table
# print(tabulate(csv_df.head(), headers='keys', tablefmt='grid'))

# #Open the CSV file automatically using the default application

# os.system(f'open -a "Microsoft Excel" /Users/ronishlamsal/Desktop/Python/Knoxville/itunes_data.csv')


# import pandas as pd
# import os

# # Read the CSV file into a DataFrame
# csv_df = pd.read_csv('/Users/ronishlamsal/Desktop/Python/Knoxville/itunes_data.csv')

# # Select the first 20 entries
# first_20_entries = csv_df.head(20)

# # Save the first 20 entries to a new Excel file
# excel_file_path = '/Users/ronishlamsal/Desktop/Python/Knoxville/itunes_data_first_20.xlsx'
# first_20_entries.to_excel(excel_file_path, index=False)

# # Open the new Excel file automatically
# os.system(f'open "{excel_file_path}"')



#Getting data from SQLite


# import pandas as pd
# from sqlalchemy import create_engine

# # Correct the engine creation with the proper SQLite dialect
# engine = create_engine('sqlite:////Users/ronishlamsal/Desktop/Python/Knoxville/chinook.db')

# # SQL query to retrieve the desired data
# query = """
# SELECT tracks.name as Track,
#        tracks.composer,
#        tracks.milliseconds,
#        tracks.bytes,
#        tracks.unitprice,
#        genres.name as Genre,
#        albums.title as Album,
#        artists.name as Artist
# FROM tracks
# JOIN genres ON tracks.genreid = genres.genreid
# JOIN albums ON tracks.albumid = albums.albumid
# JOIN artists ON albums.artistid = artists.artistid;
# """

# # Using pandas to execute the SQL query and load the data into a DataFrame
# with engine.connect() as connection:
#     sql_df = pd.read_sql_query(query, connection)

# # # Display the first 2 rows of the DataFrame, transposed
# # print(sql_df.head(2).T)


# print(sql_df.index)

# print(type(sql_df))

# #Combining data frame
# itunes_df = pd.concat([csv_df, excel_df,sql_df])

#Debugging

# import pandas as pd
# from sqlalchemy import create_engine

# # Create an engine to connect to a test SQLite database
# test_engine = create_engine('sqlite:///test.db')

# # Create a test table and insert some data
# with test_engine.connect() as conn:
#     conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
#     conn.execute("INSERT INTO test (name) VALUES ('Alice'), ('Bob')")

# # Query the test table
# query = "SELECT * FROM test"

# # Using pandas to read the SQL query
# test_df = pd.read_sql_query(query, test_engine)

# # Display the DataFrame
# print(test_df)








# #checking permisson as the code ran into permission issue

# import os

# # Verify if the file exists
# db_file_path = '/Users/ronishlamsal/Desktop/Python/Knoxville/chinook.db'
# if os.path.exists(db_file_path):
#     print(f"Database file exists: {db_file_path}")
# else:
#     print(f"Database file does not exist: {db_file_path}")

# # Verify if the file is readable
# if os.access(db_file_path, os.R_OK):
#     print(f"Database file is readable: {db_file_path}")
# else:
#     print(f"Database file is not readable: {db_file_path}")




#Using ML/ KNN(k-nearest neighbours)

# import numpy as np 
# itunes_df.loc[0,'Bytes'] = np.nan

# from sklearn.impute import KNNImputer
# imputer = KNNImputer()

# imputed = imputer.fit_transform(itunes_df [['Milliseconds', 'Bytes', 'UnitPrice']])

# itunes_df['Bytes'] = imputed[:,1]




#Wrangling and analyzing Bitcoin Price data

# import pandas as pd 

# btc_df = pd.read_csv('/Users/ronishlamsal/Desktop/Python/Knoxville/bitcoin_price.csv')
# btc_df.head()

# btc_df['symbol'].unique()

# #Dropping this column

# btc_df.drop('symbol', axis=1, inplace=True)

# #Converting the time column to pandas datetime datatype

# btc_df['time'] = pd.to_datetime(btc_df['time'], unit='ms')


# import missingo as msno
# msno.matrix(dfs)

# import pandas as pd 

# from pandas_profiling import ProfileReport
# report = ProfileReport(df)

# report.to_widgets()


# # create our own histogram of the length of the tracks
# df['Track'].str.len().plot.hist(bins=50)





# Chapter 6
# Data Wrangling Documents and Spreadsheet


# from glob import glob

# word_files = glob('/Users/ronishlamsal/Desktop/Python/Knoxville/chapter6/docx/*.docx')

# #Extracting text file with textract

# import textract
# text = textract.process(word_files[0])      #extracts text from the word file
# text = text.decode('utf-8')                 #decodes extracted binary text into a human-readble string
# print(text[:200])                           #prints first 200 words

# print(f'Above is the first 200 words and below is the last 200 words')
# print(text[-200:])



# from glob import glob
# import textract

# # Get all .docx files in the specified directory
# word_files = glob('/Users/ronishlamsal/Desktop/Python/Knoxville/chapter6/docx/*.docx')

# # Extract text from the first .docx file in the list
# text = textract.process(word_files[0])  # extracts text from the word file
# text = text.decode('utf-8')             # decodes extracted binary text into a human-readable string

# # Print the first 200 characters of the extracted text
# print(text[:200]) 

# print(f'Above is the first 200 characters and below is the last 200 characters')

# # Print the last 200 characters of the extracted text
# print(text[-200:])

# # Print the filename of the file from which the last 200 characters were extracted
# print(f'The above text was extracted from the file: {word_files[0]}')


# from glob import glob
# import textract

# # Get all the Word files in the folder, arranged by their year (assuming they are named with the year in their filenames)
# word_files = sorted(glob('/Users/ronishlamsal/Desktop/Python/Knoxville/chapter6/docx/*.docx'))

# # Extract and print the first 200 words from the first file
# first_file_text = textract.process(word_files[0]).decode('utf-8')
# print(f'First 200 words from {word_files[0]}:')
# print(first_file_text[:200])

# # Extract and print the last 200 words from the last file
# last_file_text = textract.process(word_files[-1]).decode('utf-8')
# print(f'\nLast 200 words from {word_files[-1]}:')
# print(last_file_text[-200:])


# from glob import glob

# word_files = glob('/Users/ronishlamsal/Desktop/Python/Knoxville/chapter6/docx/*.docx')

# # Extracting text file with textract
# import textract
# text = textract.process(word_files[0])      # Extracts text from the word file
# text = text.decode('utf-8')                 # Decodes extracted binary text into a human-readable string
# text[:200]                         # Prints first 200 characters

# import string
# import nltk

# translator = str.maketrans('', '', string.punctuation + string.digits)
# text = text.translate(translator)           # Removes punctuation and digits from the text

# # Check if stopwords dataset is downloaded
# if not nltk.data.find('corpora/stopwords.zip'):
#     nltk.download('stopwords')                 # Downloads stopwords data

# from nltk.corpus import stopwords

# en_stopwords = stopwords.words('english')   # Loads the English stopwords
# en_stopwords = set(en_stopwords)            # Converts it into a set for faster lookup

# words = text.lower().split()                # Converts text to lowercase and splits into individual words
# words = [w for w in words if w not in en_stopwords and len(w) > 3]  # Filters out stopwords and short words

# new_words = []
# for w in words:
#     if w not in en_stopwords and len(w) > 3:
#         new_words.append(w)

# bigrams = list([''.join(bg) for bg in nltk.bigrams(words)])  # Generates bigrams from the filtered words
# # print(bigrams[:3])  # Prints the first 3 bigrams

# ug_fdist = nltk.FreqDist(words)
# bg_fdist = nltk.FreqDist(bigrams)

# ug_fdist.most_common(20)
# bg_fdist.most_common(20)


# #We can also plot this

# import matplotlib.pyplot as plt 

# # ug_fdist.plot(20)

# from wordcloud import WordCloud

# wordcloud = WordCloud(collocations=False).generate(''.join(words))
# plt.imshow(wordcloud, interpolation ='bilinear')
# plt.axis("off")
# plt.show()



# from glob import glob
# import textract
# import string
# import nltk
# from nltk.corpus import stopwords
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud

# # Load word files
# word_files = glob('/Users/ronishlamsal/Desktop/Python/Knoxville/chapter6/docx/*.docx')

# # Extract text from the first file
# text = textract.process(word_files[0])
# text = text.decode('utf-8')
# text[:200]

# # Remove punctuation and digits
# translator = str.maketrans('', '', string.punctuation + string.digits)
# text = text.translate(translator)

# # Check and download stopwords
# if not nltk.data.find('corpora/stopwords.zip'):
#     nltk.download('stopwords')

# # Load stopwords and filter text
# en_stopwords = set(stopwords.words('english'))
# words = text.lower().split()
# words = [w for w in words if w not in en_stopwords and len(w) > 3]

# # Generate bigrams
# bigrams = list(nltk.bigrams(words))
# bigrams[:3]

# # Frequency distributions
# ug_fdist = nltk.FreqDist(words)
# bg_fdist = nltk.FreqDist(bigrams)


# print(ug_fdist.most_common(20))
# print(bg_fdist.most_common(20))

# ug_fdist.plot(20)

# # Generate and display word cloud
# if not words:
#     print("No words to generate a word cloud.")
# else:
#     wordcloud = WordCloud(width=800, height=400, collocations=False).generate(' '.join(words))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis("off")
#     plt.show()







# # Using Panda for wrangling Excel files

# import os
# import pandas as pd
# from glob import glob

# # Sort the files to ensure they are in order
# excel_files = sorted(glob('/Users/ronishlamsal/Downloads/Practical-Data-Science-with-Python-main/6-Chapter-6/data/excel/*.xlsx'))

# # Load the first Excel file
# df = pd.read_excel(excel_files[0], sheet_name='MISO', skiprows=4, nrows=17, index_col=0, usecols=range(7))

# # Print the filename of the first file
# print(f"Opening file: {excel_files[0]}")
# print(df)

# # Extract the date portion from the filename
# filename = os.path.basename(excel_files[0])  # Extracts the file name from the path
# date_str = filename.split('_')[0]  # Gets the date part (e.g., '20210123')

# # Convert the extracted date string to a datetime object
# date = pd.to_datetime(date_str, format='%Y%m%d')

# # Print the extracted date
# print(f"Extracted date: {date}")

# # Proceed with the rest of your code
# loads = df.loc['Projected Load', :].to_list()
# wind = df.loc['Renewable Forecast', :].to_list()

# # Generate column labels for the DataFrame
# load_labels = [f'load_d{d}' for d in range(1, 7)]
# wind_labels = [f'wind_d{d}' for d in range(1, 7)]

# # Using list to concatenate them easily
# data_dict = {col: val for col, val in zip(load_labels + wind_labels, loads + wind)}

# # Print the created data_dict (optional)
# print(data_dict)





# import os
# import pandas as pd 
# from glob import glob

# def extract_miso_forecasts(path):
#     """
#     Takes a filepath to .xlsx MISO MOM reports and extracts wind and load forecasts. 
#     Saves data to an Excel file - miso_forecasts.xlsx, returns the DataFrame. 
#     """
#     excel_files = glob(os.path.join(path, '/Users/ronishlamsal/Desktop/Python/Knoxville/chapter6/excel/*.xlsx'))
#     full_forecast_df = None
#     for file in excel_files:
#         df = [d.read_excel(file, sheet_name='MISO', skiprows=4, \
#                             nrows=17, index_col=0, usecols=range(7))]

#         #get data
#         loads = df.loc['Projected Load', :].to_list()
#         wind = df.loc['Renewable Forecast', :].to_list()

#         #make column labels
#         load_labels = [f'load_d{d}' for d in range(1, 7)]
#         wind_labels = [f'wind_d{d}' for d in range(1, 7)]

#         #create and append dataframe 
#         data_dict = {col: val for cal, val in zip(load_labels + wind_labels, \
#                                                     loads + winds)}
#         date = pd.to_datetime(file.split('\\')[-1].split('_')[0])
#         forecast_df = pd.DataFrame.from_records(data=data_dict, index=[date])
#         if full_forecast_df is None:
#             full_forecast_df= forecast_df.copy()
#         else:
#             full_forecast_df = full_forecast_df.append(forecast_df)

#     full_forecast_df.sort_index(inplace= True)
#     full_forecast_df,to_excel('miso_forecasts.xlsx')
#     return full_forecast_df











#Chapter 7 Web Scraping


#Using urllib




# url = 'https://en.wikipedia.org/wiki/Programming_language'
# page = urlopen(url).read()

# print(page[:50])
# print(page[:50].decode('utf-8'))



#getting a datafile

# from urllib.request import urlopen

# datafile_url = 'https://docs.misoenergy.org/marketreports/20240812_rt_bc.xls'
# mom_data = urlopen(datafile_url).read()
# print(mom_data[:20])
# print(mom_data[:20].decode('utf-8'))


# import pandas as pd
# from io import BytesIO
# from urllib.request import urlopen

# datafile_url = 'https://docs.misoenergy.org/marketreports/20240812_rt_bc.xls'
# mom_data = urlopen(datafile_url).read()

# # Use BytesIO to create a file-like object from the downloaded bytes
# xls_file = BytesIO(mom_data)

# # Read the Excel file into a Pandas DataFrame
# df = pd.read_excel(xls_file)

# # Display the first few rows of the DataFrame
# print(df.head())


# import requests as rq 

# url= 'https://en.wikipedia.org/wiki/Programming_language'

# response = rq.get(url)
# print(response.text[:50])

# import pandas as pd
# from io import BytesIO
# from urllib.request import urlopen
# import os

# import requests as rq 

# #Downloading/Scraping several files at once 

# #Define the URL and date range
# url = 'https://docs.misoenergy.org/marketreports/{}_rt_mf.xlsx'
# dates = pd.date_range(start='20240410', end = '20240812')
# dates = dates.strftime(date_format='%Y%m%d')


# #Ensure the directory exists
# output_dir = 'rt_mr'
# os.makedirs(output_dir, exist_ok=True)


# #Now looping through our dates and downloading the files

# import os
# from urllib.error import HTTPError
# from urllib.request import urlretrieve

# for d in dates:
#     filename = f'rt_mr/{d}_rt_mr.xlsx'
#     if os.path.exists(filename):
#         continue

#     try: 
#         urlretrieve(url.format(d), filename)
#     except HTTPError:
#         continue




# #parsing HTML

# from bs4 import BeautifulSoup as bs 
# import lxml
# from urllib.request import urlopen

# url = 'https://en.wikipedia.org/wiki/General-purpose_programming_language'
# wiki_text = urlopen(url).read().decode('utf-8')

# soup = bs(wiki_text)

# links = soup.find_all('a')
# print(links[102])

# soup.find_all('a', {'title': 'Programming language'})

# soup.find_all('a', text='Python')


# import lxml.html

# tree = lxml.html.fromstring(wiki_text)

# link_div = tree.xpath('////html/body')



#Web Scraping - Scraping Reddit/Subreddit for words

# import praw
# import string  # Importing the string module
# from nltk.corpus import stopwords  # Import stopwords from nltk
# from nltk import bigrams
# from nltk.probability import FreqDist

# reddit = praw.Reddit(
#     client_id="uUwkLbBwvkgYd5MDJ07fJA",
#     client_secret="qvsEORkHajFKO5nQnfRLWMSksNL8nw",
#     user_agent="python_book_demo"
# )

# post_text_list = []     # Creating empty lists
# comment_text_list = []  # Creating empty lists
# for post in reddit.subreddit("california").hot(limit=100): # Going through 'california subreddit'. Looping through posts.
#     post_text_list.append(post.selftext)
#     # removes 'show more comments' instances
#     post.comments.replace_more(limit=0)

#     for c in post.comments.list():
#         comment_text_list.append(c.body)

# all_text = ' '.join(post_text_list + comment_text_list)

# # Creating the translation table to remove punctuation and digits
# translator = str.maketrans('', '', string.punctuation + string.digits)

# cleaned_text = all_text.translate(translator)

# # Combining Reddit-specific stopwords with English stopwords
# en_stopwords = set(stopwords.words('english'))  # Ensure you have the nltk English stopwords
# reddit_stopwords = set(['removed', 'dont']) | en_stopwords
# cleaned_words = [w for w in cleaned_text.lower().split() if w not in reddit_stopwords and len(w) > 3]

# # Finding bigrams and their frequencies
# bg = [' '.join(bigr) for bigr in bigrams(cleaned_words)]
# bg_fd = FreqDist(bg)
# print(bg_fd.most_common(10))












#Chapter 8: Probability Distributions and Sampling

#Gaussian or Normal Distribution 

# import numpy as np 
# from scipy.stats import norm
# import matplotlib.pyplot as plt 

# x = np.linspace(-4,4,100)
# plt.plot(x, norm.pdf(x))
# plt.plot(x, norm.cdf(x))


# #To show the plot, have to use plt.show

# plt.title("Normal Distribution: PDF and CDF")  #adding a title
# plt.xlabel('X-axis')    #Labeling x-axis
# plt.ylabel('Probability')
# plt.legend()    #Displaying the legend
# plt.grid(True)  #Adding a grid for better readibility
# plt.show()      #Finally showing the plot



#random sampling

# import bootstrapped.bootstrap as bs 
# import bootstrapped.stats_function as bs_stats

# bs.bootstrap(df['efficiency'].values, stat_func=bs_stats.mean)







#Chapter 10: Machine learning : Preping Data for Machine Learning 



import pandas as pd
from ydata_profiling import ProfileReport

loan_df = pd.read_csv('/Users/ronishlamsal/Desktop/Python/Knoxville/10-Chapter-10/data/loan_data.csv',
                        parse_dates=['DATE_OF_BIRTH', 'DISBURSAL_DATE'],
                        infer_datetime_format=True)
report = ProfileReport(loan_df.sample(10000, random_state=42))
report.to_file('loan_dfl.html')

loan_df.drop('UNIQUEID', axis=1, inplace=True)



#Screening through our features and listing those with high numbers of unique values

for col in loan_df.columns:
    fraction_unique = loan_df[col].unique().shape[0] / loan_df.shape[0]
    if fraction_unique> 0.5:
        print(col)



#Looking for feature with too little variance
drop_cols = ['MOBILENO_AVL_FLAG']  #this one has too little variance. just 0 or 1


pri_sec_cols = [c for c in loan_df.columns if c[:3] in ['PRI', 'SEC'] and \
               c not in ['PRI_NO_ACCTS', 'PRI_OVERDUE_ACCTS']]
drop_cols.extend(pri_sec_cols)
loan_df.drop(columns = drop_cols, axis=1, inplace=True)

loan_df = pd.get_dummies(loan_df, columns=['EMPLOYMENT_TYPE', 'PERFORM_CNS_SCORE_DESCRIPTION'], drop_first=True)

import re

def convert_to_months(duration_str):
    years = re.search(r'(\d+)yrs', duration_str)
    months = re.search(r'(\d+)mon', duration_str)
    total_months = 0
    if years:
        total_months += int(years.group(1)) * 12
    if months:
        total_months += int(months.group(1))
    return total_months

loan_df['AVERAGE_ACCT_AGE'] = loan_df['AVERAGE_ACCT_AGE'].apply(convert_to_months)
loan_df['CREDIT_HISTORY_LENGTH'] = loan_df['CREDIT_HISTORY_LENGTH'].apply(convert_to_months)

corr = loan_df.corr().loc['LOAN_DEFAULT'][:-1]
corr.plot.barh()













































































































































































































































































































































