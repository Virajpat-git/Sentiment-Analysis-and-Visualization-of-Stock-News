# @Viraj2399
# The urllib.request library is used to get the url in to our database
from urllib.request import urlopen, Request
# The bs4 or BeautifulSoup is a library used to parse the HTML code
from bs4 import BeautifulSoup
# This is the package that we are using to do analysis link:https://www.nltk.org/_modules/nltk/sentiment/vader.html
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Panda is used to refactor the data to be shown in the visual format
import pandas as pd
import matplotlib.pyplot as plt

# The below-mentioned url is only the half of the url, and then we will add the stock tickers in the url by appending
# the tickers list

finviz_url = 'https://finviz.com/quote.ashx?t='
tickers = ['AMZN', 'GOOG', 'FB']

news_tables = {}

for ticker in tickers:
    url = finviz_url + ticker

    req = Request(url=url, headers={'user-agent': 'my-app'})
    response = urlopen(req)

    html = BeautifulSoup(response, features='html.parser')
    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table

parsed_data = []

for ticker, news_table in news_tables.items():
    # By using the findAll function we will find the or parse over the title and the date and time
    for row in news_table.findAll('tr'):
        date_data = []
        title = row.a.text
        date_data = row.td.text.split(' ')
        # if the title contains the date and the time then we will print it in the below-mentioned code
        if len(date_data) == 1:
            time = date_data[0]
        else:
            date = date_data[0]
            time = date_data[1]
        # In this format the data will be stored in the database
        # @Viraj2399 : Tip to self always give correct indentation
        parsed_data.append([ticker, date, time, title])

df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])
# From here sentiment analyser code starts
vader = SentimentIntensityAnalyzer()

# print(vader.polarity_scores("I  think Apple is a bad company. I think they will do fail sales this quarter."))

# we are using lambda function to only give the data structure the compound score
f = lambda title: vader.polarity_scores(title)['compound']
df['compound'] = df['title'].apply(f)
# The below line will format the date string into date format
df['date'] = pd.to_datetime(df.date).dt.date

# THe code below this is for matplotlib
plt.figure(figsize=(10, 8))

mean_df = df.groupby(['ticker', 'date']).mean()
# By unstacking it,it will be on the X-axis
mean_df = mean_df.unstack()
mean_df = mean_df.xs('compound', axis="columns").transpose()
mean_df.plot(kind='bar')
plt.show()
