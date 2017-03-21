from __future__ import print_function
from bs4 import BeautifulSoup
from urllib2 import urlopen
import re
import pandas as pd
import pickle
import threading

class FOMC (object):
    '''
    A convenient class for extracting meeting minutes from the FOMC website
    Example Usage:  
    '''

    def __init__(self, base_url='https://www.federalreserve.gov', 
                 calendar_url='https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm',
                 historical_date = 2011,
                 verbose = True):
        self.base_url = base_url
        self.calendar_url = calendar_url
        self.df = None
        self.links = None
        self.dates = None
        self.articles = None
        self.HISTORICAL_DATE = historical_date
        self.verbose = verbose

    def _get_links(self, from_year):
        '''
        private function that sets all the links for the FOMC meetings from the giving from_year
        to the current most recent year
        '''
        if self.verbose:
            print("Getting links...")
        self.links = []
        fomc_meetings_socket = urlopen(self.calendar_url)
        soup = BeautifulSoup(fomc_meetings_socket, 'html.parser')

        statements = soup.find_all('a', href=re.compile('^/newsevents/pressreleases/monetary\d{8}a.htm'))
        self.links = [statement.attrs['href'] for statement in statements] 

        if from_year <= self.HISTORICAL_DATE:        
            for year in range(from_year, self.HISTORICAL_DATE + 1):
                fomc_yearly_url = self.base_url + '/monetarypolicy/fomchistorical' + str(year) + '.htm'
                fomc_yearly_socket = urlopen(fomc_yearly_url)
                soup_yearly = BeautifulSoup(fomc_yearly_socket, 'html.parser')
                statements_historical = soup_yearly.findAll('a', text = 'Statement')
                for statement_historical in statements_historical:
                    self.links.append(statement_historical.attrs['href'])

    def _date_from_link(self, link):
        date = re.findall('[0-9]{8}', link)[0].encode('ascii')
        if date[4] == '0':
            date = date[:4] + '/' + date[5:6] + '/' + date[6:]
        else:
            date = date[:4] + '/' + date[4:6] + '/' + date[6:]
        return date

    def _get_articles(self):
        if self.verbose:
            print("Getting articles...")

        self.dates, self.articles = [], []

        for link in self.links:
            if self.verbose:
                print(".", end='')

            # date of the article content
            self.dates.append(self._date_from_link(link))
            statement_socket = urlopen(self.base_url + link)
            statement = BeautifulSoup(statement_socket, 'html.parser')
            paragraphs = statement.findAll('p')
            content = [paragraph.get_text() for paragraph in paragraphs]
            self.articles.append(content)

        for row in range(len(self.articles)):
            self.articles[row] = map(lambda x: x.strip(), self.articles[row])
            words = " ".join(self.articles[row]).split()
            self.articles[row] = " ".join(words)

    def get_statements(self, from_year=2000):
        '''
        Returns a Pandas DataFrame of meeting minutes with the date as the index
        uses a date range of from_year to the most current

        Input from_year is ignored if it is within the last 5 years as this is meant for 
        parsing much older years
        '''
        self._get_links(from_year)
        self._get_articles()
        self.df = pd.DataFrame(self.articles, index = pd.to_datetime(self.dates)).sort_index()
        return self.df

    def pick_df(self, filename="../data/minutes.pickle"):
        if filename:
            if self.verbose:
                print("Writing to", filename)        
            with open(filename, "wb") as output_file:
                    pickle.dump(self.df, output_file)
