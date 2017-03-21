# fab - fed analysis bot
## Detailed description of project
Eight times a year, the Federal Open Markets Committee (FOMC) meets to discuss the regarding rate 

## What question/problem will your project attempt to answer/solve?
What are the effects on the various asset classes from the FOMC meetings?

## How will you present your work?
### Different ways to build the model

1. Guessing majority class every time
2. Fab model with bag of words
3. Fab model with TF-IDF
4. Fab model with part of speech tagging
5. Fab model with previous market data features
6. Fab model with SARIMA / ARIMA 
7. Fab model with some or many of the above with a neural net

### Different time scales of effect
1. High frequency sub minute OHLC data
2. Hourly OHLC data
3. Daily OHLC data
4. Weekly / long term OHLC data

### effect on different asset classes
1. short term treasuries
2. volatility
3. S&P 500

## Web app - where will you host it, what kind of information will you present?
Hosted on my own server at home.  It will have at the minimum some visuals.  Idealy I would also like it to be running live and making predictions or waiting for meeting minute releases.

## Visualization - what final visuals are you aiming to produce?
- Start with basic matplotlib / seaborn plots for the above mentioned presentation
- Idealy have some interactive visuals for all the above methods in a single frame

## Presentation - slides, interpretive dance?
Slides and maybe a ritualistic dance

## What are your data sources?
- Federal Open Market Committee website for the meeting minutes, statesments and or forecasts.
- Daily stock price data from Yahoo Finance
- Higher frequency data from Interactive Brokers and IQ Feed
- 
## What is the next thing you need to work on?
- This proposal followed by a clear and workable plan and research all the previous easy to find projects online on the same topic


## Getting the data, not just some, likely all?
- All the data since 1994 are available though much of it may not be so relevant as FOMC since 2005 has become a lot more transparent regarding their motivations and concerns.  The nature of the data has changed a lot.


## Understanding the data?
- The text data will need to be processed via either NLTK or spaCy to use the text similarity ro TF-IDF for training the classifier
- We will detrend the data and use the ARIMA model to look for signal in the price data


## Building a minimum viable product?
#### Visualization
- use bag-of-words / TF-IDF 
- start with a predictions using daily data
- Understand the class balance (how many occurences are up and how many are down)
- Use a moving average on the given instrument on the previous n days as the the proxy for the expectation going into the meeting (How much have already been priced in)
- Matplotlib / Seaborn plots on predictions and actual


## Gauging how much signal might be in the data?
- A significant percentage (as high as 80%) of annual return on the broader markets are realized on the days of the FOMC meeting minutes announcements. There is definitely signal there.  Lets quantify it with some metrics:
	- F1 or AUC score on the prediction of if the price of the instrument will be moving lower or higher after the announcement

	
	