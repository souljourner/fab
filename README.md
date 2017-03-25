# *fab*
(Fed Analyser Bot)

A project in progress by John Zhu for Galvanize (Zipfian) Data Science Immersive.

## project goal
Approximately eight times a year, members of the Federal Open Markets Committee (FOMC) convene to make monetary decision at scheduled meeting dates.  With the conclusion of every meeting, they release the statements and minutes from the meetings.

The *fab* project seeks to better understand the effects of the meeting statements and the asset classes that it affects the most.

Having witnessed several insightful projects in this space (such as [this](http://www.nber.org/papers/w15367) and [this](http://blog.alexchaia.uk/home/archives/05-2016)), I will not repeat their experiments but will instead move to attempt to make predictions based on the information contained in the text of the document.

## natural language processing (nlp)
This project makes use of the amazing [spaCy package](https://github.com/explosion/spaCy)  to create a senteiment analyser as part of the prediction algorithm.


## extras

As a side project, I also created a [text extractor tool](https://github.com/souljourner/FOMC-Statements-Minutes-Scraper) to facilitate statement extraction  from the FOMC website so that you may perform your own natural language processing (NLP).
