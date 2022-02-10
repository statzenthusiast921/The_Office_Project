# The Office Project

## Description

The goal of this project was to understand how language differs and compares between all of the major characters of the tv show, The Office.  I addressed the following questions:
*How do speech patterns differ between characters and through seasons?
*Can sentiment analysis uncover trends and intensities in emotion throughout the show?
*Can we understand how certain characters are connected in any episode?
*Can we glean any insights from online sources like Twitter or IMDB?

## Data
* [Complete Transcript from The Office](https://www.kaggle.com/nasirkhalid24/the-office-us-complete-dialoguetranscript/version/1#)
* [Episode Descriptions - IMDB](https://www.imdb.com/title/tt0386676/episodes?season=1)
* [Episode Descriptions - Wiki](https://en.wikipedia.org/wiki/The_Office_(American_season_1))

## Challenges
* Analysis of Twitter data only focused on Season 6 through Season 9 as this data was not extensively available until the 6th season (2009)
* The [BERT](https://github.com/thoailinh/Sentiment-Analysis-using-BERT) algorithm would have been the best choice for sentiment analysis on character dialogue, but I had a multitude of issues getting good results (eg: converting most examples from sentiment classification to sentiment regression did not yield great results).  Therefore, I used the [VADER](https://github.com/cjhutto/vaderSentiment) algorithm instead for both character dialogue and Twitter sentiment analysis.  This algorithm is very easy to apply to any text data, but it is best applied with social media data and can be sensitive to minor changes in spelling and grammar.  

## What I Learned
* NLP is a fun branch of ML.  I definitely want to make more projects involving these types of analyses.
* Network graphing is very cool!  [VISDCC](https://github.com/jimmybow/visdcc) library is amazing!
