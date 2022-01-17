#Date Created: 08/13/21
#Date Last Modified: 01/17/22
#-----------------------------------------------#

#Import libraries
from dash_bootstrap_components._components.Card import Card
from numpy.core.numeric import True_, full
import pandas as pd
import numpy as np
import dash
import os
import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('ggplot')
from dash import dcc
from dash import html
#from dash import dbc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, Response
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import re
import contractions
from dash import dash_table as dt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import random


#!pip install visdcc
import visdcc
import networkx as nx
import itertools as it
from nltk.stem import WordNetLemmatizer 
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer

#Load data
#/Users/jonathan.zimmerman/Desktop/Office NLP/the-office-lines.xlsx
office_data = pd.read_excel("/Users/jonzimmerman/Desktop/Data Projects/The Office NLP/the-office-lines.xlsx")
imdb_data = pd.read_csv('https://raw.githubusercontent.com/statzenthusiast921/The_Office_Project/main/office_episodes.csv')
wiki_desc = pd.read_csv('/Users/jonzimmerman/Desktop/Data Projects/The Office NLP/wiki_desc.csv')

#Import Twitter data
import pandas as pd
s6_tweets = pd.read_csv('https://raw.githubusercontent.com/statzenthusiast921/The_Office_Project/main/season6_tweets.csv')
s7_tweets = pd.read_csv('https://raw.githubusercontent.com/statzenthusiast921/The_Office_Project/main/season7_tweets.csv')
s8_tweets = pd.read_csv('https://raw.githubusercontent.com/statzenthusiast921/The_Office_Project/main/season8_tweets.csv',lineterminator='\n')
s9_tweets = pd.read_csv('https://raw.githubusercontent.com/statzenthusiast921/The_Office_Project/main/season9_tweets.csv')


s6_tweets['episode'] = s6_tweets['episode ']
s8_tweets['season'] = s8_tweets['se_ep'].str.slice(start=1, stop=2)
s8_tweets['episode'] = s8_tweets['se_ep'].str.slice(start=4)

tweets = [
    s6_tweets, 
    s7_tweets,
    s8_tweets,
    s9_tweets
    
]
tweet_data = pd.concat(tweets)
tweet_data['season'] = 'Season ' + tweet_data['season'].astype(str)
tweet_data['episode'] = 'Episode ' + tweet_data['episode'].astype(str)

tweet_seasons = tweet_data['season'].unique()
tweet_episodes = tweet_data['episode'].unique()



tweet_data = tweet_data[['date','tweet','season','episode','user_id_str']]

#Get longer desc into imdb_data
imdb_data = pd.merge(imdb_data,wiki_desc,how='left',on=['season','episode'])
del imdb_data['description']
imdb_data['season'] = 'Season ' + imdb_data['season'].astype(str)
imdb_data['episode'] = 'Episode ' + imdb_data['episode'].astype(str)
imdb_data.head()

imdb_tojoin_tweets = imdb_data[['season','episode','rating']]
imdb_tojoin_tweets['season'] = imdb_tojoin_tweets['season'].str.rstrip('.0')
imdb_tojoin_tweets['episode'] = imdb_tojoin_tweets['episode'].str.rstrip('.0')




#Join data
office_data = pd.merge(office_data,imdb_data,how='left',on=['season','episode'])
tweet_data = pd.merge(tweet_data, imdb_tojoin_tweets,how='left',on=['season','episode'])



#Fix ratings, scaled rating for Episodes 1, 2, 10, 20 of Season 6-9
#1.) Season 6
tweet_data['rating'] = np.where((tweet_data['season']=='Season 6') & (tweet_data['episode']=='Episode 1'),8.8,tweet_data['rating'])
tweet_data['rating'] = np.where((tweet_data['season']=='Season 6') & (tweet_data['episode']=='Episode 2'),8.1,tweet_data['rating'])
tweet_data['rating'] = np.where((tweet_data['season']=='Season 6') & (tweet_data['episode']=='Episode 10'),8.2,tweet_data['rating'])
tweet_data['rating'] = np.where((tweet_data['season']=='Season 6') & (tweet_data['episode']=='Episode 20'),7.8,tweet_data['rating'])


#2.) Season 7
tweet_data['rating'] = np.where((tweet_data['season']=='Season 7') & (tweet_data['episode']=='Episode 1'),8.3,tweet_data['rating'])
tweet_data['rating'] = np.where((tweet_data['season']=='Season 7') & (tweet_data['episode']=='Episode 2'),8.2,tweet_data['rating'])
tweet_data['rating'] = np.where((tweet_data['season']=='Season 7') & (tweet_data['episode']=='Episode 10'),8.2,tweet_data['rating'])
tweet_data['rating'] = np.where((tweet_data['season']=='Season 7') & (tweet_data['episode']=='Episode 20'),9,tweet_data['rating'])


#3.) Season 8
tweet_data['rating'] = np.where((tweet_data['season']=='Season 8') & (tweet_data['episode']=='Episode 1'),8.1,tweet_data['rating'])
tweet_data['rating'] = np.where((tweet_data['season']=='Season 8') & (tweet_data['episode']=='Episode 2'),8.1,tweet_data['rating'])
tweet_data['rating'] = np.where((tweet_data['season']=='Season 8') & (tweet_data['episode']=='Episode 10'),7.9,tweet_data['rating'])
tweet_data['rating'] = np.where((tweet_data['season']=='Season 8') & (tweet_data['episode']=='Episode 20'),7,tweet_data['rating'])


#4.) Season 9
tweet_data['rating'] = np.where((tweet_data['season']=='Season 9') & (tweet_data['episode']=='Episode 1'),7.5,tweet_data['rating'])
tweet_data['rating'] = np.where((tweet_data['season']=='Season 9') & (tweet_data['episode']=='Episode 2'),7.1,tweet_data['rating'])
tweet_data['rating'] = np.where((tweet_data['season']=='Season 9') & (tweet_data['episode']=='Episode 10'),7.6,tweet_data['rating'])
tweet_data['rating'] = np.where((tweet_data['season']=='Season 9') & (tweet_data['episode']=='Episode 20'),8,tweet_data['rating'])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
tweet_data['scaled_rating'] = scaler.fit_transform(tweet_data['rating'].values.reshape(-1,1))



def lemmatize(text):
    lemmed = contractions.fix(str(text))
    return lemmed


def removeStopWords(str):
#select english stopwords
  cachedStopWords = set(stopwords.words("english"))
#add custom words
  cachedStopWords.update(('like','um','uh','oh',' s ','and','i','I','a','and','so','this','when','it','many','so','cant','yes','no','these'))
#remove stop words
  new_str = ' '.join([word for word in str.split() if word not in cachedStopWords]) 
  return new_str



#Define functions to clean up script column
def punct(text):
  token=RegexpTokenizer(r'\w+')#regex
  text = token.tokenize(text)
  text= " ".join(text)
  return text 

def remove_special_characters(text):
    # define the pattern to keep
    pat = r'[^a-zA-z0-9.,!?/:;\"\'\s]' 
    return re.sub(pat, '', text)


def remove_digits(text):
    pattern = r'[^a-zA-z.,!?/:;\"\'\s]' 
    return re.sub(pattern, '', text)

def remove_brackets_contents(text):
    pattern = r"\[.*?\]"
    return re.sub(pattern, "", text)


def removeStopWords_and_dumb_tweets(str):
#select english stopwords
  cachedStopWords = set(stopwords.words("english"))
#add custom words
  cachedStopWords.update(('like','um','uh','oh',' s ','and','i','I','a','and','so','this','when','it','many','so','cant','yes','no',
  'these','office','theoffice','theofficenbc','deleted','scene','watching','watch','tonight','fucking','freakfest','freakfestfucking','downtowndetroit','getglue','terryperry','terrytperry',
  'fridays','privatevip','freakygirls','badd','biotches','want','episode','season','show','last night','night','make','felonydaprince','joegunner','peek','pictures','sneak','exclusive','first'
  'promo','watched','nbcstore','officetally','clip','last','cheaturself','treaturself','nite','ladiesfreetil','parishouston','krayjuice','tinkabhottie','msebonnieb','theofficialfelz'

  ))
#remove stop words
  new_str = ' '.join([word for word in str.split() if word not in cachedStopWords]) 
  return new_str


#Clean up script
#0.) Convert column to string type
office_data['line_text'] = office_data['line_text'].astype('str')
#1.) Convert to lowercase
office_data['cleaned_text'] = office_data['line_text'].str.lower()
#2.) Remove brackets
office_data['cleaned_text'] = office_data['cleaned_text'].apply(func = remove_brackets_contents)
#3.) Remove special characters
office_data['cleaned_text'] = office_data['cleaned_text'].apply(func = remove_special_characters)
#4.) Remove digits
office_data['cleaned_text'] = office_data['cleaned_text'].apply(func = remove_digits)
#5.) Remove punctuation
office_data['cleaned_text'] = office_data['cleaned_text'].apply(func = punct)
#6.) Lemmatize words
office_data['cleaned_text'] = office_data['cleaned_text'].apply(func = lemmatize)
#7.) Remove stop words
office_data['cleaned_text'] = office_data['cleaned_text'].apply(func = removeStopWords)









#Clean the tweets
#0.) Convert everything to string
tweet_data['cleaned_tweet'] = tweet_data['tweet'].astype(str)
#1.) Lowercase
tweet_data['cleaned_tweet'] = tweet_data['tweet'].str.lower()
#2.) Remove hastags and mentions
tweet_data['cleaned_tweet'] = tweet_data['cleaned_tweet'].str.replace(r'@', '')
tweet_data['cleaned_tweet'] = tweet_data['cleaned_tweet'].str.replace(r'#', '')
#3.) Remove links
tweet_data['cleaned_tweet'] = tweet_data['cleaned_tweet'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)
#4.) Remove punctuation
tweet_data['cleaned_tweet'] = tweet_data['cleaned_tweet'].apply(func = punct)
#5.) Remove non alpha-numeric characters
tweet_data['cleaned_tweet'] = tweet_data['cleaned_tweet'].apply(func = remove_special_characters)
#6.) Remove digits
tweet_data['cleaned_tweet'] = tweet_data['cleaned_tweet'].apply(func = remove_digits)
#7.) Lemmatize words
tweet_data['cleaned_tweet'] = tweet_data['cleaned_tweet'].apply(func = lemmatize)
#8.) Remove stop words and dumb tweets
tweet_data['cleaned_tweet'] = tweet_data['cleaned_tweet'].apply(func = removeStopWords_and_dumb_tweets)
#9.) Remove certain users
tweet_data = tweet_data[tweet_data['user_id_str']!=75195164]
#10.) Remove explicit tweets
tweet_data = tweet_data[~tweet_data["cleaned_tweet"].str.contains("sex",case=False)]






#Test out metrics


sid = SentimentIntensityAnalyzer()
office_data['scores'] = office_data['line_text'].apply(lambda line_text: sid.polarity_scores(line_text))
office_data['compound']  = office_data['scores'].apply(lambda score_dict: score_dict['compound'])
office_data['comp_score'] = office_data['compound'].apply(lambda c: 'pos' if c >=0 else 'neg')




#-----------Network Graph----------#
#https://towardsdatascience.com/visualizing-networks-in-python-d70f4cbeb259

#1.) Filter down to just data with main characters
office_data['main_ind'] = np.where(
    (office_data['speaker']=='Pam')|
    (office_data['speaker']=='Jan')|
    (office_data['speaker']=='Kelly')|
    (office_data['speaker']=='Phyllis')|
    (office_data['speaker']=='Angela')|
    (office_data['speaker']=='Erin')|
    (office_data['speaker']=='Holly')|
    (office_data['speaker']=='Meredith')|
    (office_data['speaker']=='Michael')|
    (office_data['speaker']=='Jim')|
    (office_data['speaker']=='Kevin')|
    (office_data['speaker']=='Oscar')|
    (office_data['speaker']=='Stanley')|
    (office_data['speaker']=='Toby')|
    (office_data['speaker']=='Roy')|
    (office_data['speaker']=='Ryan')|
    (office_data['speaker']=='Andy')|
    (office_data['speaker']=='Creed')|
    (office_data['speaker']=='Darryl')|
    (office_data['speaker']=='Dwight'),
    1,0
)

#2.) Filter down to only scenes containing these people
size = office_data.groupby(['season','episode','scene']).size().reset_index()
sums = office_data.groupby(['season','episode','scene']).agg({'main_ind':'sum'}).reset_index()


main_metrics = pd.merge(size,sums,how='left',on=['season','episode','scene'])
main_metrics.rename(columns={0:'count'}, inplace=True )

office_data = pd.merge(office_data,main_metrics,how='left',on=['season','episode','scene'])
office_data['diff'] = office_data['count'] - office_data['main_ind_y']


#Average/Median Compound for scenes with only 2 people

#1.) Only scenes with main characters talking to other main characters
office_data['just_the_2'] = np.where(
    (office_data['speaker']=='Dwight')|
    (office_data['speaker']=='Andy'),
    1,0
)


size2 = office_data.groupby(['season','episode','scene']).size().reset_index()
sums2 = office_data.groupby(['season','episode','scene']).agg({'just_the_2':'sum'}).reset_index()



data_for_ng = office_data[office_data['diff']==0]


#3.) Make source, target, weight columns (weight - # of convos)
filtered = data_for_ng[['season','episode','scene','speaker']]



#Create a season-character dictionary
season_character_dict = {'Season 1': ['Angela', 'Darryl', 'Dwight', 'Jan', 'Jim','Kelly','Kevin','Meredith','Michael','Oscar','Pam','Phyllis','Roy','Ryan','Stanley','Toby'],
                         'Season 2': ['Angela','Creed', 'Darryl', 'Dwight', 'Jan', 'Jim','Kelly','Kevin','Meredith','Michael','Oscar','Pam','Phyllis','Roy','Ryan','Stanley','Toby'],
                         'Season 3': ['Andy', 'Angela','Creed', 'Darryl', 'Dwight', 'Jan', 'Jim','Karen','Kelly','Kevin','Meredith','Michael','Oscar','Pam','Phyllis','Roy','Ryan','Stanley','Toby'],
                         'Season 4': ['Andy', 'Angela','Creed', 'Darryl', 'Dwight','Holly', 'Jan', 'Jim','Kelly','Kevin','Meredith','Michael','Oscar','Pam','Phyllis','Roy','Ryan','Stanley','Toby'],
                         'Season 5': ['Andy', 'Angela','Creed', 'Darryl', 'Dwight','Erin','Holly', 'Jan', 'Jim','Karen','Kelly','Kevin','Meredith','Michael','Oscar','Pam','Phyllis','Roy','Ryan','Stanley','Toby'],
                         'Season 6': ['Andy', 'Angela','Creed', 'Darryl', 'Dwight','Erin','Gabe','Holly','Jan', 'Jim','Kelly','Kevin','Meredith','Michael','Oscar','Pam','Phyllis','Ryan','Stanley','Toby'],
                         'Season 7': ['Andy', 'Angela','Creed', 'Darryl', 'Dwight','Erin','Gabe','Holly','Jan', 'Jim','Karen','Kelly','Kevin','Meredith','Michael','Oscar','Pam','Phyllis','Ryan','Stanley','Toby'],
                         'Season 8': ['Andy', 'Angela','Creed', 'Darryl', 'Dwight','Erin','Gabe', 'Jim','Kelly','Kevin','Meredith','Oscar','Pam','Phyllis','Ryan','Stanley','Toby'],
                         'Season 9': ['Andy', 'Angela','Creed', 'Darryl', 'Dwight','Erin','Gabe','Jan','Jim','Kelly','Kevin','Meredith','Michael','Oscar','Pam','Phyllis','Roy','Ryan','Stanley','Toby']
}



season_episode_dict = {'Season 1': ['Episode 1', 'Episode 2', 'Episode 3', 'Episode 4', 'Episode 5','Episode 6'],
                         'Season 2': ['Episode 1', 'Episode 2', 'Episode 3', 'Episode 4', 'Episode 5','Episode 6','Episode 7', 'Episode 8', 'Episode 9', 'Episode 10', 'Episode 11','Episode 12','Episode 13', 'Episode 14', 'Episode 15', 'Episode 16', 'Episode 17','Episode 18','Episode 19', 'Episode 20', 'Episode 21', 'Episode 22'],
                         'Season 3': ['Episode 1', 'Episode 2', 'Episode 3', 'Episode 4', 'Episode 5','Episode 6','Episode 7', 'Episode 8', 'Episode 9', 'Episode 10', 'Episode 11','Episode 12','Episode 13', 'Episode 14', 'Episode 15', 'Episode 16', 'Episode 17','Episode 18','Episode 19', 'Episode 20', 'Episode 21', 'Episode 22', 'Episode 23'],
                         'Season 4': ['Episode 1', 'Episode 2', 'Episode 3', 'Episode 4', 'Episode 5','Episode 6','Episode 7', 'Episode 8', 'Episode 9', 'Episode 10', 'Episode 11','Episode 12','Episode 13', 'Episode 14'],
                         'Season 5': ['Episode 1', 'Episode 2', 'Episode 3', 'Episode 4', 'Episode 5','Episode 6','Episode 7', 'Episode 8', 'Episode 9', 'Episode 10', 'Episode 11','Episode 12','Episode 13', 'Episode 14', 'Episode 15', 'Episode 16', 'Episode 17','Episode 18','Episode 19', 'Episode 20', 'Episode 21', 'Episode 22', 'Episode 23','Episode 24','Episode 25','Episode 26'],
                         'Season 6': ['Episode 1', 'Episode 2', 'Episode 3', 'Episode 4', 'Episode 5','Episode 6','Episode 7', 'Episode 8', 'Episode 9', 'Episode 10', 'Episode 11','Episode 12','Episode 13', 'Episode 14', 'Episode 15', 'Episode 16', 'Episode 17','Episode 18','Episode 19', 'Episode 20', 'Episode 21', 'Episode 22', 'Episode 23','Episode 24'],
                         'Season 7': ['Episode 1', 'Episode 2', 'Episode 3', 'Episode 4', 'Episode 5','Episode 6','Episode 7', 'Episode 8', 'Episode 9', 'Episode 10', 'Episode 11','Episode 12','Episode 13', 'Episode 14', 'Episode 15', 'Episode 16', 'Episode 17','Episode 18','Episode 19', 'Episode 20', 'Episode 21', 'Episode 22', 'Episode 23','Episode 24'],
                         'Season 8': ['Episode 1', 'Episode 2', 'Episode 3', 'Episode 4', 'Episode 5','Episode 6','Episode 7', 'Episode 8', 'Episode 9', 'Episode 10', 'Episode 11','Episode 12','Episode 13', 'Episode 14', 'Episode 15', 'Episode 16', 'Episode 17','Episode 18','Episode 19', 'Episode 20', 'Episode 21', 'Episode 22', 'Episode 23','Episode 24'],
                         'Season 9': ['Episode 1', 'Episode 2', 'Episode 3', 'Episode 4', 'Episode 5','Episode 6','Episode 7', 'Episode 8', 'Episode 9', 'Episode 10', 'Episode 11','Episode 12','Episode 13', 'Episode 14', 'Episode 15', 'Episode 16', 'Episode 17','Episode 18','Episode 19', 'Episode 20', 'Episode 21', 'Episode 22', 'Episode 23']
}

office_data.set_index(['season','episode']).to_dict()['speaker']


s1 = season_character_dict["Season 1"]
s2 = season_character_dict["Season 2"]
s3 = season_character_dict["Season 3"]
s4 = season_character_dict["Season 4"]
s5 = season_character_dict["Season 5"]
s6 = season_character_dict["Season 6"]
s7 = season_character_dict["Season 7"]
s8 = season_character_dict["Season 8"]
s9 = season_character_dict["Season 9"]

all_peeps = set(s1+s2+s3+s4+s5+s6+s7+s8+s9)
all_main_chars = np.array(list(all_peeps))
all_main_chars = np.sort(all_main_chars)
all_main_chars_list = list(all_main_chars)

the_mains_df = office_data[office_data['speaker'].isin(all_main_chars)]
main_characters_choose = the_mains_df['speaker'].unique()
main_characters_choose_list = list(main_characters_choose)

#the_mains_dict = the_mains_df.to_dict(['season','episode','speaker'])

#season_episode_character_dict = the_mains_df.groupby('season')[['episode','speaker']].apply(lambda x: x.set_index('episode').to_dict(orient='index')).to_dict()
#print(season_episode_character_dict)


the_mains_df_cols = the_mains_df[['season','episode','speaker']]
mains_dict = the_mains_df_cols.to_dict('index')
# x=mains_dict[0].get('speaker')
# print(x)


character_choices = office_data['speaker'].sort_values().unique()
season_choices = office_data['season'].sort_values().unique()
episode_choices = office_data['episode'].sort_values().unique()


# Defining a function to visualise n-grams
def get_top_ngram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) 
                  for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:10]


season_episode_character_dictionary = {}

for season in the_mains_df['season'].unique().tolist():
    df_season = the_mains_df[the_mains_df['season'].eq(season)]
    season_episode_character_dictionary[season] = {}

    for episode in df_season['episode'].unique().tolist():
        df_episode = df_season[df_season['episode'].eq(episode)]
        characters = sorted(df_episode['speaker'].unique().tolist())
        season_episode_character_dictionary[season][episode] = characters

#season_episode_character_dictionary['Season 1']
#season_episode_character_dictionary['Season 1']['Episode 2']


tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold',
    'backgroundColor': '#222222'

}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': 'grey',
    'color': 'white',
    'padding': '6px'
}

# BS = "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"
# app = dash.Dash(external_stylesheets=[BS])

app = dash.Dash(__name__,assets_folder=os.path.join(os.curdir,"assets"))
server = app.server
app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='What is this project about?',value='tab-1',style=tab_style, selected_style=tab_selected_style,
            children=[
                html.Div([
                    html.H1(dcc.Markdown('''**Welcome to my Text Analysis of The Office!**''')),
                    html.Br()
                ]), 
                html.Div([
                    html.P(dcc.Markdown('''**What is the purpose of this dashboard?**''')),
                ],style={'text-decoration': 'underline'}),
                html.Div([
                    html.P("This analysis attempts to uncover patterns in the speech of major characters of the show.  The following questions were addressed:"),
                    html.P('1.) How do speech patterns differ between characters and through seasons?'),
                    html.P('2.) Can sentiment analysis uncover trends and intensities in emotion throughout the show?'),
                    html.P('3.) Can we understand how and why certain characters are connected in any episode?'),
                    html.P('4.) Can we glean any insights from social media data?'),
                    html.Br()
                ]),
                html.Div([
                    html.P(dcc.Markdown('''**What data is being used for this analysis?**''')),
                ],style={'text-decoration': 'underline'}),
                html.Div([
                    html.P(['The complete transcript of every episode from all 9 seasons of The Office were used from this link ', html.A('here',href='https://www.kaggle.com/nasirkhalid24/the-office-us-complete-dialoguetranscript/version/1#'), ' for the majority of the analysis.  Complementary data was obtained from the ', html.A('Internet Movie Database',href='https://www.imdb.com/title/tt0386676/episodes?season=1'), ' coupled with data from ', html.A('Wikipedia.',href='https://en.wikipedia.org/wiki/The_Office_(American_season_1)'), " Twitter data was scraped using functions under the ",html.A("TWINT",href='https://pypi.org/project/twint/')," library of Python.  The tweets were scraped for each episode from Season 6-9 from the airdate through the following date."]),
                    html.Br()
                ]),
                html.Div([
                    html.P(dcc.Markdown('''**What are the limitations of this data?**''')),
                ],style={'text-decoration': 'underline'}),
                html.Div([
                    html.P("1.) Twitter data was not extensively available until Season 6, so the analysis of Twitter data only focuses on Season 6 through 9.  Additionally, the functions under the TWINT library that performed the scraping of data were utilized by searching for any tweet that contained '#theoffice'.  Extensive cleaning of relevant tweets and discarding of irrelevant tweets was needed to obtain an useful dataset."),
                    html.P("2.) The analysis was constrained to only the core set of characters with a few exceptions and their conversations with each other.")

                ])
            ]
        ),
        dcc.Tab(label='How does their speech differ?',value='tab-2',style=tab_style, selected_style=tab_selected_style,
        children=[
                dbc.Row([
                    dbc.Col([
                        html.Label(dcc.Markdown('''**Choose season:**''')),                        
                    ],width=3),
                    dbc.Col([
                        html.Label(dcc.Markdown('''**Choose character #1:**'''))
                    ],width=3),
                    dbc.Col([
                        html.Label(dcc.Markdown('''**Choose character #2:**'''))
                    ],width=3),
                    dbc.Col([
                        html.Label(dcc.Markdown('''**Choose # of words:**''')),
                    ],width=3)
                ]),
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(
                            id='dropdown0',
                            style={'color':'black'},

                            options=[{'label': i, 'value': i} for i in season_choices],
                            value=season_choices[0]
                        )
                    ],width=3),
                    dbc.Col([
                        dcc.Dropdown(
                            id='dropdown1',
                            style={'color':'black'},
                            options=[{'label': i, 'value': i} for i in main_characters_choose],
                            value=main_characters_choose[0]
                        )
                    ],width=3),
                    dbc.Col([
                        dcc.Dropdown(
                            id='dropdown2',
                            style={'color':'black'},
                            options=[{'label': i, 'value': i} for i in main_characters_choose],
                            value=main_characters_choose[1]
                        )
                    ],width=3),
                    dbc.Col([
                        dcc.Slider(
                            id='num_words_slider',
                            min=1,max=3,step=1,value=1,
                            marks={
                                1: '1',
                                2: '2',
                                3: '3'
                            }
                        )
                    ],width=3)
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card(id="card1")
                    ],width=3),
                    dbc.Col([
                        dbc.Card(id="card2")
                    ],width=3),
                    dbc.Col([
                        dbc.Card(id="card3")
                    ],width=3),
                    dbc.Col([
                        dbc.Card(id="card4")
                    ],width=3)
                ],no_gutters=True),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='word_freq_graph1')
                    ],width=6),
                    dbc.Col([
                        dcc.Graph(id='word_freq_graph2')
                    ],width=6)
                ])


        ]),
        dcc.Tab(label='Who had a better day?',value='tab-3',style=tab_style, selected_style=tab_selected_style,
        children=[
            dbc.Row([
                dbc.Col([
                    html.Label(dcc.Markdown('''**Choose character #1:**'''))
                ],width=3),
                dbc.Col([
                    html.Label(dcc.Markdown('''**Choose character #2:**''')),
                ],width=3),
                dbc.Col([
                    html.Label(dcc.Markdown('''**Select callout option:**''')),
                ],width=3),
                dbc.Col([
                    html.Label(dcc.Markdown('''**Click below for episode info:**''')),
                ],width=3)
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                            id='dropdown3b',
                            style={'color':'black'},
                            options=[{'label': i, 'value': i} for i in all_main_chars_list],
                            value=all_main_chars_list[0]
                    )
                ],width=3),
                dbc.Col([
                    dcc.Dropdown(
                            id='dropdown3c',
                            style={'color':'black'},

                            options=[{'label': i, 'value': i} for i in all_main_chars_list],
                            value=all_main_chars_list[1]
                    )
                ],width=3),
                dbc.Col([
                     dbc.RadioItems(
                        id='radio1',
                        options=[
                            {'label': ' Show Callouts', 'value': ' Show Callouts'},
                            {'label': ' Hide Callouts', 'value': ' Hide Callouts'}
                        ],
                        value=' Show Callouts',
                        labelStyle={'display': 'inline-block','text-align': 'left'}
                    )
                ],width=3),
                dbc.Col([
                    dbc.Button("Episode Descriptions by Character",id='open0')
                ],width=3)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card(id='card5')
                ],width=3),
                dbc.Col([
                    dbc.Card(id='card6')
                ],width=3),
                dbc.Col([
                    dbc.Card(id='card7')
                ],width=3),
                dbc.Col([
                    dbc.Card(id='card8')
                ],width=3),
            ],no_gutters=True),   
            dbc.Row([     
                dbc.Col([
                    dcc.Graph(id='sentiment_line_graph'),
                ],width=12),
            ]),     
            html.Div([
                dbc.Modal(
                    children=[
                        dbc.ModalHeader("Episode Description"),
                        dbc.ModalBody(
                            children=[
                                html.P('Character 1 - Put Something here'),
                                html.P(
                                    id="table0",
                                    style={'overflow':'auto','maxHeight':'400px'}
                                ),
                                html.P('Character 2 - Put Something here'),
                                html.P(
                                    id="table0_again",
                                    style={'overflow':'auto','maxHeight':'400px'}
                                )
                            ]
                        ),
                        dbc.ModalFooter(
                            dbc.Button("Close", id="close0", className="ml-auto")
                        ),
                    ],id="modal0", size="lg"

                )
            ])           
        ]),
        dcc.Tab(label='How is everyone connected?',value='tab-4',style=tab_style, selected_style=tab_selected_style,
        children=[
            dbc.Row([
                dbc.Col([
                    html.Label(dcc.Markdown('''**Choose season:**'''))
                ],width=3),
                dbc.Col([
                    html.Label(dcc.Markdown('''**Choose episode:**'''))
                ],width=3),
                dbc.Col([
                    html.Label(dcc.Markdown('''**Choose character #1:**'''))
                ],width=3),
                dbc.Col([
                    html.Label(dcc.Markdown('''**Choose character #2:**'''))
                ],width=3),

            ]),
            dbc.Row([
                dbc.Col(
                    dcc.Dropdown(
                        id='dropdown4',
                        style={'color':'black'},
                        options=[{'label': i, 'value': i} for i in season_choices],
                        value=season_choices[0]
                    ), width=3
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id='dropdown7',
                        style={'color':'black'},
                        options=[{'label': i, 'value': i} for i in episode_choices],
                        value=episode_choices[0]
                    ), width=3
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id='dropdown5',
                        style={'color':'black'},
                        options=[{'label': i, 'value': i} for i in character_choices],
                        value=character_choices[0]
                    ), width=3
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id='dropdown6',
                        style={'color':'black'},
                        options=[{'label': i, 'value': i} for i in character_choices],
                        value=character_choices[1]
                    ), width=3
                )

            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card(id='card9')
                ],width=3),
                dbc.Col([
                    dbc.Card(id='card10')
                ],width=3),
                dbc.Col([
                    dbc.Card(id='card11')
                ],width=3),
                dbc.Col([
                    dbc.Card(id='card12')
                ],width=3),
            ],no_gutters=True),
            dbc.Row([
                dbc.Col([
                    html.Label(dcc.Markdown('''**Who was in their network?**'''),style={'text-align':'center'}),
                    visdcc.Network(
                        id='net',
                        options = dict(
                            height='600px', 
                            width='100%',
                            physics={'barnesHut': {'avoidOverlap': 0.5}},
                            maxVelocity=0,
                            stabilization={
                                'enabled': 'true',
                                'iterations': 15,
                                'updateInterval': 50,
                                'onlyDynamicEdges': 'false',
                                'fit': 'true'
                            },
                            scaling='value'
                        )
                    )
                ],width=6),
                dbc.Col([
                    html.Label(dcc.Markdown('''**Which topics connect these characters?**''')),
                    dbc.RadioItems(
                        id='radio2',
                        options=[
                            {'label': 'Show Most Frequently Discussed Topic', 'value': 'Show Most Frequently Discussed Topic'},
                            {'label': 'Show 2nd Most Frequently Discussed Topic', 'value': 'Show 2nd Most Frequently Discussed Topic'},
                            {'label': 'Show 3rd Most Frequently Discussed Topic','value': 'Show 3rd Most Frequently Discussed Topic'},

                        ],
                        value='Show Most Frequently Discussed Topic',
                        labelStyle={'display': 'inline-block','text-align': 'left'}
                    ),
                    dcc.Graph(id='episode_topic', figure={}, config={'displayModeBar': True}),

                ],width=6)
            ],no_gutters=True)
        ]),
        dcc.Tab(label='What does Twitter think?',value='tab-5',style=tab_style, selected_style=tab_selected_style,
        children=[
            dbc.Row([
                dbc.Col([
                    html.Label(dcc.Markdown('''**Choose season:**'''))
                ],width=6),
                dbc.Col([
                    html.Label(dcc.Markdown('''**Choose episode:**'''))
                ],width=6),
            ]),
            dbc.Row([
                dbc.Col(
                    dcc.Dropdown(
                        id='dropdown8',
                        style={'color':'black'},
                        options=[{'label': i, 'value': i} for i in tweet_seasons],
                        value=tweet_seasons[0]
                    ), width=6
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id='dropdown9',
                        style={'color':'black'},
                        options=[{'label': i, 'value': i} for i in tweet_episodes],
                        value=tweet_episodes[0]
                    ), width=6
                )
            ]),
            dbc.Row([
                dbc.Col(
                    dcc.Graph(id='twitter_sentiment_over_time'),
                    width=6
                ),
                dbc.Col(
                    dcc.Graph(id='word_cloud_chart', figure={}, config={'displayModeBar': False}),
                    width=6
                )
            ])
        ])
    ])
])



#Configure Reactibity for Tab Colors
@app.callback(Output('tabs-content-inline', 'children'),
              Input('tabs-styled-with-inline', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('Tab content 1')
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H3('Tab content 2')
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.H3('Tab content 2')
        ])
    elif tab == 'tab-4':
        return html.Div([
            html.H3('Tab content 3')
        ])
    elif tab == 'tab-5':
        return html.Div([
            html.H3('Tab content 4')
        ])

# @app.callback(
#     Output('first-dropdown', 'children'),
#     Input('dropdown0', 'value') #----- Select the season
# )
# def set_character_options(selected_season):
#     return dcc.Dropdown(
#         id='dropdown1',
#         options=[{'label': i, 'value': i} for i in season_character_dict[selected_season]],
#         value=season_character_dict[selected_season][0]
#         )


# @app.callback(
#     Output('second-dropdown','children'),
#     Input('dropdown1','value'),
# )
# def update_second_dropdown(value):
#     updated_choose_character = all_main_chars_list.copy()
#     updated_choose_character.remove(value)
#     return dcc.Dropdown(
#         id='dropdown2',
#         options=[{'label': i, 'value': i} for i in updated_choose_character]
#     )



@app.callback(
    Output('dropdown1', 'options'),#-----Filters the character options
    Output('dropdown1', 'value'),
    Input('dropdown0', 'value') #----- Select the season
)
def set_character_options(selected_season):
    return [{'label': i, 'value': i} for i in season_character_dict[selected_season]], season_character_dict[selected_season][0]

@app.callback(
    Output('dropdown2', 'options'), #-----Filters the character options
    Output('dropdown2', 'value'),
    Input('dropdown0', 'value') #----- Select the season
)
def set_character_options2(selected_season):
    return [{'label': i, 'value': i} for i in season_character_dict[selected_season]], season_character_dict[selected_season][1]



# @app.callback(
#     Output('dropdown2','options'),
#     Input('dropdown1','value')
# )
# def update_second_dropdown_again(value):
#     updated_choose_character = all_main_chars_list.copy()
#     updated_choose_character.remove(value)
#     return [{'label': i, 'value': i} for i in updated_choose_character]





@app.callback(
    Output('dropdown3c','options'),
    Input('dropdown3b','value')
)
def update_second_dropdown(value):
    updated_choose_character = all_main_chars_list.copy()
    updated_choose_character.remove(value)
    return [{'label': i, 'value': i} for i in updated_choose_character]



@app.callback(
    Output('dropdown7', 'options'), #--> filter episodes
    Output('dropdown7', 'value'),
    Input('dropdown4', 'value') #--> choose season
)
def set_episode_options(selected_season):
    return [{'label': i, 'value': i} for i in season_episode_dict[selected_season]], season_episode_dict[selected_season][0],





@app.callback(
    Output('dropdown5', 'options'),
    Output('dropdown5', 'value'),
    Input('dropdown4', 'value'),
    Input('dropdown7', 'value')
)
def set_character_options_network_graph(selected_season,selected_episode):
    return [{'label': i, 'value': i} for i in season_episode_character_dictionary[selected_season][selected_episode]], season_episode_character_dictionary[selected_season][selected_episode][0],


@app.callback(
    Output('dropdown6', 'options'),
    Output('dropdown6', 'value'),
    Input('dropdown4', 'value'),
    Input('dropdown7', 'value')
)
def set_character_options_network_graph(selected_season,selected_episode):
    return [{'label': i, 'value': i} for i in season_episode_character_dictionary[selected_season][selected_episode]], season_episode_character_dictionary[selected_season][selected_episode][1],




@app.callback(
    Output('dropdown9', 'options'), #--> filter episodes
    Output('dropdown9', 'value'),
    Input('dropdown8', 'value') #--> choose season
)
def set_episode_options(selected_season):
    return [{'label': i, 'value': i} for i in season_episode_dict[selected_season]], season_episode_dict[selected_season][0],



@app.callback(
    Output('word_freq_graph1','figure'),
    Output('word_freq_graph2','figure'),
    Input('dropdown0','value'),
    Input('dropdown1','value'),
    Input('dropdown2','value'),
    Input('num_words_slider','value')
)

def update_word_chart(season_select,character_select1, character_select2,slider_select):
    new_df1 = the_mains_df[(the_mains_df['season']==season_select) & (the_mains_df['speaker']==character_select1)]
    new_df2 = the_mains_df[(the_mains_df['season']==season_select) & (the_mains_df['speaker']==character_select2)]

    char1 = new_df1['speaker'].unique()[0]
    char2 = new_df2['speaker'].unique()[0]

    ch1_df = new_df1[new_df1['speaker']==char1]
    ch2_df = new_df2[new_df2['speaker']==char2]

    word_num = slider_select

    top_bigrams1 = get_top_ngram(ch1_df['cleaned_text'],word_num)[:10]
    top_bigrams2 = get_top_ngram(ch2_df['cleaned_text'],word_num)[:10]
    

    top_words1 = pd.DataFrame(top_bigrams1,columns=['word','count'])
    top_words1 = top_words1.head(10).sort_values('count',ascending=True)
    top_words1['speaker'] = char1

    top_words2 = pd.DataFrame(top_bigrams2,columns=['word','count'])
    top_words2 = top_words2.head(10).sort_values('count',ascending=True)
    top_words2['speaker'] = char2


    #Set up if then to see which of the 2 is larger
    xlimit1 = top_words1['count'].max()
    xlimit2 = top_words2['count'].max()

    if xlimit1 > xlimit2:
        bar_fig1 = px.bar(top_words1, x='count', y='word',orientation='h',title=f'{char1} Words',labels={'count':'Frequency'},template='plotly_dark')
        bar_fig2 = px.bar(top_words2, x='count', y='word',orientation='h',title=f'{char2} Words',labels={'count':'Frequency'},template='plotly_dark')

        bar_fig1.update_layout(
            coloraxis_showscale=False, 
            yaxis_title=None,
            xaxis_range=[0,xlimit1+1],
            margin=dict(l=20, r=20, t=45, b=20),
            title={
                'xanchor':'center',
                'x':0.5
            }
        )

        bar_fig2.update_layout(
            coloraxis_showscale=False, 
            yaxis_title=None,
            xaxis_range=[0,xlimit1+1],
            margin=dict(l=20, r=20, t=45, b=20),
            title={'xanchor':'center',
                   'x':0.5
            }
        )
        bar_fig1.update_traces(marker_color='#2E91E5')
        bar_fig2.update_traces(marker_color='#D7504D')

    else:
        bar_fig1 = px.bar(top_words1, x='count', y='word',orientation='h',title=f'{char1} Words',labels={'count':'Frequency'},template='plotly_dark')
        bar_fig2 = px.bar(top_words2, x='count', y='word',orientation='h',title=f'{char2} Words',labels={'count':'Frequency'},template='plotly_dark')

        bar_fig1.update_layout(
            coloraxis_showscale=False, 
            yaxis_title=None,
            xaxis_range=[0,xlimit2+1],
            margin=dict(l=20, r=20, t=45, b=20),
            title={
                'xanchor':'center',
                'x':0.5
            }
        )

        bar_fig2.update_layout(
            coloraxis_showscale=False, 
            yaxis_title=None,
            xaxis_range=[0,xlimit2+1],
            margin=dict(l=20, r=20, t=45, b=20),
            title={
                'xanchor':'center',
                'x':0.5
            }
        )
        bar_fig1.update_traces(marker_color='#2E91E5')
        bar_fig2.update_traces(marker_color='#D7504D')


    return bar_fig1, bar_fig2


@app.callback(
    Output('card1', 'children'),
    Output('card2', 'children'),
    Output('card3', 'children'),
    Output('card4', 'children'),
    Input('dropdown0', 'value'),
    Input('dropdown1', 'value'),
    Input('dropdown2', 'value')

)

def speech_stats(season_select,character_select1,character_select2):
    new_df1 = the_mains_df[(the_mains_df['season']==season_select) & (the_mains_df['speaker']==character_select1)]
    new_df2 = the_mains_df[(the_mains_df['season']==season_select) & (the_mains_df['speaker']==character_select2)]

    char1 = new_df1['speaker'].unique()[0]
    char2 = new_df2['speaker'].unique()[0]

    #Count total # of words
    top_words1 = pd.DataFrame(
            new_df1['cleaned_text'].str.split(expand=True).stack().value_counts()
        ,columns=['count']
    ).reset_index()

    top_words2 = pd.DataFrame(
            new_df2['cleaned_text'].str.split(expand=True).stack().value_counts()
        ,columns=['count']
    ).reset_index()

    TOTAL_WORDS1 = top_words1['count'].sum()
    TOTAL_WORDS2 = top_words2['count'].sum()
    larger_words1 = TOTAL_WORDS1-TOTAL_WORDS2
    larger_words2 = TOTAL_WORDS2-TOTAL_WORDS1

    #Count total # of lines
    TOTAL_LINES1 = new_df1.shape[0]
    TOTAL_LINES2 = new_df2.shape[0]
    larger_lines1 = TOTAL_LINES1-TOTAL_LINES2
    larger_lines2 = TOTAL_LINES2-TOTAL_LINES1

    #Count total # of scenes
    TOTAL_SCENES1 = len(new_df1['scene'].unique())
    TOTAL_SCENES2 = len(new_df2['scene'].unique())
    larger_scenes1 = TOTAL_SCENES1-TOTAL_SCENES2
    larger_scenes2 = TOTAL_SCENES2-TOTAL_SCENES1


    stacked = pd.concat([top_words1,top_words2])

    no_dups1 = stacked['index'].drop_duplicates(keep='first')
    no_dups2 = stacked['index'].drop_duplicates(keep=False)

    num = len(no_dups1) - len(no_dups2)
    denom = len(no_dups1)

    words_shared_perc = round((num/denom)*100,1)


   #income_formatted = f"${filtered['Per Capita Income'].median():,.2f}"


    if TOTAL_WORDS1 > TOTAL_WORDS2:
        card1 = dbc.Card([
            dbc.CardBody([
                html.H4(f'{char1}: {larger_words1:,.0f} more words'),
                html.P(f'than {char2} during {season_select}')
            ])


        ],
        style={
            'width': '100%',
            'text-align': 'center',
            'background-color': '#2E91E5',
            'color':'white',
            'fontWeight': 'bold',
            'fontSize':12},
        outline=True)
    elif TOTAL_WORDS1 < TOTAL_WORDS2:
        card1 = dbc.Card([
            dbc.CardBody([
                html.H4(f'{char2}: {larger_words2:,.0f} more words'),
                html.P(f'than {char1} during {season_select}')
            ])
        ],
        style={
            'width': '100%',
            'text-align': 'center',
            'background-color': '#D7504D',
            'color':'white',
            'fontWeight': 'bold',
            'fontSize':12},
        outline=True)
    else:
        card1 = dbc.Card([
            dbc.CardBody([
                html.H4('Same # of words spoken'),
                html.P(f'during {season_select}')

            ])
        ],
        style={
            'width': '100%',
            'text-align': 'center',
            'background-color': 'grey',
            'color':'white',
            'fontWeight': 'bold',
            'fontSize':12},
        outline=True)

    

    if TOTAL_LINES1 > TOTAL_LINES2:
        card2 = dbc.Card([
            dbc.CardBody([
                html.H4(f'{char1}: {larger_lines1:,.0f} more lines'),
                html.P(f'than {char2} during {season_select}')

            ])

        ],
        style={
            'width': '100%',
            'text-align': 'center',
            'background-color': '#2E91E5',
            'color':'white',
            'fontWeight': 'bold',
            'fontSize':12},
        outline=True)
    elif TOTAL_LINES1 < TOTAL_LINES2:
        card2 = dbc.Card([
            dbc.CardBody([
                html.H4(f'{char2}: {larger_lines2:,.0f} more lines'),
                html.P(f'than {char1} during {season_select}')
            ])
       
        ],
        style={
            'width': '100%',
            'text-align': 'center',
            'background-color': '#D7504D',
            'color':'white',
            'fontWeight': 'bold',
            'fontSize':12},
        outline=True)
    else:
        card2 = dbc.Card([
            dbc.CardBody([
                html.H4('Same # of lines spoken'),
                html.P(f'during {season_select}')

            ])
        ],
        style={
            'width': '100%',
            'text-align': 'center',
            'background-color': 'grey',
            'color':'white',
            'fontWeight': 'bold',
            'fontSize':12},
        outline=True)

    if TOTAL_SCENES1 > TOTAL_SCENES2:

        card3 = dbc.Card([
            dbc.CardBody([
                html.H4(f'{char1}: {larger_scenes1:,.0f} more scenes'),
                html.P(f'than {char2} during {season_select}')

            ])
        ],
        style={
            'width': '100%',
            'text-align': 'center',
            'background-color': '#2E91E5',
            'color':'white',
            'fontWeight': 'bold',
            'fontSize':12},
        outline=True)
    elif TOTAL_SCENES1 < TOTAL_SCENES2:
        card3 = dbc.Card([
            dbc.CardBody([
                html.H4(f'{char2}: {larger_scenes2:,.0f} more scenes'),
                html.P(f'than {char1} during {season_select}')
            ])
        ],
        style={
            'width': '100%',
            'text-align': 'center',
            'background-color': '#D7504D',
            'color':'white',
            'fontWeight': 'bold',
            'fontSize':12},
        outline=True)
    else:
        card3 = dbc.Card([
            dbc.CardBody([
                html.H4('Same # of scenes'),
                html.P(f'during {season_select}')
            ])
        ],
        style={
            'width': '100%',
            'text-align': 'center',
            'background-color': 'grey',
            'color':'white',
            'fontWeight': 'bold',
            'fontSize':12},
        outline=True)


    card4 = dbc.Card([
        dbc.CardBody([
                html.H4(f'{words_shared_perc}% words shared'),
                html.P(f'between {char1} and {char2} during {season_select}')

            ])
        ],
        style={
            'width': '100%',
            'text-align': 'center',
            'background-color': 'grey',
            'color':'white',
            'fontWeight': 'bold',
            'fontSize':12},
        outline=True)


    return card1, card2, card3, card4


@app.callback(
    Output('sentiment_line_graph','figure'),
    Output('card5','children'),
    Output('card6','children'),
    Output('card7','children'),
    Output('card8','children'),
    Output('table0','children'),
    Output('table0_again','children'),

    Input('dropdown3b','value'),
    Input('dropdown3c','value'),
    Input('radio1','value')

)

def sentiment(character_select1, character_select2,radio_select):
    new_df1 = the_mains_df[the_mains_df['speaker']==character_select1]
    new_df2 = the_mains_df[the_mains_df['speaker']==character_select2]


    sentiment_df1=new_df1.groupby(['season','episode']).agg({'compound':'mean'}).reset_index()
    sentiment_df1['label'] = sentiment_df1['season'] + ", " + sentiment_df1['episode']
    sentiment_df1['season_num'] = sentiment_df1['season'].str.slice(7, 9).astype(int)
    sentiment_df1['episode_num'] = sentiment_df1['episode'].str.slice(8).astype(int)
    sentiment_df1 = sentiment_df1.sort_values(["season_num", "episode_num"], ascending = (True, True))
    sentiment_df1 = sentiment_df1.rename(columns={'compound':f'{character_select1} Sentiment'})

    filter_pos1 = sentiment_df1[sentiment_df1[f'{character_select1} Sentiment']>=0]
    num_pos_days1 = filter_pos1.shape[0]
    num_days_total1 = sentiment_df1.shape[0]
    perc_pos_days1 = round((num_pos_days1/num_days_total1)*100,)

    filter_extreme1 = sentiment_df1[(sentiment_df1[f'{character_select1} Sentiment']>=0.5)|(sentiment_df1[f'{character_select1} Sentiment']<=-0.5)]
    numerator1 = filter_extreme1.shape[0]
    denominator1 = sentiment_df1.shape[0]
    perc_extreme_days1 = round((numerator1/denominator1)*100,1)

    sentiment_df2=new_df2.groupby(['season','episode']).agg({'compound':'mean'}).reset_index()
    sentiment_df2['label'] = sentiment_df2['season'] + ", " + sentiment_df2['episode']
    sentiment_df2['season_num'] = sentiment_df2['season'].str.slice(7, 9).astype(int)
    sentiment_df2['episode_num'] = sentiment_df2['episode'].str.slice(8).astype(int)
    sentiment_df2 = sentiment_df2.sort_values(["season_num", "episode_num"], ascending = (True, True))
    sentiment_df2 = sentiment_df2.rename(columns={'compound':f'{character_select2} Sentiment'})

    filter_pos2 = sentiment_df2[sentiment_df2[f'{character_select2} Sentiment']>=0]
    num_pos_days2 = filter_pos2.shape[0]
    num_days_total2 = sentiment_df2.shape[0]
    perc_pos_days2 = round((num_pos_days2/num_days_total2)*100,1)


    filter_extreme2 = sentiment_df2[(sentiment_df2[f'{character_select2} Sentiment']>=0.5)|(sentiment_df2[f'{character_select2} Sentiment']<=-0.5)]
    numerator2 = filter_extreme2.shape[0]
    denominator2 = sentiment_df2.shape[0]
    perc_extreme_days2 = round((numerator2/denominator2)*100,1)


    #Join datasets - full outer
    sentiment_df = pd.merge(sentiment_df1,sentiment_df2,how='outer',on=['season','episode','season_num','episode_num','label'])
    sentiment_df = sentiment_df.sort_values(["season_num", "episode_num"], ascending = (True, True))

    char1_min = sentiment_df[f'{character_select1} Sentiment'].min()
    char1_max = sentiment_df[f'{character_select1} Sentiment'].max()
    char1_se_ep_min = sentiment_df[sentiment_df[f'{character_select1} Sentiment']==char1_min]['label'].values[0]
    char1_se_ep_max = sentiment_df[sentiment_df[f'{character_select1} Sentiment']==char1_max]['label'].values[0]

    char2_min = sentiment_df[f'{character_select2} Sentiment'].min()
    char2_max = sentiment_df[f'{character_select2} Sentiment'].max()
    char2_se_ep_min = sentiment_df[sentiment_df[f'{character_select2} Sentiment']==char2_min]['label'].values[0]
    char2_se_ep_max = sentiment_df[sentiment_df[f'{character_select2} Sentiment']==char2_max]['label'].values[0]



    fig = px.line(
        sentiment_df, x="label", y=[f"{character_select1} Sentiment",f"{character_select2} Sentiment"], title='Average Sentiment Over Time',
        labels={
            "label": "Episodes"
        }, markers=True,template='plotly_dark'
    )
    fig.update_xaxes(showticklabels=False,showspikes=True)
    fig.add_hline(y=0,line_width=3, line_dash="dash", line_color="white")

    if " Show Callouts" in radio_select:
        fig.add_annotation(text=f"{character_select1} (Max)", x=char1_se_ep_max, y=char1_max, arrowhead=1, showarrow=True)
        fig.add_annotation(text=f"{character_select1} (Min)", x=char1_se_ep_min, y=char1_min, arrowhead=1, showarrow=True, ay=45)
        fig.add_annotation(text=f"{character_select2} (Max)", x=char2_se_ep_max, y=char2_max, arrowhead=1, showarrow=True)
        fig.add_annotation(text=f"{character_select2} (Min)", x=char2_se_ep_min, y=char2_min, arrowhead=1, showarrow=True, ay=45)

    fig.update_yaxes(
        range = [-1,1],
        title='Sentiment',
        showspikes=True
    )
    fig.update_traces(textposition='top center')

    fig.update_layout(
        title={
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        legend=dict(
            {'title_text':''},
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    card5 = dbc.Card([
            dbc.CardBody([
                html.H6(f'{character_select1}: {perc_pos_days1}% positive days'),
                html.P('Sentiment Score > 0')
            ])
        ],
        style={
            'width': '100%',
            'text-align': 'center',
            'background-color': '#2E91E5',
            'color':'white',
            'fontWeight': 'bold',
            'fontSize':12},
        outline=True)
    card6 = dbc.Card([
            dbc.CardBody([
                html.H6(f'{character_select1}: {perc_extreme_days1}% extreme days'),
                html.P('Sentiment Score >= 0.5 or <= -0.5')

            ])
        ],
        style={
            'width': '100%',
            'text-align': 'center',
            'background-color': '#2E91E5',
            'color':'white',
            'fontWeight': 'bold',
            'fontSize':12},
        outline=True)

    card7 = dbc.Card([
            dbc.CardBody([
                html.H6(f'{character_select2}: {perc_pos_days2}% positive days'),
                html.P('Sentiment Score > 0')

            ])
        ],
        style={
            'width': '100%',
            'text-align': 'center',
            'background-color': '#D7504D',
            'color':'white',
            'fontWeight': 'bold',
            'fontSize':12},
        outline=True)

    card8 = dbc.Card([
            dbc.CardBody([
                html.H6(f'{character_select2}: {perc_extreme_days2}% extreme days'),
                html.P('Sentiment Score >= 0.5 or <= -0.5')

            ])
        ],
        style={
            'width': '100%',
            'text-align': 'center',
            'background-color': '#D7504D',
            'color':'white',
            'fontWeight': 'bold',
            'fontSize':12},
        outline=True)
    #fig.update_traces(line_color='#2E91E5')
    #fig.update_traces(marker_color='#D7504D')


    s_df1=new_df1.groupby(['season','episode']).agg({
        'compound':'mean',
        'desc':'first',
        'title':'first'
        }).reset_index()
    s_df1['label'] = s_df1['season']+', '+s_df1['episode']

    most_pos1 = s_df1[s_df1['compound']==s_df1['compound'].max()]
    most_neg1 = s_df1[s_df1['compound']==s_df1['compound'].min()]


    most_pos1 = most_pos1.rename(columns={'label': 'Episode', 'title': 'Title','desc':'Description'})
    most_neg1 = most_neg1.rename(columns={'label': 'Episode', 'title': 'Title','desc':'Description'})

    most_pos1 = most_pos1[['Episode','Title','Description']]
    most_neg1 = most_neg1[['Episode','Title','Description']]

    most_pos1['Character'] = character_select1
    most_pos1['Sentiment'] = "Most Positive"

    most_neg1['Character'] = character_select1
    most_neg1['Sentiment'] = "Most Negative"


    s_df2=new_df2.groupby(['season','episode']).agg({
        'compound':'mean',
        'desc':'first',
        'title':'first'
        }).reset_index()
    s_df2['label'] = s_df2['season']+', '+s_df2['episode']

    most_pos2 = s_df2[s_df2['compound']==s_df2['compound'].max()]
    most_neg2 = s_df2[s_df2['compound']==s_df2['compound'].min()]


    most_pos2 = most_pos2.rename(columns={'label': 'Episode', 'title': 'Title','desc':'Description'})
    most_neg2 = most_neg2.rename(columns={'label': 'Episode', 'title': 'Title','desc':'Description'})

    most_pos2 = most_pos2[['Episode','Title','Description']]#.style.hide_index()
    most_neg2 = most_neg2[['Episode','Title','Description']]#.style.hide_index()


    most_pos2['Character'] = character_select2
    most_pos2['Sentiment'] = "Most Positive"

    most_neg2['Character'] = character_select2
    most_neg2['Sentiment'] = "Most Negative"

    extreme_episodes1 = pd.concat([most_pos1, most_neg1])
    extreme_episodes2 = pd.concat([most_pos2, most_neg2])



    episode_descs_table1 = dt.DataTable(
        columns=[{"name": i, "id": i} for i in extreme_episodes1.columns],
        data=extreme_episodes1.to_dict('records'),
        style_data={
            'whiteSpace': 'normal',
            'height': '150px',
            'color':'black',
            'backgroundColor': 'white'
        },
        style_cell={'textAlign': 'left'}
    )


    episode_descs_table2 = dt.DataTable(
        columns=[{"name": i, "id": i} for i in extreme_episodes2.columns],
        data=extreme_episodes2.to_dict('records'),
        style_data={
            'whiteSpace': 'normal',
            'height': '150px',
            'color':'black',
            'backgroundColor': 'white'
        },
        style_cell={'textAlign': 'left'}
    )
    


    return fig, card5, card6, card7, card8, episode_descs_table1, episode_descs_table2


@app.callback(
    Output('net','data'),
    Output('episode_topic','figure'),
    Output('card9','children'),
    Output('card10','children'),
    Output('card11','children'),
    Output('card12','children'),
    Input('dropdown4','value'),
    Input('dropdown7','value'),
    Input('dropdown5','value'),
    Input('dropdown6','value'),
    Input('radio2','value')
)

def network(season_select, episode_select, character_select1, character_select2, radio_select):
    
    
    filtered = data_for_ng[['season','episode','scene','speaker']]
    filtered = filtered[(filtered['season']==season_select) & (filtered['episode']==episode_select)]

    #Person with most scenes stat
    #test = filtered[(filtered['season']=="Season 5") & (filtered['episode']=="Episode 3")]
    test = filtered[['scene','speaker']]
    test['ui'] = test['scene'] + '_'+ test['speaker']
    remove_rows = pd.DataFrame(test['ui'].drop_duplicates(keep='first'))
    remove_rows['speaker'] = remove_rows['ui'].str.split('_').str[1]

    test_again = pd.DataFrame(remove_rows.groupby(['speaker']).size().reset_index(name = 'num_scenes').sort_values(by='num_scenes',ascending=False))
    test_again = test_again.reset_index()
    person_most_scenes = test_again['speaker'][0]
    person_num_scenes = test_again['num_scenes'][0]


    #Most scenes with character1
    remove_rows['scene'] = remove_rows['ui'].str.split('_').str[0]

    only_scenes_with_char1 = remove_rows[remove_rows['speaker']==character_select1]
    the_scenes1 = only_scenes_with_char1['scene'].to_list()
    peeps_with_char1 = remove_rows[remove_rows['scene'].isin(the_scenes1)]
    char1_paired = pd.DataFrame(peeps_with_char1.groupby(['speaker']).size().reset_index(name = 'num_scenes').sort_values(by='num_scenes',ascending=False))
    char1_paired = char1_paired.reset_index()
    paired1_most_scenes = char1_paired['speaker'][1]
    paried1_num_scenes = char1_paired['num_scenes'][1]

    #Most scenes with character 2
    only_scenes_with_char2 = remove_rows[remove_rows['speaker']==character_select2]
    the_scenes2 = only_scenes_with_char2['scene'].to_list()
    peeps_with_char2 = remove_rows[remove_rows['scene'].isin(the_scenes2)]
    char2_paired = pd.DataFrame(peeps_with_char2.groupby(['speaker']).size().reset_index(name = 'num_scenes').sort_values(by='num_scenes',ascending=False))
    char2_paired = char2_paired.reset_index()
    paired2_most_scenes = char2_paired['speaker'][1]
    paried2_num_scenes = char2_paired['num_scenes'][1]


    #Topic Modelling - this will be combined into a radio button selection
    df1 = the_mains_df[the_mains_df['season']==season_select]
    df2 = df1[df1['episode']==episode_select]
    df3 = df2[(df2['speaker']==character_select1)|(df2['speaker']==character_select2)]
    vectorizer = TfidfVectorizer(max_df=0.95,min_df=2,stop_words='english')
    dtm = vectorizer.fit_transform(df3['cleaned_text'].values.astype('U'))


    nmf_mod = NMF(n_components=10,random_state=42)
    nmf_mod.fit(dtm)

    office_topics = nmf_mod.transform(dtm)

    office_topic_label = {
        0:"Topic 1",
        1:"Topic 2",       
        2:"Topic 3",
        3:"Topic 4",
        4:"Topic 5",
        5:"Topic 6",
        6:"Topic 7",
        7:"Topic 8",
        8:"Topic 9",
        9:"Topic 10",

    }
    

    df3['topic_num'] = office_topics.argmax(axis=1)
    df3['topic_label'] = df3['topic_num'].map(office_topic_label)

    topic_count = df3.groupby(['topic_label']).size().reset_index(name='counts')
    topic_count = topic_count.sort_values(by='counts',ascending=False).reset_index()
    top_topics = topic_count['topic_label'][0:3].to_list()


    wc_df1 = df3[df3['topic_label']==top_topics[0]]
    wc_df2 = df3[df3['topic_label']==top_topics[1]]
    wc_df3 = df3[df3['topic_label']==top_topics[2]]

    if "Show Most Frequently Discussed Topic" in radio_select:

        df1= wc_df1.copy()
        df1 = df1.cleaned_text

        my_wordcloud1 = WordCloud(
            background_color='black',
            min_word_length = 4
        ).generate(' '.join(df1))

        fig_wordcloud = px.imshow(
            my_wordcloud1, 
            template='plotly_dark'
        )
        fig_wordcloud.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        fig_wordcloud.update_xaxes(visible=False)
        fig_wordcloud.update_yaxes(visible=False)

    if "Show 2nd Most Frequently Discussed Topic" in radio_select:

        df2= wc_df2.copy()
        df2 = df2.cleaned_text

        my_wordcloud2 = WordCloud(
            background_color='black',
            min_word_length = 4
        ).generate(' '.join(df2))

        fig_wordcloud = px.imshow(
            my_wordcloud2,
            template='plotly_dark'
        )
        fig_wordcloud.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        fig_wordcloud.update_xaxes(visible=False)
        fig_wordcloud.update_yaxes(visible=False)

    if "Show 3rd Most Frequently Discussed Topic" in radio_select:

        df3= wc_df3.copy()
        df3 = df3.cleaned_text

        my_wordcloud3 = WordCloud(
            background_color='black',
            min_word_length = 4
        ).generate(' '.join(df3))

        fig_wordcloud = px.imshow(
            my_wordcloud3,
            template='plotly_dark'
        )
        fig_wordcloud.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        fig_wordcloud.update_xaxes(visible=False)
        fig_wordcloud.update_yaxes(visible=False)


    def assets_pairs(speakers):
        unique_speakers = set(speakers)
        if len(unique_speakers) == 1:
            x = speakers.iat[0]  # get the only unique asset
            pairs = [[x, x]]
        else:
            pairs = it.permutations(unique_speakers, r=2)  # get all the unique pairs without repeated elements
        return pd.DataFrame(pairs, columns=['Source', 'Target']) 
   
    df_pairs = (
        filtered.groupby(['season', 'episode', 'scene'])['speaker']
        .apply(assets_pairs)   # create asset pairs per group 
        .groupby(['Source', 'Target'], as_index=False)  # compute the weights  by 
        .agg(Weights = ('Source', 'size'))              # counting the unique ('Source', 'Target') pairs
    )



    new_df = df_pairs[(df_pairs['Source']==character_select1)|(df_pairs['Source']==character_select2)]

#look at this for clues about how to get the color in here
    node_list = list(
        set(new_df['Source'].unique().tolist()+new_df['Target'].unique().tolist())
    )

    nodes = [{
        'id': node_name, 
        'label': node_name,
        'font':'12px arial white',
        # 'group':new_df['Source'],
        # 'color':new_df['Source'],
        'shape':'dot',
        'size':df_pairs['Weights'].max()
        }
        for i, node_name in enumerate(node_list)]

    #Create edges from df
    edges=[]
    for row in new_df.to_dict(orient='records'):
        source, target = row['Source'], row['Target']
        edges.append({
            'id':source + "__" + target,
            'from': source,
            'to': target,
            'width': 2
        })

    data = {'nodes':nodes, 'edges': edges}


    card9 = dbc.Card([
            dbc.CardBody([
                html.H4(f'{person_most_scenes} ({person_num_scenes})'),
                html.P('Most scenes in this episode.')
            ])
        ],
        style={
            'width': '100%',
            'text-align': 'center',
            'background-color': 'grey',
            'color':'white',
            'fontWeight': 'bold',
            'fontSize':12},
        outline=True)
    card10 = dbc.Card([
            dbc.CardBody([
                html.H4('X person'),
                html.P('Some metric for this episode.')

            ])
        ],
        style={
            'width': '100%',
            'text-align': 'center',
            'background-color': 'grey',
            'color':'white',
            'fontWeight': 'bold',
            'fontSize':12},
        outline=True)

    card11 = dbc.Card([
            dbc.CardBody([
                html.H4(f'{paired1_most_scenes} ({paried1_num_scenes})'),
                html.P(f'Most scenes with {character_select1}')

            ])
        ],
        style={
            'width': '100%',
            'text-align': 'center',
            'background-color': '#2391E5',
            'color':'white',
            'fontWeight': 'bold',
            'fontSize':12},
        outline=True)

    card12 = dbc.Card([
            dbc.CardBody([
                html.H4(f'{paired2_most_scenes} ({paried2_num_scenes})'),
                html.P(f'Most scenes with {character_select2}')

            ])
        ],
        style={
            'width': '100%',
            'text-align': 'center',
            'background-color': '#D7504D',
            'color':'white',
            'fontWeight': 'bold',
            'fontSize':12},
        outline=True)





    return data, fig_wordcloud, card9, card10, card11, card12

#Twitter Sentiment Over Time
@app.callback(
    Output('twitter_sentiment_over_time','figure'),
    Output('word_cloud_chart','figure'),
    Input('dropdown8','value'),
    Input('dropdown9','value')
)

def twitter_sentiment(season_select,episode_select):
    new_df = tweet_data[(tweet_data['season']==season_select)]

    wc_df = tweet_data[(tweet_data['season']==season_select) & (tweet_data['episode']==episode_select)]
    # ep_num_line = wc_df['episode'][0]
    # vert_line_num = ep_num_line.split()[1]
    # vert_line_num = int(vert_line_num)

    sid = SentimentIntensityAnalyzer()
    new_df['scores'] = new_df['tweet'].apply(lambda tweet: sid.polarity_scores(tweet))
    new_df['compound']  = new_df['scores'].apply(lambda score_dict: score_dict['compound'])
    new_df['comp_score'] = new_df['compound'].apply(lambda c: 'pos' if c >=0 else 'neg')

    avg_sent_tweets = new_df.groupby('episode')[['compound','scaled_rating']].mean()
    avg_sent_tweets = pd.DataFrame(avg_sent_tweets).reset_index()
    avg_sent_tweets = avg_sent_tweets.set_axis(['Episode', 'Twitter Sentiment','Scaled IMDB Rating'], axis=1, inplace=False)
    avg_sent_tweets['ep_num'] = avg_sent_tweets['Episode'].str.slice(start=8)
    avg_sent_tweets['ep_num'] = avg_sent_tweets['ep_num'].astype(np.int64)

    avg_sent_tweets = avg_sent_tweets.sort_values(by='ep_num',ascending=True)

    fig = px.line(
        avg_sent_tweets, x="ep_num", y=["Twitter Sentiment","Scaled IMDB Rating"], title='Average Sentiment by Season',
        labels={
            "ep_num": "Episode",
            #"compound": "Sentiment"
        },template='plotly_dark'
    )
    fig.update_layout(
        yaxis_title=None,
        legend=dict(
            {'title_text':''},
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    #Add vertical line to graph to move with episode
    #fig.add_vline(x=vert_line_num,line_width=3, line_dash="dash", line_color="black")



    
    dff = wc_df.copy()
    dff = dff.cleaned_tweet

    
    my_wordcloud = WordCloud(
        background_color='black',
        height=275,
        min_word_length = 4,

    ).generate(' '.join(dff))

    fig_wordcloud = px.imshow(my_wordcloud,template='plotly_dark',
                              title="Word Cloud by Season and Episode")
    fig_wordcloud.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    fig_wordcloud.update_xaxes(visible=False)
    fig_wordcloud.update_yaxes(visible=False)


    return fig, fig_wordcloud




@app.callback(
    Output("modal1", "is_open"),
    [Input("open1", "n_clicks"), 
    Input("close1", "n_clicks")],
    [State("modal1", "is_open")],
)

def toggle_modal1(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output("modal2", "is_open"),
    [Input("open2", "n_clicks"), 
    Input("close2", "n_clicks")],
    [State("modal2", "is_open")],
)

def toggle_modal2(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output("modal3", "is_open"),
    [Input("open3", "n_clicks"), 
    Input("close3", "n_clicks")],
    [State("modal3", "is_open")],
)

def toggle_modal3(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    Output("modal4", "is_open"),
    [Input("open4", "n_clicks"), 
    Input("close4", "n_clicks")],
    [State("modal4", "is_open")],
)

def toggle_modal4(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    Output("modal0", "is_open"),
    [Input("open0", "n_clicks"), 
    Input("close0", "n_clicks")],
    [State("modal0", "is_open")],
)

def toggle_modal0(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


app.run_server(host='0.0.0.0',port='8055')
# if __name__=='__main__':
# 	app.run_server()