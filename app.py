#from dash_html_components.Col import Col
from dash_table.DataTable import DataTable
from nltk import data
from numpy.core.numeric import full
import pandas as pd
import numpy as np
import dash
import os
import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('ggplot')
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.express as px
#import plotly.graph_objects as go
#from flask import Flask, Response
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
#from collections import Counter
import re
from sklearn.preprocessing import MinMaxScaler

import contractions
import dash_table as dt
#!pip install visdcc
import visdcc
import networkx as nx
import itertools as it
from nltk.stem import WordNetLemmatizer 
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image

#Load data
#/Users/jonathan.zimmerman/Desktop/Office NLP/the-office-lines.xlsx
office_data = pd.read_excel("/Users/jonathan.zimmerman/Desktop/Office NLP/the-office-lines.xlsx")
imdb_data = pd.read_csv('/Users/jonathan.zimmerman/Desktop/Office NLP/office_episodes.csv')
wiki_desc = pd.read_csv('/Users/jonathan.zimmerman/Desktop/Office NLP/wiki_desc.csv')

#Import Twitter data
import pandas as pd
s6_tweets = pd.read_csv('https://raw.githubusercontent.com/statzenthusiast921/Personal-Projects/main/The%20Office%20NLP/season6_tweets.csv')
s7_tweets = pd.read_csv('https://raw.githubusercontent.com/statzenthusiast921/Personal-Projects/main/The%20Office%20NLP/season7_tweets.csv')
s8_tweets = pd.read_csv('https://raw.githubusercontent.com/statzenthusiast921/Personal-Projects/main/The%20Office%20NLP/season8_tweets.csv',lineterminator='\n')
s9_tweets = pd.read_csv('https://raw.githubusercontent.com/statzenthusiast921/Personal-Projects/main/The%20Office%20NLP/season9_tweets.csv')


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
#3.) Lemmatize words
office_data['cleaned_text'] = office_data['cleaned_text'].apply(func = lemmatize)
#4.) Remove stop words
office_data['cleaned_text'] = office_data['cleaned_text'].apply(func = removeStopWords)
#5.) Remove punctuation
office_data['cleaned_text'] = office_data['cleaned_text'].apply(func = punct)
#6.) Remove special characters
office_data['cleaned_text'] = office_data['cleaned_text'].apply(func = remove_special_characters)
#7.) Remove digits
office_data['cleaned_text'] = office_data['cleaned_text'].apply(func = remove_digits)
#8.) Remove stop words again created from previous functions
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


# main_metrics2 = pd.merge(size2,sums2,how='left',on=['season','episode','scene'])
# main_metrics2.rename(columns={0:'count'}, inplace=True )

# office_data = pd.merge(office_data,main_metrics,how='left',on=['season','episode','scene'])
# office_data['diff'] = office_data['count'] - office_data['main_ind_y']




# just2 = just2[just2['diff']==0]
# just2.head()
#2.) you have the score
#3.) filter down to scenes with only the 2 selected
#4.) average for character
#5.) need something to identify other character was in scene with first character
#6.) go back to dfpair and see if you can stick (5) in there with the compound score














data_for_ng = office_data[office_data['diff']==0]






#3.) Make source, target, weight columns (weight - # of convos)
filtered = data_for_ng[['season','episode','scene','speaker']]



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

main_network = df_pairs['Source'].sort_values().unique()

df_pairs = df_pairs[(df_pairs['Source']=="Angela")|(df_pairs['Source']=="Dwight")]
# G = nx.from_pandas_edgelist(
#     df_pairs,
#     source='Source',
#     target='Target',
#     edge_attr='Weights'
# )

# nx.draw_networkx(G)


node_list = list(
    set(df_pairs['Source'].unique().tolist()+ \
        df_pairs['Target'].unique().tolist())
)

nodes = [{
        'id': node_name, 
        'label': node_name, 
        'shape':'dot',
        'size':7
        }
        for i, node_name in enumerate(node_list)]

#Create edges from df
edges=[]
for row in df_pairs.to_dict(orient='records'):
    source, target = row['Source'], row['Target']
    edges.append({
        'id':source + "__" + target,
        'from': source,
        'to': target,
        'width':20
    })



#Create a season-character dictionary
season_character_dict = {'Season 1': ['Angela', 'Darryl', 'Dwight', 'Jan', 'Jim','Kelly','Kevin','Meredith','Michael','Oscar','Pam','Phyllis','Roy','Ryan','Stanley','Toby','Todd Packer'],
                         'Season 2': ['Angela','Creed', 'Darryl', 'David Wallace', 'Dwight', 'Jan', 'Jim','Kelly','Kevin','Meredith','Michael','Oscar','Pam','Phyllis','Roy','Ryan','Stanley','Toby','Todd Packer'],
                         'Season 3': ['Andy', 'Angela','Creed', 'Darryl', 'David Wallace', 'Dwight', 'Jan', 'Jim','Karen','Kelly','Kevin','Meredith','Michael','Oscar','Pam','Phyllis','Roy','Ryan','Stanley','Toby','Todd Packer'],
                         'Season 4': ['Andy', 'Angela','Creed', 'Darryl', 'David Wallace', 'Dwight','Holly', 'Jan', 'Jim','Kelly','Kevin','Meredith','Michael','Oscar','Pam','Phyllis','Roy','Ryan','Stanley','Toby'],
                         'Season 5': ['Andy', 'Angela','Creed', 'Darryl', 'David Wallace', 'Dwight','Erin','Holly', 'Jan', 'Jim','Karen','Kelly','Kevin','Meredith','Michael','Oscar','Pam','Phyllis','Roy','Ryan','Stanley','Toby'],
                         'Season 6': ['Andy', 'Angela','Creed', 'Darryl', 'David Wallace', 'Dwight','Erin','Gabe','Holly','Jan', 'Jim','Kelly','Kevin','Meredith','Michael','Oscar','Pam','Phyllis','Ryan','Stanley','Toby','Todd Packer'],
                         'Season 7': ['Andy', 'Angela','Creed', 'Darryl', 'David Wallace', 'Dwight','Erin','Gabe','Holly','Jan', 'Jim','Karen','Kelly','Kevin','Meredith','Michael','Oscar','Pam','Pete','Phyllis','Ryan','Stanley','Toby','Todd Packer'],
                         'Season 8': ['Andy', 'Angela','Creed', 'Darryl', 'David Wallace', 'Dwight','Erin','Gabe', 'Jim','Kelly','Kevin','Meredith','Oscar','Pam','Phyllis','Ryan','Stanley','Toby','Todd Packer'],
                         'Season 9': ['Andy', 'Angela','Creed', 'Darryl', 'David Wallace', 'Dwight','Erin','Gabe','Jan','Jim','Kelly','Kevin','Meredith','Michael','Oscar','Pam','Phyllis','Roy','Ryan','Stanley','Toby','Todd Packer']
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


the_mains_df = office_data[office_data['speaker'].isin(all_main_chars)]

#the_mains_dict = the_mains_df.to_dict(['season','episode','speaker'])

#season_episode_character_dict = the_mains_df.groupby('season')[['episode','speaker']].apply(lambda x: x.set_index('episode').to_dict(orient='index')).to_dict()
#print(season_episode_character_dict)


the_mains_df.head()
the_mains_df_cols = the_mains_df[['season','episode','speaker']]
mains_dict = the_mains_df_cols.to_dict('index')
# x=mains_dict[0].get('speaker')
# print(x)


character_choices = office_data['speaker'].sort_values().unique()
season_choices = office_data['season'].sort_values().unique()
episode_choices = office_data['episode'].sort_values().unique()


tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}

# BS = "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"
# app = dash.Dash(external_stylesheets=[BS])

app = dash.Dash(__name__,assets_folder=os.path.join(os.curdir,"assets"))
server = app.server
app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Word Frequency',value='tab-1',style=tab_style, selected_style=tab_selected_style,
        children=[
                dbc.Row([
                    dbc.Col(
                        dcc.Dropdown(
                            id='dropdown1',
                            options=[{'label': i, 'value': i} for i in season_choices],
                            value=season_choices[0]
                        ), width=6
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            id='dropdown2',
                            options=[{'label': i, 'value': i} for i in character_choices],
                            value=character_choices[0]
                        ),width=6
                    )
                ]),

                dbc.Row([
                    dbc.Col(
                        dcc.Graph(id='word_freq_graph'),
                        width=6
                    ),
                    dbc.Col([
                        dbc.Card(id="card1"),
                        dbc.Card(id="card2"),
                        dbc.Card(id="card3")
                    ],width=6)
                ])
        ]),
        dcc.Tab(label='Sentiment Analysis',value='tab-2',style=tab_style, selected_style=tab_selected_style,
        children=[
            dbc.Row([
                dbc.Col(
                    dcc.Dropdown(
                            id='dropdown3',
                            options=[{'label': i, 'value': i} for i in all_main_chars],
                            value=all_main_chars[0]
                    )
                )
            ]),
            dbc.Row([
                dbc.Col(
                    dcc.Graph(id='sentiment_line_graph'),
                    width=6
                ),
                dbc.Col([
                    dcc.Graph(id='min_max_sentiment'),
                    dbc.Button('Click Here for Positive Sentiment Episode Description',size='lg',block=True,id='open1'),
                    dbc.Button('Click Here for Negative Sentiment Episode Description',size='lg',block=True,id='open2')
                    ],
                    width=6
                )
            ]),
            html.Div([
                dbc.Modal(
                    children=[
                        dbc.ModalHeader("Positive Sentiment Episode Description"),
                        dbc.ModalBody(
                            children=[
                                html.P(
                                    id="table1",
                                    style={'overflow':'auto','maxHeight':'400px'}
                                )
                            ]
                        ),
                        dbc.ModalFooter(
                            dbc.Button("Close", id="close1", className="ml-auto")
                        ),
                    ],id="modal1"
                )
            ]),
            html.Div([
                dbc.Modal(
                    children=[
                        dbc.ModalHeader("Negative Sentiment Episode Description"),
                        dbc.ModalBody(
                            children=[
                                html.P(
                                    id='table2',
                                    style={'overflow':'auto','maxHeight':'400px'}
                                )
                            ]
                        ),
                        dbc.ModalFooter(
                            dbc.Button("Close", id="close2", className="ml-auto")
                        ),
                    ],id="modal2")
            ])      
        
        ]),
        dcc.Tab(label='Network Graph',value='tab-3',style=tab_style, selected_style=tab_selected_style,
        children=[
            dbc.Row([
                dbc.Col(
                    dcc.Dropdown(
                        id='dropdown4',
                        options=[{'label': i, 'value': i} for i in season_choices],
                        value=season_choices[0]
                    ), width=3
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id='dropdown7',
                        options=[{'label': i, 'value': i} for i in episode_choices],
                        value=episode_choices[0]
                    ), width=3
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id='dropdown5',
                        options=[{'label': i, 'value': i} for i in character_choices],
                        value=character_choices[0]
                    ), width=3
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id='dropdown6',
                        options=[{'label': i, 'value': i} for i in character_choices],
                        value=character_choices[1]
                    ), width=3
                )

            ]),
            dbc.Row([
                dbc.Col(
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
                        )
                    )
                )
            ])
        ]),
        dcc.Tab(label='What Does Twitter Think?',value='tab-4',style=tab_style, selected_style=tab_selected_style,
        children=[
            dbc.Row([
                dbc.Col(
                    dcc.Dropdown(
                        id='dropdown8',
                        options=[{'label': i, 'value': i} for i in tweet_seasons],
                        value=tweet_seasons[0]
                    ), width=6
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id='dropdown9',
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
            html.H3('Tab content 3')
        ])
    elif tab == 'tab-4':
        return html.Div([
            html.H3('Tab content 4')
        ])




#Configure reactivity for dynamic dropboxes - 1st one informs the 2nd
@app.callback(
    Output('dropdown2', 'options'),
    Output('dropdown2', 'value'),
    Input('dropdown1', 'value')
)
def set_character_options(selected_season):
    return [{'label': i, 'value': i} for i in season_character_dict[selected_season]], season_character_dict[selected_season][0]



@app.callback(
    Output('dropdown5', 'options'),
    Output('dropdown5', 'value'),
    Input('dropdown4', 'value') #--> choose season
)
def set_character_options2(selected_season):
    return [{'label': i, 'value': i} for i in season_character_dict[selected_season]], season_character_dict[selected_season][0],


@app.callback(
    Output('dropdown6', 'options'),
    Output('dropdown6', 'value'),
    Input('dropdown4', 'value') #--> choose season
)
def set_character_options2(selected_season):
    return [{'label': i, 'value': i} for i in season_character_dict[selected_season]], season_character_dict[selected_season][1],

@app.callback(
    Output('dropdown7', 'options'), #--> filter episodes
    Output('dropdown7', 'value'),
    Input('dropdown4', 'value') #--> choose season
)
def set_episode_options(selected_season):
    return [{'label': i, 'value': i} for i in season_episode_dict[selected_season]], season_episode_dict[selected_season][0],

@app.callback(
    Output('dropdown9', 'options'), #--> filter episodes
    Output('dropdown9', 'value'),
    Input('dropdown8', 'value') #--> choose season
)
def set_episode_options(selected_season):
    return [{'label': i, 'value': i} for i in season_episode_dict[selected_season]], season_episode_dict[selected_season][0],





@app.callback(
    Output('word_freq_graph','figure'),
    Input('dropdown1','value'),
    Input('dropdown2','value')
)

def update_word_chart(season_select,character_select):
    new_df = office_data[(office_data['season']==season_select) & (office_data['speaker']==character_select)]

    top_words = pd.DataFrame(
        new_df['cleaned_text'].str.split(expand=True).stack().value_counts()
    ,columns=['count']
    ).reset_index()

    top_words = top_words.head(10).sort_values('count',ascending=True)

    bar_fig = px.bar(top_words, x='count', y='index',
                     orientation='h',

                     #color='count',
                     #color_continuous_scale="blues",
                     labels={'index':'Word',
                             'count':'Count'}
                    )
    bar_fig.update_layout(coloraxis_showscale=False)
    return bar_fig


#Configure reactivity for dynamic dropboxes - 1st one informs the 2nd
@app.callback(
    Output('card1', 'children'),
    Output('card2', 'children'),
    Output('card3', 'children'),
    Input('dropdown1', 'value'),
    Input('dropdown2', 'value')
)

def speech_stats(season_select,character_select):
    new_df = office_data[(office_data['season']==season_select) & (office_data['speaker']==character_select)]


    #Count total # of words
    top_words = pd.DataFrame(
            new_df['cleaned_text'].str.split(expand=True).stack().value_counts()
        ,columns=['count']
    ).reset_index()

    TOTAL_WORDS = top_words['count'].sum()

    #Count total # of lines
    TOTAL_LINES = new_df.shape[0]

    #Count total # of scenes
    TOTAL_SCENES = len(new_df['scene'].unique())



    card1 = dbc.Card([
        dbc.CardBody([
            html.H4(TOTAL_WORDS, className="card-title"),
            html.P(f"Total Words Spoken")
        ])
    ],
    style={#'display': 'inline-block',
           'width': '100%',
           'text-align': 'center',
           'background-color': 'rgba(104,108,252)',
           'color':'white',
           'fontWeight': 'bold',
           'fontSize':20},
    outline=True)

    card2 = dbc.Card([
        dbc.CardBody([
            html.H4(TOTAL_LINES, className="card-title"),
            html.P(f"Total Lines")
        ])
    ],
    style={#'display': 'inline-block',
           'width': '100%',
           'text-align': 'center',
           'background-color': 'rgba(104,108,252)',
           'color':'white',
           'fontWeight': 'bold',
           'fontSize':20},
    outline=True)

    card3 = dbc.Card([
        dbc.CardBody([
            html.H4(TOTAL_SCENES, className="card-title"),
            html.P(f"Total Scenes")
        ])
    ],
    style={#'display': 'inline-block',
           'width': '100%',
           'text-align': 'center',
           'background-color': 'rgb(104,108,252)',
           'color':'white',
           'fontWeight': 'bold',
           'fontSize':20},
    outline=True)

    return (card1, card2, card3)  


@app.callback(
    Output('sentiment_line_graph','figure'),
    Output('min_max_sentiment','figure'),
    Output('table1','children'),
    Output('table2','children'),
    Input('dropdown3','value')
)

def sentiment(character_select):
    new_df = office_data[office_data['speaker']==character_select]

    sentiment_df=new_df.groupby(['season','episode']).agg({'compound':'mean'}).reset_index()
    sentiment_df['label'] = sentiment_df['season'] + ", " + sentiment_df['episode']
    sentiment_df['season_num'] = sentiment_df['season'].str.slice(7, 9).astype(int)
    sentiment_df['episode_num'] = sentiment_df['episode'].str.slice(8).astype(int)
    sentiment_df = sentiment_df.sort_values(["season_num", "episode_num"], ascending = (True, True))



    fig = px.line(
        sentiment_df, x="label", y="compound", title='Average Sentiment',
        labels={
            "label": "Episodes",
            "compound": "Sentiment"
        }
    )
    fig.update_xaxes(showticklabels=False)
    fig.add_hline(y=0,line_width=3, line_dash="dash", line_color="black")
    fig.update_yaxes(range = [-1,1])

    fig.update_layout(
        title={
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )



    a = pd.DataFrame(sentiment_df[sentiment_df['compound']==sentiment_df['compound'].min()])
    b = pd.DataFrame(sentiment_df[sentiment_df['compound']==sentiment_df['compound'].max()])

    frames = [a,b]
    c = pd.concat(frames)
    c['label'] = c['season'] + ', '+ c['episode']
    c["color"] = np.where(c["compound"]<0, 'red', 'blue')


    fig2 = px.bar(c, y='label',x='compound',color='color',orientation='h',
                color_discrete_map={
                    "red":"red",
                    "blue":"blue"
                },
                title='Sentiment Range',
                height=300,
                hover_data={
                    "color":False,
                    "label":True,
                    "compound":True
                },
                labels={
                     "label": "Episodes",
                     "compound": "Sentiment"
                 })
    fig2.update_xaxes(range = [-1,1])
    fig2.update_yaxes(title='Episodes', visible=True, showticklabels=True)
    fig2.update_layout(
        showlegend=False,
        title={
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )



    s_df=new_df.groupby(['season','episode']).agg({
        'compound':'mean',
        'desc':'first',
        'title':'first'
        }).reset_index()
    s_df['label'] = s_df['season']+', '+s_df['episode']

    most_pos = s_df[s_df['compound']==s_df['compound'].max()]
    most_neg = s_df[s_df['compound']==s_df['compound'].min()]


    most_pos = most_pos.rename(columns={'label': 'Episode', 'title': 'Title','desc':'Description'})
    most_neg = most_neg.rename(columns={'label': 'Episode', 'title': 'Title','desc':'Description'})

    most_pos = most_pos[['Episode','Title','Description']]#.style.hide_index()
    most_neg = most_neg[['Episode','Title','Description']]#.style.hide_index()

    pos_dt = dt.DataTable(
        columns=[{"name": i, "id": i} for i in most_pos.columns],
        data=most_pos.to_dict('records'),
        style_data={
            'whiteSpace': 'normal',
            'height': '150px',
        },
        style_cell={'textAlign': 'left'}
    )
    neg_dt = dt.DataTable(
        columns=[{"name": i, "id": i} for i in most_neg.columns],
        data=most_neg.to_dict('records'),
        style_data={
            'whiteSpace': 'normal',
            'height': '150px',
        },
        style_cell={'textAlign': 'left'}
    )


    return fig, fig2, pos_dt, neg_dt


@app.callback(
    Output('net','data'),
    Input('dropdown4','value'),
    Input('dropdown7','value'),
    Input('dropdown5','value'),
    Input('dropdown6','value'),
)

def network(season_select, episode_select, character_select1, character_select2):
    
    
    filtered = data_for_ng[['season','episode','scene','speaker']]
    filtered = filtered[filtered['season']==season_select]
    filtered = filtered[filtered['episode']==episode_select]

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



    node_list = list(
        set(new_df['Source'].unique().tolist()+new_df['Target'].unique().tolist())
    )

    nodes = [{
        'id': node_name, 
        'label': node_name, 
        'shape':'dot',
        'size':15
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
    return data

#Twitter Sentiment Over Time
@app.callback(
    Output('twitter_sentiment_over_time','figure'),
    Output('word_cloud_chart','figure'),
    Input('dropdown8','value'),
    Input('dropdown9','value')
)

def twitter_sentiment(season_select,episode_select):
    new_df = tweet_data[(tweet_data['season']==season_select)]


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
        }
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


    wc_df = tweet_data[(tweet_data['season']==season_select) & (tweet_data['episode']==episode_select)]
    
    dff = wc_df.copy()
    dff = dff.cleaned_tweet

    
    my_wordcloud = WordCloud(
        background_color='black',
        height=275,
        min_word_length = 4,

    ).generate(' '.join(dff))

    fig_wordcloud = px.imshow(my_wordcloud, template='ggplot2',
                              title="Word Cloud by Season and Episode")
    fig_wordcloud.update_layout(margin=dict(l=20, r=20, t=30, b=20))
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



#app.run_server(host='0.0.0.0',port='8055')
if __name__=='__main__':
	app.run_server()