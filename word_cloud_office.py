from numpy.core.numeric import full
import pandas as pd
import numpy as np
import dash
import os
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('ggplot')
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output 
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, Response
import base64
from io import BytesIO
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import re
import contractions
import operator

from nltk.stem import WordNetLemmatizer 




office_data = pd.read_excel("/Users/jonzimmerman/Desktop/Data Projects/The Office NLP/the-office-lines.xlsx")
# sheet_url ="https://docs.google.com/spreadsheets/d/18wS5AAwOh8QO95RwHLS95POmSNKA2jjzdt0phrxeAE0/edit#gid=747974534"
# url_1 = sheet_url.replace('/edit#gid=', '/export?format=csv&gid=')
# office_data = pd.read_csv(url_1)


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



character_choices = office_data['speaker'].sort_values().unique()
season_choices = office_data['season'].sort_values().unique()


app = dash.Dash(__name__,assets_folder=os.path.join(os.curdir,"assets"))
server = app.server
app.layout = html.Div([
                html.Div([
                    dcc.Dropdown(
                        id='dropdown1',
                        options=[{'label': i, 'value': i} for i in season_choices],
                        value=season_choices[0]
                    )
                ],style={'width': '50%','display': 'inline-block','text-align': 'center'}),
                html.Div([
                    dcc.Dropdown(
                        id='dropdown2',
                        options=[{'label': i, 'value': i} for i in character_choices],
                        value=character_choices[0]
                    )
                ],style={'width': '50%','display': 'inline-block','text-align': 'center'}),

                html.Div([
                    dcc.Graph(id='word_cloud_chart', figure={}, config={'displayModeBar': False})
                ],style={'width': '100%','display': 'inline-block','text-align': 'center'}),

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
    Output('word_cloud_chart','figure'),

    Input('dropdown1','value'),
    Input('dropdown2','value')
)

def update_word_cloud(season_select,character_select):
    new_df = office_data[(office_data['season']==season_select) & (office_data['speaker']==character_select)]

    dff = new_df.copy()
    dff = dff.cleaned_text
    
    my_wordcloud = WordCloud(
        background_color='black',
        height=275,
        min_word_length = 4,

    ).generate(' '.join(dff))

    fig_wordcloud = px.imshow(my_wordcloud, template='ggplot2',
                              title="Word Cloud by Season and Character")
    fig_wordcloud.update_layout(margin=dict(l=20, r=20, t=30, b=20))
    fig_wordcloud.update_xaxes(visible=False)
    fig_wordcloud.update_yaxes(visible=False)


    return fig_wordcloud



#app.run_server(host='0.0.0.0',port='8055')
if __name__=='__main__':
	app.run_server()