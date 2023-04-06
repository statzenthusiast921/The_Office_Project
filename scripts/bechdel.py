import pandas as pd
import numpy as np
import re
import contractions
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
#Load data
office_data = pd.read_excel("/Users/jonzimmerman/Desktop/Data Projects/The Office NLP/the-office-lines.xlsx")



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



#Bechdel test
#Step 1: Indicator for when male is mentioned in conversation
office_data['male_subject'] = np.where(# where(condition, [x, y])
    office_data['cleaned_text'].str.contains('he|him|his|michael|jim|kevin|oscar|stanley|toby|roy|ryan|andy|creed|darryl|gabe|david wallace|dwight|todd packer|todd|david'), 1, 0)

#Step 2: Indicator for when a woman is in the scene
office_data['women_speaking'] = np.where((office_data['speaker']=='Pam') |
                                         (office_data['speaker']=='Jan') |
                                         (office_data['speaker']=='Kelly') |
                                         (office_data['speaker']=="Phyllis") |
                                         (office_data['speaker']=="Angela") |
                                         (office_data['speaker']=="Erin") |
                                         (office_data['speaker']=="Holly") |
                                         (office_data['speaker']=="Karen") |
                                         (office_data['speaker']=="Meredith") 
                                         , 1, 0)

#All women characters
women = office_data[office_data['women_speaking']==1]
all_women_mains = women['speaker'].sort_values().unique()

#Filter down to the lines spoken by ONLY the main characters
main_characters = office_data[office_data['speaker'].str.contains('Andy|Angela|Creed|Darryl|David Wallace|Dwight|Erin|Gabe|Pam|Jan|Jim|Holly|Karen|Kelly|Kevin|Meredith|Michael|Oscar|Pam|Phyllis|Roy|Ryan|Stanley|Toby|Todd Packer')]

#Identify lines spoken per scene and merge metric back into full main character dataset
lines_per_scene = main_characters.groupby(['season', 'episode','scene']).size().reset_index(name='counts')
new_df = pd.merge(main_characters, lines_per_scene,  how='left', left_on=['season','episode','scene'], right_on = ['season','episode','scene'])

#Sum down to # of lines spoken by women in scene and merge back into full main character dataset
total_women_lines = new_df.groupby(['season', 'episode','scene'])['women_speaking'].agg('sum').reset_index(name='tot_wom_lines')
full_bechdel = pd.merge(new_df, total_women_lines,  how='left', left_on=['season','episode','scene'], right_on = ['season','episode','scene'])

#Filter down to the scenes with only women
full_bechdel['all_women_scene'] = full_bechdel['counts'] - full_bechdel['tot_wom_lines']
only_women = full_bechdel[full_bechdel['all_women_scene']==0]

#Get rid of all testimonial scenes
only_women = only_women[only_women['counts']>1]


#Calculate the bechdel score
bechdel_scores = only_women.groupby(['season','speaker'])['male_subject'].agg('sum').reset_index(name='bech_sum')
total_lines = only_women.groupby(['season','speaker']).agg('count').reset_index()
total_lines = total_lines[['season','speaker','counts']]
bechdel_percs = pd.merge(bechdel_scores, total_lines,  how='left', left_on=['season','speaker'], right_on = ['season','speaker'])

bechdel_percs['b_percent'] = round((bechdel_percs['bech_sum']/bechdel_percs['counts'])*100,2)
bechdel_percs
bechdel_percs.to_csv('/Users/jonzimmerman/Desktop/Data Projects/The Office NLP/bechdel_season.csv', index=False)

#only_women_df = pd.merge(only_women,bechdel_percs,how='left',on=['speaker'])
#only_women_df.to_csv('/Users/jonzimmerman/Desktop/Data Projects/The Office NLP/bechdel.csv', index=False)

