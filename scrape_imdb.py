from requests import get
from bs4 import BeautifulSoup
import pandas as pd


#initialize series that the loop will populate
office_eps=[]

for season in range(1,10):
    url = 'https://www.imdb.com/title/tt0386676/episodes?season='+str(season)
    response = get(url)
    #Parse content of the request with Beautiful Soup
    html_soup = BeautifulSoup(response.text,'html.parser')
    #Select all the episode containers from the season's page
    episode_container = html_soup.find_all('div',class_='info')

    #For each episode in each season
    for episodes in episode_container:
            season = season
            episode = episodes.meta['content']

            title = episodes.a['title']
            airdate = episodes.find('div',class_='airdate').text.strip()
            rating = episodes.find('span',class_='ipl-rating-star__rating').text
            total_votes = episodes.find('span',class_='ipl-rating-star__total-votes').text
            description = episodes.find('div',class_='item_description').text.strip()

            #Compiling the episode info
            episode_data = [season, episode, title, airdate, rating, total_votes, description]
            
            #Append the episode info to the complete dataset
            office_eps.append(episode_data)

episodes_df = pd.DataFrame(office_eps,columns = ['season','episode','title','airdate','rating','total_votes','description'])


def remove_str(votes):
    for r in ((',',''), ('(',''),(')','')):
        votes = votes.replace(*r)
    return votes


episodes_df['total_votes'] = episodes_df['total_votes'].apply(remove_str).astype(int)
episodes_df['total_votes']



episodes_df['rating'] = episodes_df['rating'].astype(float)

episodes_df['airdate'] = pd.to_datetime(episodes_df['airdate'])
episodes_df.info()


episodes_df.head()




episodes_df.to_csv('/Users/jonzimmerman/Desktop/Data Projects/The Office NLP/office_episodes.csv',index=False)