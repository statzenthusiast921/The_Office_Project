from requests import get
from bs4 import BeautifulSoup
import pandas as pd


#initialize series that the loop will populate
office_eps=[]
for season in range(1,10):
    url = 'https://en.wikipedia.org/wiki/The_Office_(American_season_'+str(season)+')'
    response = get(url)
    #Parse content of the request with Beautiful Soup
    html_soup = BeautifulSoup(response.text,'html.parser')
    #Select all the episode containers from the season's page --> parent container for which all HTML code is under
    episode_container = html_soup.find_all('tr',class_='expand-child')
    #episode_container2 = html_soup.find_all('tr',class_='vevent')

    #For each episode in each season
    for episodes in episode_container:
        season = season
            #episode = episodes.find()
        desc = episodes.find('td',class_='description').text.strip()
        #episode_num = episodes.find('td',class_='vevent').text.strip()
            # episode = episodes.meta['content']

            # title = episodes.a['title']
            # airdate = episodes.find('div',class_='airdate').text.strip()
            # rating = episodes.find('span',class_='ipl-rating-star__rating').text
            # total_votes = episodes.find('span',class_='ipl-rating-star__total-votes').text
            # description = episodes.find('div',class_='item_description').text.strip()

            #Compiling the episode info
        episode_data = [season, desc]# episode, title, airdate, rating, total_votes, description]
            
            #Append the episode info to the complete dataset
        office_eps.append(episode_data)



episodes_df = pd.DataFrame(office_eps,columns = ['season','desc'])



episodes_df.to_csv('/Users/jonzimmerman/Desktop/Data Projects/The Office NLP/wiki_desc.csv',index=False)
