# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 23:53:03 2023

@author: clin4
"""

from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
from unidecode import unidecode

def get_projections(position, week='draft'):
    url = 'https://www.fantasypros.com/nfl/projections/{}.php?week={}'.format(position.lower(), week)
    html = urlopen(url)
    soup = BeautifulSoup(html, features = 'lxml')
    tables = soup.find(id='data')
    
    data = pd.read_html(str(tables))[0]
    players = data[('Unnamed: 0_level_0', 'Player')]
    fantasy_projections = data[('MISC', 'FPTS')]
    
    tmp = list(players.str.split())
    player_teams = [i[-1] for i in tmp]
    players = [' '.join(i[:-1]) for i in tmp]
    df = pd.DataFrame(data = {'Player': players, 'Team': player_teams, 'Points': fantasy_projections})
    df['Player'] = df['Player'].astype(str)
    df['Position'] = position
    #df[position] = True
    return df

def get_advanced_stats(position, ):
    url = 'https://www.fantasypros.com/nfl/advanced-stats-{}.php'.format(position.lower())
    html = urlopen(url)
    soup = BeautifulSoup(html, )#features = 'lxml')
    tables = soup.find(id='data')
    data = pd.read_html(str(tables))[0]
    
    return data    

