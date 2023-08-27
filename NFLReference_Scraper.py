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
    soup = BeautifulSoup(html, features = 'lxml')
    tables = soup.find(id='data')
    data = pd.read_html(str(tables))[0]
    
    return data    
"""
def new_point_system(stats, position):
    #incorporate player's projected points with advanced stats 
    #player column: ('Unnamed: 0_level_0',        'Rank')
    if position=='RB':
        #('Unnamed: 1_level_0',      'Player'),
        #(           'RUSHING',   'YBCON/ATT'),
        #(           'RUSHING',   'YACON/ATT'),
        #(           'RUSHING',      'BRKTKL'),
        #(           'RUSHING',     'TK LOSS'),
        #(    'BIG RUSH PLAYS',     '10+ YDS'),
        #(    'BIG RUSH PLAYS',         'LNG'),
        #(         'RECEIVING',         'TGT'),
        #(         'RECEIVING',       'YACON')
    elif position=='WR':
        #('Unnamed: 1_level_0',    'Player'),
        #(         'RECEIVING',     'YBC/R'),
        #(         'RECEIVING',     'AIR/R'),
        #(         'RECEIVING',     'YAC/R'),
        #(         'RECEIVING',   'YACON/R'),
        #(         'RECEIVING',    'BRKTKL'),
        #(           'TARGETS',       'TGT'),
        #(           'TARGETS',      '% TM'),
        #(           'TARGETS', 'CATCHABLE'),
        #(           'TARGETS',      'DROP'),
        #(           'TARGETS',    'RZ TGT'),
        #(         'BIG PLAYS',   '10+ YDS'),
        #(         'BIG PLAYS',       'LNG')
    elif position=='QB':
        #('Unnamed: 1_level_0',   'Player'),
        #(           'PASSING',      'Y/A'),
        #(           'PASSING',    'AIR/A'),
        #( 'DEEP BALL PASSING',  '20+ YDS'),
        #(          'PRESSURE', 'PKT TIME'),
        #(              'MISC',   'RZ ATT'),
        
    else:
        # TE
        #('Unnamed: 1_level_0',    'Player'),
        #(         'RECEIVING',     'YBC/R'),
        #(         'RECEIVING',     'AIR/R'),
        #(         'RECEIVING',     'YAC/R'),
        #(         'RECEIVING',   'YACON/R'),
        #(         'RECEIVING',    'BRKTKL'),
        #(           'TARGETS',       'TGT'),
        #(           'TARGETS',      '% TM'),
        #(           'TARGETS', 'CATCHABLE'),
        #(           'TARGETS',      'DROP'),
        #(           'TARGETS',    'RZ TGT'),
        #(         'BIG PLAYS',   '10+ YDS'),
        #(         'BIG PLAYS',       'LNG')
        
        
    return
"""