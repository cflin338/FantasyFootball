"""
@author: clin4

https://www.fantasypros.com/nfl/auction-values/calculator.php

construct multiple optimal fantasy football rosters
this is similar to a knapsac problem with various contraints

given:
    total_cost
    min_qbs
    min_rbs
    min_wrs
    min_tes
    flex_positions
    bench_positions
    roster_size = min_qbs + min_rbs + min_wrs + min_tes + flex_postions + bench_positions
    
optimization function:
    max C*X = total projected fantasy points 
    C = weekly projected fantasy points
    
constraints:
    sum(player_costs) <= total_cost
    QB_count >= min_qbs
    RB_count >= min_rbs
    WR_count >= min_wrs
    TE_count >= min_tes
    QB_count + RB_count + WR_count + TE_count == roster_size
    

iteration 1: no bench players, reduce total salary
iteration 2: bench players>0
"""

import pandas as pd
import numpy as np
import os
import cvxpy
import random
import matplotlib.pyplot as plt

def load_data(data_loc = os.getcwd() + '/Data/FF_2023_Data.csv', add_variability = False, random_cost = 5, prev_selected = None):    
    # import data
    player_info = pd.read_csv(data_loc)

    # for all players with NA as projecteed points, change them to 0 for now
    player_info['Points'] = player_info['Points'].fillna(0)

    # create binary for positions
    for position in ['QB', 'RB', 'WR', 'TE']:
        player_info[position] = player_info['Position']==position
    
    if prev_selected:
        #add variability only to players previously selected
        players_to_raise_value = player_info['Player'].isin(prev_selected)
        
        new_cost = player_info['Cost'] + [random.random()*random_cost*i for i in players_to_raise_value]
        player_info['Cost'] = new_cost.round()
            
    elif add_variability:
        #add variability to all players
        new_cost = player_info['Cost'] + [random.random() * random_cost for i in range(len(player_info))]
        player_info['Cost'] = new_cost.round()
    #elif dynamic_variability:
        #add variability to all previously selected players, adding more weight to those selected more times
        #possibility 1: increase by weight
    return player_info


def optimize_team(player_info, min_qb, min_rb, min_wr, min_te, flex_positions, bench_positions, total_cost, adjust_cost=True, adjust_cost_val = 5):    
    roster_size = min_qb + min_rb + min_wr + min_te + flex_positions + bench_positions
    if adjust_cost: 
        total_cost = 300 - bench_positions*adjust_cost_val

    player_selection = cvxpy.Variable(shape = len(player_info), boolean = True)

    # equality constraints
    eq_constraint1 = player_info['QB'].to_numpy() @ player_selection == min_qb
    eq_constraint2 = np.ones(len(player_info)) @ player_selection == roster_size

    ineq_constraint1 = player_info['WR'].to_numpy() @ player_selection >= min_wr
    ineq_constraint2 = player_info['RB'].to_numpy() @ player_selection >= min_rb
    ineq_constraint3 = player_info['TE'].to_numpy() @ player_selection >= min_te
    ineq_constraint4 = player_info['Cost'].to_numpy() @ player_selection <= total_cost - 4*3
    
    #optimization function
    total_points = player_info['Points'].to_numpy() @  player_selection

    #optimization problem
    team_selection = cvxpy.Problem(cvxpy.Maximize(total_points), [eq_constraint1, eq_constraint2, ineq_constraint1, ineq_constraint2, ineq_constraint3, ineq_constraint4])

    team_selection.solve()

    #print(player_info[np.round(np.abs(player_selection.value)).astype(bool)])
    return team_selection, player_info[np.round(np.abs(player_selection.value)).astype(bool)]

counts = {}
totals = []
iterations = 1000
prev_selected = None
adjust_by_prev = True
for i in range(iterations):
    #after every iteration, include a make players that were previously selected more expensive
    player_info = load_data(add_variability = True, prev_selected=prev_selected)

    #team, players = optimize_team(player_info, min_qb = 2, min_rb = 3, min_wr = 5, min_te = 2, 
    #                             flex_positions = 2, bench_positions = 0, total_cost = 300, )
    team, players = optimize_team(player_info, 
                                  min_qb = 2, min_rb = 3, min_wr = 5, min_te = 2, 
                                  flex_positions = 2, bench_positions = 0,#7, 
                                  total_cost = 300-7*4, 
                                  adjust_cost = False,)
    if adjust_by_prev:
        prev_selected = list(players['Player'])
    
    for player in players['Player']:
        if player in counts:
            counts[player]+=1
        else:
            counts[player]=1
            
    totals.append(team.value)

#sort by ascending order, most appearances
counts = sorted(counts.items(), key = lambda x:x[1])

# iteration 2:
#    implementing weights for starters, bench

fig, ax = plt.subplots()
ax.scatter(range(len(counts)), [i[1] for i in counts]) #counts.values(), )
for idx,i in enumerate(counts):
    ax.annotate(text = '{}, {}'.format(i[0], i[1]), 
                xy = (idx, i[1]),
                textcoords = 'offset points',
                xytext = (5,0),
                fontsize = 'small',
                rotation='vertical')
plt.grid()
plt.title('Number of Sims a Player is Selected as Part of Optimal Team')
plt.ylabel('# Appearances')
plt.show()

