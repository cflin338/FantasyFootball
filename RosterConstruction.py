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
import numpy as np
import random

def load_data(data_loc = os.getcwd() + '/Data/FF_2023_Data.csv', add_variability = False, random_cost = 5):    
    # import data
    player_info = pd.read_csv(data_loc)

    # for all players with NA as projecteed points, change them to 0 for now
    player_info['Points'] = player_info['Points'].fillna(0)

    # create binary for positions
    for position in ['QB', 'RB', 'WR', 'TE']:
        player_info[position] = player_info['Position']==position
        
    if add_variability:
        new_cost = player_info['Cost'] + [random.random() * random_cost for i in range(len(player_info))]
        player_info['Cost'] = new_cost.round()
        
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
totals = 0
iterations = 100
for i in range(iterations):
    player_info = load_data(add_variability = True)

    #team, players = optimize_team(player_info, min_qb = 2, min_rb = 3, min_wr = 5, min_te = 2, 
    #                             flex_positions = 2, bench_positions = 0, total_cost = 300, )
    team, players = optimize_team(player_info, min_qb = 2, min_rb = 3, min_wr = 5, min_te = 2, 
                                 flex_positions = 2, bench_positions = 7, total_cost = 300, 
                                 adjust_cost = False,)

    for player in players['Player']:
        if player in counts:
            counts[player]+=1
        else:
            counts[player]=1
    totals+=team.value

print(sorted(counts.items(), key = lambda x:x[1]))


# iteration 2:
#    implementing weights for starters, bench



