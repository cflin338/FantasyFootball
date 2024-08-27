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
iteration 3: added variable point projections

todo: include keepers; remove variability in cost for keeper players
"""

import pandas as pd
import numpy as np
import os
import cvxpy
import random
import matplotlib.pyplot as plt
import settings
from scipy.stats import skewnorm

from NFLReference_Scraper import get_projections, get_advanced_stats

def load_costs(data_loc = os.getcwd() + '/Data/FF24.csv', 
               add_variability = True, 
               prev_selected = None):
    player_costs = pd.read_csv(data_loc, encoding='latin-1')
        
    player_costs['Player'] = player_costs['Player'].astype(str)
    # player_costs['Team'] = player_costs['Team'].str.upper()
    player_costs['Min Cost'] = player_costs['Min Cost']
    player_costs['Max Cost'] = player_costs['Max Cost']
    
    if len(settings.keepers)>0:
        for keeper in settings.keepers:
            keeper_price = settings.keepers[keeper]
            player_costs.loc[player_costs['Player'] == keeper, 'Min Cost'] = keeper_price
            player_costs.loc[player_costs['Player'] == keeper, 'Max Cost'] = keeper_price

    
    return player_costs

def load_base_stats(week=None):
    costs = load_costs()
    if week is None:
        print('loading data for full season projections')
        qb = get_projections('QB',week='draft').drop('Team', axis = 1)
        wr = get_projections('WR',week='draft').drop('Team', axis = 1)
        rb = get_projections('RB',week='draft').drop('Team', axis = 1)
        te = get_projections('TE',week='draft').drop('Team', axis = 1)
    else:
        print('loading data for first {} weeks'.format(week))
        qb = pd.DataFrame(columns = ['Player', 'Points'])
        wr = pd.DataFrame(columns = ['Player', 'Points'])
        rb = pd.DataFrame(columns = ['Player', 'Points'])
        te = pd.DataFrame(columns = ['Player', 'Points'])
        for i in range(1,week+1):
            qb_ = get_projections('QB',week=i).drop(['Team','Position'], axis = 1)
            if len(qb_)<2:
                settings.missing_weeks['QB']+=1
                print('Week {} has no QB projections yet'.format(i))
            else:
                qb = pd.merge(qb, qb_, on=['Player','Team'], how='outer', suffixes=('_t1','_t2'))
                qb.fillna(0, inplace=True)
                qb['Points'] = qb['Points_t1'] + qb['Points_t2']
                qb = qb.drop(['Points_t1','Points_t2'],axis=1)
                
            wr_ = get_projections('WR',week=i).drop(['Position'], axis = 1)
            if len(wr_)<2:
                settings.missing_weeks['WR']+=1
                print('Week {} has no WR projections yet'.format(i))
            else:
                wr = pd.merge(wr, wr_, on=['Player','Team'], how='outer', suffixes=('_t1','_t2'))
                wr.fillna(0, inplace=True)
                wr['Points'] = wr['Points_t1'] + wr['Points_t2']
                wr = wr.drop(['Points_t1','Points_t2'],axis=1)
                
            
            rb_ = get_projections('RB',week=i).drop(['Position'], axis = 1)
            if len(rb_)<2:
                settings.missing_weeks['RB']+=1
                print('Week {} has no RB projections yet'.format(i))
            else:
                rb = pd.merge(rb, rb_, on=['Player','Team'], how='outer', suffixes=('_t1','_t2'))
                rb.fillna(0, inplace=True)
                rb['Points'] = rb['Points_t1'] + rb['Points_t2']
                rb = rb.drop(['Points_t1','Points_t2'],axis=1)
                
            te_ = get_projections('TE',week=i).drop(['Position'], axis = 1)
            if len(te_)<2:
                settings.missing_weeks['TE']+=1
                print('Week {} has no TE projections yet'.format(i))
            else:
                te = pd.merge(te, te_, on=['Player','Team'], how='outer', suffixes=('_t1','_t2'))
                te.fillna(0, inplace=True)
                te['Points'] = te['Points_t1'] + te['Points_t2']
                te = te.drop(['Points_t1','Points_t2'],axis=1)
        qb['Position']='QB'
        wr['Position']='WR'
        rb['Position']='RB'
        te['Position']='TE'
    all_projections = pd.concat([qb, wr, rb, te])

    costs = costs.merge(right = all_projections, how = 'left', on = 'Player')
    
    for position in ['QB', 'RB', 'WR', 'TE']:
        costs[position] = costs['Position']==position
    
    costs['Min Cost'] = costs['Min Cost'].fillna(0)
    costs['Max Cost'] = costs['Max Cost'].fillna(0)
    costs['Points'] = costs['Points'].fillna(0)
    return costs

def add_cost_variability(costs, prev_selected, mode = 0, variable_cost = 5, total_sims = 1000):
    #given base dataframe of players and their costs, add variability to pricing
    #pricings will vary dependent on prev_players
    #if mode = 0, randomly increase for all pre players at a fixed value
    #if mode = 1, randomly increase for pre players only if selected in any previous 
    #   iteration, more of a weight for players more selected more often.
    #   more weight = more price raise
    if prev_selected:
        if mode==0:
            if prev_selected:
                players_to_raise_value = costs['Player'].isin(prev_selected)
                players_to_raise_value = costs['Player'].isin(prev_selected) & ~costs['Player'].isin(settings.keepers)

                tmp = [random.random()*variable_cost*i for i in players_to_raise_value]
                new_mincost = costs['Min Cost'] + tmp
                new_maxcost = costs['Max Cost'] + tmp
                costs['Min Cost'] = new_mincost.round()
                costs['Max Cost'] = new_maxcost.round()
                
        if mode==1:
            for idx,row in costs.iterrows():
                if row['Player'] in prev_selected and row['Player'] not in settings.keepers:
                    min_adjust = np.ceil((prev_selected[row['Player']]/total_sims)*variable_cost)
                    tmp = random.randint(min_adjust, variable_cost)
                    new_mincost = row['Min Cost'] + tmp
                    new_maxcost = row['Max Cost'] + tmp
                    costs.loc[idx,'Min Cost'] = new_mincost
                    costs.loc[idx,'Max Cost'] = new_maxcost
        
    return costs


def optimize_team(player_info, min_qb, min_rb, min_wr, min_te, flex_positions, bench_positions, total_cost, adjust_cost=True, adjust_cost_val = 5):    
    roster_size = min_qb + min_rb + min_wr + min_te + flex_positions + bench_positions*settings.incude_bench
    if adjust_cost: 
        total_cost = total_cost - bench_positions*adjust_cost_val

    player_selection = cvxpy.Variable(shape = len(player_info), boolean = True)

    # equality constraints
    eq_constraint1 = player_info['QB'].to_numpy() @ player_selection == min_qb
    eq_constraint2 = np.ones(len(player_info)) @ player_selection == roster_size
    
    # ineq constraints
    ineq_constraint1 = player_info['WR'].to_numpy() @ player_selection >= min_wr
    ineq_constraint2 = player_info['RB'].to_numpy() @ player_selection >= min_rb
    ineq_constraint3 = player_info['TE'].to_numpy() @ player_selection >= min_te
    ineq_constraint4 = player_info['Cost'].to_numpy() @ player_selection <= total_cost - 4*3
    
    #optimization function
    total_points = player_info['ProjPoints'].to_numpy() @  player_selection

    #optimization problem
    team_selection = cvxpy.Problem(cvxpy.Maximize(total_points), [eq_constraint1, eq_constraint2, ineq_constraint1, ineq_constraint2, ineq_constraint3, ineq_constraint4])
    team_selection.solve()

    return team_selection, player_info[np.round(np.abs(player_selection.value)).astype(bool)]

def perform_sim(sims = 1000, variable_cost = 5, random_mode = 0,weeks=None):
    
    #perform sims
    base_stats = load_base_stats(week=weeks)
    base_stats = base_stats[~base_stats['Player'].isin(settings.unavailable)]

    counts = {}
    prices = {}
    totals = []
    prev_selected = None

    def random_in_range(row, skew_baseline):
        #increase upper range by 50%
        #randomly select but with a skew, which is higher for higher values
        #will be relative to max
        if row['Player'] in settings.keepers:
            return row['Max Cost']
        
        adjustment = max(1, (row['Max Cost']-row['Min Cost'])*.5)
        max_cost = row['Max Cost']+adjustment
        min_cost = row['Min Cost']
        if max_cost > skew_baseline:
            skewness = (max_cost - skew_baseline) / (max_cost - min_cost)
        else:
            skewness = -(skew_baseline-max_cost) / (max_cost - min_cost)
        
        skewed_distribution = skewnorm(skewness, loc=min_cost, scale=max_cost - min_cost)
        random_value = skewed_distribution.rvs()
    
        # Ensure the value is within the integer range and round it
        random_int = int(np.round(random_value))
        
        # Clip to ensure it stays within the range
        return int(np.ceil(max(min(random_int, row['Max Cost']+.5*(row['Max Cost']-row['Min Cost'])), row['Min Cost'])))
        
        #return random.randint(np.floor(row['Min Cost']), np.ceil(row['Max Cost']))
    
    def variable_points(row, weeks=weeks):
        if weeks is None:
            return np.random.normal(row['Points'],settings.position_pt_var[row['Position']])
        else:
            return np.random.normal(row['Points'],settings.var[row['Position']]/np.sqrt(17/(weeks-settings.missing_weeks[row['Position']])))
    
    for i in range(sims):
        if settings.verbose and i%100==0:
            print('sim {} of {} total sims'.format(i+1,sims))
            
        #after every iteration, make players that were previously selected more expensive
        player_info = add_cost_variability(costs=base_stats.copy(), 
                                           prev_selected=prev_selected,
                                           mode = random_mode,
                                           variable_cost = variable_cost)
        # using min/max possible cost, generate cost to be used for optimization
        
        # player_info['Cost'] = player_info.apply(random_in_range,axis=1)
        skew_baseline = ((player_info['Max Cost'] - player_info['Min Cost'])*.5+player_info['Max Cost']).median()
        # skew_baseline=player_info['Max Cost'].max()
        player_info['Cost'] = player_info.apply(lambda row: random_in_range(row, skew_baseline), 
                                                axis=1)
        
        # add variation to player projected points
        player_info['ProjPoints'] = player_info.apply(variable_points, axis=1)
        team, players = optimize_team(player_info, 
                                      min_qb = settings.min_qb, min_rb = settings.min_rb, 
                                      min_wr = settings.min_wr, min_te = settings.min_te, 
                                      flex_positions = settings.flex_positions, 
                                      bench_positions = settings.bench_positions,
                                      total_cost = settings.total_cost, 
                                      adjust_cost = settings.adjust_cost,)
        if i==sims//2 and settings.verbose:
            print('--------halway check, sample team--------')
            print(players[['Player', 'Rank', 'Position','Min Cost', 'Max Cost', 'Cost', 'Points','ProjPoints']])
            
        for player,cost in zip(players['Player'],players['Cost']) :
            if player in counts:
                counts[player]+=1
                prices[player].append(cost)
            else:
                counts[player]=1
                prices[player]=[cost]
        
        totals.append(team.value)
        
        if random_mode==0:
            prev_selected = list(players['Player'])
        elif random_mode==1:
            prev_selected = counts
    
    #sort by ascending order, most appearances
    counts = sorted(counts.items(), key = lambda x:x[1], reverse=True)
    price_range = {player:(min(prices[player]), max(prices[player])) for player in prices}
    return counts, totals, price_range

def main(sims, cost, mode,keepers=None,weeks=None):    
    counts, totals, prices = perform_sim(sims=sims, variable_cost = cost, 
                                         random_mode = mode, weeks=weeks)
    counts = [c for c in counts if c[1]>1]
    
    fig, ax = plt.subplots(figsize=(12,6))
    ax.scatter(range(len(counts)), [i[1] for i in counts]) #counts.values(), )
    for idx,i in enumerate(counts):
        ax.annotate(text = '{}, {}'.format(i[0], i[1]), 
                    xy = (idx, i[1]),
                    textcoords = 'offset points',
                    xytext = (5,0),
                    fontsize = 'small',
                    rotation='vertical')
    plt.grid()
    if weeks is None: 
        length='Full'
        title='Full Season Projections'
    else: 
        length='{}wks'.format(weeks)
        title='First {} Weeks Projections'.format(weeks)
    plt.title('${} variable, mode {}: Number of Sims a Player is Selected as Part of Optimal Team, {}'.format(cost,mode,title))
    plt.ylabel('# Appearances')
    
    if settings.save_plots:
        plt.savefig(os.getcwd() + '/Plots/{}sims_dol{}mode{}_{}.png'.format(sims, cost, mode,length), dpi=300, bbox_inches='tight')
    
    fig2, ax2 = plt.subplots(figsize=(12,6))
    ax2.boxplot(totals, vert = False, )
    plt.title('Range of Team Total Outcomes for Simulations, {}'.format(title))
    plt.xlabel('Total Points')
    
    if settings.save_plots:
        plt.savefig(os.getcwd() + '/Plots/{}sims_dol{}mode{}_points_{}.png'.format(sims, cost, mode,length), dpi=300, bbox_inches='tight')
    
    return counts, prices
    
if __name__=='__main__':
    with pd.ExcelWriter(os.getcwd() + '/Plots/{}.xlsx'.format(settings.result_excel), engine='openpyxl') as writer:
        idx=0
        for cost in settings.variable_costs:
            for mode in [0,1]:
                counts,prices = main(settings.sims, cost, mode,weeks=settings.pull_weeks)
                prices = pd.DataFrame([(i,prices[i][0], prices[i][1]) for i in prices],columns=['Player','Min Price','Max Price'])
                counts = pd.DataFrame(counts,columns=['Player','Count'])
                comb = pd.merge(left=counts, right=prices,how='inner', on='Player')
                comb.to_excel(writer, sheet_name='Sheet1', startcol=idx, index=False)
                idx+=4