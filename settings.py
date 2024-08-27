# -*- coding: utf-8 -*-
"""
settings for roster construction
"""
save_plots = False
verbose=True
position_pt_var = {
    'QB': 22.568,
    'RB': 28.27,
    'WR': 19.354,
    'TE': 15.70989
    }

result_excel = 'results_v2_keepers'
league = 'personal' #'harshin'
variable_costs = [5, 10]

sims=1000

pull_weeks=None


# other player's keepers that will be unavailable to draft

#settings for optimization roster size
if league=='personal':
    # list of your own keepers
    keepers = {'Breece Hall':15, 'Brandon Aiyuk':5}
    unavailable = {'James Cook', 'DJ Moore', 
                   'Isiah Pacheco', 'Christian McCaffrey',
                   'Mike Evans', 'Amon-Ra St. Brown',
                   'Diontae Johnson', 'Travis Etienne Jr.',
                   'Ja\'Marr Chase', 'Jahmyr Gibbs'}



    min_qb = 2
    min_rb = 3
    min_wr = 5
    min_te = 2
    flex_positions = 2
    bench_positions = 6
    incude_bench=False
    bench_allocations=4
    
    starting_salary=350
    total_cost = starting_salary-(1-incude_bench)*bench_positions*bench_allocations
    adjust_cost = False
else:
    keepers = {}
    unavailable = {}
    min_qb = 1
    min_rb = 2
    min_wr = 3
    min_te = 1
    flex_positions = 1
    bench_positions = 6
    incude_bench=False
    bench_allocations=3
    
    starting_salary=200
    total_cost = starting_salary-(1-incude_bench)*bench_positions*bench_allocations
    adjust_cost = False
missing_weeks={'QB':0,'RB':0,'WR':0,'TE':0}

