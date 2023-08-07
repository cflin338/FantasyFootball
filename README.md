# FantasyFootball
requirements:
pandas, numpy, cvxpy
Construct optimal fantasy football team based on auction cost

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
\n   sum(player_costs) <= total_cost
\n    QB_count >= min_qbs
\n    RB_count >= min_rbs
\n    WR_count >= min_wrs
\n    TE_count >= min_tes
\n    QB_count + RB_count + WR_count + TE_count == roster_size
    
optimize using cvxpy package

includes variability in costs

