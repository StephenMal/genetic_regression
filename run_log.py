import subprocess, sys

'''
params = \
    {'n_runs':[200],\
     'n_gens':[200],\
     'xov_rate':[0.7, 0.8, 0.9, 1.0],\
     'pop_size':[200],\
     'L1':[0, 0.1, 0.01, 0.001],\
     'L2':[0, 0.1, 0.01, 0.001],\
     'w_range':[1,5,10],\
     'togs':[True, False]}

vga_params = \
    {'dtype':[float],\
     'min':[-1],\
     'max':[1],\
     'mut_op':['uniform_mutation'],\
     'xov_op':['twopt', 'onept'],\
     'mut_rate':[0, 0.015, 0.05, 0.1]}

pga_params = \
    {'dtype':[float],\
     'min':[-1],\
     'max':[1],\
     'map_fxn':[1,2,3],\
     'mut_op':['uniform_mutation'],\
     'mut_rate':[0, 0.015, 0.05, 0.1],\
     'xov_op':['twopt', 'onept', 'homo'],\
     'length':[-1, 30, 100, 200],\
     'n_noncoding_chars':[0,1,3,5]}

bsga_params = \
    {'dtype':[float],\
     'min':[-1],\
     'max':[1],\
     'mut_op':['flipbit'],\
     'mut_rate':[0, 0.015, 0.05, 0.1],\
     'xov_op':['twopt', 'onept'],\
     'gene_size':[4,8,16,32]}
'''

'''

params = \
    {'n_runs':[200],\
     'n_gens':[200],\
     'xov_rate':[0.9],\
     'pop_size':[200],\
     'L1':[0],\
     'L2':[0],\
     'w_range':[1],\
     'togs':[True]}

vga_params = \
    {'dtype':[float],\
     'min':[-1],\
     'max':[1],\
     'mut_op':['uniform_mutation'],\
     'mut_rate':[0.05],\
     'xov_op':['twopt']}

pga_params = \
    {'dtype':[float],\
     'min':[-1],\
     'max':[1],\
     'map_fxn':[1,2,3],\
     'mut_op':['uniform_mutation'],\
     'mut_rate':[0.015],\
     'xov_op':['twopt', 'onept', 'homo'],\
     'length':[-5],\
     'n_noncoding_chars':[0,1,3,5]}

bsga_params = \
    {'dtype':[float],\
     'min':[-1],\
     'max':[1],\
     'mut_op':['flipbit'],\
     'xov_op':['twopt'],\
     'gene_size':[16]}

'''

n_runs, n_gens = 200, 200
xov_rate = 0.85

inlst = []
outlst = []

if sys.argv[1] == 1: #VGA

    for inp, out in (inlst, outlst):
        for product(*[])
    mut_rate = 0.2
    xov_op, mut_op = 'twopt', 'uniform_mutation'
    ln = f'python run_log_cmd {inp} {out} vga '+\
         f'--n_runs {n_runs} --n_gens {n_gens} '+\
         f'--mut_rate {mut_rate} --xov_rate {xov_rate} '+\
         f'--mut_op {mut_op} --xov_op {xov_op} '+\
         f'--'
    run = subprocess.run(\
            )

run = subprocess.run(\
    'python run_log_cmd.py datasets/medicare results/test vga '+\
    '--n_runs 2 --n_gens 5 --togs', shell=True, \
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
