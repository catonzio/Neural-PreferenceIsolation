from files.functions.results_rocs_maker import *

base_path = r"/home/catonz/Neural-PreferenceIsolation/results/pif_benchmark/test_22-07-16-51-58"
root_path = r"/home/catonz/Neural-PreferenceIsolation"

make_rocs_barplot(base_path=base_path, towrite=True)
make_scores_rocs_plots(root_path, base_path, towrite=True)
