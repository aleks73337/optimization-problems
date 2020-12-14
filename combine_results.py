import os

results_folder = "results"
result_fpath = os.path.join(results_folder, 'result.txt')
with open(result_fpath, 'w+') as res_file:
    for fname in os.listdir(results_folder):
        fpath = os.path.join(results_folder, fname)
        graph_name = fname.split('.')[0]
        with open(fpath, 'r') as f:
            data = f.readlines()
        res_file.write(f'Graph name: {graph_name}\n')
        for line in data[0:2]:
            res_file.write(line)
        res_file.write('\n')