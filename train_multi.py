import subprocess
import itertools
import time

n_layers = 5
n_units = [784, 500, 500, 500, 10]
W_range_hidden_vals = [0.5 1.0]
W_range_output_vals = [0.01, 0.02, 0.03, 0.005]
Z_range_vals = [0.01]
Y_range_vals = [1.0, 5.0, 0.5]
f_etas_hidden_vals = [0.3, 0.2, 0.1]
f_etas_output_vals = [0.01, 0.02, 0.03]
r_etas_vals = [0]
b_etas_vals = [0]
desired_u = 0.05
# hard_m_vals = [0, 0.001, 0.01, 0.2]
# hard_v_vals = [0.001, 0.005, 0.01, 0.1, 0.5]

# W_range = [0, 0.1, 0.1, 0.01]
# f_etas = [0, 0.1, 0.1, 0.01]
# Y_range = [0, 10.0, 10.0]
Z_range = [0.01, 0.01, 0.01, 0.01]
b_etas = [0, 0, 0, 0]
r_etas = [0, 0, 0, 0]
hard_m_vals = [0.1]
hard_v_vals = [0.05]


# total_count = (len(W_range_vals)**(n_layers-1))*(len(Z_range_vals)**(n_layers-2))*(len(Y_range_vals)**(n_layers-2))*(len(f_etas_vals)**(n_layers-1))*(len(r_etas_vals)**(n_layers-2))*(len(b_etas_vals)**(n_layers-2))*len(hard_m_vals)*len(hard_v_vals)
# total_count = len(hard_m_vals)*len(hard_v_vals)
total_count = len(W_range_hidden_vals)*len(W_range_output_vals)*len(Y_range_vals)*len(f_etas_hidden_vals)*len(f_etas_output_vals)
count = 0
for W_range_hidden in W_range_hidden_vals:
    for W_range_output in W_range_output_vals:
#     for Z_range in itertools.product(Z_range_vals, repeat=n_layers-2):
        for Y_range in Y_range_vals:
            for f_etas_hidden in f_etas_hidden_vals:
                for f_etas_output in f_etas_output_vals:
#                 for r_etas in itertools.product(r_etas_vals, repeat=n_layers-2):
#                     for b_etas in itertools.product(b_etas_vals, repeat=n_layers-2):
                    for hard_m in hard_m_vals:
                        for hard_v in hard_v_vals:
                            n_units_string = ",".join([ str(i) for i in n_units ])
                            W_range_string = ",".join(["0"] + [ str(W_range_hidden) for i in range(n_layers-2) ] + [str(W_range_output)])
                            Z_range_string = ",".join(["0"] + [ str(i) for i in Z_range ])
                            Y_range_string = ",".join(["0"] + [ str(Y_range) for i in range(n_layers-2) ])
                            f_etas_string  = ",".join(["0"] + [ str(f_etas_hidden) for i in range(n_layers-2) ] + [str(f_etas_output)])
                            r_etas_string  = ",".join(["0"] + [ str(i) for i in r_etas ])
                            b_etas_string  = ",".join(["0"] + [ str(i) for i in b_etas ])
                            print("{}/{}. W_range: {}. Y_range: {}. Z_range: {}.f_etas: {}. r_etas: {}. b_etas: {}. hard_m: {}. hard_v: {}.".format(count+1, total_count, W_range_string, Y_range_string, Z_range_string, f_etas_string, r_etas_string, b_etas_string, hard_m, hard_v))
                            subprocess.call('sbatch train.sh "5LayerTesting" {} {} {} {} {} {} {} {} {} {} {}'.format(n_layers, n_units_string, W_range_string, Z_range_string, Y_range_string, f_etas_string, r_etas_string, b_etas_string, desired_u, hard_m, hard_v).split())
                            time.sleep(1)

                            count += 1