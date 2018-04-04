import numpy as np
import os
import subprocess

def generate_bash_file(filename, n_units, W_std, Z_std, Y_std, f_etas, b_etas, r_etas, n_epochs):
    new_filename = "new_" + filename

    if os.path.isfile(new_filename):
        os.remove(new_filename)

    with open(filename) as script_file:
        with open(new_filename, "w+") as output_file:
            for num, line in enumerate(script_file, 1):
                if "python3 train.py" in line:
                    n_units_string = " ".join([ str(i) for i in n_units ])
                    W_std_string   = " ".join([ str(i) for i in W_std ])
                    Z_std_string   = " ".join([ str(i) for i in Z_std ])
                    Y_std_string   = " ".join([ str(i) for i in Y_std ])
                    f_etas_string  = " ".join([ str(i) for i in f_etas ])
                    b_etas_string  = " ".join([ str(i) for i in b_etas ])
                    r_etas_string  = " ".join([ str(i) for i in r_etas ])
                    output_file.write("python3 train.py $SCRATCH '{}' '{}' '{}' '{}' '{}' '{}' '{}' {}".format(n_units_string, W_std_string, Z_std_string, Y_std_string, f_etas_string, b_etas_string, r_etas_string, n_epochs))
                else:
                    output_file.write(line)
    return new_filename

def submit_bash_file(filename):
    subprocess.check_call("qsub " + filename, shell=True)

if __name__ == "__main__":
    filename = "grid_search.sh"

    n_units  = [784, 500, 300, 10]
    W_std    = [0, 0.1, 0.1, 0.1]
    Z_std    = [0, 0.1, 0.1]
    Y_std    = [0, 1.0, 1.0]
    b_etas   = [0, 0.01, 0.01]
    r_etas   = [0, 0.01, 0.01]
    n_epochs = 1

    f_eta_options = [0.01, 0.1, 1.0]
    for i in f_eta_options:
        for j in f_eta_options:
            for k in f_eta_options:
                f_etas = [0, i, j, k]
                new_filename = generate_bash_file(filename, n_units, W_std, Z_std, Y_std, f_etas, b_etas, r_etas, n_epochs)
                submit_bash_file(new_filename)
