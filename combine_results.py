import os
import glob

directories = glob.glob("Tensorboard/*/") + glob.glob("/scratch/jordang/*/")

with open("combined_results.csv", "w+") as file:
    line = "n_layers,n_units,f_etas,b_etas,r_etas,W_range,Z_range,Y_range,n_epochs,n_examples,n_test_examples,validation,output_burst_prob,desired_u,hard_m,hard_v,test_error,test_cost,train_error,train_cost\n"
    file.write(line)

    for directory in directories:
        filename = os.path.join(directory, "results.csv")
        if os.path.exists(filename):
            with open(filename, "r") as f:
                lines = f.readlines()
                if len(lines) > 1:
                    for i in range(1, len(lines)):
                        line = lines[i]

                        file.write(line)
