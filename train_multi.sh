#!/bin/bash

for n_layers in 3
do
    for n_units in 784,500,10
    do
        for W_range in 0,0.05,0.01
        do
            for Z_range in 0,0.01
            do
                for Y_range in 0,0.1
                do
                    for f_etas in 0,0.1,0.0001 0,0.05,0.0001 0,0.5,0.0001 0,0.1,0.0005 0,0.1,0.001 0,0.1,0.00005
                    do
                        for r_etas in 0.0,0.0
                        do
                            for b_etas in 0,0.0
                            do
                                for desired_u in 0.05
                                do
                                    for hard_m in 0.1 0.01 0.05 0.5
                                    do
                                        for hard_v in 0.1 0.01 0.05 0.5
                                        do
                                            sbatch train.sh "BatchTesting" $n_layers $n_units $W_range $Z_range $Y_range $f_etas $r_etas $b_etas $desired_u $hard_m $hard_v
                                        done
                                    done
                                done
                            done
                        done 
                    done
                done
            done
        done
    done
done

