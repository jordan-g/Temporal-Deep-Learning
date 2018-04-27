import numpy as np
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('directory', help='Job ID')
parser.add_argument("-n_epochs",type=int, help="Number of epochs",default=50)
parser.add_argument("-store",type=int, help="Frequency save",default=1000)
parser.add_argument("-n_layers",type=int, help="Number Layers",default=3)
parser.add_argument("-n_units", help="Number Units",type=lambda s: [int(item) for item in s.split(',')], default=[784, 500, 10])
parser.add_argument("-w_std", help="W Std",type=lambda s: [float(item) for item in s.split(',')], default=[0, 0.05, 0.01])
parser.add_argument("-z_std", help="Z std",type=lambda s: [float(item) for item in s.split(',')], default=[0, 0.01])
parser.add_argument("-y_std", help="Y std",type=lambda s: [float(item) for item in s.split(',')], default=[0, 0.005])
parser.add_argument("-f_etas", help="Forward Etas",type=lambda s: [float(item) for item in s.split(',')], default=[0, 0.5, 0.01])
parser.add_argument("-r_etas", help="Recurrent Etas",type=lambda s: [float(item) for item in s.split(',')], default=[0, 0.0])
parser.add_argument("-b_etas", help="Backward Etas",type=lambda s: [float(item) for item in s.split(',')], default=[0, 0.0])
parser.add_argument("-output_burst_prob",type=float, help="Output Burst Prob",default=0.2)
parser.add_argument("-min_Z",type=float, help="Min Z",default=0.1)
parser.add_argument("-u_range",type=float, help="U Range",default=2.)
parser.add_argument("-w_decay",type=float, help="Number Layers",default=0.)
parser.add_argument("-dynamic_plot",type=bool, help="Dynamic Plot",default=False)
parser.add_argument("-tensorB",type=bool, help="Tensorboard",default=True)
parser.add_argument("-use_comet",type=bool, help="Use Comet",default=False)
parser.add_argument("-info",type=str, help="Information",default="Normal")
parser.add_argument("-hard_m",type=float, help="Hard derivative mean",default=0.)
parser.add_argument("-hard_vl",type=float, help="Hard derivative variance Low",default=4.)
parser.add_argument("-hard_vh",type=float, help="Hard derivative variance High",default=4.)
parser.add_argument("-target_onsets", help="Onset times of target",type=lambda s: np.array([float(item) for item in s.split(',')]), default=np.array([2]))
parser.add_argument("-target_on_time", type=int, help="Number of timesteps to show the target",default=1)
parser.add_argument("-time_per_example", type=int, help="Number of timesteps to show each example",default=4)
args=parser.parse_args()


tmp = args.info
if tmp == "Normal":
	import network0 as net
	net.ref = 0
elif tmp == "Sigmoid":
	import network1 as net
	net.ref = 1
elif tmp == "Exp_deriv":
	import network2 as net
	net.ref = 2
elif tmp == "Sigmoid_Exp_deriv":
	import network3 as net
	net.ref = 3
elif tmp == "Exp_activation_deriv":
	import network4 as net
	net.ref = 4
elif tmp == "Sigmoid_Exp_activation_deriv":
	import network5 as net
	net.ref = 5
elif tmp == "Sigmoid_Fix_Recur":
	import network6 as net
	net.ref = 6
elif tmp == "Sigmoid_Exp_activation_deriv_Fix_Recur":
	import network7 as net
	net.ref = 7
elif tmp == "Sigmoid_Learn_Recur_New_Cost":
	import network8 as net
	net.ref = 8
elif tmp == "Sigmoid_Learn_Recur_New_Cost_c_stats":
	import network9 as net
	net.ref = 9
elif tmp == "Baseline_Fixed_RFW":
	import network10 as net
	net.ref = 10
elif tmp == "Baseline_Fixed_RFW_Sigmoid":
	import network11 as net
	net.ref = 11
elif tmp == "Baseline_Fixed_RFW_Sigmoid_Fix_Recur":
	import network12 as net
	net.ref = 12
elif tmp == "Baseline_Fixed_RFW_Sigmoid_Learn_Recur":
	import network13 as net
	net.ref = 13
elif tmp == "Baseline_Fixed_RFW_Sigmoid_Learn_Recur_Exp_deriv":
	import network14 as net
	net.ref = 14
elif tmp == "Baseline_Fixed_RFW_Sigmoid_Learn_Recur_Exp_activation_deriv":
	import network15 as net
	net.ref = 15
elif tmp == "Baseline_Fixed_RFW_Sigmoid_Learn_Recur_Sigmoid_activation_Hard_deriv":
	import network16 as net
	net.ref = 16
elif tmp == "Baseline_Fixed_RFW_Sigmoid_Learn_Recur_SoftPlus_activation_Hard_deriv":
	import network17 as net
	net.ref = 17
elif tmp == "network25":
	import network25 as net
	net.ref = 25
elif tmp == "network26":
	import network26 as net
	net.ref = 26
elif tmp == "network27":
	import network27 as net
	net.ref = 27
elif tmp == "network28":
	import network28 as net
	net.ref = 28
elif tmp == "network29":
	import network29 as net
	net.ref = 29
else:
	raise ValueError("Unknown parameter")

folder_prefix = args.directory
net.n_epochs  = args.n_epochs
net.store     = args.store
net.n_layers  = args.n_layers

net.n_units   = args.n_units
net.W_std     = args.w_std
net.Z_std     = args.z_std
net.Y_std     = args.y_std
net.f_etas    = args.f_etas
net.r_etas    = args.r_etas
net.b_etas    = args.b_etas

net.output_burst_prob = args.output_burst_prob
net.min_Z             = args.min_Z 
net.u_range           = args.u_range
net.W_decay           = args.w_decay
net.dynamic_plot      = args.dynamic_plot
net.TensorB           = args.tensorB
net.use_comet         = args.use_comet
net.info              = args.info
net.hard_m            = args.hard_m 
net.hard_vl           = args.hard_vl
net.hard_vh           = args.hard_vh
net.target_onsets     = args.target_onsets
net.target_on_time    = args.target_on_time
net.time_per_example  = args.time_per_example

net.train(folder_prefix=folder_prefix)