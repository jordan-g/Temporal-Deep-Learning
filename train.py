import numpy as np
import network as net
import sys

if len(sys.argv) > 1:
    folder_prefix = str(sys.argv[1])
else:
    folder_prefix = None

if len(sys.argv) > 2:
    net.n_units   = [ int(i) for i in sys.argv[2].split(" ") ]
    net.W_std     = [ float(i) for i in sys.argv[3].split(" ") ]
    net.Z_std     = [ float(i) for i in sys.argv[4].split(" ") ]
    net.Y_std     = [ float(i) for i in sys.argv[5].split(" ") ]
    net.f_etas    = [ float(i) for i in sys.argv[6].split(" ") ]
    net.b_etas    = [ float(i) for i in sys.argv[7].split(" ") ]
    net.r_etas    = [ float(i) for i in sys.argv[8].split(" ") ]
    net.n_epochs  = int(sys.argv[9])

net.train(folder_prefix=folder_prefix)