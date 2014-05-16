#!/usr/bin/env python

# run as: ./sedov_2d_cyl_in_cart.py castro_dir plotfle

import sys
import os
import shutil
import numpy as np
import pylab

def process(castro_dir, plotfile):

    run_dir = os.getcwd()

    # 1. make sure that the analysis tool is built
    build_dir = castro_dir + "/Diagnostics/Sedov/"
    os.chdir(build_dir)
    os.system("make programs=fsedov2d_cyl_in_cartcoords >& /dev/null")

    # find the executable
    for file in os.listdir(build_dir):
        if (os.path.isfile(file) and 
            file.startswith("fsedov2d_cyl_in_cartcoords") and
            file.endswith(".exe")):
            analysis_routine = file
            break

    print "analysis_routine = ", analysis_routine

    shutil.copy(analysis_routine, run_dir)

    os.chdir(run_dir)


    # 2. analyze the data
    
    
    # output the average profile
    os.system("./{} -p {} -s {}".format(analysis_routine, plotfile, "sedov_2d_cyl_in_cart.out"))


    analytic = castro_dir + "/Exec/Sedov/Verification/cylindrical_sedov.dat"
    analytic_data = np.loadtxt(analytic)

    data = np.loadtxt("sedov_2d_cyl_in_cart.out")


    # 3. make the plot
    pylab.subplot(221)

    pylab.plot(analytic_data[:,1], analytic_data[:,2])
    pylab.scatter(data[:,0], data[:,1], marker="+", color="r")

    pylab.xlabel("x")
    pylab.ylabel("density")
    pylab.xlim(0,0.4)


    pylab.subplot(222)

    pylab.plot(analytic_data[:,1], analytic_data[:,5])
    pylab.scatter(data[:,0], data[:,2], marker="+", color="r")

    pylab.xlabel("x")
    pylab.ylabel("velocity")
    pylab.xlim(0,0.4)


    pylab.subplot(223)

    pylab.plot(analytic_data[:,1], analytic_data[:,4])
    pylab.scatter(data[:,0], data[:,3], marker="+", color="r")

    pylab.xlabel("x")
    pylab.ylabel("pressure")
    pylab.xlim(0,0.4)



    pylab.subplot(224)

    pylab.plot(analytic_data[:,1], analytic_data[:,3])
    pylab.scatter(data[:,0], data[:,4], marker="+", color="r")

    pylab.xlabel("x")
    pylab.ylabel("internal energy")
    pylab.xlim(0,0.4)

    ax = pylab.gca()
    ax.set_yscale("log")

    pylab.tight_layout()

    pylab.savefig("sedov_2d_cyl_in_cart.png")



if __name__ == "__main__":

    castro_dir = str(sys.argv[1])
    plotfile = str(sys.argv[2])

    process(castro_dir, plotfile)

