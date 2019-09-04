#! /usr/bin/env python3

""" Plots convergence histories.
    Use `python3 plotconv.py --help` to see all available options.
"""

import sys
import argparse
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator

def setAxisParams(ax, baseLineWidth):
    """ Sets line style and grid lines for a given pyplot axis."""
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(which='major', axis='x', lw=0.5*baseLineWidth, linestyle='-', color='0.75')
    ax.grid(which='minor', axis='x', lw=0.5*baseLineWidth, linestyle=':', color='0.75')
    ax.grid(which='major', axis='y', lw=0.5*baseLineWidth, linestyle='-', color='0.75')
    ax.grid(which='minor', axis='y', lw=0.5*baseLineWidth, linestyle=':', color='0.75')

def plotquantity(filelist, quantname, numits, labellist, opts, imageformatstring):
    plt.close()
    markdivisor = 20
    for i in range(len(filelist)):
        filename = filelist[i]
        data = np.genfromtxt(filename)
        numsteps = data.shape[0]
        opts['markinterval'] = int(numsteps/markdivisor)

        # number of points to plot
        pltdatalen = len(data[:,0])-numits

        plt.xlabel("Asynchronous sweeps", fontsize="medium")
        if quantname == "residual":
            plt.plot(data[numits:,0], np.log10(data[numits:,3]), \
                    lw=opts['linewidth'], ls=opts['linetype'][i], color=opts['colorlist'][i], \
                    marker=opts['marklist'][i], ms=opts['marksize'], \
                    mew=opts['markedgewidth'], \
                    markevery=list(range(0,pltdatalen,opts['markinterval'])), \
                    label=labellist[i])
            plt.ylabel("Log vector 1-norm of ILU fixed-point residual", fontsize="medium")
        elif quantname == "lowerdiff":
            plt.plot(data[numits:,0], np.log10(data[numits:,1]), \
                    lw=opts['linewidth'], ls=opts['linetype'][i], color=opts['colorlist'][i], \
                    marker=opts['marklist'][i], ms=opts['marksize'], \
                    mew=opts['markedgewidth'], \
                    markevery=list(range(0,pltdatalen,opts['markinterval'])), \
                    label=labellist[i])
            plt.ylabel("Log vector max-norm of error in L matrix", fontsize="medium")
        elif quantname == "upperdiff":
            plt.plot(data[numits:,0], np.log10(data[numits:,2]), \
                    lw=opts['linewidth'], ls=opts['linetype'][i], color=opts['colorlist'][i], \
                    marker=opts['marklist'][i], ms=opts['marksize'], \
                    mew=opts['markedgewidth'], \
                    markevery=list(range(0,pltdatalen,opts['markinterval'])), \
                    label=labellist[i])
            plt.ylabel("Log vector max-norm of error in U matrix", fontsize="medium")

        ax = plt.axes()
        setAxisParams(ax,opts['linewidth'])
        plt.legend(loc="best", fontsize="medium")

    plt.savefig(filename.split('/')[-1].split('.')[0]+"-"+quantname+"." + imageformatstring, dpi=200)

if __name__ == "__main__":

    opts = { \
            "marklist" : ['.', 'x', '+', '^', 'v', '<', '>', 'd'],
            "colorlist" : ['k', 'b', 'r', 'g', 'c', 'm', 'orange', 'pink'],
            "linetype" : ['-', '--', '-.', ':', '--', '-.', '--',':'],
            "linewidth" : 0.75,
            "marksize" : 5,
            "markedgewidth" : 1 \
            }

    if(len(sys.argv) < 2):
        print("Error. Please provide input file name.")
        sys.exit(-1)

    parser = argparse.ArgumentParser(description="Plots async ILU history w.r.t. sweeps starting at a specified iteration")
    parser.add_argument("files", nargs='+')
    parser.add_argument("--labels", nargs='+', help = "Legend strings")
    parser.add_argument("--labelstr", default="", help = "Common suffix for legend strings")
    parser.add_argument("--start_iter", type=int, default=0, help = "Iteration to start plotting from")
    parser.add_argument("--format", default="eps", help = "Output format")
    args = parser.parse_args(sys.argv)

    plotquantity(args.files[1:], "residual", args.start_iter, args.labels, opts, args.format)
    plotquantity(args.files[1:], "lowerdiff", args.start_iter, args.labels, opts, args.format)
    plotquantity(args.files[1:], "upperdiff", args.start_iter, args.labels, opts, args.format)

