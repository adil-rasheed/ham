"""
temperature_profiles.py

Written by Sindre Stenen Blakseth, 2021.

Visualizing temperature profiles of different systems.
"""

########################################################################################################################
# Package imports.

import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib import cm
from matplotlib.ticker import LinearLocator

########################################################################################################################
# File imports.

import config

########################################################################################################################

def main():
    xs = np.linspace(0,1,1001, endpoint=True)
    X = np.linspace(0,1,1001, endpoint=True)
    t = np.linspace(0,5,5001, endpoint=True)
    X, t = np.meshgrid(X, t)

    T_exacts = [
        lambda x, t: t + 0.5*(x**2),
        lambda x, t: np.sqrt(t) + 10*(x**2)*(x-1)*(x+2),
        lambda x, t: 2*np.sqrt(t) + 7*(x**2)*(x-1)*(x+2),
        lambda x, t: 2*(x**2) - (t**2)*x*(x-1),
        lambda x, t: np.sin(2*np.pi*x)*np.exp(-t),
        lambda x, t: -2*(x**3)*(x-1)/(t+0.5),
        lambda x, t: -(x**3)*(x-1)/(t+0.1),
        lambda x, t: 2 + (x-1)*np.tanh(x/(t+0.1)),
        lambda x, t: np.sin(2*np.pi*t) + np.sin(2*np.pi*x),
        lambda x, t: 1 + np.sin(2*np.pi*t) * np.cos(2*np.pi*x),
        lambda x, t: 1 + np.sin(3*np.pi*t) * np.cos(4*np.pi*x),
        lambda x, t: 1 + np.sin(2*np.pi*x*(t**2)),
        lambda x, t: 5 + x*(x-1)/(t+0.1) + 0.1*t*np.sin(2*np.pi*x),
        lambda x, t: 1 + np.sin(5*x*t)*np.exp(-0.2*x*t),
        lambda x, t: 5*t*(x**2)*np.sin(10*np.pi*t) + np.sin(2*np.pi*x)/(t + 0.2),
        lambda x, t: 1 + t/(1+((x-0.5)**2)),
        lambda x, t: 1 + t*np.exp(-1000*(x-0.5)**2)
    ]
    timess   = [
        [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
        [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
        [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
        [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
        [0, 0.1, 0.2, 0.3, 0.5, 1.0],
        [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
        [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
        [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
        [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
        [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
        [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
        [0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0],
        [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
        [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
        [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 3.0, 3.003, 3.01],
        [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
        [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    ]

    for i in range(len(T_exacts)):
        plt.figure()
        for time in timess[i]:
            plt.plot(xs, T_exacts[i](xs, time), label=str(time))
            plt.grid()
        plt.legend()
        plt.xlabel(r'$x$ (m)')
        plt.ylabel(r'$T$ (K)')
        plt.savefig(os.path.join(config.results_dir, 'system' + str(i+1) + '_2D.pdf'), bbox_inches='tight')

    for i in range(len(T_exacts)):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, t, T_exacts[i](X,t), cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        #ax.zaxis.set_major_locator(LinearLocator(10))
        # A StrMethodFormatter is used automatically
        #ax.zaxis.set_major_formatter('{x:.02f}')
        ax.set_xlabel(r'$x$ (m)')
        ax.set_ylabel(r'$t$ (s)')
        ax.set_zlabel(r'$T$ (K)')

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.savefig(os.path.join(config.results_dir, 'system' + str(i+1) + '_3D.pdf'), bbox_inches='tight')


########################################################################################################################

if __name__ == "__main__":
    main()

########################################################################################################################