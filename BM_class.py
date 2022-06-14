#%%
import numpy as np
import pandas as pd
import numpy.random as r
import scipy.stats as stats
import matplotlib.pyplot as plt


class BrownianMotion:


    def __init__(self, timepoints = 100, drift = 1, sigma = 1, T = 1, n_paths = 1, type = "BM"):

        if type == "BM":
            f = {}
            t = np.arange(0, T + T/timepoints, T/timepoints)
            for j in range(1, n_paths+1):
                nrand = r.standard_normal(timepoints)

                W = list(range(0,timepoints+1))
                W[0] = 0

                for i in range(1, timepoints+1):
                    W[i] = W[i-1] + sigma * np.sqrt(t[i] - t[i-1]) * nrand[i-1] + drift * (t[i] - t[i-1])

                f["BM{a}".format(a = j)] = W

            df = pd.DataFrame(f).set_index(t)    
            
            self.sim = df
            

        if type == "GBM":
            f = {}
            t = np.arange(0, T + T/timepoints , T/timepoints)
            for j in range(1, n_paths):
                nrand = r.standard_normal(timepoints)

                S = list(range(0,timepoints+1))
                S[0] = 1

                for i in range(1, timepoints+1):
                    S[i] = S[i-1] * np.exp(sigma*np.sqrt(t[i]-t[i-1])*nrand[i-1]+drift*(t[i]-t[i-1]))

                S = np.vstack(S).tolist()
                S = [item for sublist in S for item in sublist]

                f["BM{a}".format(a = j)] = W

            df = pd.DataFrame(f).set_index(t)

            self.sim = df
            

    def plot(self):
        plt.xlabel("t")
        plt.ylabel("W(t)")
        plt.title("{a} Brownian Motions paths".format(a = len(self.sim.columns)))
        if len(self.sim) > 15:
            plt.plot(self.sim)
        else:
            plt.plot(self.sim)

    def rel_hist(self, n_bins = 10):
        self.sim.iloc[-1].hist(bins = n_bins, weights=np.zeros_like(self.sim.iloc[-1]) + 1. / self.sim.iloc[-1].size)

    
x = BrownianMotion(100, -0.1, 0.25, 5, 1000)
x.plot()
    
# %%
