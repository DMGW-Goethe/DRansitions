import numpy as np
from scipy.integrate import solve_ivp

class BetaFunctions:
    '''
    This class contains the beta functions for the gauge singlet-extended SM (xSM).
    It is used to run the 4D input parameters from the scale mu = M_Z to the EFT matching scale mu_4
    '''
    def __init__(self,lambdaMix,lambdaS,inputParams_MZ_4D):
        '''
        Value of scalar singlet couplings at mu = M_z
        '''
        self.lambdaMix = lambdaMix
        self.lambdaS = lambdaS
        '''
        Z mass and number of fermion generations
        '''
        self.MZ = 91.1876
        self.Nf = 3
        '''
        Standard Model (SM) input parameters at scale mu = M_Z.
        '''
        self.g, self.gPrime, self.yt, self.mPhiSq, self.mSSq, self.lambdaPhi, self.gsSq = inputParams_MZ_4D

    def BetaFuncs(self,mu,X):
        '''
        Beta Functions of the gauge singlet extended SM
        Input: scale mu, X = [gSq, gPrimeSq, ytSq, mPhiSq, lambdaPhi, mSSq, lambdaS, lambdaMix]
        Output: Y = d/d(mu) X
        '''
        gSq, gPrimeSq, ytSq, mPhiSq, lambdaPhi, mSSq, lambdaS, lambdaMix, gsSq = X

        dgsSq = -7 * gsSq ** 2 / (8 * np.pi ** 2 * mu)
        dgSq = -gSq**2 * (43/6 - 4/3 * self.Nf) /(8*np.pi**2*mu)
        dgPrimeSq = gPrimeSq**2 * (1/6 + 20/9 * self.Nf) /(8*np.pi**2*mu)
        dytSq = (9/2 * ytSq**2 - 8 * gsSq * ytSq - 9/4 * gSq * ytSq - 17/12 * gPrimeSq * ytSq) /(8*np.pi**2*mu)
        dmPhiSq = (1/2 * lambdaMix * mSSq + mPhiSq * (-9/4 * gSq - 3/4 * gPrimeSq + 6 * lambdaPhi + 3 * ytSq) )/(8*np.pi**2*mu)
        dlambdaPhi = (12 * lambdaPhi**2 + 1/4 * lambdaMix**2 + 9/16 * gSq**2 + 3/8 * gSq * gPrimeSq + 3/16 * gPrimeSq**2 - 3 * ytSq**2 - lambdaPhi *  3/2 * (3 * gSq + gPrimeSq) + 2 * lambdaPhi * 3 * ytSq)/(8*np.pi**2*mu)
        dmSSq = (3 * lambdaS * mSSq + 2 * lambdaMix * mPhiSq) /(8*np.pi**2*mu)
        dlambdaS = (lambdaMix**2 + 9 * lambdaS**2)/(8*np.pi**2*mu)
        dlambdaMix = lambdaMix * (-9/4 * gSq - 3/4 * gPrimeSq + 3 * ytSq + 2 * lambdaMix + 6 * lambdaPhi + 3*lambdaS)/(8*np.pi**2*mu)
        return dgSq,dgPrimeSq,dytSq,dmPhiSq,dlambdaPhi,dmSSq,dlambdaS,dlambdaMix,dgsSq

    def solveBetaFuncs(self,muBar):
        '''
        Evolve set of input variables X0 from scale mu = mZ to the matching scale of the EFT mu_4 = muBar.
        Output: Values at muBar.
        '''
        X0 = self.g**2, self.gPrime**2, self.yt**2, self.mPhiSq, self.lambdaPhi, self.mSSq, self.lambdaS, self.lambdaMix, self.gsSq
        sol = solve_ivp(self.BetaFuncs, y0=X0, t_span=[self.MZ,muBar], t_eval=np.linspace(self.MZ,muBar,10000),atol=1e-10,rtol=1e-10)
        return sol.t[-1], sol.y[:,-1]

    def solveBetaFuncs2(self, muBar):
        '''
        Evolve set of input variables X0 from scale mu = mZ to the matching scale of the EFT mu = muBar.
        Output: not only values at mu = muBar, but complete evolution from mu=M_Z to mu=muBar.
        '''
        X0 = self.g**2, self.gPrime**2, self.yt**2, self.mPhiSq, self.lambdaPhi, self.mSSq, self.lambdaS, self.lambdaMix, self.gsSq
        #sol = solve_ivp(self.BetaFuncs, y0=X0, t_span=[self.MZ, muBar], t_eval = np.linspace(self.MZ,muBar,10000))
        t = np.logspace(np.log10(self.MZ),np.log10(muBar),10000)
        sol = solve_ivp(self.BetaFuncs, y0=X0, t_span = [t[0],t[-1]],t_eval = t)
        return sol.t, sol.y

