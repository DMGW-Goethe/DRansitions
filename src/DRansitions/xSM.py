import numpy as np
from GenericPotential3D import * 
from mpmath import *


class xSM(GenericPotential3D):
    '''
    This class computes the 3D effective potential for the gauge singlet-extended SM at the ultrasoft scale g^2 T up to two-loops.
    The necessary functions to compute Veff are contained in the class GenericPotential_3D
    The function Veff_3D can directly be used as an input for CosmoTransitions.
    '''
    def __init__(self,mh2,lambdaMix,lambdaS,inputParams_MZ_4D,
            mu3Bar: float, mu4D: float,
            LoopOrderParameters,
            LoopOrderPotential: int):
        super().__init__(LoopOrderPotential, mu4D)
        self.c = -0.348723
        '''
        BSM input at scale mu = M_Z
        '''
        self.mh2 = mh2
        self.lambdaMix = lambdaMix
        self.lambdaS = lambdaS

        self.thetaBar = 0

        '''
        Number of Higgs doublets, fermion generations, strong gauge coupling (neglect running)
        '''
        self.Nd = 1
        self.Nf = 3

        '''
        Specify the desired loop order (1,2) at which effective parameters and the effective potential are computed
        '''
        self.LoopOrderParameters = LoopOrderParameters
        self.LoopOrderPotential = LoopOrderPotential

        '''
        Input parameters at the scale mu = M_Z
        '''
        #g, gPrime, yt, mPhiSq, mSSq, lambdaPhi, gsSq = inputParams_MZ_4D
        self.inputParams_MZ_4D = inputParams_MZ_4D

        '''
        3D and 4D RG scale of EFT as fraction of temperature
        '''
        self.mu3Bar = mu3Bar
        self.mu4D = mu4D

        '''
        Fix dimension
        '''
        self.dim = 3

        '''
        Fix loop counting
        '''
        self.N0LO, self.N1LO = 0, 0
        if self.LoopOrderParameters >= 1:
            self.N0LO = 1
        if self.LoopOrderParameters >= 2:
            self.N1LO = 1


    def Get4Dparams(self, T):
        '''
        From the 4D input parameters at the scale mu = M_Z, this function runs the couplings
        to the matching scale muBar = 4 * pi * exp(-gamma) * mu4D * T.
        '''
        muBar = 4 * np.pi * np.exp(-np.euler_gamma) * self.mu4D * T
        g2Sq, g1Sq, yt1Sq, mPhiSq, lambdaPhi, mSSq, lambdaS, lambdaMix, g3Sq = BetaFunctions(self.lambdaMix, self.lambdaS, self.inputParams_MZ_4D).solveBetaFuncs(muBar)[1]

        return g2Sq, g1Sq, yt1Sq, mPhiSq, lambdaPhi, mSSq, lambdaS, lambdaMix, g3Sq

    def Soft_params(self, T):
        '''
        From a set of 4D parameters at the matching scale mu_4, this function returns effective parameters at the
        soft scale ~g T. The desired precision can be chosen by adjusting LoopOrderParameters when initializing the class.
        '''
        g2Sq, g1Sq, yt1Sq, mPhiSq, lambdaPhi, mSSq, lambdaS, lambdaMix, g3Sq = self.Get4Dparams(T)
        # TODO Why sqrt here. The soft matching parameters should all depend on the squared paremeters
        # could be better put out by DRalgo mathematica file
        g1 = np.sqrt(g1Sq)
        g2 = np.sqrt(g2Sq)
        g3 = np.sqrt(g3Sq)
        yt1 = np.sqrt(yt1Sq)

        mu3Bar = self.mu3Bar * T
        mu4D = 4 * np.pi * np.exp(-np.euler_gamma) * self.mu4D * T

        Lb = 2 * np.log(mu4D / T) - 2 * (np.log(4 * np.pi) - np.euler_gamma)  # check
        Lf = Lb + 4 * np.log(2)  # check
        Pi = np.pi
        Nf = self.Nf
        EulerGamma = np.euler_gamma
        Glaisher = float(mp.glaisher)

        g13Sq = g1 ** 2 * T - (g1 ** 4 * Lb * T) / (96. * Pi ** 2) - (5 * g1 ** 4 * Lf * Nf * T) / (36. * Pi ** 2)
        g13d = csqrt(g13Sq)
        g23Sq = g2 ** 2 * T + (g2 ** 4 * T) / (24. * Pi ** 2) + (43 * g2 ** 4 * Lb * T) / (96. * Pi ** 2) - (
                    g2 ** 4 * Lf * Nf * T) / (12. * Pi ** 2)
        g23d = csqrt(g23Sq)
        g33Sq = g3 ** 2 * T + (g3 ** 4 * T) / (16. * Pi ** 2) + (11 * g3 ** 4 * Lb * T) / (16. * Pi ** 2) - (
                    g3 ** 4 * Lf * Nf * T) / (12. * Pi ** 2)
        g33d = csqrt(g33Sq)

        lambdaMix3d = lambdaMix * T + (3 * g1 ** 2 * lambdaMix * Lb * T) / (64. * Pi ** 2) + (
                    9 * g2 ** 2 * lambdaMix * Lb * T) / (64. * Pi ** 2) - (lambdaMix ** 2 * Lb * T) / (
                                  8. * Pi ** 2) - (3 * lambdaMix * lambdaPhi * Lb * T) / (8. * Pi ** 2) - (
                                  3 * lambdaMix * lambdaS * Lb * T) / (16. * Pi ** 2) - (
                                  3 * lambdaMix * Lf * T * yt1 ** 2) / (16. * Pi ** 2)
        lambdaPhi3d = lambdaPhi * T + (g1 ** 4 * T) / (128. * Pi ** 2) + (g1 ** 2 * g2 ** 2 * T) / (
                    64. * Pi ** 2) + (3 * g2 ** 4 * T) / (128. * Pi ** 2) - (3 * g1 ** 4 * Lb * T) / (
                                  256. * Pi ** 2) - (3 * g1 ** 2 * g2 ** 2 * Lb * T) / (128. * Pi ** 2) - (
                                  9 * g2 ** 4 * Lb * T) / (256. * Pi ** 2) - (lambdaMix ** 2 * Lb * T) / (
                                  64. * Pi ** 2) + (3 * g1 ** 2 * lambdaPhi * Lb * T) / (32. * Pi ** 2) + (
                                  9 * g2 ** 2 * lambdaPhi * Lb * T) / (32. * Pi ** 2) - (
                                  3 * lambdaPhi ** 2 * Lb * T) / (4. * Pi ** 2) - (
                                  3 * lambdaPhi * Lf * T * yt1 ** 2) / (8. * Pi ** 2) + (3 * Lf * T * yt1 ** 4) / (
                                  16. * Pi ** 2)
        lambdaS3d = lambdaS * T - (lambdaMix ** 2 * Lb * T) / (16. * Pi ** 2) - (9 * lambdaS ** 2 * Lb * T) / (
                    16. * Pi ** 2)

        lambdaVLL1 = -0.25 * (g2 ** 2 * g3 ** 2 * Nf * T) / Pi ** 2
        lambdaVLL2 = (9 * g3 ** 4 * T) / (4. * Pi ** 2) - (g3 ** 4 * Nf * T) / (2. * Pi ** 2)
        lambdaVLL3 = (17 * g2 ** 4 * T) / (8. * Pi ** 2) - (g2 ** 4 * Nf * T) / (2. * Pi ** 2)
        lambdaVLL4 = 0
        lambdaVLL5 = 0
        lambdaVLL6 = -1 / 12. * (g1 * g3 ** 3 * Nf * T) / Pi ** 2
        lambdaVLL7 = (-11 * g1 ** 2 * g3 ** 2 * Nf * T) / (36. * Pi ** 2)
        lambdaVLL8 = (g1 ** 2 * g2 ** 2 * T) / (8. * Pi ** 2) - (g1 ** 2 * g2 ** 2 * Nf * T) / (6. * Pi ** 2)
        lambdaVLL9 = (g1 ** 4 * T) / (8. * Pi ** 2) - (95 * g1 ** 4 * Nf * T) / (54. * Pi ** 2)
        lambdaVL1 = (g2 ** 2 * lambdaMix * T) / (8. * Pi ** 2)
        lambdaVL2 = -0.25 * (g3 ** 2 * T * yt1 ** 2) / Pi ** 2
        lambdaVL3 = (g1 ** 2 * lambdaMix * T) / (8. * Pi ** 2)
        lambdaVL4 = (g2 ** 2 * T) / 2. + (g1 ** 2 * g2 ** 2 * T) / (64. * Pi ** 2) + (17 * g2 ** 4 * T) / (
                    64. * Pi ** 2) + (3 * g2 ** 2 * lambdaPhi * T) / (8. * Pi ** 2) + (43 * g2 ** 4 * Lb * T) / (
                                192. * Pi ** 2) + (g2 ** 4 * Nf * T) / (24. * Pi ** 2) - (g2 ** 4 * Lf * Nf * T) / (
                                24. * Pi ** 2) - (3 * g2 ** 2 * T * yt1 ** 2) / (16. * Pi ** 2)
        lambdaVL5 = (g1 * g2 * T) / 2. + (g1 ** 3 * g2 * T) / (96. * Pi ** 2) - (g1 * g2 ** 3 * T) / (
                    32. * Pi ** 2) + (g1 * g2 * lambdaPhi * T) / (8. * Pi ** 2) - (g1 ** 3 * g2 * Lb * T) / (
                                384. * Pi ** 2) + (43 * g1 * g2 ** 3 * Lb * T) / (384. * Pi ** 2) + (
                                5 * g1 ** 3 * g2 * Nf * T) / (144. * Pi ** 2) + (g1 * g2 ** 3 * Nf * T) / (
                                48. * Pi ** 2) - (5 * g1 ** 3 * g2 * Lf * Nf * T) / (144. * Pi ** 2) - (
                                g1 * g2 ** 3 * Lf * Nf * T) / (48. * Pi ** 2) + (g1 * g2 * T * yt1 ** 2) / (
                                16. * Pi ** 2)
        lambdaVL6 = (g1 ** 2 * T) / 2. + (g1 ** 4 * T) / (192. * Pi ** 2) + (3 * g1 ** 2 * g2 ** 2 * T) / (
                    64. * Pi ** 2) + (3 * g1 ** 2 * lambdaPhi * T) / (8. * Pi ** 2) - (g1 ** 4 * Lb * T) / (
                                192. * Pi ** 2) + (5 * g1 ** 4 * Nf * T) / (72. * Pi ** 2) - (
                                5 * g1 ** 4 * Lf * Nf * T) / (72. * Pi ** 2) - (17 * g1 ** 2 * T * yt1 ** 2) / (
                                48. * Pi ** 2)

        N0LO, N1LO = self.N0LO, self.N1LO

        musqSU2 = (g2 ** 2 * mPhiSq * N1LO) / (8. * Pi ** 2) + (5 * g2 ** 2 * N0LO * T ** 2) / 6. + (
                    g2 ** 2 * N0LO * Nf * T ** 2) / 3. + (g1 ** 2 * g2 ** 2 * N1LO * T ** 2) / (128. * Pi ** 2) + (
                              23 * g2 ** 4 * N1LO * T ** 2) / (128. * Pi ** 2) + (
                              g2 ** 2 * lambdaMix * N1LO * T ** 2) / (192. * Pi ** 2) + (
                              g2 ** 2 * lambdaPhi * N1LO * T ** 2) / (16. * Pi ** 2) + (
                              215 * g2 ** 4 * Lb * N1LO * T ** 2) / (576. * Pi ** 2) - (
                          g1 ** 2 * g2 ** 2 * N1LO * Nf * T ** 2) / (96. * Pi ** 2) + (
                              11 * g2 ** 4 * N1LO * Nf * T ** 2) / (288. * Pi ** 2) - (
                              g2 ** 2 * g3 ** 2 * N1LO * Nf * T ** 2) / (8. * Pi ** 2) + (
                              43 * g2 ** 4 * Lb * N1LO * Nf * T ** 2) / (288. * Pi ** 2) - (
                              5 * g2 ** 4 * Lf * N1LO * Nf * T ** 2) / (72. * Pi ** 2) + (
                              g2 ** 4 * N1LO * Nf ** 2 * T ** 2) / (36. * Pi ** 2) - (
                              g2 ** 4 * Lf * N1LO * Nf ** 2 * T ** 2) / (36. * Pi ** 2) - (
                              g2 ** 2 * N1LO * T ** 2 * yt1 ** 2) / (64. * Pi ** 2)
        musqSU3 = g3 ** 2 * N0LO * T ** 2 + (g3 ** 2 * N0LO * Nf * T ** 2) / 3. + (5 * g3 ** 4 * N1LO * T ** 2) / (
                    16. * Pi ** 2) + (11 * g3 ** 4 * Lb * N1LO * T ** 2) / (16. * Pi ** 2) - (
                              11 * g1 ** 2 * g3 ** 2 * N1LO * Nf * T ** 2) / (576. * Pi ** 2) - (
                              3 * g2 ** 2 * g3 ** 2 * N1LO * Nf * T ** 2) / (64. * Pi ** 2) + (
                              g3 ** 4 * N1LO * Nf * T ** 2) / (48. * Pi ** 2) + (
                              11 * g3 ** 4 * Lb * N1LO * Nf * T ** 2) / (48. * Pi ** 2) - (
                              g3 ** 4 * Lf * N1LO * Nf * T ** 2) / (12. * Pi ** 2) + (
                          g3 ** 4 * N1LO * Nf ** 2 * T ** 2) / (36. * Pi ** 2) - (
                              g3 ** 4 * Lf * N1LO * Nf ** 2 * T ** 2) / (36. * Pi ** 2) - (
                              g3 ** 2 * N1LO * T ** 2 * yt1 ** 2) / (16. * Pi ** 2)
        musqU1 = (g1 ** 2 * mPhiSq * N1LO) / (8. * Pi ** 2) + (g1 ** 2 * N0LO * T ** 2) / 6. + (
                    5 * g1 ** 2 * N0LO * Nf * T ** 2) / 9. + (5 * g1 ** 4 * N1LO * T ** 2) / (1152. * Pi ** 2) + (
                             3 * g1 ** 2 * g2 ** 2 * N1LO * T ** 2) / (128. * Pi ** 2) + (
                             g1 ** 2 * lambdaMix * N1LO * T ** 2) / (192. * Pi ** 2) + (
                             g1 ** 2 * lambdaPhi * N1LO * T ** 2) / (16. * Pi ** 2) - (
                             g1 ** 4 * Lb * N1LO * T ** 2) / (576. * Pi ** 2) - (
                             85 * g1 ** 4 * N1LO * Nf * T ** 2) / (864. * Pi ** 2) - (
                             g1 ** 2 * g2 ** 2 * N1LO * Nf * T ** 2) / (32. * Pi ** 2) - (
                         11 * g1 ** 2 * g3 ** 2 * N1LO * Nf * T ** 2) / (72. * Pi ** 2) - (
                             5 * g1 ** 4 * Lb * N1LO * Nf * T ** 2) / (864. * Pi ** 2) - (
                             5 * g1 ** 4 * Lf * N1LO * Nf * T ** 2) / (216. * Pi ** 2) + (
                             25 * g1 ** 4 * N1LO * Nf ** 2 * T ** 2) / (324. * Pi ** 2) - (
                             25 * g1 ** 4 * Lf * N1LO * Nf ** 2 * T ** 2) / (324. * Pi ** 2) - (
                             11 * g1 ** 2 * N1LO * T ** 2 * yt1 ** 2) / (192. * Pi ** 2)

        mPhiSq3d = mPhiSq * N0LO + (3 * g1 ** 2 * Lb * mPhiSq * N1LO) / (64. * Pi ** 2) + (
                    9 * g2 ** 2 * Lb * mPhiSq * N1LO) / (64. * Pi ** 2) - (3 * lambdaPhi * Lb * mPhiSq * N1LO) / (
                               8. * Pi ** 2) - (lambdaMix * Lb * mSSq * N1LO) / (32. * Pi ** 2) + (
                               g1 ** 2 * N0LO * T ** 2) / 16. + (3 * g2 ** 2 * N0LO * T ** 2) / 16. + (
                               lambdaMix * N0LO * T ** 2) / 24. + (lambdaPhi * N0LO * T ** 2) / 2. + (
                               g1 ** 4 * N1LO * T ** 2) / (4608. * Pi ** 2) - (
                               7 * EulerGamma * g1 ** 4 * N1LO * T ** 2) / (512. * Pi ** 2) - (
                               3 * g1 ** 2 * g2 ** 2 * N1LO * T ** 2) / (256. * Pi ** 2) - (
                               15 * EulerGamma * g1 ** 2 * g2 ** 2 * N1LO * T ** 2) / (256. * Pi ** 2) + (
                               167 * g2 ** 4 * N1LO * T ** 2) / (1536. * Pi ** 2) + (
                           81 * EulerGamma * g2 ** 4 * N1LO * T ** 2) / (512. * Pi ** 2) - (
                               EulerGamma * lambdaMix ** 2 * N1LO * T ** 2) / (64. * Pi ** 2) + (
                               g1 ** 2 * lambdaPhi * N1LO * T ** 2) / (64. * Pi ** 2) + (
                               3 * EulerGamma * g1 ** 2 * lambdaPhi * N1LO * T ** 2) / (32. * Pi ** 2) + (
                               3 * g2 ** 2 * lambdaPhi * N1LO * T ** 2) / (64. * Pi ** 2) + (
                               9 * EulerGamma * g2 ** 2 * lambdaPhi * N1LO * T ** 2) / (32. * Pi ** 2) - (
                           3 * EulerGamma * lambdaPhi ** 2 * N1LO * T ** 2) / (8. * Pi ** 2) + (
                               11 * g1 ** 4 * Lb * N1LO * T ** 2) / (1536. * Pi ** 2) + (
                               3 * g1 ** 2 * g2 ** 2 * Lb * N1LO * T ** 2) / (64. * Pi ** 2) - (
                               47 * g2 ** 4 * Lb * N1LO * T ** 2) / (512. * Pi ** 2) + (
                               g1 ** 2 * lambdaMix * Lb * N1LO * T ** 2) / (512. * Pi ** 2) + (
                               3 * g2 ** 2 * lambdaMix * Lb * N1LO * T ** 2) / (512. * Pi ** 2) + (
                               lambdaMix ** 2 * Lb * N1LO * T ** 2) / (384. * Pi ** 2) - (
                               3 * g1 ** 2 * lambdaPhi * Lb * N1LO * T ** 2) / (64. * Pi ** 2) - (
                               9 * g2 ** 2 * lambdaPhi * Lb * N1LO * T ** 2) / (64. * Pi ** 2) - (
                               lambdaMix * lambdaPhi * Lb * N1LO * T ** 2) / (64. * Pi ** 2) - (
                               lambdaMix * lambdaS * Lb * N1LO * T ** 2) / (128. * Pi ** 2) + (
                           5 * g1 ** 4 * N1LO * Nf * T ** 2) / (1728. * Pi ** 2) + (
                               g2 ** 4 * N1LO * Nf * T ** 2) / (192. * Pi ** 2) - (
                               5 * g1 ** 4 * Lb * N1LO * Nf * T ** 2) / (384. * Pi ** 2) - (
                               3 * g2 ** 4 * Lb * N1LO * Nf * T ** 2) / (128. * Pi ** 2) + (
                               5 * g1 ** 4 * Lf * N1LO * Nf * T ** 2) / (1152. * Pi ** 2) + (
                               g2 ** 4 * Lf * N1LO * Nf * T ** 2) / (128. * Pi ** 2) - (
                               3 * Lf * mPhiSq * N1LO * yt1 ** 2) / (16. * Pi ** 2) + (
                               N0LO * T ** 2 * yt1 ** 2) / 4. - (11 * g1 ** 2 * N1LO * T ** 2 * yt1 ** 2) / (
                               768. * Pi ** 2) - (3 * g2 ** 2 * N1LO * T ** 2 * yt1 ** 2) / (256. * Pi ** 2) - (
                               g3 ** 2 * N1LO * T ** 2 * yt1 ** 2) / (8. * Pi ** 2) + (
                               47 * g1 ** 2 * Lb * N1LO * T ** 2 * yt1 ** 2) / (4608. * Pi ** 2) + (
                               21 * g2 ** 2 * Lb * N1LO * T ** 2 * yt1 ** 2) / (512. * Pi ** 2) - (
                           g3 ** 2 * Lb * N1LO * T ** 2 * yt1 ** 2) / (24. * Pi ** 2) - (
                               9 * lambdaPhi * Lb * N1LO * T ** 2 * yt1 ** 2) / (64. * Pi ** 2) + (
                               55 * g1 ** 2 * Lf * N1LO * T ** 2 * yt1 ** 2) / (4608. * Pi ** 2) - (
                               3 * g2 ** 2 * Lf * N1LO * T ** 2 * yt1 ** 2) / (512. * Pi ** 2) + (
                               g3 ** 2 * Lf * N1LO * T ** 2 * yt1 ** 2) / (6. * Pi ** 2) - (
                               lambdaMix * Lf * N1LO * T ** 2 * yt1 ** 2) / (128. * Pi ** 2) - (
                               3 * lambdaPhi * Lf * N1LO * T ** 2 * yt1 ** 2) / (64. * Pi ** 2) + (
                           3 * Lb * N1LO * T ** 2 * yt1 ** 4) / (128. * Pi ** 2) + (
                               21 * g1 ** 4 * N1LO * T ** 2 * np.log(Glaisher)) / (128. * Pi ** 2) + (
                               45 * g1 ** 2 * g2 ** 2 * N1LO * T ** 2 * np.log(Glaisher)) / (64. * Pi ** 2) - (
                               243 * g2 ** 4 * N1LO * T ** 2 * np.log(Glaisher)) / (128. * Pi ** 2) + (
                               3 * lambdaMix ** 2 * N1LO * T ** 2 * np.log(Glaisher)) / (16. * Pi ** 2) - (
                               9 * g1 ** 2 * lambdaPhi * N1LO * T ** 2 * np.log(Glaisher)) / (8. * Pi ** 2) - (
                               27 * g2 ** 2 * lambdaPhi * N1LO * T ** 2 * np.log(Glaisher)) / (8. * Pi ** 2) + (
                               9 * lambdaPhi ** 2 * N1LO * T ** 2 * np.log(Glaisher)) / (2. * Pi ** 2) + (
                               5 * g13d ** 4 * N1LO * np.log(mu3Bar / mu4D)) / (256. * Pi ** 2) + (
                               9 * g13d ** 2 * g23d ** 2 * N1LO * np.log(mu3Bar / mu4D)) / (128. * Pi ** 2) - (
                           39 * g23d ** 4 * N1LO * np.log(mu3Bar / mu4D)) / (256. * Pi ** 2) + (
                               lambdaMix3d ** 2 * N1LO * np.log(mu3Bar / mu4D)) / (32. * Pi ** 2) - (
                               3 * g13d ** 2 * lambdaPhi3d * N1LO * np.log(mu3Bar / mu4D)) / (16. * Pi ** 2) - (
                               9 * g23d ** 2 * lambdaPhi3d * N1LO * np.log(mu3Bar / mu4D)) / (16. * Pi ** 2) + (
                               3 * lambdaPhi3d ** 2 * N1LO * np.log(mu3Bar / mu4D)) / (4. * Pi ** 2) - (
                               3 * g33d ** 2 * N1LO * lambdaVL2 * np.log(mu3Bar / mu4D)) / (2. * Pi ** 2) + (
                               N1LO * lambdaVL2 ** 2 * np.log(mu3Bar / mu4D)) / (4. * Pi ** 2) - (
                               3 * g23d ** 2 * N1LO * lambdaVL4 * np.log(mu3Bar / mu4D)) / (8. * Pi ** 2) + (
                               3 * N1LO * lambdaVL4 ** 2 * np.log(mu3Bar / mu4D)) / (32. * Pi ** 2) + (
                               3 * N1LO * lambdaVL5 ** 2 * np.log(mu3Bar / mu4D)) / (16. * Pi ** 2) + (
                               N1LO * lambdaVL6 ** 2 * np.log(mu3Bar / mu4D)) / (32. * Pi ** 2)

        mSSq3d = mSSq * N0LO - (lambdaMix * Lb * mPhiSq * N1LO) / (8. * Pi ** 2) - (
                    3 * lambdaS * Lb * mSSq * N1LO) / (16. * Pi ** 2) + (lambdaMix * N0LO * T ** 2) / 6. + (
                             lambdaS * N0LO * T ** 2) / 4. + (g1 ** 2 * lambdaMix * N1LO * T ** 2) / (
                             192. * Pi ** 2) + (EulerGamma * g1 ** 2 * lambdaMix * N1LO * T ** 2) / (
                             32. * Pi ** 2) + (g2 ** 2 * lambdaMix * N1LO * T ** 2) / (64. * Pi ** 2) + (
                             3 * EulerGamma * g2 ** 2 * lambdaMix * N1LO * T ** 2) / (32. * Pi ** 2) - (
                             EulerGamma * lambdaMix ** 2 * N1LO * T ** 2) / (16. * Pi ** 2) - (
                             3 * EulerGamma * lambdaS ** 2 * N1LO * T ** 2) / (16. * Pi ** 2) - (
                         3 * g1 ** 2 * lambdaMix * Lb * N1LO * T ** 2) / (128. * Pi ** 2) - (
                             9 * g2 ** 2 * lambdaMix * Lb * N1LO * T ** 2) / (128. * Pi ** 2) + (
                             5 * lambdaMix ** 2 * Lb * N1LO * T ** 2) / (192. * Pi ** 2) - (
                             lambdaMix * lambdaPhi * Lb * N1LO * T ** 2) / (16. * Pi ** 2) - (
                             lambdaMix * lambdaS * Lb * N1LO * T ** 2) / (32. * Pi ** 2) + (
                             3 * lambdaS ** 2 * Lb * N1LO * T ** 2) / (64. * Pi ** 2) - (
                         3 * lambdaMix * Lb * N1LO * T ** 2 * yt1 ** 2) / (64. * Pi ** 2) + (
                             lambdaMix * Lf * N1LO * T ** 2 * yt1 ** 2) / (64. * Pi ** 2) - (
                             3 * g1 ** 2 * lambdaMix * N1LO * T ** 2 * np.log(Glaisher)) / (8. * Pi ** 2) - (
                             9 * g2 ** 2 * lambdaMix * N1LO * T ** 2 * np.log(Glaisher)) / (8. * Pi ** 2) + (
                         3 * lambdaMix ** 2 * N1LO * T ** 2 * np.log(Glaisher)) / (4. * Pi ** 2) + (
                             9 * lambdaS ** 2 * N1LO * T ** 2 * np.log(Glaisher)) / (4. * Pi ** 2) - (
                             g13d ** 2 * lambdaMix3d * N1LO * np.log(mu3Bar / mu4D)) / (16. * Pi ** 2) - (
                             3 * g23d ** 2 * lambdaMix3d * N1LO * np.log(mu3Bar / mu4D)) / (16. * Pi ** 2) + (
                             lambdaMix3d ** 2 * N1LO * np.log(mu3Bar / mu4D)) / (8. * Pi ** 2) + (
                         3 * lambdaS3d ** 2 * N1LO * np.log(mu3Bar / mu4D)) / (8. * Pi ** 2) - (
                             3 * g23d ** 2 * N1LO * lambdaVL1 * np.log(mu3Bar / mu4D)) / (8. * Pi ** 2) + (
                             3 * N1LO * lambdaVL1 ** 2 * np.log(mu3Bar / mu4D)) / (32. * Pi ** 2) + (
                             N1LO * lambdaVL3 ** 2 * np.log(mu3Bar / mu4D)) / (32. * Pi ** 2)

        return lambdaMix3d, lambdaS3d, lambdaPhi3d, mSSq3d, mPhiSq3d, musqU1, musqSU2, musqSU3, g13Sq, g23Sq, g33Sq, lambdaVLL1, lambdaVLL2, lambdaVLL3, lambdaVLL4, lambdaVLL5, lambdaVLL6, lambdaVLL7, lambdaVLL8, lambdaVLL9, lambdaVL1, lambdaVL2, lambdaVL3, lambdaVL4, lambdaVL5, lambdaVL6

    def GetUltrasoftParams(self, T):
        '''
        From a set of 3D parameters at the soft scale ~gT, this function returns effective parameters at the
        ultrasoft scale ~g^2 T. The desired precision can be chosen by adjusting LoopOrderParameters when initializing the class.
        '''
        lambdaMix3d, lambdaS3d, lambdaPhi3d, mSSq3d, mPhiSq3d, musqU1, musqSU2, musqSU3, g13Sq, g23Sq, g33Sq, lambdaVLL1, lambdaVLL2, lambdaVLL3, lambdaVLL4, lambdaVLL5, lambdaVLL6, lambdaVLL7, lambdaVLL8, lambdaVLL9, lambdaVL1, lambdaVL2, lambdaVL3, lambdaVL4, lambdaVL5, lambdaVL6 = self.Soft_params(T)
        Pi = np.pi
        g23d = np.sqrt(g23Sq)

        mu3Bar = self.mu3Bar * T

        N0LO, N1LO = self.N0LO, self.N1LO

        lambdaMix3dUS = lambdaMix3d - (3 * lambdaVL1 * lambdaVL4) / (16. * csqrt(musqSU2) * Pi) - (
                    lambdaVL3 * lambdaVL6) / (16. * csqrt(musqU1) * Pi)
        lambdaPhi3dUS = lambdaPhi3d - (3 * lambdaVL4 ** 2) / (32. * csqrt(musqSU2) * Pi) - lambdaVL2 ** 2 / (
                    4. * csqrt(musqSU3) * Pi) - lambdaVL5 ** 2 / (
                                    8. * (csqrt(musqSU2) + csqrt(musqU1)) * Pi) - lambdaVL6 ** 2 / (
                                    32. * csqrt(musqU1) * Pi)
        lambdaS3dUS = lambdaS3d - (3 * lambdaVL1 ** 2) / (32. * csqrt(musqSU2) * Pi) - lambdaVL3 ** 2 / (
                    32. * csqrt(musqU1) * Pi)
        g13dUSSq = g13Sq
        g23dUSSq = g23Sq - g23d ** 4 / (24. * csqrt(musqSU2) * Pi)
        mPhiSq3dUS = mPhiSq3d * N0LO + (3 * g33Sq * lambdaVL2 * N1LO) / (8. * Pi ** 2) - (lambdaVL2 ** 2 * N1LO) / (
                    8. * Pi ** 2) + (3 * g23Sq * lambdaVL4 * N1LO) / (32. * Pi ** 2) - (
                                 3 * lambdaVL4 ** 2 * N1LO) / (64. * Pi ** 2) - (3 * lambdaVL5 ** 2 * N1LO) / (
                                 32. * Pi ** 2) - (lambdaVL6 ** 2 * N1LO) / (64. * Pi ** 2) + (
                                 5 * lambdaVL2 * lambdaVLL2 * N1LO) / (24. * Pi ** 2) + (
                                 5 * lambdaVL4 * lambdaVLL3 * N1LO) / (128. * Pi ** 2) + (
                                 lambdaVL6 * lambdaVLL9 * N1LO) / (128. * Pi ** 2) + (
                             3 * lambdaVL2 * lambdaVLL1 * csqrt(musqSU2) * N1LO) / (
                                 16. * csqrt(musqSU3) * Pi ** 2) + (
                                 3 * lambdaVL4 * lambdaVLL1 * csqrt(musqSU3) * N1LO) / (
                                 16. * csqrt(musqSU2) * Pi ** 2) + (
                                 3 * lambdaVL6 * lambdaVLL8 * csqrt(musqSU2) * N1LO) / (
                                 128. * csqrt(musqU1) * Pi ** 2) + (
                                 lambdaVL6 * lambdaVLL7 * csqrt(musqSU3) * N1LO) / (
                                 16. * csqrt(musqU1) * Pi ** 2) + (
                                 3 * lambdaVL4 * lambdaVLL8 * csqrt(musqU1) * N1LO) / (
                             128. * csqrt(musqSU2) * Pi ** 2) + (lambdaVL2 * lambdaVLL7 * csqrt(musqU1) * N1LO) / (
                                 16. * csqrt(musqSU3) * Pi ** 2) - (3 * lambdaVL4 * csqrt(musqSU2) * N0LO) / (
                                 8. * Pi) - (lambdaVL2 * csqrt(musqSU3) * N0LO) / Pi - (
                                 lambdaVL6 * csqrt(musqU1) * N0LO) / (
                             8. * Pi) - (3 * g23d ** 4 * N1LO * np.log(mu3Bar / (2. * csqrt(musqSU2)))) / (
                                 64. * Pi ** 2) + (
                                 3 * g23Sq * lambdaVL4 * N1LO * np.log(mu3Bar / (2. * csqrt(musqSU2)))) / (
                                 8. * Pi ** 2) - (
                                 3 * lambdaVL4 ** 2 * N1LO * np.log(mu3Bar / (2. * csqrt(musqSU2)))) / (
                                 32. * Pi ** 2) + (
                                 3 * g33Sq * lambdaVL2 * N1LO * np.log(mu3Bar / (2. * csqrt(musqSU3)))) / (
                                 2. * Pi ** 2) - (
                             lambdaVL2 ** 2 * N1LO * np.log(mu3Bar / (2. * csqrt(musqSU3)))) / (4. * Pi ** 2) - (
                                 3 * lambdaVL5 ** 2 * N1LO * np.log(mu3Bar / (csqrt(musqSU2) + csqrt(musqU1)))) / (
                                 16. * Pi ** 2) - (
                                 lambdaVL6 ** 2 * N1LO * np.log(mu3Bar / (2. * csqrt(musqU1)))) / (32. * Pi ** 2)

        mSSq3dUS = mSSq3d * N0LO + (3 * g23Sq * lambdaVL1 * N1LO) / (32. * Pi ** 2) - (
                    3 * lambdaVL1 ** 2 * N1LO) / (64. * Pi ** 2) - (lambdaVL3 ** 2 * N1LO) / (64. * Pi ** 2) + (
                               5 * lambdaVL1 * lambdaVLL3 * N1LO) / (128. * Pi ** 2) + (
                               lambdaVL3 * lambdaVLL9 * N1LO) / (128. * Pi ** 2) + (
                               3 * lambdaVL1 * lambdaVLL1 * csqrt(musqSU3) * N1LO) / (
                               16. * csqrt(musqSU2) * Pi ** 2) + (
                               3 * lambdaVL3 * lambdaVLL8 * csqrt(musqSU2) * N1LO) / (
                               128. * csqrt(musqU1) * Pi ** 2) + (
                               lambdaVL3 * lambdaVLL7 * csqrt(musqSU3) * N1LO) / (16. * csqrt(musqU1) * Pi ** 2) + (
                               3 * lambdaVL1 * lambdaVLL8 * csqrt(musqU1) * N1LO) / (
                               128. * csqrt(musqSU2) * Pi ** 2) - (3 * lambdaVL1 * csqrt(musqSU2) * N0LO) / (
                               8. * Pi) - (lambdaVL3 * csqrt(musqU1) * N0LO) / (
                           8. * Pi) + (3 * g23Sq * lambdaVL1 * N1LO * np.log(mu3Bar / (2. * csqrt(musqSU2)))) / (
                               8. * Pi ** 2) - (
                               3 * lambdaVL1 ** 2 * N1LO * np.log(mu3Bar / (2. * csqrt(musqSU2)))) / (
                               32. * Pi ** 2) - (lambdaVL3 ** 2 * N1LO * np.log(mu3Bar / (2. * csqrt(musqU1)))) / (
                               32. * Pi ** 2)

        # return g3BarSq,g3PrimeBarSq,lambdaPhi3Bar,lambdaMix3Bar,lambdaS3Bar,mS3BarSq,mPhi3BarSq
        self.UltrasoftParams = g23dUSSq, g13dUSSq, lambdaPhi3dUS, lambdaMix3dUS, lambdaS3dUS, mSSq3dUS, mPhiSq3dUS
     
    def V0(self, fields, temperature: float):
        '''
        Tree-level part of the potential
        '''
        T = np.asanyarray(temperature)
        phiBar = fields[...,0]
        sBar = fields[...,1]


        _, _, lambdaPhi3Bar, lambdaMix3Bar, lambdaS3Bar, mS3BarSq, mPhi3BarSq = self.UltrasoftParams

        V = (
            + 1 / 2 * mPhi3BarSq * phiBar ** 2
            + 1 / 4 * lambdaPhi3Bar * phiBar ** 4
            + 1 / 2 * mS3BarSq * sBar ** 2
            + 1 / 4 * lambdaS3Bar * sBar ** 4 
            + 1 / 4 * lambdaMix3Bar * phiBar ** 2 * sBar ** 2)
        return V

    def particleMassSq(self, fields, temperature: float):
        """
        Calculate the boson particle spectrum. Should be overridden by
        subclasses.

        Parameters
        ----------
        fields : array_like
            Field value(s).
            Either a single point (with length `Ndim`), or an array of points.
        temperature : float or array_like
            The temperature at which to calculate the particle masses.

        Returns
        -------
        massSq : array_like
            A list of the particle masses at each input point `fields`.
        degrees_of_freedom : float or array_like
            The number of degrees of freedom for each particle.
        """ 
        T = np.asanyarray(temperature)
        fields = np.asanyarray(fields)
        phiBar = fields[...,0]
        sBar = fields[...,1]

        g3BarSq, g3PrimeBarSq, lambdaPhi3Bar, lambdaMix3Bar, lambdaS3Bar, mS3BarSq, mPhi3BarSq = self.UltrasoftParams 

        thetaBar=self.thetaBar
        D=self.dim

        degrees_of_freedom = np.array([1,1,3,2*(D-1),(D-1)]) #h,s,chi,W,Z

        """
        mass eigenvalues 
        TODO diagonalise mass matrix
        """

        mh1BarSq = 1 / 2 * (
            + (6 * lambdaPhi3Bar * phiBar ** 2 + 2 * mPhi3BarSq + lambdaMix3Bar * sBar ** 2) * np.cos(thetaBar) ** 2 
            + (lambdaMix3Bar * phiBar ** 2 + 2 * mS3BarSq + 6 * lambdaS3Bar * sBar ** 2) * np.sin(thetaBar) ** 2 
            - 2 * lambdaMix3Bar * phiBar * sBar * np.sin(2 * thetaBar))
        mh2BarSq = 1 / 2 * (
            + (6 * lambdaPhi3Bar * phiBar ** 2 + 2 * mPhi3BarSq + lambdaMix3Bar * sBar ** 2) * np.sin(thetaBar) ** 2 
            + (lambdaMix3Bar * phiBar ** 2 + 2 * mS3BarSq + 6 * lambdaS3Bar * sBar ** 2) * np.cos(thetaBar) ** 2
            + 2 * lambdaMix3Bar * phiBar * sBar * np.sin(2 * thetaBar))

        mWBarSq = 1 / 4 * g3BarSq * phiBar ** 2
        mZBarSq = 1 / 4 * (g3BarSq + g3PrimeBarSq) * phiBar ** 2
        mGBarSq = mPhi3BarSq + lambdaPhi3Bar * phiBar ** 2 + 1 / 2 * lambdaMix3Bar * sBar ** 2

        massSq = np.stack((mh1BarSq, mh2BarSq, mGBarSq, mWBarSq, mZBarSq), axis=-1)
        return massSq, degrees_of_freedom



    def V2(self, fields, particles, temperature: float):
        '''
        3D Thermal effective potential up to two loops at the ultrasoft scale.
        This function returns T * Veff_3D, such that the output has mass dimension 4 and can directly be used for the
        computation of the phase transition parameters with CosmoTransitions.
        '''
        T = np.asanyarray(temperature)
        phiBar = fields[...,0]
        sBar = fields[...,1]

        g3BarSq, g3PrimeBarSq, lambdaPhi3Bar, lambdaMix3Bar, lambdaS3Bar,_,_ = self.UltrasoftParams

        mu3Bar = self.mu3Bar * T
        muBar = mu3Bar
        D = self.dim
        massSq,_ = particles 

        thetaBar = self.thetaBar
        ct = np.cos(thetaBar)
        st = np.sin(thetaBar)

        mh1Bar = csqrt(massSq[...,0])
        mh2Bar = csqrt(massSq[...,1])
        mGBar = csqrt(massSq[...,2])
        mWBar = csqrt(massSq[...,3])
        mZBar = csqrt(massSq[...,4])

        '''
        Two-loop part of the potential
        '''
        C_h1h1h1h1 = -6 * (lambdaPhi3Bar * ct ** 4 + lambdaMix3Bar * ct ** 2 * st ** 2 + lambdaS3Bar * st ** 4)
        C_h2h2h2h2 = -6 * (lambdaPhi3Bar * st ** 4 + lambdaMix3Bar * ct ** 2 * st ** 2 + lambdaS3Bar * ct ** 4)
        C_GGGG = -6 * lambdaPhi3Bar
        C_h1h1h2h2 = 1 / 4 * (-3 * lambdaPhi3Bar - lambdaMix3Bar - 3 * lambdaS3Bar + 3 * (
                    lambdaPhi3Bar - lambdaMix3Bar + lambdaS3Bar) * np.cos(4 * thetaBar))
        C_h1h1GG = C_h1h1GpGm = -2 * lambdaPhi3Bar * ct ** 2 - lambdaMix3Bar * st ** 2
        C_h2h2GG = C_h2h2GpGm = -2 * lambdaPhi3Bar * st ** 2 - lambdaMix3Bar * ct ** 2
        C_GpGmGpGm = -4 * lambdaPhi3Bar
        C_GGGpGm = -2 * lambdaPhi3Bar
        C_ZZh1h1 = -1 / 2 * (g3BarSq + g3PrimeBarSq) * ct ** 2
        C_ZZh2h2 = -1 / 2 * (g3BarSq + g3PrimeBarSq) * st ** 2
        C_ZZGG = -1 / 2 * (g3BarSq + g3PrimeBarSq)
        C_WpWmh1h1 = -1 / 2 * g3BarSq * ct ** 2
        C_WpWmh2h2 = -1 / 2 * g3BarSq * st ** 2
        C_WpWmGG = C_WpWmGpGm = -1 / 2 * g3BarSq
        C_ZZGpGm = -1 / 2 * (g3BarSq - g3PrimeBarSq) ** 2 / (g3BarSq + g3PrimeBarSq)
        C_WpWmWpWm = -g3BarSq
        C_WpWmZZ = g3BarSq ** 2 / (g3BarSq + g3PrimeBarSq)
        C_h1h1h1 = -6 * phiBar * lambdaPhi3Bar * ct ** 3 + 3 * lambdaMix3Bar * sBar * ct ** 2 * st + 3 * lambdaMix3Bar * phiBar * ct * st ** 2 + 6 * lambdaS3Bar * sBar * st ** 3
        C_h2h2h2 = -6 * phiBar * lambdaPhi3Bar * st ** 3 - 3 * lambdaMix3Bar * sBar * ct * st ** 2 - 3 * lambdaMix3Bar * phiBar * ct ** 2 * st - 6 * lambdaS3Bar * sBar * ct ** 3
        C_h1GG = C_h1GpGm = -2 * phiBar * lambdaPhi3Bar * ct + lambdaMix3Bar * sBar * st
        C_h1h1h2 = -lambdaMix3Bar * sBar * ct ** 3 + 2 * phiBar * (
                    -3 * lambdaPhi3Bar + lambdaMix3Bar) * ct ** 2 * st + (
                                2 * lambdaMix3Bar * sBar - 6 * lambdaS3Bar * sBar) * ct * st ** 2 - phiBar * lambdaMix3Bar * st ** 3
        C_h2h2h1 = -lambdaMix3Bar * sBar * st ** 3 + 2 * phiBar * (
                    -3 * lambdaPhi3Bar + lambdaMix3Bar) * st ** 2 * ct + (
                                2 * lambdaMix3Bar * sBar - 6 * lambdaS3Bar * sBar) * st * ct ** 2 - phiBar * lambdaMix3Bar * ct ** 3
        C_GGh2 = C_h2GpGm = -2 * phiBar * lambdaPhi3Bar * st - lambdaMix3Bar * sBar * ct
        C_ZZh1 = -1 / 2 * (g3BarSq + g3PrimeBarSq) * phiBar * ct
        C_ZZh2 = -1 / 2 * (g3BarSq + g3PrimeBarSq) * phiBar * st
        C_WpWmh1 = -1 / 2 * g3BarSq * phiBar * ct
        C_WpWmh2 = -1 / 2 * g3BarSq * phiBar * st
        C_WmZGp = C_WpZGm = +phiBar / 2 * csqrt(g3BarSq) * g3PrimeBarSq / csqrt(g3BarSq + g3PrimeBarSq)
        C_WmAGp = C_WpAGm = -phiBar / 2 * csqrt(g3PrimeBarSq) * g3BarSq / csqrt(g3BarSq + g3PrimeBarSq)
        C_h1GZ = -1j / 2 * csqrt(g3BarSq + g3PrimeBarSq) * ct
        C_h2GZ = -1j / 2 * csqrt(g3BarSq + g3PrimeBarSq) * st
        C_GpGmZ = 1 / 2 * (g3PrimeBarSq - g3BarSq) / csqrt(g3PrimeBarSq + g3BarSq)
        C_GpGmA = -csqrt(g3BarSq) * csqrt(g3PrimeBarSq) / csqrt(g3PrimeBarSq + g3BarSq)
        C_h1GpWm = 1 / 2 * csqrt(g3BarSq) * ct
        C_h2GpWm = 1 / 2 * csqrt(g3BarSq) * st
        C_h1GmWp = -C_h1GpWm
        C_h2GmWp = -C_h2GpWm
        C_GGpWm = C_GGmWp = -1j / 2 * csqrt(g3BarSq)
        C_WpWmZ = g3BarSq / csqrt(g3BarSq + g3PrimeBarSq)
        C_WpWmA = csqrt(g3PrimeBarSq) * csqrt(g3BarSq) / csqrt(g3BarSq + g3PrimeBarSq)
        C_WpcbarmcZ = C_WmcbarZcp = -g3BarSq / csqrt(g3BarSq + g3PrimeBarSq)
        C_WpcbarZcm = C_WmcbarpcZ = -C_WpcbarmcZ
        C_WpcbarmcA = C_WmcbarAcp = -csqrt(g3BarSq) * csqrt(g3PrimeBarSq) / csqrt(g3BarSq + g3PrimeBarSq)
        C_WpcbarAcm = C_WmcbarpcA = -C_WpcbarmcA
        C_Zcbarpcm = -g3BarSq / csqrt(g3BarSq + g3PrimeBarSq)
        C_Zcbarmcp = -C_Zcbarpcm

        SSS = (
            + 1 / 12 * C_h1h1h1 ** 2 * self.D_SSS(mh1Bar, mh1Bar, mh1Bar, muBar)
            + 1 / 12 * C_h2h2h2 ** 2 * self.D_SSS(mh2Bar, mh2Bar, mh2Bar, muBar)
            + 1 / 4 * C_h1GG ** 2 * self.D_SSS(mh1Bar, mGBar, mGBar, muBar)
            + 1 / 4 * C_h1h1h2 ** 2 * self.D_SSS(mh1Bar, mh1Bar, mh2Bar, muBar)
            + 1 / 4 * C_h2h2h1 ** 2 * self.D_SSS(mh2Bar, mh2Bar, mh1Bar, muBar)
            + 1 / 4 * C_GGh2 ** 2 * self.D_SSS(mGBar, mGBar, mh2Bar, muBar)
            + 1 / 2 * C_h1GpGm ** 2 * self.D_SSS(mh1Bar, mGBar, mGBar, muBar)
            + 1 / 2 * C_h2GpGm ** 2 * self.D_SSS(mh2Bar, mGBar, mGBar, muBar))

        VSS = (
            - 1 / 2 * C_h1GZ ** 2 * self.D_VSS(mh1Bar, mGBar, mZBar, muBar)
            - 1 / 2 * C_h2GZ ** 2 * self.D_VSS(mh2Bar, mGBar, mZBar, muBar)
            + 1 / 2 * C_GpGmZ ** 2 * self.D_VSS(mGBar, mGBar, mZBar, muBar)
            + 1 / 2 * C_GpGmA ** 2 * self.D_VSS_1(mGBar, mGBar, muBar)
            - C_h1GpWm * C_h1GmWp * self.D_VSS(mh1Bar, mGBar, mWBar, muBar)
            - C_h2GpWm * C_h2GmWp * self.D_VSS(mh2Bar, mGBar, mWBar, muBar)
            - C_GGpWm * C_GGmWp * self.D_VSS(mGBar, mGBar, mWBar, muBar))

        VVS = (
            + 1 / 4 * C_ZZh1 ** 2 * self.D_VVS(mh1Bar, mZBar, mZBar, muBar)
            + 1 / 4 * C_ZZh2 ** 2 * self.D_VVS(mh2Bar, mZBar, mZBar, muBar)
            + 1 / 2 * C_WpWmh1 ** 2 * self.D_VVS(mh1Bar, mWBar, mWBar, muBar)
            + 1 / 2 * C_WpWmh2 ** 2 * self.D_VVS(mh2Bar, mWBar, mWBar, muBar)
            + C_WmZGp * C_WpZGm * self.D_VVS(mGBar, mWBar, mZBar, muBar)
            + C_WmAGp * C_WpAGm * self.D_VVS_1(mGBar, mWBar, muBar))

        VVV = (
            + 1 / 2 * C_WpWmZ ** 2 * self.D_VVV(mWBar, mZBar, muBar)
            + 1 / 2 * C_WpWmA ** 2 * self.D_VVV_1(mWBar, muBar))

        VGG = (
            - C_WpcbarmcZ * C_WmcbarZcp * self.D_VGG(mWBar, muBar)
            - C_WpcbarZcm * C_WmcbarpcZ * self.D_VGG(mWBar, muBar)
            - C_WpcbarmcA * C_WmcbarAcp * self.D_VGG(mWBar, muBar)
            - C_WpcbarAcm * C_WmcbarpcA * self.D_VGG(mWBar, muBar)
            - 1 / 2 * C_Zcbarpcm ** 2 * self.D_VGG(mZBar, muBar)
            - 1 / 2 * C_Zcbarmcp ** 2 * self.D_VGG(mZBar, muBar))

        SS = (
            + 1 / 8 * C_h1h1h1h1 * self.I_3(mh1Bar) ** 2
            + 1 / 8 * C_h2h2h2h2 * self.I_3(mh2Bar) ** 2
            + 1 / 8 * C_GGGG * self.I_3(mGBar) ** 2
            + 1 / 4 * C_h1h1h2h2 * self.I_3(mh1Bar) * self.I_3(mh2Bar)
            + 1 / 4 * C_h1h1GG * self.I_3(mh1Bar) * self.I_3(mGBar)
            + 1 / 4 * C_h2h2GG * self.I_3(mGBar) * self.I_3(mh2Bar)
            + 1 / 2 * C_GpGmGpGm * self.I_3(mGBar) ** 2
            + 1 / 2 * C_h1h1GpGm * self.I_3(mh1Bar) * self.I_3(mGBar)
            + 1 / 2 * C_h2h2GpGm * self.I_3(mh2Bar) * self.I_3(mGBar)
            + 1 / 2 * C_GGGpGm * self.I_3(mGBar) ** 2)

        VS = (
            + 1 / 4 * C_ZZh1h1 * self.I_3(mh1Bar) * self.I_3(mZBar)
            + 1 / 4 * C_ZZh2h2 * self.I_3(mh2Bar) * self.I_3(mZBar)
            + 1 / 4 * C_ZZGG * self.I_3(mGBar) * self.I_3(mZBar)
            + 1 / 2 * C_WpWmh1h1 * self.I_3(mh1Bar) * self.I_3(mWBar)
            + 1 / 2 * C_WpWmh2h2 * self.I_3(mh2Bar) * self.I_3(mWBar)
            + 1 / 2 * C_WpWmGG * self.I_3(mGBar) * self.I_3(mWBar)
            + 1 / 2 * C_ZZGpGm * self.I_3(mGBar) * self.I_3(mZBar)
            + C_WpWmGpGm * self.I_3(mGBar) * self.I_3(mWBar))
        VS *= (D-1)

        VV = (
            + 1 / 2 * C_WpWmWpWm * self.D_VV(mWBar, mWBar) 
            - C_WpWmZZ * self.D_VV(mWBar, mZBar))

        V = -(SSS + VSS + VVS + VVV + VGG + SS + VS + VV)

        return np.real(V)
