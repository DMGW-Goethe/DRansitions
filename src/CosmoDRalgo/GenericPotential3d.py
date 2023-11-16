import scipy.special
from BetaFunctions import *
from scipy import special
from numpy.lib.scimath import sqrt as csqrt

class GenericPotential3D:
    '''
    This class comprises all necessary functions to compute the effective potential in the 3d effective theory up to two-loops
    '''
    def __init__(self):
        pass

    def J_3(self, m):
        return -m ** 3 / (12 * np.pi)

    def D_SSS(self, m1, m2, m3, Lambda):
        return 1 / (4 * np.pi) ** 2 * (1 / 2 + np.log(Lambda / (m1 + m2 + m3)))

    def D_VSS(self, m1, m2, m3, Lambda):
        return 1 / m3 ** 2 * ((-m1 ** 2 + m2 ** 2 + m3 ** 2) * self.I_3(m2) * self.I_3(m3) + (-m3 ** 2 * self.I_3(m2) + (m1 ** 2 - m2 ** 2 + m3 ** 2) * self.I_3(m3)) * self.I_3(m1) - (m1 ** 2 - m2 ** 2) ** 2 * self.D_SSS(m1, m2, 0, Lambda) + (m1 - m2 - m3) * (m1 + m2 - m3) * (m1 - m2 + m3) * (m1 + m2 + m3) * self.D_SSS(m1, m2, m3, Lambda))

    def D_VSS_1(self, m1, m2, Lambda):
        D = 3
        return -(D - 1) * ((m1 ** 2 + m2 ** 2) * self.D_SSS(m1, m2, 0, Lambda) + self.I_3(m1) * self.I_3(m2))

    def D_VVS(self, m1, m2, m3, Lambda):
        D = 3
        return 1 / (4 * m2 ** 2 * m3 ** 2) * (-m3 ** 2 * self.I_3(m1) * self.I_3(m2) + (-m2 ** 2 * self.I_3(m1) + (-m1 ** 2 + m2 ** 2 + m3 ** 2) * self.I_3(m2)) * self.I_3( m3) + m1 ** 4 * self.D_SSS(m1, 0, 0, Lambda) - (m2 ** 2 - m1 ** 2) ** 2 * self.D_SSS(m1, m2, 0,Lambda) - (m3 ** 2 - m1 ** 2) ** 2 * self.D_SSS(m1, m3, 0, Lambda) + ((m2 ** 2 - m1 ** 2) ** 2 + (- 2 * m1 ** 2 + (4 * D - 6) * m2 ** 2) * m3 ** 2 + m3 ** 4) * self.D_SSS(m1, m2, m3, Lambda))

    def D_VVS_1(self, m1, m2, Lambda):
        D = 3
        return -(D - 1) / (4 * m2 ** 2) * ((m1 ** 2 - 3 * m2 ** 2) * self.D_SSS(m1, m2, 0, Lambda) - m1 ** 2 * self.D_SSS(m1, 0, 0,Lambda) + self.I_3(m1) * self.I_3(m2))

    def D_VVV(self, m1, m2, Lambda):
        D = 3
        return -self.I_3(m1) * self.I_3(m2) * (D * m1 ** 4 - (5 * D - 4) * m1 ** 2 * m2 ** 2 - D * (4 * D - 7) * m2 ** 4) / (2 * D * m1 ** 2 * m2 ** 2) + self.I_3(m1) ** 2 * (4 * (3 * D ** 2 - 4 * D - 1) * m1 ** 4 - 2 * D * (4 * D - 7) * m1 ** 2 * m2 ** 2 - D * m2 ** 4) / (4 * D * m1 ** 4) - self.D_SSS(m1, m2, 0, Lambda) * (m1 ** 2 - m2 ** 2) ** 2 * (m1 ** 4 + 2 * (2 * D - 3) * m1 ** 2 * m2 ** 2 + m2 ** 4) / (2 * m1 ** 4 * m2 ** 2) - self.D_SSS(m1, m1, m2, Lambda) * (4 * m1 ** 2 - m2 ** 2) * (4 * (D - 1) * m1 ** 4 + 4 * (2 * D - 3) * m1 ** 2 * m2 ** 2 + m2 ** 4) / (4 * m1 ** 4) + self.D_SSS(m2, 0, 0, Lambda) * m2 ** 6 / (4 * m1 ** 4) + self.D_SSS(m1, 0, 0, Lambda) * m1 ** 4 / (2 * m2 ** 2)

    def D_VVV_1(self, m, Lambda):
        d = 3
        second_term = -(3 * d - 5) / 2 * m ** 2 * self.D_SSS(m, 0, 0, Lambda)
        first_term_finite = -(5 * d ** 3 - 19 * d ** 2 + 15 * d + 3) / d * np.log(np.exp(np.euler_gamma) * Lambda ** 2 / (4 * np.pi)) * m ** 2 / (4 * np.pi) ** 3 * scipy.special.gamma(-1 / 2)
        return first_term_finite + second_term

    def D_VGG(self, m, Lambda):
        return 1 / 4 * m ** 2 * self.D_SSS(m, 0, 0, Lambda)

    def D_VV(self, m1, m2):
        D = 3
        return (D - 1) ** 3 / D * self.I_3(m1) * self.I_3(m2)

    def I_3(self, m):
        alpha = 1
        D = 3
        return csqrt(m ** 2) / (4 * np.pi) ** (D / 2) * special.gamma(alpha - D / 2) / special.gamma(alpha)