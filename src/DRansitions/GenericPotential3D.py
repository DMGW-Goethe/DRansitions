import scipy.special
from BetaFunctions import *
from scipy import special
from numpy.lib.scimath import sqrt as csqrt
from cosmoTransitions import generic_potential
from abc import ABC, abstractmethod ## Abstract Base Class

class GenericPotential3D(ABC,generic_potential.generic_potential):
    '''
    This class comprises all necessary functions to compute the effective potential in the 3d effective theory up to two-loops
    '''
    def __init__(self, order: int, RGscale: float):
        self.LoopOrderPotential = order
        self.RGscale = RGscale
        """Initialisation
        """


    # @staticmethod
    def J_3(self, msq: float) -> complex:
        """
        Parameters
        ----------
        msq : float 
            Mass squared of given particle

        Returns
        -------
        J_3 : float 
            An object of the Model class.
        """
        # keep track of imaginary part
        # 3d units
        return -(msq + 0j)**(3/2) / (12.*np.pi)

    def D_SSS(self, m1, m2, m3, Lambda):
        return 1 / (4 * np.pi) ** 2 * (1 / 2 + np.log(Lambda / (m1 + m2 + m3)))

    def D_VSS(self, m1, m2, m3, Lambda):
        return 1 / m3 ** 2 * (
            + (-m1 ** 2 + m2 ** 2 + m3 ** 2) * self.I_3(m2) * self.I_3(m3) 
            + (-m3 ** 2 * self.I_3(m2) + (m1 ** 2 - m2 ** 2 + m3 ** 2) * self.I_3(m3)) * self.I_3(m1) 
            - (m1 ** 2 - m2 ** 2) ** 2 * self.D_SSS(m1, m2, 0, Lambda) 
            + (m1 - m2 - m3) * (m1 + m2 - m3) * (m1 - m2 + m3) * (m1 + m2 + m3) * self.D_SSS(m1, m2, m3, Lambda))

    def D_VSS_1(self, m1, m2, Lambda):
        D = self.dim
        return -(D - 1) * (
            + (m1 ** 2 + m2 ** 2) * self.D_SSS(m1, m2, 0, Lambda) 
            + self.I_3(m1) * self.I_3(m2))

    def D_VVS(self, m1, m2, m3, Lambda):
        D = self.dim
        return 1 / (4 * m2 ** 2 * m3 ** 2) * (
            - m3 ** 2 * self.I_3(m1) * self.I_3(m2) 
            + (-m2 ** 2 * self.I_3(m1) + (-m1 ** 2 + m2 ** 2 + m3 ** 2) * self.I_3(m2)) * self.I_3( m3) 
            + m1 ** 4 * self.D_SSS(m1, 0, 0, Lambda) 
            - (m2 ** 2 - m1 ** 2) ** 2 * self.D_SSS(m1, m2, 0, Lambda) 
            - (m3 ** 2 - m1 ** 2) ** 2 * self.D_SSS(m1, m3, 0, Lambda) 
            + ((m2 ** 2 - m1 ** 2) ** 2 + (- 2 * m1 ** 2 + (4 * D - 6) * m2 ** 2) * m3 ** 2 + m3 ** 4) * self.D_SSS(m1, m2, m3, Lambda))

    def D_VVS_1(self, m1, m2, Lambda):
        D = self.dim
        return -(D - 1) / (4 * m2 ** 2) * (
            + (m1 ** 2 - 3 * m2 ** 2) * self.D_SSS(m1, m2, 0, Lambda) 
            - m1 ** 2 * self.D_SSS(m1, 0, 0,Lambda) 
            + self.I_3(m1) * self.I_3(m2))

    def D_VVV(self, m1, m2, Lambda):
        D = self.dim
        return (
            - self.I_3(m1) * self.I_3(m2) * (D * m1 ** 4 - (5 * D - 4) * m1 ** 2 * m2 ** 2 - D * (4 * D - 7) * m2 ** 4) / (2 * D * m1 ** 2 * m2 ** 2) 
            + self.I_3(m1) ** 2 * (4 * (3 * D ** 2 - 4 * D - 1) * m1 ** 4 - 2 * D * (4 * D - 7) * m1 ** 2 * m2 ** 2 - D * m2 ** 4) / (4 * D * m1 ** 4) 
            - self.D_SSS(m1, m2, 0, Lambda) * (m1 ** 2 - m2 ** 2) ** 2 * (m1 ** 4 + 2 * (2 * D - 3) * m1 ** 2 * m2 ** 2 + m2 ** 4) / (2 * m1 ** 4 * m2 ** 2) 
            - self.D_SSS(m1, m1, m2, Lambda) * (4 * m1 ** 2 - m2 ** 2) * (4 * (D - 1) * m1 ** 4 + 4 * (2 * D - 3) * m1 ** 2 * m2 ** 2 + m2 ** 4) / (4 * m1 ** 4) 
            + self.D_SSS(m2, 0, 0, Lambda) * m2 ** 6 / (4 * m1 ** 4) 
            + self.D_SSS(m1, 0, 0, Lambda) * m1 ** 4 / (2 * m2 ** 2))

    def D_VVV_1(self, m, Lambda):
        d = self.dim
        second_term = -(3 * d - 5) / 2 * m ** 2 * self.D_SSS(m, 0, 0, Lambda)
        first_term_finite = -(5 * d ** 3 - 19 * d ** 2 + 15 * d + 3) / d * np.log(np.exp(np.euler_gamma) * Lambda ** 2 / (4 * np.pi)) * m ** 2 / (4 * np.pi) ** 3 * scipy.special.gamma(-1 / 2)
        return first_term_finite + second_term

    def D_VGG(self, m, Lambda):
        return 1 / 4 * m ** 2 * self.D_SSS(m, 0, 0, Lambda)

    def D_VV(self, m1, m2):
        D = self.dim
        return (D - 1) ** 3 / D * self.I_3(m1) * self.I_3(m2)

    def I_3(self, m: complex):
        # TODO should be made for squared masses
        D = self.dim
        alpha = 1
        return csqrt(m ** 2) / (4 * np.pi) ** (D / 2) * special.gamma(alpha - D / 2) / special.gamma(alpha)

    @abstractmethod
    def GetUltrasoftParams(self, temperature: float):
        pass

    @abstractmethod
    def particleMassSq(self, fields, temperature: float):
        """
        Particle mass spectrum and
        degrees of freedom

        Returns
        -------
        massSq : array_like

        degrees_of_freedom : float or array_like
        """
        pass

    def V1(self, particles, temperature: float):
        """
        The one-loop corrections to the one-loop
        EFT potential using MS-bar renormalization.

        This is generally not called directly, but is instead used by
        :func:`Vtot`.

        Parameters
        ----------
        particles : array of floats
            EFT particle spectrum (here: masses, number of dofs)

        Returns
        -------
        V1 : 1loop vacuum contribution to the pressure

        """
        msq, dof = particles 
        V = np.sum(dof*self.J_3(msq), axis=-1)

        return np.real(V)

    @abstractmethod
    def V2(self, particles, temperature: float):
        """
        Two loop effective potential
        """
        pass

    def Vtot(self, fields, temperature: float, include_radiation=False):
        """
        The total finite temperature effective potential.

        Parameters
        ----------
        fields : array_like
            Field value(s).
            Either a single point (with length `Ndim`), or an array of points.
        temperature : float or array_like
            broadcastable (that is, ``X[0,...]*T`` is a valid operation).
        include_radiation : bool, optional
            If False, this will drop all field-independent radiation
            terms from the effective potential. Useful for calculating
            differences or derivatives.

        Returns
        -------
        Vtot : total effective potential
        """
        T = np.asanyarray(temperature)
        _=self.GetUltrasoftParams(T)
        fields = np.asanyarray(fields)
        fields = fields/np.sqrt(T + 1e-100)


        particles = self.particleMassSq(fields,T)
        if self.LoopOrderPotential >= 0:
            V = self.V0(fields, T)
        if self.LoopOrderPotential >= 1:
            V += self.V1(particles, T)
        if self.LoopOrderPotential >= 2:
            V += self.V2(fields, particles, T)

        if include_radiation:
            # TODO
            pass
        return T*np.real(V)