#
# Copyright (c) 2021 TECHNICAL UNIVERSITY OF MUNICH,
# DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
# AUTHOR: Christian Meyer
#

class RBF:
    """
    Class to provide strings that can be passed to as rbf_function to Stages

    """
    @staticmethod
    def gaussian_spline(epsilon):
        r"""
        Returns Gaussian spline function.

        .. math::

            \phi(r) = \mathrm{e}^{-\left(\frac{r}{\varepsilon}\right)^2}

        Parameters
        ----------
        epsilon : float
            epsilon that controls the effect radius of the RBF

        Returns
        -------
        rbf : str
        """
        return r'exp(-(r*r) / (epsilon*epsilon))'.replace('epsilon', str(epsilon))

    @staticmethod
    def multi_quadratic_biharmonic_spline(epsilon):
        r"""
        Returns Multi Quadratic Biharmonic Spline string.

        ..  math::

            \phi(r) = \sqrt{r^2 + \varepsilon^2}

        Parameters
        ----------
        epsilon : float
            epsilon that controls the effect radius of the RBF

        Returns
        -------
        rbf : str
        """
        return r'sqrt((r * r) + (epsilon * epsilon))'.replace('epsilon', str(epsilon))

    @staticmethod
    def thin_plate_spline(epsilon):
        r"""
        Returns Thin Plate Spline (TPS).

        This is RBF is very recommended for first studies.

        .. math::

            \phi(r) = \begin{cases}
                \left(\frac{r}{\varepsilon}\right)^2 \ln \left(\frac{r}{\varepsilon}\right)^2 & \text{for } r/\varepsilon > 0 \\
                \left(\frac{r}{\varepsilon}\right)^2 & \text{ else}
            \end{cases}

        Parameters
        ----------
        epsilon : float
            epsilon that controls the effect radius of the RBF

        Returns
        -------
        rbf : str

        """
        return r'where((r/epsilon) > 0, (r*r/(epsilon*epsilon))*log(r*r/(epsilon*epsilon)), r*r/(epsilon*epsilon))'.replace('epsilon', str(epsilon))

    @staticmethod
    def beckert_wendland_c2_basis(epsilon):
        r"""
        Beckert Wendland Splin C2. This is a compact RBF, it can be used for
        sparse assembly.

        .. math::

            \phi(r) = \begin{cases}
                \left(4 \cdot \frac{r}{\varepsilon} + 1 \right) \cdot \left( 1 - \frac{r}{\varepsilon}\right)^4 & \text{for } \left(1 - \frac{r}{\varepsilon}\right) > 0 \\
                0 & \text{else}
            \end{cases}

        Parameters
        ----------
        epsilon : float
            epsilon that controls the effect radius of the RBF

        Returns
        -------
        rbf : str

        """
        return r'where((1.0 - r/epsilon) > 0.0, (r/epsilon*4.0 + 1.0)*(1.0 - r/epsilon)**4, 0.0)'.replace('epsilon', str(epsilon))
