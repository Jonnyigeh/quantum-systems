import numba
import numpy as np
import scipy.special
import scipy.linalg

from quantum_systems import BasisSet

from quantum_systems.quantum_dots.one_dim.one_dim_potentials import (
    HOPotential,
    DWPotential,
    DWPotentialSmooth,
    SymmetricDWPotential,
    AsymmetricDWPotential,
    GaussianPotential,
    AtomicPotential,
    MorsePotentialDW
)

import warnings
import logging
logging.basicConfig(level=logging.WARNING)

@numba.njit
def _trapz_prep(vec, dx):
    # The trapezoidal method applied to a vector can be implemented as multiplying it by dx, halving the ends, and taking the sum.
    # Since we are applying the trapezoidal method to a product of 3 vectors,
    # we can first perform the first two steps on this vector, which is reused, to greatly speed up the calculation.
    prepped_vec = vec * dx
    prepped_vec[0] *= 0.5
    prepped_vec[-1] *= 0.5
    return prepped_vec


@numba.njit
def _shielded_coulomb(x_1, x_2, alpha, a):
    return alpha / np.sqrt((x_1 - x_2) ** 2 + a**2)


@numba.njit
def _compute_inner_integral(spf, l, num_grid_points, grid, alpha, a):
    inner_integral = np.zeros((l, l, num_grid_points), dtype=np.complex128)
    dx = grid[1] - grid[0]
    for i in range(num_grid_points):
        coulomb = _shielded_coulomb(grid[i], grid, alpha, a)
        trapz_prepped_coulomb = _trapz_prep(coulomb, dx)
        for q in range(l):
            trapz_prod = np.conjugate(spf[q]) * trapz_prepped_coulomb
            for s in range(l):
                inner_integral[q, s, i] = np.dot(
                    trapz_prod, spf[s]
                )  # _trapz(np.conjugate(spf[q]) * coulomb * spf[s], grid)

    return inner_integral


@numba.njit
def _compute_orbital_integrals(spf, l, inner_integral, grid):
    u = np.zeros((l, l, l, l), dtype=np.complex128)
    dx = grid[1] - grid[0]

    for q in range(l):
        for s in range(l):
            trapz_prepped_inner = _trapz_prep(inner_integral[q, s], dx)
            for p in range(l):
                trapz_prod = np.conjugate(spf[p]) * trapz_prepped_inner
                for r in range(l):
                    u[p, q, r, s] = np.dot(
                        trapz_prod, spf[r].astype(np.complex128)
                    )  # _trapz(np.conjugate(spf[p]) * inner_integral[q, s] * spf[r], grid)

    return u


class ODHO(BasisSet):
    """Create matrix elements and grid representation associated with the harmonic
    oscillator basis.

    Example
    -------

    >>> odho = ODHO(20, 11, 201, omega=1)
    >>> odho.l == 20
    True
    >>> abs(0.5 - odho.h[0, 0]) # doctest.ELLIPSIS
    0.0
    """

    def __init__(
        self,
        l,
        grid_length,
        num_grid_points,
        omega=0.25,
        a=0.25,
        alpha=1.0,
        beta=0,
        grid=None,
        **kwargs,
    ):
        super().__init__(l, dim=1, **kwargs)

        self.omega = omega
        self.a = a
        self.alpha = alpha

        self.grid_length = grid_length
        self.num_grid_points = num_grid_points
        if grid is None:
            grid = np.linspace(
                -self.grid_length, self.grid_length, self.num_grid_points
            )
        self.grid = grid
        self.beta = beta

        self.setup_basis()

    def setup_basis(self):
        dx = self.grid[1] - self.grid[0]
        self.eigen_energies = self.omega * (np.arange(self.l) + 0.5)
        self.h = np.diag(self.eigen_energies).astype(np.complex128)
        self.s = np.eye(self.l)
        self.spf = np.zeros((self.l, self.num_grid_points))

        for p in range(self.l):
            self.spf[p] = self.ho_function(self.grid, p)

        inner_integral = _compute_inner_integral(
            self.spf,
            self.l,
            self.num_grid_points,
            self.grid,
            self.alpha,
            self.a,
        )

        self.u = _compute_orbital_integrals(
            self.spf, self.l, inner_integral, self.grid
        )

        self.construct_position_integrals()

    def ho_function(self, x, n):
        return (
            self.normalization(n)
            * np.exp(-0.5 * self.omega * x**2)
            * scipy.special.hermite(n)(np.sqrt(self.omega) * x)
        )

    def normalization(self, n):
        return (
            1.0
            / np.sqrt(2**n * scipy.special.factorial(n))
            * (self.omega / np.pi) ** 0.25
        )

    def construct_position_integrals(self):
        self.position = np.zeros((1, self.l, self.l), dtype=self.spf.dtype)

        for n in range(self.l - 1):
            Nn = self.normalization(n)
            Nn_up = self.normalization(n + 1)
            pos = (
                Nn
                * Nn_up
                * (n + 1)
                * np.sqrt(np.pi)
                * 2**n
                * scipy.special.factorial(n)
                / self.omega
            )
            self.position[0, n, n + 1] = pos
            self.position[0, n + 1, n] = pos


class ODQD(BasisSet):
    """Create 1D quantum dot system

    Parameters
    ----------
    l : int
        Number of basis functions
    grid_length : int or float
        Space over which to model wavefunction
    num_grid_points : int or float
        Defines resolution of numerical representation
        of wavefunction
    a : float, default 0.25
        Screening parameter in the shielded Coulomb interation.
    alpha : float, default 1.0
        Strength parameter in the shielded Coulomb interaction.
    beta : float, default 0.0
        Strength parameter of the non-dipole term in the laser interaction matrix.

    Attributes
    ----------
    h : np.array
        One-body matrix
    f : np.array
        Fock matrix
    u : np.array
        Two-body matrix

    Methods
    -------
    setup_basis()
        Must be called to set up quantum system.  The method will revert to
        regular harmonic oscillator potential if no potential is provided. It
        is also possible to use double well potentials.
    construct_position_integrals()
        Constructs position matrix elements. This method is called by
        setup_basis().

    Example
    -------

    >>> odqd = ODQD(20, 11, 201, potential=ODQD.HOPotential(omega=1))
    >>> odqd.l == 20
    True
    >>> abs(0.5 - odqd.h[0, 0]) # doctest.ELLIPSIS
    0.0003...
    """

    HOPotential = HOPotential
    DWPotential = DWPotential
    DWPotentialSmooth = DWPotentialSmooth
    SymmetricDWPotential = SymmetricDWPotential
    AsymmetricDWPotential = AsymmetricDWPotential
    GaussianPotential = GaussianPotential
    AtomicPotential = AtomicPotential

    def __init__(
        self,
        l,
        grid_length,
        num_grid_points,
        a=0.25,
        alpha=1.0,
        beta=0,
        potential=None,
        **kwargs,
    ):
        super().__init__(l, dim=1, **kwargs)

        self.a = a
        self.alpha = alpha

        self.grid_length = grid_length
        self.num_grid_points = num_grid_points
        self.grid = np.linspace(
            -self.grid_length, self.grid_length, self.num_grid_points
        )
        self.beta = beta

        if potential is None:
            omega = (
                0.25  # Default frequency corresponding to Zanghellini article
            )
            potential = HOPotential(omega)

        self.potential = potential

        self.setup_basis()

    def setup_basis(self):
        dx = self.grid[1] - self.grid[0]

        h_diag = 1.0 / (dx**2) + self.potential(self.grid[1:-1])
        h_off_diag = -1.0 / (2 * dx**2) * np.ones(self.num_grid_points - 3)

        eps, C = scipy.linalg.eigh_tridiagonal(
            h_diag, h_off_diag, select="i", select_range=(0, self.l - 1)
        )

        self.spf = np.zeros((self.l, self.num_grid_points), dtype=np.complex128)
        self.spf[:, 1:-1] = C.T / np.sqrt(dx)
        self.eigen_energies = eps

        self.h = np.diag(eps).astype(np.complex128)
        self.s = np.eye(self.l)

        u = _shielded_coulomb(
            self.grid[None, 1:-1], self.grid[1:-1, None], self.alpha, self.a
        )
        self.u = np.einsum(
            "pa, qb, pc, qd, pq -> abcd", C, C, C, C, u, optimize=True
        )

        self.position = np.zeros((1, self.l, self.l), dtype=self.spf.dtype)
        self.position[0] = np.einsum(
            "pa, p, pb -> ab",
            C,
            self.grid[1:-1] + self.beta * self.grid[1:-1] ** 2,
            C,
            optimize=True,
        )


class ODMorse(BasisSet):
    """
    something something
    Analytical expressions are from the book "Ideas of Quantum Chemistry" by Lucjan Piela, 2014
    Equations are from the article "The Morse oscillator in position space, momentum space, and phase space" by Dahl and Springborg,
    in doi:10.1063/1.453761
    """

    MorsePotentialDW = MorsePotentialDW

    def __init__(
            self,
            l,
            grid_length,
            num_grid_points,
            _a=0.25,
            alpha=1.0,
            D_a=10.0,
            D_b=10.0,
            k_a=1.0,
            k_b=1.0,
            d=15.0,
            potential=None,
            visualize=False,
            **kwargs,
    ):
        super().__init__(l, dim=1, **kwargs)
        if potential is None:
            potential = MorsePotentialDW(D_a=D_a, D_b=D_b, k_a=k_a, k_b=k_b, d=d)
        self.potential = potential
        self.D_a = potential.D_a
        self.D_b = potential.D_b
        self._a = _a
        self.alpha = alpha
        self.k_a = k_a
        self.k_b = k_b
        self.a = potential.a
        self.b = potential.b
        self.d = potential.d
        self.a_lmbda = np.sqrt(2 * self.D_a) / self.a
        self.b_lmbda = np.sqrt(2 * self.D_b) / self.b
        # Check that the number of basis functions in each well is less than lambda - 0.5
        try: 
            assert l < np.floor(self.a_lmbda - 0.5) and l < np.floor(self.b_lmbda - 0.5)
        except AssertionError:
            print("The number of basis functions in well A and B must be less than lambda - 0.5. Reducing the number of basis functions in well A (left) and B (right)...")
            l = int(np.min(np.floor([self.a_lmbda - 0.5, self.b_lmbda - 0.5]))) # Equal number of basis functions in each well
            print(f'Put {l} functions into each well.')
        # Set total number of basis functions for the composite system, i.e l ^ 2, since we are using product states.
        self.l_sub = l
        self.l = l ** 2
        print(f"Total number of composite functions for bipartite system: {self.l}")
        self.grid_length = grid_length
        # assert self.grid_length > self.d, "Grid length must be larger than the distance between the wells. Twice the distance is recommended."
        self.num_grid_points = num_grid_points
        self.grid = np.linspace(
            -self.grid_length, self.grid_length, self.num_grid_points
        )
        ## Visualization, and comparison of analytical and numerical basis functions
        # self.setup_analytical_basis()
        self.setup_basis()
        if visualize:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 2)
            compare_ax = fig.add_subplot(212)
            for f in range(self.spf_a.shape[0]):
                ax[0,0].plot(self.grid, np.abs(self.spf_a[f]) ** 2, linestyle='-')
                compare_ax.plot(self.grid, np.abs(self.spf_a[f]) ** 2, linestyle='-')
            compare_ax.set_prop_cycle(None)
            for f in range(self.spf_l.shape[0]):
                ax[0,1].plot(self.grid, np.abs(self.spf_l[f]) ** 2, linestyle='--')
                compare_ax.plot(self.grid, np.abs(self.spf_l[f]) ** 2, linestyle='--' )
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], linestyle='-', color='black', lw=2, label='Analytical'),
                    Line2D([0], [0], linestyle='--', color='black', lw=2, label='Numerical')]
            fig.legend(handles=legend_elements)
            plt.show()
            print("Analytical energies: left well: ", self.a_eigen_energies, "right well: ", self.b_eigen_energies)
            print('\n')
            print("numerical energies: left well: ", self.eigen_energies_l, "right well: ", self.eigen_energies_r)
        print("Basis initialized..")
        self.a_position = self.construct_position_integrals(self.a_lmbda, self.spf_l, self.l_sub)
        self.b_position = self.construct_position_integrals(self.b_lmbda, self.spf_r, self.l_sub)
        print("Position integrals computed..")


    def setup_basis(self):
        dx = self.grid[1] - self.grid[0]
        # Find the eigenbasis for each well separately, left = A, right = B - accounting for Dirichlet BC, i.e the WF should go to zero at the end-points
        V_l = self.potential.left_pot(self.grid[1:-1])
        V_r = self.potential.right_pot(self.grid[1:-1]) 
        h_l_diag = 1 / (dx**2) + V_l
        h_l_off_diag = - 1 / (2 * dx**2) * np.ones(self.num_grid_points - 3)
        h_r_diag = 1 / (dx**2) + V_r
        h_r_off_diag = - 1 / (2 * dx**2) * np.ones(self.num_grid_points - 3)
        eps_l, C_l = scipy.linalg.eigh_tridiagonal(h_l_diag, h_l_off_diag, select="i", select_range=(0, self.l_sub - 1))
        eps_r, C_r = scipy.linalg.eigh_tridiagonal(h_r_diag, h_r_off_diag, select="i", select_range=(0, self.l_sub - 1))
        # Enforce single-particle functions goes to zero on boundary, and that they are normalized (on a discrete grid)
        self.spf_l = np.zeros((self.l_sub, self.num_grid_points), dtype=np.complex128)
        self.spf_l[:, 1:-1] = C_l.T / np.sqrt(dx)
        self.eigen_energies_l = eps_l
        self.spf_r = np.zeros((self.l_sub, self.num_grid_points), dtype=np.complex128)
        self.spf_r[:, 1:-1] = C_r.T / np.sqrt(dx)
        self.eigen_energies_r = eps_r

        # Set up the Hamiltonian matrices for each well
        self.h_l = np.diag(eps_l)
        self.h_r = np.diag(eps_r)
        
        self._ulr = self.find_u_matrix(grid=self.grid, alpha=self.alpha, a=self._a, spf_l=self.spf_l, spf_r=self.spf_r)
        self.h = np.kron(self.h_l, np.eye(*self.h_l.shape)) + np.kron(np.eye(*self.h_r.shape), self.h_r)
        self.u = self._ulr.reshape(16,16)
        self.s = np.eye(self.l)
        # Find the unified basis \psi_{lr}(x) = \psi_r(x) * \psi_l(x) by element-wise multiplication
        self.spf = np.zeros((self.l, self.num_grid_points), dtype=np.complex128)
        idx = 0
        for i in range(self.l_sub):
            for j in range(self.l_sub):
                self.spf[idx, :] = self.spf_l[i,:] * self.spf_r[j,:]
                idx += 1
        # # Find the matrix elements in two steps: first the inner integral <p|V|q> for each pair of states p and q
        # # then the outer integral <pq|V|rs> for each pair of states p, q, r and s. Do this for each well. Or together?
        # # Must do integration, direct evaluation on the grid would not suffice. Here V is the shielded coulomb interaction.
        # inner_integral = _compute_inner_integral(
        #     self.spf,
        #     self.l,
        #     self.num_grid_points,
        #     self.grid,
        #     self.alpha,
        #     self._a
        # )
        
        # self.u_trapz = _compute_orbital_integrals(
        #     self.spf,
        #     self.l,
        #     inner_integral,
        #     self.grid
        # )        



    def setup_analytical_basis(self):
        self.a_eigen_energies = self.compute_eigenenergies(self.a, self.D_a, self.l_sub)
        self.b_eigen_energies = self.compute_eigenenergies(self.b, self.D_b, self.l_sub)

        # Construct the Hamiltonian matrices for the two wells
        self.h_a = np.diag(self.a_eigen_energies).astype(np.complex128)
        self.h_b = np.diag(self.b_eigen_energies).astype(np.complex128)
        self.s = np.eye(self.l_sub) # The eigenstates are orthogonal, so the overlap matrix is the identity matrix

        # Allocate memory for the single-particle functions for each well
        self.spf_a = np.zeros((self.l_sub, self.num_grid_points), dtype=np.complex128)
        self.spf_b = np.zeros((self.l_sub, self.num_grid_points), dtype=np.complex128)
        # Evaluate the single-particle functions on the grid, and store them in the spf matrices
        for p in range(self.l_sub):
            self.spf_a[p] = self.morse_function(self.grid, p, self.a_lmbda, -self.d / 2, self.a)
            self.spf_b[p] = self.morse_function(self.grid, p, self.b_lmbda, self.d / 2, self.b, reversed=True)

        # Find the unified basis \psi_{lr}(x) = \psi_r(x) * \psi_l(x) by element-wise multiplication
        self.spf_unified = np.zeros((self.l_sub * self.l_sub, self.num_grid_points), dtype=np.complex128)
        idx = 0
        for i in range(self.l_sub):
            for j in range(self.l_sub):
                self.spf_unified[idx, :] = self.spf_a[i,:] * self.spf_b[j,:]
                idx += 1
        
        # Find the matrix elements in two steps: first the inner integral <p|V|q> for each pair of states p and q
        # then the outer integral <pq|V|rs> for each pair of states p, q, r and s. Do this for each well. Or together?
        inner_integral = _compute_inner_integral(
            self.spf_unified,
            self.l,
            self.num_grid_points,
            self.grid,
            self.alpha,
            self._a
        )
        
        self.u = _compute_orbital_integrals(
            self.spf_unified,
            self.l,
            inner_integral,
            self.grid
        )        
        

    def morse_function(self, x, n, lmbda, x_e, c, reversed=False):
        """
        Single-well Morsepotential eigenfunction of degree n. Analytical expressions from "Ideas of Quantum Chemistry" by Lucjan Piela.
        
        
        params:
        x: np.array
            Grid points
        n: int
            Degree of the Morse potential
        lmbda: float
            potential variable lambda = sqrt(2D)/a
        x_e: float
            Center of the Morse potential
        c: float
            Width of the Morse potential (not directly, but controls the width)
        """
        if reversed:
            x = x[::-1]
            x_e *= -1.0
        z = 2 * lmbda * np.exp(-c * (x - x_e))
        return (
            self.normalization(n, lmbda, c) *
             z**(lmbda - n - 0.5) * np.exp(-z / 2) * scipy.special.genlaguerre(n, 2 * lmbda - 2 * n - 1)(z)
        )
    
    def normalization(self, n, lmbda, c):
        return (
            (scipy.special.factorial(n) * (2 * lmbda - 2 * n - 1) * c / scipy.special.gamma(2 * lmbda - n))**0.5 # Gamma(n+1) = factorial(n)
        )

    def compute_eigenenergies(self, c, D, l):
        hnu = 2 * c * np.sqrt(D / 2)
        E_n = np.zeros(l)
        for n in range(l):
            E_n[n] = hnu * (n + 0.5) - (c * hnu * (n + 0.5)**2) / np.sqrt(8 * D)

        return E_n

    def find_u_matrix(self, grid, alpha, a, spf_l, spf_r)->np.ndarray:
        """
        Calculate the interaction matrix elements of shielded Coulomb (1D) by einstein-summation. A bit slower, and less accurate than the trapezoidal numeric integration
        but is more easily done for a bipartite system when this is required. Shielded coulomb is alpha / |x1 - x2 + a|

        args:
        grid: np.array
            Grid (in position space) where the interactions should be computed
        alpha: float
            Scaling parameter in coulomb
        a: float
            Shielding parameter in coulomb
        spf_l: np.array
            Single-particle functions in the left well.
        spf_r: np.array
            Single-particle functions in the right well.
        """
        l = spf_l.shape[0] # number of basis functions
        num_grid = spf_l.shape[1] # number of grid points
        u = np.zeros((l,l,l,l), dtype=np.complex128)
        # Find the Coulomb-interactions on the grid
        dx = grid[1] - grid[0]
        coulomb = np.zeros((num_grid, num_grid), dtype=np.complex128)
        for i in range(num_grid):
            coulomb[i] = _shielded_coulomb(grid[i], grid, alpha, a)
        

        # from opt_einsum import contract
        # # Make the integration (a sum over indices)
        # u = contract('ix, jy, xy, kx, ly -> ijkl', spf_l.conj(), spf_r.conj(), coulomb, spf_l, spf_r)
        u = np.einsum('ix, jy, xy, kx, ly -> ijkl', spf_l, spf_r, coulomb, spf_l, spf_r, optimize=True)

        return u




    def construct_position_integrals(self, lmbda, spf, l):
        """
        Analytical expressions for the position integrals in the Morse potential basis.
        Taken from doi:10.1088/0953-4075/38/7/004
        """
        position = np.zeros((1, self.l, self.l), dtype=spf.dtype)
        N = lmbda - 0.5
        # Loop through all n < m, to find elements <n|x|m>
        for n in range(l - 1):
            for m in range(n+1, l):
                pre_factor = 2 * (-1)**(m-n+1) /((m-n) * (2 * N -m))
                position[0, n, m] = (
                    pre_factor * np.sqrt((N-n) * (N-m) * scipy.special.gamma(2 * N - m + 1) * scipy.special.factorial(m) / 
                    (scipy.special.gamma(2 * N - n + 1) * scipy.special.factorial(n)))
                )
        # Symmetrize the position matrix with the hermitian conjugate, i.e adding <m|x|n> (since <n|x|m> = <m|x|n>^H)
        position[0, :, :] += position[0, :, :].conj().T
        # Diagonal elements <n|x|n>
        for n in range(self.l):
            position[0, n, n] = (
                np.log(2 * N + 1) + scipy.special.psi(2 * N - n + 1) - scipy.special.psi(2 * N - 2 * n + 1) - scipy.special.psi(2 * N - 2 * n)
            )

        return position