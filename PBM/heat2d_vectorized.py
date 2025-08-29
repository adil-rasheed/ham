# 2D Heat Conduction (Crank–Nicolson + Picard), vectorized assembly with spdiags
# - Dirichlet / Neumann / Robin BCs on all sides (vectorized along edges)
# - Variable properties rho(X,Y,t,T), cp(X,Y,t,T), k(X,Y,t,T)
# - Source qdot(X,Y,t)
# - Uniform rect grid; conservative 5-point stencil; harmonic means at faces

import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Literal, Optional

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

BCType = Literal["Dirichlet", "Neumann", "Robin"]  # Robin: -k dT/dn = h (T_surface - T_inf); flux INTO domain positive


# ---------------- Problem definition containers ----------------

@dataclass
class Material2D:
    rho: Callable[[np.ndarray, np.ndarray, float, np.ndarray], np.ndarray]  # rho(X,Y,t,T)
    cp:  Callable[[np.ndarray, np.ndarray, float, np.ndarray], np.ndarray]  # cp(X,Y,t,T)
    k:   Callable[[np.ndarray, np.ndarray, float, np.ndarray], np.ndarray]  # k(X,Y,t,T) (isotropic)

@dataclass
class Source2D:
    qdot: Callable[[np.ndarray, np.ndarray, float], np.ndarray]             # qdot(X,Y,t) [W/m^3]

@dataclass
class BC2D:
    # Vectorized edge BCs:
    #   left(y,t)  -> (type, value) where value is:
    #       Dirichlet: array (Ny,)
    #       Neumann:   array (Ny,)  [W/m^2], positive INTO domain
    #       Robin:     (h_array, Tinf_array), each (Ny,)
    # Similarly:
    #   right(y,t), bottom(x,t), top(x,t)
    left:   Callable[[np.ndarray, float], Tuple[BCType, object]]
    right:  Callable[[np.ndarray, float], Tuple[BCType, object]]
    bottom: Callable[[np.ndarray, float], Tuple[BCType, object]]
    top:    Callable[[np.ndarray, float], Tuple[BCType, object]]

@dataclass
class Grid2D:
    Lx: float
    Ly: float
    Nx: int
    Ny: int
    def __post_init__(self):
        assert self.Nx >= 2 and self.Ny >= 2, "Need at least 2 nodes in each direction"
        self.x = np.linspace(0.0, self.Lx, self.Nx)
        self.y = np.linspace(0.0, self.Ly, self.Ny)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")  # (Nx, Ny)


# ---------------- Helpers ----------------

def _harm(a, b, eps=1e-20):
    """Harmonic mean with protection against division by zero."""
    return 2.0 * a * b / np.maximum(a + b, eps)

def _spdiags_from_diagonals(diags_vals, offsets, shape):
    """Create CSR matrix from list of diagonal arrays (length Ntot each) and offsets."""
    if _HAVE_SCIPY:
        A = sp.spdiags(diags_vals, offsets, shape[0], shape[1]).tocsr()
        return A
    # Dense fallback
    A = np.zeros(shape)
    Ntot = shape[0]
    for vals, off in zip(diags_vals, offsets):
        vals = np.asarray(vals)
        if off >= 0:
            rows = np.arange(0, Ntot - off)
            A[rows + off, rows] += vals[:Ntot - off]
        else:
            rows = np.arange(0, Ntot + off)  # off is negative
            A[rows, rows - off] += vals[-off:Ntot]
    return A


# ---------------- Solver ----------------

class Heat2DSolverVectorized:
    r"""
    rho(x,y,t,T)*cp(x,y,t,T) dT/dt = ∂/∂x(k ∂T/∂x) + ∂/∂y(k ∂T/∂y) + qdot(x,y,t)

    - Uniform grid, conservative 5-point stencil (harmonic means at faces)
    - Crank–Nicolson (theta) time integration
    - Picard iterations for T-dependent properties
    - Dirichlet / Neumann / Robin on each of the four sides (vectorized along edges)
    """

    def __init__(
        self,
        grid: Grid2D,
        material: Material2D,
        source: Source2D,
        bc: BC2D,
        theta: float = 0.5,
        max_iter: int = 15,
        tol: float = 1e-8,
    ):
        assert 0.0 < theta <= 1.0
        self.g = grid
        self.mat = material
        self.src = source
        self.bc = bc
        self.theta = theta
        self.max_iter = max_iter
        self.tol = tol

    def _assemble_CN(self, Tn: np.ndarray, dt: float, t_n: float, t_np1: float) -> np.ndarray:
        Nx, Ny = self.g.Nx, self.g.Ny
        dx, dy = self.g.dx, self.g.dy
        X, Y   = self.g.X, self.g.Y
        th     = self.theta
        Ntot   = Nx * Ny

        # --- Properties at t^n (explicit) ---
        rho_n = self.mat.rho(X, Y, t_n, Tn)
        cp_n  = self.mat.cp (X, Y, t_n, Tn)
        k_n   = self.mat.k  (X, Y, t_n, Tn)
        Cn    = rho_n * cp_n

        # Face diffusivities at t^n (for explicit L(Tn))
        De_n = _harm(k_n[1:, :], k_n[:-1, :]) / (dx*dx)   # between i and i+1 -> shape (Nx-1, Ny)
        Dn_n = _harm(k_n[:, 1:], k_n[:, :-1]) / (dy*dy)   # between j and j+1 -> shape (Nx, Ny-1)

        # Explicit diffusion operator L(Tn)
        L_n = np.zeros_like(Tn)
        # x-direction
        L_n[1:-1, :] += De_n[1:, :] * (Tn[2:, :] - Tn[1:-1, :])
        L_n[1:-1, :] -= De_n[:-1, :] * (Tn[1:-1, :] - Tn[:-2, :])
        # y-direction
        L_n[:, 1:-1] += Dn_n[:, 1:] * (Tn[:, 2:] - Tn[:, 1:-1])
        L_n[:, 1:-1] -= Dn_n[:, :-1] * (Tn[:, 1:-1] - Tn[:, :-2])

        # Sources
        q_n   = self.src.qdot(X, Y, t_n)
        q_np1 = self.src.qdot(X, Y, t_np1)

        # --- Picard iterations for T-dependent properties ---
        Tstar = Tn.copy()
        for _ in range(self.max_iter):
            rho_np1 = self.mat.rho(X, Y, t_np1, Tstar)
            cp_np1  = self.mat.cp (X, Y, t_np1, Tstar)
            k_np1   = self.mat.k  (X, Y, t_np1, Tstar)
            Cnp1    = rho_np1 * cp_np1

            # Face diffusivities at t^{n+1} (for implicit couplings)
            De = _harm(k_np1[1:, :], k_np1[:-1, :]) / (dx*dx)    # (Nx-1, Ny)
            Dn = _harm(k_np1[:, 1:], k_np1[:, :-1]) / (dy*dy)    # (Nx, Ny-1)

            # --- Build 5 diagonals as full arrays ---
            # Coeffs to neighbors (negative values); diagonal accumulates positives
            diag0 = np.ones((Nx, Ny), dtype=float)
            diag_w = np.zeros((Nx, Ny), dtype=float)   # offset -Ny (west)
            diag_e = np.zeros((Nx, Ny), dtype=float)   # offset +Ny (east)
            diag_s = np.zeros((Nx, Ny), dtype=float)   # offset -1  (south)
            diag_n = np.zeros((Nx, Ny), dtype=float)   # offset +1  (north)

            # Implicit neighbor couplings (interior faces)
            # West: contribution from face between i-1 and i  → rows i=1..Nx-1 use De[i-1,:]
            diag_w[1:, :]  = - th * dt * De[:,    :] / Cnp1[1:, :]

            # East: contribution from face between i and i+1  → rows i=0..Nx-2 use De[i,:]
            diag_e[:-1, :] = - th * dt * De[:,    :] / Cnp1[:-1, :]

            # South: face between j-1 and j  → rows j=1..Ny-1 use Dn[:, j-1]
            diag_s[:, 1:]  = - th * dt * Dn[:, :-1] / Cnp1[:, 1:]

            # North: face between j and j+1  → rows j=0..Ny-2 use Dn[:, j]
            diag_n[:, :-1] = - th * dt * Dn[:,    :] / Cnp1[:, :-1]


            # Diagonal accrues positive sums (same denominators as each row)
            diag0[1:,  :] += -diag_w[1:,  :]
            diag0[:-1, :] += -diag_e[:-1, :]
            diag0[:, 1:]  += -diag_s[:, 1:]
            diag0[:, :-1] += -diag_n[:, :-1]

            # RHS vector (flattened)
            d = (Tn + (1 - th) * dt * (L_n / Cn) + dt * (th * (q_np1 / Cnp1) + (1 - th) * (q_n / Cn))).ravel()

            # ------------- Boundary conditions (vectorized masks) -------------
            left_mask   = np.zeros((Nx, Ny), dtype=bool); left_mask[0, :]  = True
            right_mask  = np.zeros((Nx, Ny), dtype=bool); right_mask[-1, :] = True
            bottom_mask = np.zeros((Nx, Ny), dtype=bool); bottom_mask[:, 0] = True
            top_mask    = np.zeros((Nx, Ny), dtype=bool); top_mask[:, -1] = True

            y_edge = self.g.y
            x_edge = self.g.x

            # --- LEFT edge ---
            bL_type, bL_np1 = self.bc.left(y_edge, t_np1)
            _,      bL_n    = self.bc.left(y_edge, t_n)
            if bL_type == "Dirichlet":
                vals = np.asarray(bL_np1, dtype=float)
                diag_w[left_mask] = diag_e[left_mask] = diag_s[left_mask] = diag_n[left_mask] = 0.0
                diag0[left_mask]  = 1.0
                d[left_mask.ravel()] = vals
            elif bL_type == "Neumann":
                qL_np1 = np.asarray(bL_np1, dtype=float)  # W/m^2 into domain
                qL_n   = np.asarray(bL_n,   dtype=float)
                Cnp1_L = Cnp1[0, :]; Cn_L = Cn[0, :]
                add = dt * ( th * (2.0 * qL_np1 / (Cnp1_L * dx)) + (1 - th) * (2.0 * qL_n / (Cn_L * dx)) )
                d[left_mask.ravel()] += add
            elif bL_type == "Robin":
                h_np1, Tinf_np1 = bL_np1
                h_np1   = np.asarray(h_np1, dtype=float)
                Tinf_np1= np.asarray(Tinf_np1, dtype=float)
                h_n, Tinf_n = bL_n
                h_n     = np.asarray(h_n, dtype=float)
                Tinf_n  = np.asarray(Tinf_n, dtype=float)
                Cnp1_L = Cnp1[0, :]; Cn_L = Cn[0, :]
                diag0[left_mask] += th * dt * (2.0 * h_np1 / (Cnp1_L * dx))
                d[left_mask.ravel()] += th * dt * (2.0 * h_np1 * Tinf_np1) / (Cnp1_L * dx)
                d[left_mask.ravel()] += (1 - th) * dt * (2.0 * h_n * (Tinf_n - Tn[0, :])) / (Cn_L * dx)
            else:
                raise ValueError("Unknown BC type on left edge")

            # --- RIGHT edge ---
            bR_type, bR_np1 = self.bc.right(y_edge, t_np1)
            _,      bR_n    = self.bc.right(y_edge, t_n)
            if bR_type == "Dirichlet":
                vals = np.asarray(bR_np1, dtype=float)
                diag_w[right_mask] = diag_e[right_mask] = diag_s[right_mask] = diag_n[right_mask] = 0.0
                diag0[right_mask]  = 1.0
                d[right_mask.ravel()] = vals
            elif bR_type == "Neumann":
                qR_np1 = np.asarray(bR_np1, dtype=float)
                qR_n   = np.asarray(bR_n,   dtype=float)
                Cnp1_R = Cnp1[-1, :]; Cn_R = Cn[-1, :]
                add = dt * ( th * (2.0 * qR_np1 / (Cnp1_R * dx)) + (1 - th) * (2.0 * qR_n / (Cn_R * dx)) )
                d[right_mask.ravel()] += add
            elif bR_type == "Robin":
                h_np1, Tinf_np1 = bR_np1
                h_np1   = np.asarray(h_np1, dtype=float)
                Tinf_np1= np.asarray(Tinf_np1, dtype=float)
                h_n, Tinf_n = bR_n
                h_n     = np.asarray(h_n, dtype=float)
                Tinf_n  = np.asarray(Tinf_n, dtype=float)
                Cnp1_R = Cnp1[-1, :]; Cn_R = Cn[-1, :]
                diag0[right_mask] += th * dt * (2.0 * h_np1 / (Cnp1_R * dx))
                d[right_mask.ravel()] += th * dt * (2.0 * h_np1 * Tinf_np1) / (Cnp1_R * dx)
                d[right_mask.ravel()] += (1 - th) * dt * (2.0 * h_n * (Tinf_n - Tn[-1, :])) / (Cn_R * dx)
            else:
                raise ValueError("Unknown BC type on right edge")

            # --- BOTTOM edge ---
            bB_type, bB_np1 = self.bc.bottom(self.g.x, t_np1)
            _,      bB_n    = self.bc.bottom(self.g.x, t_n)
            if bB_type == "Dirichlet":
                vals = np.asarray(bB_np1, dtype=float)
                diag_w[bottom_mask] = diag_e[bottom_mask] = diag_s[bottom_mask] = diag_n[bottom_mask] = 0.0
                diag0[bottom_mask]  = 1.0
                d[bottom_mask.ravel()] = vals
            elif bB_type == "Neumann":
                qB_np1 = np.asarray(bB_np1, dtype=float)
                qB_n   = np.asarray(bB_n,   dtype=float)
                Cnp1_B = Cnp1[:, 0]; Cn_B = Cn[:, 0]
                add = dt * ( th * (2.0 * qB_np1 / (Cnp1_B * dy)) + (1 - th) * (2.0 * qB_n / (Cn_B * dy)) )
                d[bottom_mask.ravel()] += add
            elif bB_type == "Robin":
                h_np1, Tinf_np1 = bB_np1
                h_np1   = np.asarray(h_np1, dtype=float)
                Tinf_np1= np.asarray(Tinf_np1, dtype=float)
                h_n, Tinf_n = bB_n
                h_n     = np.asarray(h_n, dtype=float)
                Tinf_n  = np.asarray(Tinf_n, dtype=float)
                Cnp1_B = Cnp1[:, 0]; Cn_B = Cn[:, 0]
                diag0[bottom_mask] += th * dt * (2.0 * h_np1 / (Cnp1_B * dy))
                d[bottom_mask.ravel()] += th * dt * (2.0 * h_np1 * Tinf_np1) / (Cnp1_B * dy)
                d[bottom_mask.ravel()] += (1 - th) * dt * (2.0 * h_n * (Tinf_n - Tn[:, 0])) / (Cn_B * dy)
            else:
                raise ValueError("Unknown BC type on bottom edge")

            # --- TOP edge ---
            bT_type, bT_np1 = self.bc.top(self.g.x, t_np1)
            _,      bT_n    = self.bc.top(self.g.x, t_n)
            if bT_type == "Dirichlet":
                vals = np.asarray(bT_np1, dtype=float)
                diag_w[top_mask] = diag_e[top_mask] = diag_s[top_mask] = diag_n[top_mask] = 0.0
                diag0[top_mask]  = 1.0
                d[top_mask.ravel()] = vals
            elif bT_type == "Neumann":
                qT_np1 = np.asarray(bT_np1, dtype=float)
                qT_n   = np.asarray(bT_n,   dtype=float)
                Cnp1_T = Cnp1[:, -1]; Cn_T = Cn[:, -1]
                add = dt * ( th * (2.0 * qT_np1 / (Cnp1_T * dy)) + (1 - th) * (2.0 * qT_n / (Cn_T * dy)) )
                d[top_mask.ravel()] += add
            elif bT_type == "Robin":
                h_np1, Tinf_np1 = bT_np1
                h_np1   = np.asarray(h_np1, dtype=float)
                Tinf_np1= np.asarray(Tinf_np1, dtype=float)
                h_n, Tinf_n = bT_n
                h_n     = np.asarray(h_n, dtype=float)
                Tinf_n  = np.asarray(Tinf_n, dtype=float)
                Cnp1_T = Cnp1[:, -1]; Cn_T = Cn[:, -1]
                diag0[top_mask] += th * dt * (2.0 * h_np1 / (Cnp1_T * dy))
                d[top_mask.ravel()] += th * dt * (2.0 * h_np1 * Tinf_np1) / (Cnp1_T * dy)
                d[top_mask.ravel()] += (1 - th) * dt * (2.0 * h_n * (Tinf_n - Tn[:, -1])) / (Cn_T * dy)
            else:
                raise ValueError("Unknown BC type on top edge")

            # ---- Build sparse matrix with spdiags ----
            # Flatten diagonals (row-major) with offsets relative to k=i*Ny + j
            diag0_f = diag0.ravel()
            west_f  = diag_w.ravel()     # offset -Ny
            east_f  = diag_e.ravel()     # offset +Ny
            south_f = diag_s.ravel()     # offset -1
            north_f = diag_n.ravel()     # offset +1

            offsets = np.array([-self.g.Ny, -1, 0, +1, +self.g.Ny])
            diags_vals = [west_f, south_f, diag0_f, north_f, east_f]
            A = _spdiags_from_diagonals(diags_vals, offsets, (Ntot, Ntot))

            # Solve A T = d
            if _HAVE_SCIPY:
                Tvec = spla.spsolve(A, d)
            else:
                Tvec = np.linalg.solve(A, d)
            Tnew = Tvec.reshape((Nx, Ny))

            # Convergence
            if np.linalg.norm(Tnew - Tstar, ord=np.inf) < self.tol:
                return Tnew
            Tstar = Tnew

        return Tstar  # last iterate if not converged

    def step(self, Tn: np.ndarray, dt: float, t_n: float):
        Tnp1 = self._assemble_CN(Tn, dt, t_n, t_n + dt)
        return Tnp1, t_n + dt

    def run(self, T0: np.ndarray, t0: float, tf: float, dt: float,
            callback: Optional[Callable[[float, np.ndarray], None]] = None):
        T = T0.copy()
        t = t0
        while t < tf - 1e-15:
            dt_eff = min(dt, tf - t)
            T, t = self.step(T, dt_eff, t)
            if callback is not None:
                callback(t, T)
        return T, t


# ---------------- Example usage ----------------
if __name__ == "__main__":
    # Grid
    Lx, Ly = 0.01, 0.002          # 10 mm x 2 mm
    Nx, Ny = 81, 21
    grid = Grid2D(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny)

    # Material (constants here; can be T/x/y/t-dependent)
    k0, rho0, cp0 = 0.26, 1300.0, 2000.0
    rho = lambda X, Y, t, T: np.full_like(X, rho0)
    cp  = lambda X, Y, t, T: np.full_like(X, cp0)
    k   = lambda X, Y, t, T: np.full_like(X, k0)
    material = Material2D(rho=rho, cp=cp, k=k)

    # Source
    qdot = lambda X, Y, t: np.zeros_like(X)
    source = Source2D(qdot=qdot)

    # BCs: 250 C on left/right, insulated top/bottom
    TL = TR = 250.0
    bc = BC2D(
        left   = lambda y, t: ("Dirichlet", np.full_like(y, TL, dtype=float)),
        right  = lambda y, t: ("Dirichlet", np.full_like(y, TR, dtype=float)),
        bottom = lambda x, t: ("Neumann",   np.zeros_like(x, dtype=float)),  # adiabatic
        top    = lambda x, t: ("Neumann",   np.zeros_like(x, dtype=float)),  # adiabatic
    )

    # Initial condition
    T0 = np.full((Nx, Ny), 30.0, dtype=float)

    # Solver
    solver = Heat2DSolverVectorized(grid, material, source, bc, theta=0.5, max_iter=20, tol=1e-9)

    # Time step: choose for accuracy (CN is unconditionally stable for linear problems)
    alpha = k0 / (rho0 * cp0)
    dt = 0.25 * min(grid.dx**2, grid.dy**2) / alpha

    # Run
    T_end, _ = solver.run(T0, 0.0, 600.0, dt)
    print("Done. T(center) =", T_end[Nx//2, Ny//2])
