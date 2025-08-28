import torch
import torch.nn as nn

# ---------------------------
# Global double precision
# ---------------------------
torch.set_default_dtype(torch.float64)

# ---------------------------
# Utilities
# ---------------------------

def custom_cdist(x1, x2):
    """
    Computes squared Euclidean distances between x1 and x2.
    x1: [N, 1], x2: [M, 1]
    Returns [N, M]
    """
    x1 = x1.double()
    x2 = x2.double()
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    x1_pad = torch.ones_like(x1_norm, dtype=torch.float64)
    x2_pad = torch.ones_like(x2_norm, dtype=torch.float64)
    x1_ = torch.cat([-2. * x1, x1_norm, x1_pad], dim=-1)
    x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
    res = x1_.matmul(x2_.transpose(-2, -1))
    return res.clamp_min(0.0)

class NonStationaryGP(nn.Module):
    def __init__(self):
        super().__init__()
        # Kernel parameters
        self.ell_raw = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.max_val_raw = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.slope_raw = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.loc_raw = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.sigma_n_raw = nn.Parameter(torch.tensor(0.01, dtype=torch.float64))
        
        # Exponential mean parameters
        self.A_raw = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))      # amplitude
        self.tau_raw = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))    # decay time

    # Kernel properties
    @property
    def ell(self): return torch.exp(self.ell_raw)
    @property
    def max_val(self): return torch.exp(self.max_val_raw)
    @property
    def slope(self): return torch.exp(self.slope_raw)
    @property
    def loc(self): return torch.exp(self.loc_raw)
    @property
    def sigma_n(self): return torch.exp(self.sigma_n_raw)

    # Mean properties
    @property
    def A(self): return torch.exp(self.A_raw)
    @property
    def tau(self): return torch.exp(self.tau_raw)

    # Exponential mean function
    def mean_fxn(self, t):
        return self.A * torch.exp(-t / self.tau)

    # Width function for Gibbs kernel
    def width_fxn(self, t):
        return self.max_val / (1 + torch.exp(self.slope * (t - self.loc)))

    # Gibbs kernel
    def f_kernel(self, X1, X2):
        Xdd = custom_cdist(X1, X2)
        σ1 = self.width_fxn(X1)
        σ2 = self.width_fxn(X2)
        prefactor = torch.outer(σ1.squeeze(), σ2.squeeze())
        exponential = torch.exp(-Xdd / (2 * self.ell**2))
        return prefactor * exponential

    # Negative log marginal likelihood
    def NEG_LMLH(self, X, Y):
        n = len(X)
        m = self.mean_fxn(X).squeeze()  # mean vector
        Y_centered = Y - m
        K = self.f_kernel(X, X) + (self.sigma_n ** 2) * torch.eye(n, dtype=torch.float64)
        K = 0.5 * (K + K.T)  # symmetrize
        L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(Y_centered.unsqueeze(-1), L)
        nll = 0.5 * (Y_centered.unsqueeze(-1).T @ alpha) + torch.sum(torch.log(torch.diag(L))) + 0.5 * n * torch.log(torch.tensor(2 * torch.pi, dtype=torch.float64))
        return nll.squeeze()

    # Posterior prediction
    def predict(self, X_train, Y_train, X_test):
        m_train = self.mean_fxn(X_train).squeeze()
        Y_centered = Y_train - m_train
        n_train = len(X_train)
        n_test = len(X_test)
        K = self.f_kernel(X_train, X_train) + (self.sigma_n**2) * torch.eye(n_train, dtype=torch.float64)
        K_s = self.f_kernel(X_test, X_train)
        K_ss = self.f_kernel(X_test, X_test) + 1e-8 * torch.eye(n_test, dtype=torch.float64)
        K = 0.5 * (K + K.T)
        K_ss = 0.5 * (K_ss + K_ss.T)
        L = torch.linalg.cholesky(K)
        mu_post = K_s @ torch.cholesky_solve(Y_centered.unsqueeze(-1), L)
        mu_post = mu_post.squeeze() + self.mean_fxn(X_test).squeeze()  # add mean back
        v = torch.cholesky_solve(K_s.T.double(), L)
        cov_post = K_ss - K_s @ v
        return mu_post, cov_post

    def print_params(self):
        print("Optimized parameters:")
        print(f"  ell      = {self.ell.item():.4f}")
        print(f"  max_val  = {self.max_val.item():.4f}")
        print(f"  slope    = {self.slope.item():.4f}")
        print(f"  loc      = {self.loc.item():.4f}")
        print(f"  sigma_n  = {self.sigma_n.item():.6f}")
        print(f"  A        = {self.A.item():.4f}")
        print(f"  tau      = {self.tau.item():.4f}")