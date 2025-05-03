import torch


class TVLDA:
    """
    Channel-wise, time-variant LDA (two classes) with full covariance.

    * xA, xB  : (N_A, C, T) and (N_B, C, T)   - trials x channels x time
    * time_windows : [(start, end), ...]       - end exclusive
    """

    def __init__(self, time_windows, *, lamb=1e-4, device="cpu"):
        self.time_windows = time_windows            # list[(s,e)]
        self.lamb   = lamb
        self.device = device
        self.w      = None      # (C, W, L)   projection vectors
        self.b      = None      # (C, W)      intercepts

    # -------------------------------------------------------------- #
    def _stack_windows(self, x):
        """(N, C, T) → (N, C, W, L)"""
        return torch.stack([x[..., s:e] for s, e in self.time_windows], 2)

    # -------------------------------------------------------------- #
    def fit(self, xA: torch.Tensor, xB: torch.Tensor):
        xA, xB = xA.to(self.device), xB.to(self.device)
        XA = self._stack_windows(xA)          # (N_A, C, W, L)
        XB = self._stack_windows(xB)          # (N_B, C, W, L)

        N_A, C, W, L = XA.shape
        N_B = XB.size(0)
        n_cls = 2
        eye_L = torch.eye(L, device=self.device)

        # ---------- means -------------------------------------------------- #
        muA = XA.mean(0)                     # (C, W, L)
        muB = XB.mean(0)
        mu  = (N_A * muA + N_B * muB) / (N_A + N_B)

        # ---------- within-class scatter Sw (C, W, L, L) ------------------- #
        # class-A
        XA_c = (XA - muA.unsqueeze(0)).permute(2, 1, 0, 3)   # W,C,N_A,L
        Sw_A = torch.matmul(XA_c.transpose(-1, -2), XA_c) / (N_A - 1)  # W,C,L,L
        # class-B
        XB_c = (XB - muB.unsqueeze(0)).permute(2, 1, 0, 3)   # W,C,N_B,L
        Sw_B = torch.matmul(XB_c.transpose(-1, -2), XB_c) / (N_B - 1)  # W,C,L,L
        # average over classes
        Sw   = (Sw_A + Sw_B) / n_cls                          # W,C,L,L
        Sw   = Sw.permute(1, 0, 2, 3)                         # C,W,L,L

        # ---------- total scatter St  -------------------------------------- #
        X_all = torch.cat([XA, XB], 0)                        # (N,C,W,L)
        X_c   = (X_all - mu.unsqueeze(0)).permute(2, 1, 0, 3) # W,C,N,L
        St    = torch.matmul(X_c.transpose(-1, -2), X_c) / (N_A + N_B - 1)
        St    = St.permute(1, 0, 2, 3)                        # C,W,L,L

        # ---------- between-class scatter Sb = St - Sw --------------------- #
        Sb = St - Sw                                          # C,W,L,L

        # ---------- regularise and solve eig-problem ----------------------- #
        Sw += self.lamb * eye_L                                # broadcast
        Sw_inv = torch.linalg.pinv(Sw)                         # batched pinv
        temp   = Sw_inv @ Sb                                   # (C,W,L,L)

        # eig across last 2 dims; PyTorch 2.1+ supports batched eig
        eigvals, eigvecs = torch.linalg.eig(temp)              # complex
        eigvals = eigvals.real                                 # (C,W,L)
        eigvecs = eigvecs.real

        # pick eigenvector of largest eigenvalue per (C,W)
        idx_max = eigvals.argmax(-1)                           # (C,W)
        idx_exp = idx_max.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, L, 1)
        w = torch.gather(eigvecs, -1, idx_exp).squeeze(-1)     # (C,W,L)
        w = torch.nn.functional.normalize(w, dim=-1)           # unit-norm

        # ---------- intercept  b = −½(μ_A+μ_B)·w --------------------------- #
        b = -0.5 * ((muA + muB) * w).sum(-1)                   # (C,W)

        self.w, self.b = w, b

    # -------------------------------------------------------------- #
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (N, C, T)  →  scores : (N, C, W)
        """
        if self.w is None:
            raise RuntimeError("Call .fit() first.")
        X = self._stack_windows(x.to(self.device))             # (N,C,W,L)
        scores = (X * self.w.unsqueeze(0)).sum(-1) + self.b.unsqueeze(0)
        return scores

    # -------------------------------------------------------------- #
    def score(self, x: torch.Tensor) -> torch.Tensor:
        """scalar TVLDA score per trial"""
        return self.transform(x).sum(dim=(1, 2))
