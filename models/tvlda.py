import torch


class TVLDA:
    """
    Time-variant LDA (two classes).

    Input shape
    -----------
    N : n_observations (trials)
    F : n_features (channels x features)
    W : n_windows (already separated)
    xA , xB : (N_A, F, W) and (N_B, F, W)
        N_A / N_B   trials (observations) in class-A / class-B
        F           features per window   (e.g., channels x samples)
        W           number of time windows  (already separated)

    After .fit():
        self.w : (W, F)   projection vectors, unit-norm
        self.b : (W,)     intercepts
    After .fit_score(X, expected_label):
        self.expected_sign : 1 or -1
        self.label         : chosen label (1 or -1)
    After .get_label(X):
        return the label based on X
    """

    def __init__(self, *, lamb: float = 1e-4, device: str = "cpu"):
        self.lamb   = lamb
        self.device = device
        self.w:  torch.Tensor | None = None
        self.b:  torch.Tensor | None = None
        self.expected_sign: int | None = None  # sign (1 or -1) from fit_score
        self.label: int | None = None  # chosen label after fit_score


    # ------------------------------------------------------------------ #
    def fit(self, xA: torch.Tensor, xB: torch.Tensor):
        """Train one LDA per window."""
        xA, xB = xA.to(self.device), xB.to(self.device)       # (N_A,F,W), (N_B,F,W)
        NA, F, W = xA.shape
        NB       = xB.size(0)
        n_cls    = 2

        # ---- reshape to (W, N, F) for concise maths -------------------- #
        XA = xA.permute(2, 0, 1)                              # (W, N_A, F)
        XB = xB.permute(2, 0, 1)                              # (W, N_B, F)

        # ---- class means per window ----------------------------------- #
        muA = XA.mean(1)                                      # (W, F)
        muB = XB.mean(1)                                      # (W, F)
        mu  = (NA * muA + NB * muB) / (NA + NB)               # (W, F)

        # ---- within-class scatter Sw (W,F,F) -------------------------- #
        XA_c = XA - muA[:, None, :]                           # (W,N_A,F)
        XB_c = XB - muB[:, None, :]
        Sw_A = XA_c.transpose(-1, -2) @ XA_c / (NA - 1)       # (W,F,F)
        Sw_B = XB_c.transpose(-1, -2) @ XB_c / (NB - 1)
        Sw   = (Sw_A + Sw_B) / n_cls                          # average

        # ---- total scatter St  ---------------------------------------- #
        X_all = torch.cat([XA, XB], 1)                        # (W,N_A+N_B,F)
        X_c   = X_all - mu[:, None, :]
        St    = X_c.transpose(-1, -2) @ X_c / (NA + NB - 1)   # (W,F,F)

        # ---- between-class scatter ------------------------------------ #
        Sb = St - Sw                                          # (W,F,F)

        # ---- regularise Sw and solve eigen-problem -------------------- #
        Sw += self.lamb * torch.eye(F, device=self.device)    # broadcast
        Sw_inv = torch.linalg.pinv(Sw)                        # (W,F,F)
        M      = Sw_inv @ Sb                                  # (W,F,F)

        # batched eig (complex); keep real parts
        eigvals, eigvecs = torch.linalg.eig(M)
        eigvals, eigvecs = eigvals.real, eigvecs.real         # (W,F), (W,F,F)

        # select eigenvector with largest eigenvalue per window
        idx_max   = eigvals.argmax(-1)                        # (W,)
        gather_ix = idx_max[:, None, None].expand(-1, F, 1)   # (W,F,1)
        w         = torch.gather(eigvecs, 2, gather_ix).squeeze(-1)  # (W,F)
        w         = torch.nn.functional.normalize(w, dim=1)   # unit-norm

        # intercept   b = 0.5(u_A+u_B)·w   for every window
        b = -0.5 * (muA + muB).mul(w).sum(1)                  # (W,)

        self.w, self.b = w, b

    # ------------------------------------------------------------------ #
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (N, F, W)  →  per-window scores : (N, W)
        """
        if self.w is None:
            raise RuntimeError("Call .fit() first.")

        x = x.to(self.device)                                 # (N,F,W)
        scores = torch.einsum("nfw,wf->nw", x, self.w) + self.b  # (N,W)
        return scores

    # ------------------------------------------------------------------ #
    def score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scalar TVLDA score per trial: sum of all window scores.
        Shape: (N,)
        """
        return self.transform(x).sum(1)

    # ------------------------------------------------------------------ #
    def fit_score(self, X: torch.Tensor, label: int):
        """
        Evaluate the scores for input X (must have shape (N, F, W)) and
        record the expected sign based on the average score.
        The provided label is taken as positive if the average score is >= 0,
        otherwise negative.
        """
        # Compute the scalar scores for all trials.
        s = self.score(X)
        avg_score = s.mean().item()
        # Store the expected sign and label.
        self.expected_sign = 1 if avg_score >= 0 else -1
        self.label = label if self.expected_sign > 0 else -label
        return avg_score

    # ------------------------------------------------------------------ #
    def get_label(self, X: torch.Tensor) -> int:
        """
        Compute the score for input X and return:
          label   if the score sign matches the expected sign
         -label   otherwise.
        """
        if self.expected_sign is None or self.label is None:
            raise RuntimeError("Call fit_score() first.")
        s = self.score(X)  # s has shape (N_samples,)
        trial_sign = torch.where(s >= 0, 1, -1)
        labels = torch.where(trial_sign == self.expected_sign, self.label, -self.label)
        return labels