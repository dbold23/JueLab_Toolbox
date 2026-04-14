"""
Posterior-predictive sampler for synthetic growth-curve augmentation (Phase B.1).

Loads a fitted hierarchical Bayesian Gompertz trace (produced by
``06_advanced_fitting.py``) and samples new (A, μ, λ) parameter tuples from
the joint posterior, then generates curves with residual-bootstrapped noise.

This respects the A/μ/λ *joint* correlations that hardcoded uniform sampling
destroys (Zinsli 2024, Fang 2023 — see tasks plan), and uses real TECAN noise
via ResidualBootstrapNoise.

Interface mirrors ``SyntheticGrowthCurveGenerator`` so the existing scenario
pipeline can route ``pattern='ppc'`` here.

Primary entry:

    sampler = PosteriorPredictiveSampler()
    sampler.load(trace_path)
    params = sampler.sample_parameters(n=1000)             # (A, μ, λ) DataFrame
    curves = sampler.sample_curves(
        n=1000,
        grids=[(t_array, n_points)],                        # empirical grid samples
        noise=ResidualBootstrapNoise(residual_pool=pool),   # real-noise bootstrap
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    import arviz as az  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - arviz is a hard dep in production
    az = None  # noqa: F841


def gompertz(time: np.ndarray, A: float, mu: float, lam: float) -> np.ndarray:
    """Gompertz growth model — same parameterization as 01_growth_curve_analysis.py."""
    return A * np.exp(-np.exp((mu * np.e / A) * (lam - time) + 1))


class PosteriorPredictiveSampler:
    """Sample growth curves from a fitted Bayesian Gompertz trace."""

    #: posterior variable names expected in the trace (see 06_advanced_fitting.py)
    A_VAR = "A_strain"
    MU_VAR = "mu_strain"
    LAM_VAR = "lam_strain"

    def __init__(self) -> None:
        self._samples: Optional[pd.DataFrame] = None  # (n_draws × n_strains) columns
        self._strain_names: list[str] = []
        self._trace_path: Optional[Path] = None

    # ---------------------------------------------------------------
    # Loading
    # ---------------------------------------------------------------

    def load(self, trace_path: str | Path, strain_names: Optional[list[str]] = None) -> None:
        """Load an ArviZ ``InferenceData`` trace from netCDF.

        Args:
            trace_path: path to gompertz_trace.nc
            strain_names: optional list of strain labels, ordered to match the
                posterior's strain dimension. If None, uses integer indices.
        """
        if az is None:
            raise ImportError(
                "arviz is required to load PyMC traces. `pip install arviz`"
            )
        path = Path(trace_path)
        if not path.exists():
            raise FileNotFoundError(f"Trace not found: {path}")

        idata = az.from_netcdf(str(path))
        self._trace_path = path

        # Extract (chain, draw, strain) arrays → flatten chain+draw → (n_draws, n_strains)
        def _flatten(var: str) -> np.ndarray:
            arr = idata.posterior[var].values  # (chain, draw, strain)
            n_chain, n_draw, n_strain = arr.shape
            return arr.reshape(n_chain * n_draw, n_strain)

        A = _flatten(self.A_VAR)
        mu = _flatten(self.MU_VAR)
        lam = _flatten(self.LAM_VAR)

        n_draws, n_strains = A.shape
        self._strain_names = (
            strain_names if strain_names is not None
            else [f"strain_{i}" for i in range(n_strains)]
        )
        # Store as long-format DataFrame for easy joint sampling
        self._samples = pd.DataFrame({
            "draw": np.repeat(np.arange(n_draws), n_strains),
            "strain_idx": np.tile(np.arange(n_strains), n_draws),
            "strain": [self._strain_names[i] for _ in range(n_draws) for i in range(n_strains)],
            "A": A.ravel(),
            "mu": mu.ravel(),
            "lam": lam.ravel(),
        })

    # ---------------------------------------------------------------
    # Sampling
    # ---------------------------------------------------------------

    def sample_parameters(
        self,
        n: int,
        strain: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> pd.DataFrame:
        """Return ``n`` (A, μ, λ) tuples sampled from the joint posterior.

        Each row corresponds to one posterior draw for one strain — so the
        marginal and joint distributions of (A, μ, λ) are preserved without
        assuming independence (the #1 augmentation pitfall).

        Args:
            n: number of parameter tuples to draw.
            strain: if given, restrict to that strain's posterior; otherwise
                pool across all strains (preserves between-strain variability).
            rng: numpy Generator for reproducibility.

        Returns:
            DataFrame with columns ``['strain', 'A', 'mu', 'lam', 'draw']``.
        """
        if self._samples is None:
            raise RuntimeError("Call load(trace_path) before sample_parameters().")

        rng = rng or np.random.default_rng()
        pool = (self._samples if strain is None
                else self._samples[self._samples["strain"] == strain])
        if len(pool) == 0:
            raise KeyError(f"strain {strain!r} not found in posterior")

        idx = rng.integers(0, len(pool), size=n)
        return pool.iloc[idx][["strain", "A", "mu", "lam", "draw"]].reset_index(drop=True)

    def sample_curves(
        self,
        n: int,
        grids: Iterable[tuple[np.ndarray, float]] | Iterable[np.ndarray],
        noise=None,
        rng: Optional[np.random.Generator] = None,
    ) -> pd.DataFrame:
        """Generate ``n`` synthetic curves by sampling params then evaluating Gompertz.

        Args:
            n: number of curves.
            grids: iterable of time grids (np.ndarray each) — sampler cycles
                through these to match the empirical length/duration
                distribution of the real dataset. A single grid may be passed
                as ``[grid]``.
            noise: optional noise model with ``.apply(signal, seed=None)``.
                Recommended: ResidualBootstrapNoise built from real fit
                residuals. If None, clean curves are returned.
            rng: Generator for reproducibility.

        Returns:
            long-format DataFrame with columns
            ``['curve_id', 'time', 'od_clean', 'od', 'A', 'mu', 'lam', 'strain']``.
        """
        rng = rng or np.random.default_rng()
        grids_list = [np.asarray(g if not isinstance(g, tuple) else g[0]) for g in grids]
        if not grids_list:
            raise ValueError("At least one time grid is required")

        params = self.sample_parameters(n, rng=rng)
        rows: list[dict] = []
        for i, row in params.iterrows():
            t = grids_list[i % len(grids_list)]
            clean = gompertz(t, row["A"], row["mu"], row["lam"])
            observed = (noise.apply(clean, seed=int(rng.integers(0, 2**31 - 1)))
                        if noise is not None else clean.copy())
            for j, (t_j, c_j, o_j) in enumerate(zip(t, clean, observed)):
                rows.append({
                    "curve_id": i,
                    "time": float(t_j),
                    "od_clean": float(c_j),
                    "od": float(o_j),
                    "A": float(row["A"]),
                    "mu": float(row["mu"]),
                    "lam": float(row["lam"]),
                    "strain": row["strain"],
                })
        return pd.DataFrame(rows)

    # ---------------------------------------------------------------
    # Residual pool loader (Phase Step 4)
    # ---------------------------------------------------------------

    @staticmethod
    def build_residual_pool_from_dir(residuals_dir: str | Path,
                                      od_bins=None) -> dict:
        """Load all per-strain residuals from step 01's `residuals/` output.

        Pairs each curve's `od_predicted` (clean Gompertz) with `od_observed`
        (noisy), then delegates to ``ResidualBootstrapNoise.build_pool_from_fits``
        for OD-binned bootstrapping.

        Args:
            residuals_dir: path to a `residuals/` directory containing
                ``{strain}.csv`` files with columns time, od_observed,
                od_predicted, residual (as produced by step 01 post-Phase-A.1).
            od_bins: optional bin edges; uses
                ``ResidualBootstrapNoise.DEFAULT_OD_BINS`` if None.

        Returns:
            dict[int -> np.ndarray] suitable for
            ``ResidualBootstrapNoise(residual_pool=...)``.
        """
        from pathlib import Path as _P
        import pandas as _pd
        # Local import to avoid circular if noise_models imports from here later
        from noise_models import ResidualBootstrapNoise

        residuals_dir = _P(residuals_dir)
        if not residuals_dir.exists():
            raise FileNotFoundError(f"Residuals directory not found: {residuals_dir}")

        curves = []
        for fpath in sorted(residuals_dir.glob("*.csv")):
            try:
                df = _pd.read_csv(fpath)
                if not {"od_observed", "od_predicted"}.issubset(df.columns):
                    continue
                curves.append((df["od_predicted"].values, df["od_observed"].values))
            except Exception:
                continue

        if not curves:
            raise ValueError(f"No usable residual files in {residuals_dir}")

        return ResidualBootstrapNoise.build_pool_from_fits(curves, od_bins=od_bins)

    # ---------------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------------

    @property
    def n_strains(self) -> int:
        return len(self._strain_names)

    @property
    def n_draws(self) -> int:
        if self._samples is None:
            return 0
        return int(self._samples["draw"].nunique())

    def __repr__(self) -> str:
        if self._samples is None:
            return "PosteriorPredictiveSampler(unloaded)"
        return (f"PosteriorPredictiveSampler("
                f"strains={self.n_strains}, draws={self.n_draws}, "
                f"trace={self._trace_path})")
