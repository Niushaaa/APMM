#!/usr/bin/env python3
"""
parlay_mm_sim.py
================
Simulate and compare three automated-market-maker (AMM) designs for parlay
trading over M = 3 binary base events, driven by an IDENTICAL stream of trades
per pass, and measure the market maker's (MM's) loss under each design.

Pure Python + numpy, single file, seeded RNG, no external services.

--------------------------------------------------------------------------------
LMSR PRIMITIVES (shared by all designs)
--------------------------------------------------------------------------------
Every market is an LMSR with liquidity parameter b (default b = 1) over a finite
set of outcomes. With share-count vector Q over those outcomes:

  * prices :  pi[x] = softmax(Q/b)[x]
              implemented with a NUMERICALLY STABLE softmax (subtract max(Q/b)
              before exponentiating).
  * cost   :  C(Q) = b * log( sum_x exp(Q[x]/b) )     (stable log-sum-exp)
  * cash   :  buying delta contracts of outcome x increases ONLY Q[x] by delta;
              the trader pays  dC = C(Q_after) - C(Q_before).
  * entropy:  H(pi) = - sum_x pi[x] * log(pi[x])   (nats)

--------------------------------------------------------------------------------
DESIGN A -- Hierarchical LMSR with SHARED parameters (the design under study)
--------------------------------------------------------------------------------
One LMSR market per non-empty leg-set S of {1,2,3} -> 7 markets:
  level 1: {1},{2},{3}            (2 outcomes each)
  level 2: {1,2},{1,3},{2,3}      (4 outcomes each)
  level 3: {1,2,3}                (8 outcomes)

Parameters: one residual scalar per (leg-set, outcome), all init 0:
  q[{i}][a]  (a in {0,1}),  r[{i,j}][ab],  r[{1,2,3}][abc]
Total 3*2 + 3*4 + 1*8 = 26 parameters.

Each market's share counts aggregate its OWN residual plus all consistent
sub-leg-set residuals (sum over every non-empty subset T of S, indexing each
residual by the bits of its own legs within the outcome x):

  Q[{i}][a]     = q[{i}][a]
  Q[{i,j}][ab]  = q[{i}][a] + q[{j}][b] + r[{i,j}][ab]
  Q[{1,2,3}][abc] = q[{1}][a]+q[{2}][b]+q[{3}][c]
                    + r[{1,2}][ab]+r[{1,3}][ac]+r[{2,3}][bc]
                    + r[{1,2,3}][abc]
  (general:  Q[S][x] = sum over nonempty T subset of S of residual[T][x|_T] )

A trade buying delta contracts of outcome x in market S increments ONLY that
residual (r[S][x] += delta, or q for singletons). This automatically propagates
into every super-market's Q because they share the residual, but the cash dC is
charged in market S only, computed from S's own Q vector before/after.

--------------------------------------------------------------------------------
DESIGN B -- Single MONOLITHIC LMSR over all 2^M = 8 atoms (the ideal floor)
--------------------------------------------------------------------------------
One LMSR over the 8 joint outcomes abc, share counts Q8[abc], all init 0. There is
exactly ONE cost function; the lower-leg-parlay markets are consistent marginal
INTERFACES into it (a trade on outcome x of leg-set S maps to the bundle of all
atoms consistent with x; its quoted price is the summed marginal
sum_{atoms a consistent with x} pi8[a]).

B is driven from the latent EFFICIENTLY: each timestep it matches ONLY the top/full
leg-set to pi*. Because the full leg-set's outcomes ARE the 8 atoms, that single
match pins the entire joint -- and hence every lower marginal -- in one trade, so
no redundant lower-level parlay matches are needed. The loss of a clean single LMSR
book equals b*(H_init - H_final), a function of the final joint only, so B realizes
the theoretical loss floor with minimal cash turnover (matching every interface
bottom-up instead would give the IDENTICAL loss but far more churn).

--------------------------------------------------------------------------------
DESIGN C -- INDEPENDENT per-level LMSRs (no parameter sharing)
--------------------------------------------------------------------------------
Same 7-market structure as Design A, but each market is a regular, standalone
LMSR with its OWN full parameter set: q[{i}][a], independent Q[{i,j}][ab] (4 free
params each), Q[{1,2,3}][abc] (8 free params), all init 0, with NO sharing -- a
trade on market S touches only market S's own params and cost function. The
markets are independent cost functions.

--------------------------------------------------------------------------------
TRADE STREAM  (identical across A, B, C within each pass / seed)
--------------------------------------------------------------------------------
Latent "true" joint price process -- MULTIVARIATE PROBIT / GAUSSIAN COPULA.
We maintain a single latent multivariate-Gaussian random walk in R^M:

      Y(t) = Y(t-1) + Sigma^{1/2} N(0, I_M)        (Y(0) = 0,  Y in R^M)

with innovation covariance Sigma (off-diagonals = innovation correlation). The
target joint over the 2^M atoms is the orthant distribution of a correlated normal
Z ~ N(Y(t), R) (R = copula correlation matrix, unit diagonal):

      pi*(b) = Pr[ 1(Z_i > 0) = b_i  for all i ],   Z ~ N(Y(t), R).

Every market S's target is the MARGINAL of this single pi* onto S's legs:

      tau[S][x] = sum_{atoms consistent with x} pi*(atom).

The off-diagonals of Sigma (innovation correlation) and R (copula correlation)
make the joint carry EVOLVING higher-order interaction: a random nudge to the walk
changes both marginals and correlations -- no independence assumption. Defaults:
innov_sigma = 0.5, innov_rho = 0.3, copula_rho = 0.4 (Sigma = innov_sigma^2 *
equicorr(innov_rho), R = equicorr(copula_rho)). The latent -> target-joint map is
the SWAPPABLE latent_to_target_joint (helper joint_over_atoms, which evaluates the
orthant probabilities via scipy's multivariate-normal CDF); the final realized
atom is a categorical draw from pi*.

At each timestep this defines a target distribution for EVERY market (the target
joint and its marginals onto each leg-set).

Trade execution -- BOTTOM-UP level sweep to the target joint (NO volume halving).
At each timestep, with target joint pi* and marginals tau_S, process the levels in
INCREASING order k = 1, 2, ..., M. Within level k, take that level's markets in a
uniform-random order (this order, and the latent path, are fixed across A/B/C),
and for each market S of size k execute BUYS-ONLY trades to move S's current quoted
distribution to its target marginal tau_S AS CLOSELY AS THE DESIGN ALLOWS, computed
against the design's LIVE books at that moment.

  Buys-only full-distribution match. To reach a target distribution tau from the
  current quoted prices m of a market, prices are invariant to a constant added to
  all share counts, so the minimal buys-only increment is, per outcome x,
        delta[x] = b*(log tau[x] - log m[x]) - min_y b*(log tau[y] - log m[y]),
  giving delta >= 0 (the argmin outcome gets 0) and post-trade quoted prices = tau
  exactly. All outcomes' deltas are applied SIMULTANEOUSLY (one cost difference).

  Design C: each market is an independent book; matching S to tau_S touches ONLY S.
    Lower levels were already matched; level-k trades inject the order-k increment.
  Design B: does NOT sweep -- it matches ONLY the top/full leg-set to pi* (one trade
    on the single book pins the whole joint and every marginal at once). The ideal
    floor: same represented distribution as a full bottom-up sweep would give, but
    with no redundant lower-level churn.
  Design A: matching S to tau_S sets S's SHARED residual, which propagates upward
    into all super-markets immediately. Higher-level markets are therefore matched
    starting from the prices LEFT BY lower-level propagation, adding only the new
    granularity (the order-k residual) needed to reach tau_S from there. This reuse
    of propagated low-order information is the design under study.

The shared objects across the designs are the target joint pi* and the latent path
(plus the within-level market order, used by the sweeping designs A and C). B is the
efficient single-book floor; A differs from it through upward propagation.

--------------------------------------------------------------------------------
MEASUREMENTS (per design, averaged over N independent seeds, default N = 200)
--------------------------------------------------------------------------------
1. Net MM loss = total settlement payout - total cash collected.
   A single final realized atom is sampled from the final latent joint (same atom
   for all three designs in a seed); the MM pays $1 per contract held on every
   market outcome consistent with the realized atom. Reported: mean +/- std across
   seeds, AND the EXPECTED loss (averaging settlement over the final latent joint
   analytically: expected payout = sum_S sum_x contracts[S][x] * tau_final[S][x]).

2. Summed entropy-drop loss = b * sum_markets [ H(pi_S^init) - H(pi_S^final) ] in
   nats. pi^init is uniform (all Q init 0) so H_init = log(#outcomes). For Design B
   it is the single book's b*[H(pi8^init) - H(pi8^final)]. Broken down by level for
   A and C.

3. Marginal inconsistency (A and C only; B is consistent by construction): for each
   lower market S (|S| < 3), the L1 distance between its own quoted distribution and
   the marginal of the triple market onto S's legs. Reported per-market and summed.

4. Total cash collected and total dC paid, per design. (By construction these are
   equal: each trader's cash IS the cost difference dC, so cash_collected == sum dC.)

Per-trade logs are written to CSV (by default only for the first `log_seeds` seeds
to keep the file manageable; configurable).

M = 3 is the default but leg-sets / markets / atoms are generated programmatically
so the construction generalizes to other M.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import multivariate_normal as _mvnorm   # orthant probabilities


# =============================================================================
# LMSR primitives
# =============================================================================
def stable_softmax(Q: np.ndarray, b: float) -> np.ndarray:
    z = Q / b
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def lmsr_cost(Q: np.ndarray, b: float) -> float:
    z = Q / b
    m = z.max()
    return float(b * (m + np.log(np.exp(z - m).sum())))


def entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-300, 1.0)
    return float(-np.sum(p * np.log(p)))


def stable_weights(Q: np.ndarray, b: float) -> np.ndarray:
    """Unnormalized, numerically-stable exp weights proportional to prices."""
    z = Q / b
    z = z - z.max()
    return np.exp(z)


# =============================================================================
# Combinatorial structure (generated programmatically for general M)
# =============================================================================
@dataclass
class Structure:
    M: int
    legsets: list                      # ordered: by size, then lexicographic
    full: tuple                        # the full leg-set (the "triple" for M=3)
    outcomes: dict                     # legset -> list of outcome tuples
    out_index: dict                    # legset -> {outcome tuple: idx}
    pos_in: dict                       # legset -> {leg: position within outcome}
    subsets: dict                      # legset S -> list of non-empty subsets T<=S
    proj_idx: dict                     # (S,T) -> np.array mapping S-outcome -> T-outcome idx
    atoms: list                        # outcomes of the full leg-set
    consistent: dict                   # (S, o_idx) -> np.array of atom indices consistent
    atom_signs: np.ndarray             # (2^M, M) array of +-1 = 2*bit-1 per atom


def build_structure(M: int = 3) -> Structure:
    legs = list(range(1, M + 1))
    legsets = []
    for k in range(1, M + 1):
        for combo in itertools.combinations(legs, k):
            legsets.append(combo)
    full = tuple(legs)

    outcomes, out_index, pos_in = {}, {}, {}
    for S in legsets:
        outs = list(itertools.product([0, 1], repeat=len(S)))
        outcomes[S] = outs
        out_index[S] = {o: i for i, o in enumerate(outs)}
        pos_in[S] = {leg: i for i, leg in enumerate(S)}

    # non-empty subsets of each leg-set, and projection-index arrays
    subsets, proj_idx = {}, {}
    for S in legsets:
        subs = []
        for k in range(1, len(S) + 1):
            subs.extend(itertools.combinations(S, k))
        subsets[S] = subs
        for T in subs:
            posS = pos_in[S]
            arr = np.array(
                [out_index[T][tuple(o[posS[leg]] for leg in T)] for o in outcomes[S]],
                dtype=np.intp,
            )
            proj_idx[(S, T)] = arr

    atoms = outcomes[full]
    consistent = {}
    for S in legsets:
        for oi, o in enumerate(outcomes[S]):
            idxs = [
                ai
                for ai, atom in enumerate(atoms)
                if tuple(atom[leg - 1] for leg in S) == o
            ]
            consistent[(S, oi)] = np.array(idxs, dtype=np.intp)

    # sign vector per atom: s_i = +1 if bit i is 1 else -1 (orthant the atom picks)
    atom_signs = np.array([[2 * bit - 1 for bit in atom] for atom in atoms],
                          dtype=float)

    return Structure(
        M=M, legsets=legsets, full=full, outcomes=outcomes, out_index=out_index,
        pos_in=pos_in, subsets=subsets, proj_idx=proj_idx, atoms=atoms,
        consistent=consistent, atom_signs=atom_signs,
    )


# =============================================================================
# Designs
# =============================================================================
class DesignBase:
    """Common interface. Each design exposes:
       interface_w(S) -> unnormalized price weights over S's outcomes
       buy(S, idx, delta) -> cash paid (dC); mutates params + contracts ledger
    """

    name = "base"

    def __init__(self, struct: Structure, b: float):
        self.s = struct
        self.b = b
        self.contracts = {S: np.zeros(len(struct.outcomes[S])) for S in struct.legsets}

    def interface_w(self, S):
        raise NotImplementedError

    def prices(self, S):
        w = self.interface_w(S)
        return w / w.sum()

    def match(self, S, delta):
        """Apply a full delta-vector to market S simultaneously; return cash dC."""
        raise NotImplementedError


class DesignA(DesignBase):
    """Hierarchical LMSR with shared residual parameters."""

    name = "A"

    def __init__(self, struct, b):
        super().__init__(struct, b)
        self.residual = {S: np.zeros(len(struct.outcomes[S])) for S in struct.legsets}

    def Q(self, S):
        s = self.s
        q = np.zeros(len(s.outcomes[S]))
        for T in s.subsets[S]:
            q = q + self.residual[T][s.proj_idx[(S, T)]]
        return q

    def interface_w(self, S):
        return stable_weights(self.Q(S), self.b)

    def match(self, S, delta):
        Qb = self.Q(S)
        cb = lmsr_cost(Qb, self.b)
        self.residual[S] += delta          # propagates upward into super-markets
        Qa = Qb + delta                    # S's own Q rises by exactly delta
        ca = lmsr_cost(Qa, self.b)
        self.contracts[S] += delta
        return ca - cb


class DesignB(DesignBase):
    """Single monolithic 8-atom LMSR; parlay markets are marginal interfaces."""

    name = "B"

    def __init__(self, struct, b):
        super().__init__(struct, b)
        self.Q8 = np.zeros(len(struct.atoms))

    def interface_w(self, S):
        e = stable_weights(self.Q8, self.b)
        outs = self.s.outcomes[S]
        w = np.empty(len(outs))
        for oi in range(len(outs)):
            w[oi] = e[self.s.consistent[(S, oi)]].sum()
        return w

    def match(self, S, delta):
        cb = lmsr_cost(self.Q8, self.b)
        for oi, d in enumerate(delta):     # each outcome's bundle of consistent atoms
            if d != 0.0:
                self.Q8[self.s.consistent[(S, oi)]] += d
        ca = lmsr_cost(self.Q8, self.b)
        self.contracts[S] += delta
        return ca - cb

    def prices8(self):
        return stable_softmax(self.Q8, self.b)


class DesignC(DesignBase):
    """Independent per-market standalone LMSRs (no sharing)."""

    name = "C"

    def __init__(self, struct, b):
        super().__init__(struct, b)
        self.Q = {S: np.zeros(len(struct.outcomes[S])) for S in struct.legsets}

    def interface_w(self, S):
        return stable_weights(self.Q[S], self.b)

    def match(self, S, delta):
        Qb = self.Q[S]
        cb = lmsr_cost(Qb, self.b)
        self.Q[S] = Qb + delta             # touches only this independent book
        ca = lmsr_cost(self.Q[S], self.b)
        self.contracts[S] += delta
        return ca - cb


# =============================================================================
# Latent -> target-joint map (SWAPPABLE):  MULTIVARIATE PROBIT / GAUSSIAN COPULA
# =============================================================================
def equicorr(M: int, rho: float) -> np.ndarray:
    """MxM equicorrelation matrix (1 on diagonal, rho off-diagonal)."""
    C = np.full((M, M), float(rho))
    np.fill_diagonal(C, 1.0)
    return C


_FROZEN_ORTHANT = {}        # cache: (R.tobytes(), M) -> list of frozen N(0, S R S)
# scipy's MVN cdf uses RANDOMIZED quasi-Monte Carlo; a FIXED seed per call makes
# the orthant probabilities deterministic (so runs are reproducible and every
# design in a seed sees the identical target stream).
_ORTHANT_SEED = 0


def _frozen_orthant(struct: Structure, R: np.ndarray):
    """Frozen N(0, S R S) per atom (depends only on R), cached. Then
    Pr[orthant b] = Pr[N(-s.Y, SRS) <= 0] = Pr[N(0, SRS) <= s.Y] = frozen.cdf(s.Y).
    """
    key = (R.tobytes(), struct.M)
    fr = _FROZEN_ORTHANT.get(key)
    if fr is None:
        fr = [_mvnorm(mean=np.zeros(struct.M), cov=R * np.outer(s, s))
              for s in struct.atom_signs]
        _FROZEN_ORTHANT[key] = fr
    return fr


def joint_over_atoms(Y: np.ndarray, struct: Structure, R: np.ndarray) -> np.ndarray:
    """Target joint over the 2^M atoms = ORTHANT PROBABILITIES of Z ~ N(Y, R)
    (a multivariate-probit / Gaussian-copula model):

        pi*(b) = Pr[ 1(Z_i > 0) = b_i  for all i ],   Z ~ N(Y, R).

    For atom b with signs s_i = 2 b_i - 1 the orthant probability equals
    frozen.cdf(s.Y) with frozen ~ N(0, S R S), S = diag(s). The last orthant is
    inferred from the partition-of-unity (orthants tile R^M); all are renormalized
    for numerical safety.
    """
    fr = _frozen_orthant(struct, R)
    s_all = struct.atom_signs
    n = len(fr)
    pi = np.empty(n)
    for i in range(n - 1):
        pi[i] = fr[i].cdf(s_all[i] * Y, rng=_ORTHANT_SEED)
    pi[n - 1] = max(0.0, 1.0 - pi[:n - 1].sum())
    pi = np.clip(pi, 1e-15, None)
    return pi / pi.sum()


def latent_to_target_joint(Y: np.ndarray, struct: Structure,
                           R: np.ndarray) -> dict:
    """SWAPPABLE latent -> target map. Each market S's target is the MARGINAL of
    the single Gaussian-copula joint pi*(Y, R) onto S's legs, so targets are
    mutually consistent and carry evolving order-1, -2 and -3 correlation.
    """
    pi = joint_over_atoms(Y, struct, R)
    targets = {}
    for S in struct.legsets:
        outs = struct.outcomes[S]
        targets[S] = np.array(
            [pi[struct.consistent[(S, oi)]].sum() for oi in range(len(outs))]
        )
    return targets


# =============================================================================
# Trade execution
# =============================================================================
TOL = 1e-9
T_CLIP = 1e-6


def match_to_target(design: DesignBase, S, tau, b):
    """Buys-only full-distribution match of market S to target marginal tau,
    against the design's LIVE books. Returns (cash dC, delta_vec, pre_L1)."""
    w = design.interface_w(S)
    m = w / w.sum()                                   # current quoted distribution
    pre_l1 = float(np.abs(m - tau).sum())
    log_ratio = b * (np.log(np.clip(tau, T_CLIP, 1.0))
                     - np.log(np.clip(m, 1e-300, 1.0)))
    delta = log_ratio - log_ratio.min()               # shift so min delta = 0
    delta = np.maximum(delta, 0.0)                     # numerical guard (buys only)
    cash = design.match(S, delta)
    return cash, delta, pre_l1


# =============================================================================
# Stream generation (one shared stream + latent per seed)
# =============================================================================
def build_seed_schedule(struct, rng, T, L, R):
    """Return (steps, targets_final, pi_final, realized_atom).

    steps: list (one per timestep) of (targets, order) where `targets` maps each
    leg-set S to its target marginal tau_S, and `order` is the bottom-up processing
    list [(level k, market S), ...] with a uniform-random market order within each
    level. The same `steps` (and the same latent path) drive all three designs.

    Latent: a single multivariate-Gaussian random walk Y(t) = Y(t-1) + L N(0,I)
    with L = Sigma^{1/2} (innovation covariance Sigma); the target joint is the
    Gaussian-copula (multivariate-probit) orthant distribution of Z ~ N(Y, R).
    """
    M = struct.M
    by_level = {k: [S for S in struct.legsets if len(S) == k] for k in range(1, M + 1)}

    Y = np.zeros(M)                                     # latent multivariate state
    steps = []
    targets = None
    for _t in range(1, T + 1):
        Y = Y + L @ rng.standard_normal(M)              # correlated Gaussian step
        targets = latent_to_target_joint(Y, struct, R)
        order = []
        for k in range(1, M + 1):                       # bottom-up
            markets = list(by_level[k])
            rng.shuffle(markets)                         # uniform order within level
            order.extend((k, S) for S in markets)
        steps.append((targets, order))

    # final realized atom: categorical draw from the final copula joint pi*
    pi_final = joint_over_atoms(Y, struct, R)
    realized = struct.atoms[int(rng.choice(len(struct.atoms), p=pi_final))]
    return steps, targets, pi_final, realized


# =============================================================================
# Metrics
# =============================================================================
def win_index(struct, S, realized):
    proj = tuple(realized[leg - 1] for leg in S)
    return struct.out_index[S][proj]


def settlement_payout(struct, contracts, realized):
    total = 0.0
    for S in struct.legsets:
        total += contracts[S][win_index(struct, S, realized)]
    return total


def expected_payout(struct, contracts, targets_final):
    total = 0.0
    for S in struct.legsets:
        total += float(np.dot(contracts[S], targets_final[S]))
    return total


def entropy_drop_by_level(design: DesignBase, struct: Structure):
    """b * (H_init - H_final) summed, and per level. (Used for A and C.)"""
    b = design.b
    per_level = {}
    total = 0.0
    for S in struct.legsets:
        n = len(struct.outcomes[S])
        h_init = np.log(n)
        h_final = entropy(design.prices(S))
        val = b * (h_init - h_final)
        per_level[len(S)] = per_level.get(len(S), 0.0) + val
        total += val
    return total, per_level


def marginal_inconsistency(design: DesignBase, struct: Structure):
    """L1 between each lower market's own dist and the triple-marginal onto it."""
    full = struct.full
    tri = design.prices(full)
    per_market = {}
    total = 0.0
    for S in struct.legsets:
        if S == full:
            continue
        own = design.prices(S)
        marg = np.array(
            [tri[struct.consistent[(S, oi)]].sum() for oi in range(len(own))]
        )
        l1 = float(np.abs(own - marg).sum())
        per_market[S] = l1
        total += l1
    return total, per_market


# =============================================================================
# Simulation driver
# =============================================================================
@dataclass
class Config:
    M: int = 3
    b: float = 1.0
    # latent multivariate-probit / Gaussian-copula parameters:
    innov_sigma: float = 0.5      # per-step innovation std of the Y random walk
    innov_rho: float = 0.3        # innovation correlation (off-diag of Sigma's corr)
    copula_rho: float = 0.4       # copula correlation (off-diag of R)
    T: int = 40
    N: int = 200
    base_seed: int = 12345
    log_seeds: int = 1
    csv_path: str = "parlay_trades.csv"


@dataclass
class DesignAccumulator:
    name: str
    net_loss: list = field(default_factory=list)
    exp_loss: list = field(default_factory=list)
    cash: list = field(default_factory=list)
    entropy_total: list = field(default_factory=list)
    entropy_level: dict = field(default_factory=dict)       # level -> list
    incons_total: list = field(default_factory=list)
    incons_market: dict = field(default_factory=dict)       # S -> list


# Design variants: (name, class, execution mode).
#   "sweep" -> match every market bottom-up (the parlay-level order flow).
#   "top"   -> match ONLY the top/full leg-set; since the joint's own outcomes ARE
#              the atoms, this one trade pins the whole joint and hence every
#              marginal -- the efficient single-book floor with no redundant
#              lower-level churn.
# Design B IS the single monolithic book, traded only at the top ("top" mode):
# the theoretical loss floor.
DESIGN_SPECS = (
    ("A", DesignA, "sweep"),
    ("B", DesignB, "top"),
    ("C", DesignC, "sweep"),
)
SINGLE_BOOK = {"B"}      # one cost function -> consistent by construction


def latent_matrices(cfg: Config):
    """Return (L, R): L = Sigma^{1/2} (Cholesky of the innovation covariance) and
    R the copula correlation. Sigma = innov_sigma^2 * equicorr(innov_rho)."""
    Sigma = (cfg.innov_sigma ** 2) * equicorr(cfg.M, cfg.innov_rho)
    L = np.linalg.cholesky(Sigma)
    R = equicorr(cfg.M, cfg.copula_rho)
    return L, R


def run(cfg: Config):
    struct = build_structure(cfg.M)
    accs = {nm: DesignAccumulator(nm) for nm, _, _ in DESIGN_SPECS}
    L, R = latent_matrices(cfg)

    csv_file = open(cfg.csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(
        ["seed", "timestep", "level", "design", "market", "pre_L1",
         "total_delta", "cash"]
    )

    for seed in range(cfg.N):
        rng = np.random.default_rng(cfg.base_seed + seed)
        steps, targets_final, _pi, realized = build_seed_schedule(
            struct, rng, cfg.T, L, R
        )

        for nm, klass, mode in DESIGN_SPECS:
            design = klass(struct, cfg.b)
            total_cash = 0.0
            log_this = seed < cfg.log_seeds
            for ti, (targets, order) in enumerate(steps, start=1):
                # efficient B trades only the top leg-set; others sweep all markets
                seq = order if mode == "sweep" else [(cfg.M, struct.full)]
                for (k, S) in seq:          # bottom-up; fixed order across designs
                    cash, delta, pre_l1 = match_to_target(
                        design, S, targets[S], cfg.b
                    )
                    total_cash += cash
                    if log_this:
                        writer.writerow(
                            [seed, ti, k, nm, "".join(map(str, S)),
                             f"{pre_l1:.6f}", f"{delta.sum():.6f}", f"{cash:.6f}"]
                        )

            payout = settlement_payout(struct, design.contracts, realized)
            exp_pay = expected_payout(struct, design.contracts, targets_final)

            acc = accs[nm]
            acc.net_loss.append(payout - total_cash)
            acc.exp_loss.append(exp_pay - total_cash)
            acc.cash.append(total_cash)

            if nm in SINGLE_BOOK:
                # single book: entropy drop of the one 8-atom joint
                h_init = np.log(len(struct.atoms))
                drop = cfg.b * (h_init - entropy(design.prices8()))
                acc.entropy_total.append(drop)
                acc.entropy_level.setdefault(3, []).append(drop)
                acc.incons_total.append(0.0)   # consistent by construction
            else:
                etot, eperlvl = entropy_drop_by_level(design, struct)
                acc.entropy_total.append(etot)
                for lvl, v in eperlvl.items():
                    acc.entropy_level.setdefault(lvl, []).append(v)
                itot, iper = marginal_inconsistency(design, struct)
                acc.incons_total.append(itot)
                for S, v in iper.items():
                    acc.incons_market.setdefault(S, []).append(v)

    csv_file.close()
    return struct, accs


# =============================================================================
# Reporting
# =============================================================================
def ms(arr):
    a = np.asarray(arr, dtype=float)
    return a.mean(), a.std()


def print_report(cfg: Config, struct: Structure, accs):
    print("=" * 78)
    print("PARLAY MARKET-MAKER SIMULATION  --  M = %d binary base events" % cfg.M)
    print("=" * 78)
    print(
        f"Config: b={cfg.b}  T={cfg.T}  N={cfg.N} seeds  base_seed={cfg.base_seed}\n"
        f"        innov_sigma={cfg.innov_sigma}  innov_rho={cfg.innov_rho}  "
        f"copula_rho={cfg.copula_rho}"
    )
    print("Latent: multivariate Gaussian random walk Y -> Gaussian-copula joint")
    print("        pi*(b)=orthant prob of N(Y,R); targets = CONSISTENT marginals,")
    print("        evolving order-1/2/3 correlation (no independence assumption).")
    print(f"Per-trade logs (first {cfg.log_seeds} seed[s]) -> {cfg.csv_path}")
    print()

    order = ("A", "B", "C")
    label = {"A": "A", "B": "B", "C": "C"}

    # ---- main comparison table ----
    print("MAIN COMPARISON   (B = single monolithic book, traded only at the top)")
    print("-" * 78)
    header = f"{'Design':<8}{'Net loss (mean+/-std)':<26}{'Entropy loss':<16}" \
             f"{'Margin. incons.':<16}{'Cash':<12}"
    print(header)
    print("-" * 78)
    for nm in order:
        acc = accs[nm]
        nl_m, nl_s = ms(acc.net_loss)
        ent_m, _ = ms(acc.entropy_total)
        inc_m, _ = ms(acc.incons_total)
        cash_m, _ = ms(acc.cash)
        incons_str = "n/a (exact)" if nm in SINGLE_BOOK else f"{inc_m:.4f}"
        print(
            f"{label[nm]:<8}{nl_m:>9.4f} +/- {nl_s:<11.4f}{ent_m:>13.4f}   "
            f"{incons_str:<16}{cash_m:>10.4f}"
        )
    print("-" * 78)
    print()

    # ---- expected loss ----
    print("EXPECTED MM LOSS  (settlement averaged over final latent joint)")
    print("-" * 78)
    for nm in order:
        em, es = ms(accs[nm].exp_loss)
        print(f"  Design {label[nm]:<6}:  {em:>9.4f} +/- {es:.4f}")
    print()

    # ---- cash / dC note ----
    print("CASH COLLECTED  ==  TOTAL dC PAID  (equal by construction)")
    print("-" * 78)
    for nm in order:
        cm, cs = ms(accs[nm].cash)
        print(f"  Design {label[nm]:<6}:  cash = sum(dC) = {cm:>9.4f} +/- {cs:.4f}")
    print()

    # ---- entropy by level ----
    print("ENTROPY-DROP LOSS BY LEVEL  ( b*[H_init - H_final], nats )")
    print("-" * 78)
    print(f"{'Design':<8}{'Level1':<12}{'Level2':<12}{'Level3':<12}{'Total':<12}")
    for nm in order:
        acc = accs[nm]
        cells = []
        for lvl in (1, 2, 3):
            if lvl in acc.entropy_level:
                cells.append(f"{np.mean(acc.entropy_level[lvl]):.4f}")
            else:
                cells.append("-")
        tot = np.mean(acc.entropy_total)
        note = "  (single 8-atom book)" if nm in SINGLE_BOOK else ""
        print(f"{label[nm]:<8}{cells[0]:<12}{cells[1]:<12}{cells[2]:<12}"
              f"{tot:<12.4f}{note}")
    print()

    # ---- marginal inconsistency per market ----
    print("MARGINAL INCONSISTENCY PER LOWER MARKET  (L1 vs triple-marginal)")
    print("-" * 78)
    lowers = [S for S in struct.legsets if S != struct.full]
    head = f"{'Design':<8}" + "".join(
        f"{'{'+','.join(map(str,S))+'}':<12}" for S in lowers
    ) + f"{'Sum':<10}"
    print(head)
    for nm in ("A", "C"):
        acc = accs[nm]
        row = f"{nm:<8}"
        for S in lowers:
            row += f"{np.mean(acc.incons_market[S]):<12.4f}"
        row += f"{np.mean(acc.incons_total):<10.4f}"
        print(row)
    print(f"{'B':<8}" + "single book: consistent by construction (all L1 = 0).")
    print("=" * 78)


# =============================================================================
# Main
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Parlay AMM design comparison (A/B/C).")
    p.add_argument("--b", type=float, default=1.0, help="LMSR liquidity parameter")
    p.add_argument("--innov-sigma", type=float, default=0.5,
                   help="per-step innovation std of the latent Y random walk")
    p.add_argument("--innov-rho", type=float, default=0.3,
                   help="innovation correlation (off-diagonal of Sigma's corr)")
    p.add_argument("--copula-rho", type=float, default=0.4,
                   help="Gaussian-copula correlation (off-diagonal of R)")
    p.add_argument("--T", type=int, default=40, help="number of timesteps")
    p.add_argument("--N", type=int, default=200, help="number of seeds")
    p.add_argument("--base-seed", type=int, default=12345)
    p.add_argument("--log-seeds", type=int, default=1,
                   help="how many seeds to write per-trade logs for")
    p.add_argument("--csv", type=str, default="parlay_trades.csv")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        b=args.b, innov_sigma=args.innov_sigma, innov_rho=args.innov_rho,
        copula_rho=args.copula_rho, T=args.T, N=args.N,
        base_seed=args.base_seed, log_seeds=args.log_seeds, csv_path=args.csv,
    )
    struct, accs = run(cfg)
    print_report(cfg, struct, accs)


if __name__ == "__main__":
    main()
