# Experiment 1 — Loss scaling under synthetic low-order informed flow

**Goal.** Using synthetic *informed* order flow that satisfies the low-order
assumption, test whether the operator's realized net loss (equivalently, the
informed trader's profit — see §5) scales as the theory predicts: **APMM** stays
pinned near/below the **base** floor while the independent baseline **ind** grows
steeply as more orders enter.

Implemented in `simulation/experiment1_loworder.py` — self-contained apart from the
shared mechanics it imports from `parlay_mm_sim.py` (`DesignA`, `DesignC`,
`match_to_target`, `stable_softmax`, `entropy_drop_by_level`, `T_CLIP`,
`Structure`). First of three scripts; the flow generator and executor carry over to
Exp 2 (order/sparsity/magnitude stress) and Exp 3 (Kalshi flow).

---

## 1. Model roster (3 measured curves)

| Name | Class | What it is | Books | Role |
|------|-------|-----------|-------|------|
| **APMM** | `DesignA` | hierarchical shared-residual LMSR (eq. Q-agg) | all traded leg-sets | design under study |
| **ind**  | `DesignC` | independent per-market LMSRs | all traded leg-sets | SOTA baseline |
| **base** | `DesignC`, singletons only | M independent binary LMSRs, parlays unavailable | M singletons | lower reference / floor |

**Loss metric:** total net loss `= settlement_payout − cash` over all books,
Monte-Carlo expected (§5). B (monolithic) is dropped. No exponential (`c·2^M` / `g^M`)
fits are reported — just the raw curves and H1/H3/H4 verdicts.

---

## 2. Canonical parameterization — Walsh, anchored at all-ones

Sign `s_i(ω)=2ω_i−1` (ω=1 → +1; all-ones is the reference corner, from
`atom_signs`). Walsh characters `χ_S(ω)=∏_{i∈S} s_i(ω)`. Natural parameters are
scalars `θ_S`, one per leg-set (complete basis): `Q(ω)=Σ_S θ_S χ_S(ω)`,
`P=softmax(Q/b)`. These `θ_S` are APMM's `residual[S]` in the Walsh basis. A
k-order belief change is exact: only write `θ_S` for `|S|≤k` ⇒ `θ_S≡0` for `|S|>k`.

---

## 3. Flow generation — `build_low_order_flow` (one trade per step)

Settled config (defaults for Exp 1; Exp 2 sweeps these):

| Symbol | Config | Value | Meaning |
|--------|--------|-------|---------|
| `M`    | `--M`  | 2…20 | base events |
| `k`    | `--k_auto` | `⌈√M⌉` per M | order cap (grows with M) |
| support | (fixed) | **flat: 4M** | see step 1 |
| `ρ`    | `--rho`   | `1.0` | per-level increment decay |
| `ρ_0`  | `--rho_0` | `5.0` | increment scale ⇒ flat bound 5 |
| `T`    | `--steps_per_M` | `10` ⇒ `T=10M` | steps = trades (1 trade/step) |
| `b`    | `--b`     | `10.0` | LMSR liquidity |
| settles | `--n_settle` | `2000` | MC base-market draws (§5) |
| seeds | `--seeds` | `40` | for tight CIs |

Run: `python3 experiment1_loworder.py --k_auto --steps_per_M 10 --rho 1 --rho_0 5 --seeds 40 --M 2 … 20`

**Step 1 — support `D` (flat, `|D|=O(M)`), `pick_support`.** All M singletons, plus
`4M` leg-sets drawn **uniformly at random from every level 2…k pooled together**
(seeded, so `D` differs per seed). `θ_S≠0` only for `S∈D`, `|S|≤k`. (M=8,k=3 → |D|=40;
M=20,k=5 → |D|=100.) Concentrating a fixed `O(M)` support is what produces the clean
scaling — spreading over more markets under-trades them and blurs it.

**Step 2 — one θ-change → one trade per step.** `Q=0` at open (uniform P̂). For each
of `T` steps: pick **one** leg-set `S` uniformly from `D`, draw
`Δθ_S ~ U[−ρ^{|S|}ρ_0, +ρ^{|S|}ρ_0]` (with `ρ=1` this is a flat bound `±5`), update
`Q += Δθ_S·χ_S`. **Only Δθ is bounded — θ itself is an unbounded walk.** This is the
key knob: few re-quotes (small `T`) keep APMM's buys-only leak negligible (§8).

**Step 3 — the trade.** After the θ-update, `P=softmax(Q/b)`; compute the top
marginal on `S` once (bincount over the 2^M atoms), derive its sub-marginals via
`proj_idx`, and emit `(S, {S'⊆S : τ_{S'}})`. `T = n_steps` trades total.

**Step 4 — settlement law.** Store `base_p1` = final base marginals `P(ω_i=1)` (§5).

Flow artifact `{trades, base_p1, D, k, ρ, ρ_0, …}` is cached to
`flow_cache/…pkl` and shared across the 3 models. **Clear the cache when a config
that changes the support (M, k, ρ_0, …) changes** — the run script `rm -rf`s the
output dir at start.

**Sparse structure (`build_sparse_structure`, enables large M).** Builds only
leg-sets of size `≤k` (no full leg-set; ind naturally clipped at k) with the full
2^M atom space. Marginals use `bincount` (no `consistent` dict → no `O(M^k·2^M)`
memory); sub-marginals reuse the one top-S marginal. Makes M up to 20 feasible
(~13s/seed at M=20,k=5). The `incons` metric is dropped (needs `consistent`).

---

## 4. Trade execution — confined to S's sub-lattice (`sweep_books`)

Each trade re-quotes only `subsets(S)` bottom-up to the current marginals,
buys-only (`match_to_target`):
- **APMM:** matching `Sp` writes `residual[Sp]` to the eq. route value
  `b·log τ_{Sp} − Σ_{T⊊Sp} r^{(T)}` (up to the loss-neutral buys-only gauge);
  lower orders propagate up for free. Guarded by `--check_route`.
- **ind:** same sub-trades on independent books (each re-learns).
- **base:** singleton sub-trades only.

---

## 5. Settlement & loss — `settle_flow` (Monte-Carlo, independent base draws)

`n_settle` instances, each draws every base market independently
`ω_i ~ Bernoulli(base_p1_i)`; that one base realization settles **every** book —
parlays via projecting the base outcome onto their legs (base resolutions alone
settle parlays). Per model `net = payout − cash`, averaged over draws (same draws
for all models; only nonzero-contract books iterated). Net loss can be **negative**
(APMM profits on the correlation premium: cash collected on correlated parlays,
settlement independent-base). Per-seed = MC mean; aggregate over seeds = mean ± 95% CI.

**Trader ↔ operator (zero-sum):** the informed trader pays the premiums (`cash`) and
receives the payout, so **trader net profit = payout − cash = operator net loss**.
Thus the reported net-loss curves are exactly the informed trader's profit against
each design. (`trader_profit_hist.py`-style tooling histograms this; at M=8 the
trader nets ~+ind's loss against ind but ~0/slightly-negative against APMM.)

---

## 6. Outputs

`results_exp1_loworder/`: `flow_cache/`, `per_seed.csv`, `summary_aggregate.csv`,
`results.txt` (summary table + H1/H3/H4 verdicts), `plot_net_loss_lowM.png`
(APMM/ind/base ± CI, zero line since net loss can be negative). Live per-M progress
via `[running M=… k=… ]` prints.

**Success criteria:** H1 APMM near floor, gap `ind−APMM` widens with M; H3
`KL(APMM) ≲ KL(ind)`; H4 `base ≤ APMM ≤ ind` (may fail *favorably* — APMM below base).

---

## 7. Key finding — the leak scales with re-quote frequency

APMM's buys-only "leak" (shared low-order residuals, re-quoted to a drifting target,
forcing corrective buys on sibling books) accumulates with the **number of
re-quotes**, not with book inconsistency. Confirmed: many trades ⇒ APMM leaks and
loses to ind; an active-set *consistent* full-menu sweep did **not** fix it; **few
trades (one per step, `T=10M`) fixes it** — APMM's sharing advantage then dominates.
This is why `T` (via `steps_per_M`) is the lever, not the executor.

---

## 8. Validated result (flat, `k=⌈√M⌉`, 40 seeds, M=2…20)

- **APMM stays flat and drifts negative** (≈3 → −36): prices the low-order structure
  via sharing, so the operator ~breaks even / profits and the informed trader gets no
  edge.
- **ind explodes** ≈5 → ~1200 (≈250×), with sharp jumps exactly where `k` steps up
  (M=10→k=4 jumps to ~142; M=17→k=5 jumps to ~880) — each new order forces ind to
  independently price a fresh stratum of parlay books.
- **base** grows slowly (~linear, 3.5 → 14). Gap `ind−APMM` widens monotonically to
  ~1360 (H1=YES). H3 holds; H4 fails favorably.
- 40 seeds give clean curves. Trader-profit view: informed traders extract ~ind's
  loss against ind but ~lose against APMM.

`k=⌈√M⌉` (vs the earlier `⌈log M⌉`) makes k climb with M so higher orders enter and
ind's growth accelerates. `ρ=1` (flat `b_theta=5`) gives a stronger ind/APMM
separation than a decaying `ρ^k`, because higher-order θ then move as much as base.

---

## 9. Reuse & guards

**Imported unchanged from `parlay_mm_sim.py`:** `DesignA`, `DesignC`,
`match_to_target`, `stable_softmax`, `entropy_drop_by_level`, `T_CLIP`, `Structure`.
Everything else (support, flow, sparse structure, executor, settlement, `agg`, `kl`,
reporting) lives in `experiment1_loworder.py`. **Guard:** `--check_route` asserts eq.
route after each APMM sub-trade. **Determinism:** one `default_rng(base_seed+seed)`
for the flow, a separate stream for settlement; flows cached; runs reproducible.
