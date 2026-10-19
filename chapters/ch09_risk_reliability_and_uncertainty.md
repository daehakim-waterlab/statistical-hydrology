---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# 9. Risk, reliability and uncertainty in design

```{admonition} Core ideas
:class: tip
1. The **risk** that a design event is exceeded at least once during a structure's life is much larger than its annual probability. Design return periods should be chosen from the acceptable risk over the design life, or from the balance of construction cost and **expected damage**.
2. The performance of a water system over time is summarized by its **reliability** (how often it succeeds), **resilience** (how quickly it recovers) and **vulnerability** (how bad failures are).
3. Uncertainty comes from natural variability, data, parameters and model choice. It is **propagated** to design quantities by first-order (FOSM) approximations or by **Monte Carlo simulation**.
4. The probability of failure of a structure can be expressed through a **reliability index**, the safety margin divided by its standard deviation.
5. When the climate or the catchment changes, exceedance probabilities change over time, and return period and risk must be redefined for **nonstationary** conditions.
```

```{code-cell} python
:tags: [remove-cell]
# Plot style shared by all figures in this chapter.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from myst_nb import glue
from IPython.display import Markdown
%config InlineBackend.figure_format = "retina"

c1, c2, c3 = "#2a78d6", "#eb6834", "#1baf7a"   # categorical colors
ink, ink2, bg = "#0b0b0b", "#52514e", "#fcfcfb" # text, secondary text, background
plt.rcParams.update({
    "figure.facecolor": bg, "axes.facecolor": bg, "savefig.facecolor": bg,
    "text.color": ink, "axes.labelcolor": ink2, "axes.edgecolor": ink2,
    "xtick.color": ink2, "ytick.color": ink2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e3df", "axes.axisbelow": True,
    "font.size": 11, "axes.titlesize": 11, "axes.titlelocation": "left",
    "legend.frameon": False, "figure.constrained_layout.use": True,
})
```

## 9.1 Risk over a design life

### Hydrological risk

A structure designed for the $T$-year event, with annual exceedance probability $p = 1/T$, faces that probability every year of its design life of $n$ years. Under independence, the number of exceedances is binomial (Chapter 5), and the **hydrological risk**, the probability of at least one exceedance, is

\begin{equation*}
R = 1 - \left(1 - \frac{1}{T}\right)^n .
\tag{9.1}
\end{equation*}

Solving for $T$ gives the return period needed to keep the risk at $R$:

\begin{equation*}
T = \frac{1}{1 - (1 - R)^{1/n}} .
\tag{9.2}
\end{equation*}

For $T = n$, the risk is about 63% for any long design life, since $(1 - 1/n)^n \to e^{-1}$. Keeping a 50-year structure's risk at 10% requires a 475-year design event (Figure 9.1). The choice of $T$ is therefore a choice of acceptable risk, which depends on the consequences of failure: a farm culvert may accept a high risk, a dam spillway must not.

```{code-cell} python
:tags: [hide-input]
T = np.logspace(0.5, 4, 300)
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for n, col in zip([10, 25, 50, 100], [c3, c1, c2, ink]):
    ax.semilogx(T, 1 - (1 - 1 / T) ** n, color=col, lw=2.2, label=f"n = {n} years")
ax.axhline(0.1, color=ink2, lw=1, ls=":"); ax.text(4, 0.12, "R = 0.1", color=ink2, fontsize=9.5)
ax.set_xlabel("Design return period T (years)"); ax.set_ylabel("Risk over design life"); ax.set_ylim(0, 1)
ax.legend(loc="upper right", fontsize=9.5)
plt.show()
```

*Figure 9.1. Risk of at least one exceedance of the $T$-year event during a design life of $n$ years, Equation (9.1).*

### Expected damage and economic design

Risk ignores the size of the consequences. If $D(x)$ is the damage caused by an event of magnitude $x$, the **expected annual damage** is

\begin{equation*}
\text{EAD} = \int D(x)\,f(x)\,dx = \int_0^1 D\!\left(x(p)\right)dp ,
\tag{9.3}
\end{equation*}

where the second form integrates damage over the annual exceedance probability $p$, as is done with a damage–probability curve. Raising the design level costs more to build but reduces the residual EAD; the economically optimal design minimizes the sum of the annualized construction cost and the residual expected damage (Example 9.3). In practice, minimum standards set by regulation, and the uncertainty in both damages and probabilities, also shape the choice.

## 9.2 Reliability, resilience and vulnerability

For systems that operate continuously, such as a water supply, a reservoir or a treatment plant, performance is judged over a time series. Classify each time step as satisfactory (S) or unsatisfactory (failure, F), for example according to whether supply meets demand. Hashimoto et al. (1982) proposed three indices:

- **Reliability**: the probability of a satisfactory state, estimated as the fraction of time steps in S.
- **Resilience**: the probability that a failure step is followed by a satisfactory one, that is, how quickly the system recovers. It is the number of transitions from F to S divided by the number of failure steps; its reciprocal is the mean duration of failure events.
- **Vulnerability**: the severity of failures, measured for example by the mean or maximum deficit per failure event.

A system can be highly reliable but recover slowly, or fail often but mildly. Figure 9.2 compares two synthetic supply systems with the same demand: one draws directly from the river, the other has a small reservoir that smooths the shortfalls.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(92)
nm = 40 * 12; mon = np.arange(nm) % 12
seas = 1 + 0.9 * np.sin(2 * np.pi * (mon - 3) / 12)
z = np.zeros(nm)
for t in range(1, nm): z[t] = 0.5 * z[t - 1] + np.sqrt(0.75) * rng.normal()
inflow = 10 * seas * np.exp(0.45 * z - 0.1)                              # million m3 per month
demand = 7.0
def rrv(supply):
    fail = supply < demand - 1e-9; deficit = np.maximum(demand - supply, 0)
    rel = 1 - fail.mean()
    res = np.sum(fail[:-1] & ~fail[1:]) / max(fail.sum(), 1)
    events, cur = [], 0.0
    for t in range(nm):
        if fail[t]: cur += deficit[t]
        elif cur > 0: events.append(cur); cur = 0.0
    if cur > 0: events.append(cur)
    return rel, res, (np.mean(events) if events else 0.0), fail
sup_a = np.minimum(inflow, demand)                                        # run-of-river
S, cap, sup_b = 8.0, 15.0, np.zeros(nm)                                  # small reservoir
for t in range(nm):
    S += inflow[t]; sup_b[t] = min(S, demand); S = min(S - sup_b[t], cap)
ra, rb = rrv(sup_a), rrv(sup_b)
fig, ax = plt.subplots(figsize=(6.4, 3.6))
w = slice(0, 96); tt = np.arange(nm)[w] / 12
ax.plot(tt, inflow[w], color=ink2, lw=1, label="Inflow")
ax.axhline(demand, color=ink, lw=1.2, ls="--", label="Demand")
ax.plot(tt, sup_a[w], color=c2, lw=1.8, label="Supply, run-of-river")
ax.plot(tt, sup_b[w], color=c1, lw=1.8, label="Supply, with reservoir")
ax.set_ylim(0, 30); ax.set_xlabel("Year"); ax.set_ylabel("Million m³ per month"); ax.legend(loc="upper right", fontsize=8.5, ncol=2)
plt.show()
rows = [f"| Run-of-river | {ra[0]:.2f} | {ra[1]:.2f} | {ra[2]:.1f} |", f"| With 15 million m³ reservoir | {rb[0]:.2f} | {rb[1]:.2f} | {rb[2]:.1f} |"]
```

*Figure 9.2. Eight of 40 years of synthetic monthly inflow and the supply delivered against a constant demand, without and with a reservoir. Table 9.1 gives the performance indices over the whole record.*

```{code-cell} python
:tags: [remove-input]
Markdown("| System | Reliability | Resilience | Vulnerability (mean deficit per event, million m³) |\n|---|---|---|---|\n" + "\n".join(rows))
```

*Table 9.1. Reliability, resilience and vulnerability of the two systems in Figure 9.2.*

## 9.3 Sources and measures of uncertainty

The design value computed in Chapter 8 is uncertain for several reasons:

- **Natural variability**: the future sequence of floods is random; it is described by the fitted distribution itself.
- **Data uncertainty**: measurement errors, especially of high flows computed from rating curves extrapolated beyond the highest gauging, as well as gaps and inhomogeneities.
- **Parameter uncertainty**: short records give uncertain parameters (Chapter 7).
- **Model uncertainty**: the choice of distribution, of estimation method, and of the assumption of stationarity (Figure 8.1).

Uncertainty is described by a **standard deviation** or **coefficient of variation**, by **confidence intervals** for estimated quantities, or by a full probability distribution. Bayesian methods treat parameters as random variables and give a **predictive distribution** of future events that combines natural variability and parameter uncertainty in a single statement.

## 9.4 Propagating uncertainty

### First-order second-moment method

A design quantity $Y = g(X_1, \dots, X_k)$ depends on uncertain inputs with means $\mu_i$ and standard deviations $\sigma_i$. Expanding $g$ in a first-order Taylor series about the means gives the **first-order second-moment** (FOSM) approximation:

\begin{equation*}
\mu_Y \approx g(\mu_1, \dots, \mu_k), \qquad \sigma_Y^2 \approx \sum_{i} \left(\frac{\partial g}{\partial x_i}\right)^2 \sigma_i^2 + 2\sum_{i < j} \frac{\partial g}{\partial x_i}\frac{\partial g}{\partial x_j}\operatorname{Cov}(X_i, X_j),
\tag{9.4}
\end{equation*}

with derivatives evaluated at the means. For a product of independent factors, $Y = \prod X_i$, it reduces to adding squared coefficients of variation: $C_{v,Y}^2 \approx \sum C_{v,i}^2$. FOSM is quick and shows which input contributes most, but it is accurate only when $g$ is nearly linear over the range of uncertainty.

### Monte Carlo simulation

**Monte Carlo simulation** makes no such approximation: draw many sets of inputs from their distributions (using the probability integral transform, Chapter 6, or library generators), compute $Y$ for each, and use the resulting sample as the distribution of $Y$. With $N$ simulations, a probability $P$ is estimated with standard error $\sqrt{P(1 - P)/N}$, so rare failure probabilities need many runs. Correlated inputs must be generated with their correlation (Chapter 12). Figure 9.3 compares the two methods for a peak discharge computed with the rational method.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(94)
N = 100000
mC, cvC = 0.55, 0.15; mi, cvi = 62.0, 0.20; mA, cvA = 3.2, 0.05            # runoff coefficient, intensity (mm/h), area (km2)
Cs = rng.normal(mC, cvC * mC, N); ii = rng.lognormal(np.log(mi) - 0.5 * np.log(1 + cvi**2), np.sqrt(np.log(1 + cvi**2)), N)
As = rng.normal(mA, cvA * mA, N)
Qs = Cs * ii * As / 3.6                                                   # rational method, m3/s
mQ = mC * mi * mA / 3.6; sQ = mQ * np.sqrt(cvC**2 + cvi**2 + cvA**2)
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.hist(Qs, bins=120, density=True, color=c1, alpha=0.6, edgecolor="none", label="Monte Carlo")
q = np.linspace(0, Qs.max(), 400)
ax.plot(q, stats.norm.pdf(q, mQ, sQ), color=c2, lw=2.2, label="FOSM (normal)")
ax.set_xlim(0, 70); ax.set_xlabel("Peak discharge (m³ s⁻¹)"); ax.set_ylabel("Density"); ax.legend(loc="upper right", fontsize=9.5)
plt.show()
glue("mc_m", float(Qs.mean()), display=False); glue("mc_s", float(Qs.std()), display=False)
glue("fo_m", float(mQ), display=False); glue("fo_s", float(sQ), display=False)
glue("mc_p95", float(np.percentile(Qs, 95)), display=False); glue("fo_p95", float(stats.norm.ppf(0.95, mQ, sQ)), display=False)
```

*Figure 9.3. Uncertainty of a rational-method peak discharge $Q = CiA/3.6$ with uncertain runoff coefficient (mean 0.55, CV 15%), rainfall intensity (62 mm h⁻¹, CV 20%, lognormal) and area (3.2 km², CV 5%). Monte Carlo: mean {glue:text}`mc_m:.1f`, standard deviation {glue:text}`mc_s:.1f` m³ s⁻¹; FOSM: {glue:text}`fo_m:.1f` and {glue:text}`fo_s:.1f` m³ s⁻¹. The 95th percentiles are {glue:text}`mc_p95:.1f` (Monte Carlo) and {glue:text}`fo_p95:.1f` m³ s⁻¹ (FOSM), which misses the skewness.*

### Reliability index

Structural reliability methods, borrowed from civil engineering, express failure as a negative **safety margin** $M = C - L$, the difference between capacity (levee height, channel capacity) and load (flood level, discharge). If $M$ is normal, the probability of failure is

\begin{equation*}
P_f = P(M < 0) = \Phi(-\beta), \qquad \beta = \frac{\mu_M}{\sigma_M},
\tag{9.5}
\end{equation*}

where $\beta$ is the **reliability index**, the number of standard deviations by which the mean margin exceeds zero. A design with $\beta$ = 3 has $P_f \approx$ 0.0013. For non-normal variables, more refined methods (first-order reliability methods) or Monte Carlo simulation are used.

## 9.5 Design under nonstationarity

All of the above assumes that the distribution of annual maxima does not change over the design life. Urbanization, river engineering and climate change can violate this assumption ("stationarity is dead", Milly et al., 2008). If the annual exceedance probability of a fixed design level changes from year to year, $p_t$, the risk over a design life of $n$ years generalizes Equation (9.1):

\begin{equation*}
R = 1 - \prod_{t=1}^{n} (1 - p_t) .
\tag{9.6}
\end{equation*}

The return period loses its simple meaning, because $1/p$ differs from year to year. Two generalizations are used (Salas and Obeysekera, 2014): the **expected waiting time** until the first exceedance, and the design level for which the **expected number of exceedances** over the design life is one. Both depend on the design life and on the assumed trend, which must be estimated (Chapter 13) and is itself uncertain.

Figure 9.4 illustrates a GEV whose location and scale parameters increase by 0.5% per year. A level that is the 100-year flood today is exceeded with increasing probability, and its risk over a 50-year design life rises from {glue:text}`R50_s:.2f` to {glue:text}`R50_n:.2f`.

```{code-cell} python
:tags: [hide-input]
mu0, sg, xi = 300.0, 90.0, 0.1
x100 = stats.genextreme.ppf(0.99, -xi, loc=mu0, scale=sg)
years = np.arange(1, 101)
p_t = stats.genextreme.sf(x100, -xi, loc=mu0 * (1 + 0.005 * years), scale=sg * (1 + 0.005 * years))
R_stat = 1 - 0.99 ** years; R_non = 1 - np.cumprod(1 - p_t)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 3.5))
a1.plot(years, p_t, color=c2, lw=2.2); a1.axhline(0.01, color=c1, lw=1.6, ls="--")
a1.set_xlabel("Year of design life"); a1.set_ylabel("Annual exceedance probability"); a1.set_title("(a) Exceedance probability")
a2.plot(years, R_stat, color=c1, lw=2.2, label="Stationary"); a2.plot(years, R_non, color=c2, lw=2.2, label="Nonstationary")
a2.set_xlabel("Design life n (years)"); a2.set_ylabel("Risk"); a2.set_title("(b) Risk over n years"); a2.legend(loc="lower right", fontsize=9)
plt.show()
glue("R50_s", float(R_stat[49]), display=False); glue("R50_n", float(R_non[49]), display=False); glue("p50", float(p_t[49]), display=False)
```

*Figure 9.4. A level equal to today's 100-year flood under a GEV whose location and scale increase by 0.5% per year. (a) Its annual exceedance probability rises to {glue:text}`p50:.3f` after 50 years. (b) The risk over a 50-year design life is {glue:text}`R50_n:.2f` instead of {glue:text}`R50_s:.2f`.*

Nonstationary design requires a model of how the distribution changes, usually parameters that depend on time or on covariates such as an urbanization index or a climate variable, fitted by maximum likelihood. Projecting such trends decades ahead adds a large uncertainty of its own, which is why scenario-based approaches and adaptive designs that can be upgraded are often preferred.

## 9.6 Worked examples

**Example 9.1 — Choosing a design return period.** A bridge has a design life of 75 years, and the acceptable risk of its design flood being exceeded is 20%. What return period should be used?

```{code-cell} python
R, n = 0.20, 75
print(f"T = {1 / (1 - (1 - R) ** (1 / n)):.0f} years")
```

**Example 9.2 — Reliability index of a levee.** The annual maximum water level at a levee has mean 6.2 m and standard deviation 0.7 m; the crest elevation, uncertain because of settlement, has mean 8.4 m and standard deviation 0.2 m. Assuming independent normal variables, find the reliability index and the annual probability of overtopping.

```{code-cell} python
mC_, sC_, mL_, sL_ = 8.4, 0.2, 6.2, 0.7
beta = (mC_ - mL_) / np.sqrt(sC_**2 + sL_**2)
print(f"beta = {beta:.2f}; P_f = {stats.norm.cdf(-beta):.4f} per year")
```

**Example 9.3 — Economic design.** A levee designed for return period $T$ costs $0.9\,T^{0.35}$ million won per year (annualized), and the residual expected annual damage is $120/T$ million won. Find the return period that minimizes the total annual cost.

```{code-cell} python
Tg = np.logspace(0.5, 3.5, 2000)
total = 0.9 * Tg ** 0.35 + 120 / Tg
k = np.argmin(total)
print(f"optimal T ≈ {Tg[k]:.0f} years; total annual cost = {total[k]:.1f} million won")
```

**Example 9.4 — FOSM for a product.** A runoff volume is $V = C P A$ with coefficients of variation 0.15 for $C$, 0.10 for rainfall depth $P$ and 0.03 for area $A$, all independent. Find the coefficient of variation of $V$ and the largest contributor.

```{code-cell} python
cv = {"C": 0.15, "P": 0.10, "A": 0.03}
tot = np.sqrt(sum(v ** 2 for v in cv.values()))
print(f"CV of V = {tot:.3f}; shares of variance: " + ", ".join(f"{k} {100 * v**2 / tot**2:.0f}%" for k, v in cv.items()))
```

## 9.7 Common mistakes

- **Choosing $T$ equal to the design life**, which implies a 63% risk.
- **Ignoring the consequences of failure** when choosing a design standard.
- **Treating FOSM results as exact** for nonlinear functions or skewed inputs.
- **Using too few Monte Carlo runs** to estimate small failure probabilities.
- **Applying stationary return periods** to catchments with strong trends, or extrapolating fitted trends far beyond the data.

## 9.8 Exercises

```{code-cell} python
:tags: [remove-cell]
glue("r1", float(1 - (1 - 1 / 200) ** 100), display=False)
glue("r2", float(1 / (1 - 0.95 ** (1 / 30))), display=False)
glue("r3", float(stats.norm.cdf(-2.5)), display=False)
glue("r4", float(np.sqrt(0.2 ** 2 + 0.1 ** 2)), display=False)
glue("r5", float(1 - (1 - 0.01) ** 10 * (1 - 0.02) ** 10 * (1 - 0.03) ** 10), display=False)
```

1. What is the risk that the 200-year flood is exceeded at least once in 100 years?
2. Find the return period that limits the risk over 30 years to 5%.
3. A design has reliability index 2.5. What is its probability of failure?
4. $Y = X_1 X_2$ with independent factors whose coefficients of variation are 0.2 and 0.1. Estimate the coefficient of variation of $Y$ by FOSM.
5. A level's annual exceedance probability is 0.01 in years 1–10, 0.02 in years 11–20 and 0.03 in years 21–30. What is the risk over 30 years?

```{admonition} Answers
:class: dropdown
1. {glue:text}`r1:.3f`.
2. About {glue:text}`r2:.0f` years.
3. {glue:text}`r3:.4f`.
4. {glue:text}`r4:.3f`.
5. {glue:text}`r5:.3f`.
```

## 9.9 Further reading

- Ang, A. H.-S., and Tang, W. H. (2007). *Probability Concepts in Engineering* (2nd ed.). Wiley.
- Hashimoto, T., Stedinger, J. R., and Loucks, D. P. (1982). Reliability, resiliency, and vulnerability criteria for water resource system performance evaluation. *Water Resources Research*, 18, 14–20.
- Maity, R. (2022). *Statistical Methods in Hydrology and Hydroclimatology* (2nd ed.). Springer.
- Milly, P. C. D., et al. (2008). Stationarity is dead: whither water management? *Science*, 319, 573–574.
- Salas, J. D., and Obeysekera, J. (2014). Revisiting the concepts of return period and risk for nonstationary hydrologic extreme events. *Journal of Hydrologic Engineering*, 19, 554–568.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
