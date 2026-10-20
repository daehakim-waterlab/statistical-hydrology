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

# 15. Copulas and joint extremes

```{admonition} Core ideas
:class: tip
1. Many hydrological hazards depend on several variables together: flood peak and volume, storm depth and duration, drought duration and severity, river flow and sea level. Their **joint** behavior, especially in the extremes, decides the risk.
2. **Rank correlations** (Kendall's τ, Spearman's ρ) measure dependence independently of the marginal distributions; **tail dependence** measures whether extremes tend to occur together.
3. **Sklar's theorem** splits a joint distribution into its marginals and a **copula**, a joint distribution of uniform variables that carries all the dependence. Marginals and dependence can then be modeled separately.
4. **Elliptical** (Gaussian, t) and **Archimedean** (Clayton, Gumbel, Frank) families differ mainly in their tails. They are fitted by inverting Kendall's τ or by pseudo-likelihood and compared by goodness of fit.
5. Joint return periods depend on the definition: an event in which **either** variable is exceeded is more frequent than one in which **both** are. Copulas also give **conditional** distributions for simulation and prediction.
```

```{code-cell} python
:tags: [remove-cell]
# Plot style shared by all figures in this chapter.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, integrate, optimize
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
# Copula families: cdf, density, sampler and Kendall's tau <-> parameter
def C_gumbel(u, v, t): return np.exp(-((-np.log(u)) ** t + (-np.log(v)) ** t) ** (1 / t))
def c_gumbel(u, v, t):
    x, y = -np.log(u), -np.log(v); A = x ** t + y ** t
    return C_gumbel(u, v, t) / (u * v) * (x * y) ** (t - 1) * A ** (2 / t - 2) * ((t - 1) * A ** (-1 / t) + 1)
def C_clayton(u, v, t): return (u ** -t + v ** -t - 1) ** (-1 / t)
def c_clayton(u, v, t): return (1 + t) * (u * v) ** (-t - 1) * (u ** -t + v ** -t - 1) ** (-2 - 1 / t)
def C_frank(u, v, t): return -np.log(1 + np.expm1(-t * u) * np.expm1(-t * v) / np.expm1(-t)) / t
def c_frank(u, v, t):
    e = np.exp(-t); return t * (1 - e) * np.exp(-t * (u + v)) / ((1 - e) - (1 - np.exp(-t * u)) * (1 - np.exp(-t * v))) ** 2
def C_gauss(u, v, r):
    return stats.multivariate_normal(cov=[[1, r], [r, 1]]).cdf(np.dstack([stats.norm.ppf(u), stats.norm.ppf(v)]))
def c_gauss(u, v, r):
    a, b = stats.norm.ppf(u), stats.norm.ppf(v); return np.exp(-(r * r * (a * a + b * b) - 2 * r * a * b) / (2 * (1 - r * r))) / np.sqrt(1 - r * r)
def r_gumbel(t, n, rng):                                                   # Marshall-Olkin with a positive stable frailty
    a = 1 / t; U = rng.uniform(0, np.pi, n); W = rng.exponential(size=n)
    S = (np.sin(a * U) / np.sin(U) ** (1 / a)) * (np.sin((1 - a) * U) / W) ** ((1 - a) / a)
    return np.exp(-(rng.exponential(size=(n, 2)) / S[:, None]) ** a)
def r_clayton(t, n, rng):                                                  # Marshall-Olkin with a gamma frailty
    V = rng.gamma(1 / t, 1, n); return (1 + rng.exponential(size=(n, 2)) / V[:, None]) ** (-1 / t)
def r_frank(t, n, rng):                                                    # conditional inversion
    u, w = rng.uniform(size=n), rng.uniform(size=n)
    return np.c_[u, -np.log(1 + w * np.expm1(-t) / (w + (1 - w) * np.exp(-t * u))) / t]
def r_gauss(r, n, rng):
    z = rng.multivariate_normal([0, 0], [[1, r], [r, 1]], n); return stats.norm.cdf(z)
debye1 = lambda t: integrate.quad(lambda s: s / np.expm1(s), 0, t)[0] / t
tau_from = {"Gumbel": lambda t: 1 - 1 / t, "Clayton": lambda t: t / (t + 2),
            "Frank": lambda t: 1 - 4 / t * (1 - debye1(t)), "Gaussian": lambda r: 2 / np.pi * np.arcsin(r)}
par_from_tau = {"Gumbel": lambda k: 1 / (1 - k), "Clayton": lambda k: 2 * k / (1 - k),
                "Frank": lambda k: optimize.brentq(lambda t: tau_from["Frank"](t) - k, 1e-4, 80), "Gaussian": lambda k: np.sin(np.pi * k / 2)}
```

## 15.1 Why marginals are not enough

A flood is described by more than its peak. A long, moderately high flood with a large volume can fill a reservoir or overtop a levee by seepage, while a sharp peak of small volume may pass harmlessly; spillway design depends on both. Likewise, a storm's damage depends on depth and duration, a drought's on duration and deficit (Chapter 8), and coastal flooding on river flow and sea level together (Chapter 2). Analyzing each variable separately and assuming independence, or assuming both are normal and correlated, can badly misstate the probability of the combinations that matter.

Two difficulties arise. The variables usually have different, skewed marginal distributions (a GEV peak, a gamma volume), so the bivariate normal does not apply. And the dependence may be stronger in the extremes than in the bulk of the data: the largest peaks often come with the largest volumes. Copulas address both.

## 15.2 Rank correlation and tail dependence

The Pearson correlation measures linear association and changes under nonlinear transformations of the variables. **Rank correlations** depend only on the ordering of the data and are therefore unchanged by any increasing transformation of either variable:

- **Spearman's ρ** is the Pearson correlation of the ranks.
- **Kendall's τ** is the difference between the probabilities of concordance and discordance of two randomly chosen pairs: a pair of observations $(x_i, y_i)$ and $(x_j, y_j)$ is **concordant** if $x$ and $y$ move in the same direction from one observation to the other. With $n_c$ concordant and $n_d$ discordant pairs among the $n(n - 1)/2$ pairs, $\hat\tau = (n_c - n_d)/[n(n - 1)/2]$.

Because they do not depend on the marginals, rank correlations are the natural measures of the dependence carried by a copula.

**Tail dependence** describes the extremes. The upper tail dependence coefficient is the limiting probability that one variable is extreme given that the other is:

\begin{equation*}
\lambda_U = \lim_{q \to 1} P\left(V > q \mid U > q\right),
\tag{15.1}
\end{equation*}

where $U$ and $V$ are the variables transformed to uniform scale by their own cdfs. If $\lambda_U > 0$, the largest values tend to occur together even at the highest levels; if $\lambda_U = 0$, they become independent in the far tail. The lower tail dependence $\lambda_L$ is defined similarly. Two models with the same overall correlation can have very different tail dependence, and therefore very different joint return periods.

## 15.3 Sklar's theorem

A **copula** is a joint distribution function $C(u, v)$ of two variables, each uniform on $[0, 1]$. **Sklar's theorem** (Sklar, 1959) states that any joint distribution $H$ of continuous variables $X$ and $Y$, with marginals $F$ and $G$, can be written as

\begin{equation*}
H(x, y) = C\left(F(x), G(y)\right),
\tag{15.2}
\end{equation*}

for a unique copula $C$, and conversely that any copula combined with any marginals gives a valid joint distribution. The joint density is then $h(x, y) = c(F(x), G(y))\,f(x)\,g(y)$, where $c = \partial^2 C/\partial u\,\partial v$ is the **copula density**. The marginals describe each variable; the copula describes everything about how they depend on each other.

Three copulas mark the range of possibilities. The **independence copula** $C(u, v) = uv$; the **comonotonic** upper bound $C(u, v) = \min(u, v)$, when $Y$ is an increasing function of $X$; and the **countermonotonic** lower bound $\max(u + v - 1, 0)$. Every copula lies between the two bounds (the Fréchet–Hoeffding bounds).

In practice, the data are transformed to the copula scale before the marginals are fitted, using their ranks: the **pseudo-observations** $u_i = R_i/(n + 1)$ and $v_i = S_i/(n + 1)$, where $R_i$ and $S_i$ are the ranks of $x_i$ and $y_i$. Their scatter plot shows the dependence structure stripped of the marginals, and their empirical distribution is the **empirical copula**.

## 15.4 Elliptical and Archimedean families

### Elliptical copulas

The **Gaussian copula** is the dependence structure of the bivariate normal distribution, with correlation parameter $r$ (related to Kendall's τ by $\tau = (2/\pi)\arcsin r$). It is easy to extend to many variables (Chapter 12) but has no tail dependence: extremes become independent in the limit. The **t copula** adds a degrees-of-freedom parameter and has symmetric tail dependence in both tails.

### Archimedean copulas

**Archimedean** copulas are built from a generator function $\varphi$ as $C(u, v) = \varphi^{-1}[\varphi(u) + \varphi(v)]$, giving one-parameter families with closed forms:

\begin{equation*}
\begin{aligned}
\text{Gumbel:}\quad & C(u, v) = \exp\left\{-\left[(-\ln u)^\theta + (-\ln v)^\theta\right]^{1/\theta}\right\}, & \tau &= 1 - 1/\theta, & \lambda_U &= 2 - 2^{1/\theta}, \\
\text{Clayton:}\quad & C(u, v) = \left(u^{-\theta} + v^{-\theta} - 1\right)^{-1/\theta}, & \tau &= \frac{\theta}{\theta + 2}, & \lambda_L &= 2^{-1/\theta}, \\
\text{Frank:}\quad & C(u, v) = -\frac{1}{\theta}\ln\left[1 + \frac{(e^{-\theta u} - 1)(e^{-\theta v} - 1)}{e^{-\theta} - 1}\right], & \tau &= 1 - \frac{4}{\theta}\left[1 - D_1(\theta)\right], & \lambda &= 0,
\end{aligned}
\tag{15.3}
\end{equation*}

where $D_1(\theta) = \theta^{-1}\int_0^\theta s/(e^s - 1)\,ds$ is the Debye function. The **Gumbel** copula has upper tail dependence, which suits flood peaks and volumes or rainfall at nearby gauges during extreme storms; the **Clayton** copula has lower tail dependence, which suits droughts that are jointly severe; the **Frank** copula is symmetric and has no tail dependence. Figure 15.1 shows samples from four families with the same Kendall's τ: they differ almost only in their corners.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(151)
tau0, nsim = 0.5, 1500
samplers = {"Gaussian": r_gauss, "Gumbel": r_gumbel, "Clayton": r_clayton, "Frank": r_frank}
fig, axs = plt.subplots(1, 4, figsize=(6.4, 2.0))
for ax, (name, rs), col in zip(axs, samplers.items(), [ink2, c2, c1, c3]):
    UV = rs(par_from_tau[name](tau0), nsim, rng)
    ax.plot(UV[:, 0], UV[:, 1], ".", color=col, ms=1.6, alpha=0.7)
    ax.set_title(name, fontsize=9.5); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal"); ax.grid(False)
    for s in ax.spines.values(): s.set_visible(True); s.set_color("#c9c8c3")
plt.show()
```

*Figure 15.1. 1500 samples on the uniform scale from four copula families with the same Kendall's τ = 0.5. The Gumbel copula concentrates points in the upper-right corner (joint high extremes), the Clayton copula in the lower-left (joint low extremes), and the Gaussian and Frank copulas in neither.*

## 15.5 Fitting and selecting a copula

The usual procedure has four steps:

1. Fit the **marginal** distributions separately (Chapters 6–8).
2. Compute the **pseudo-observations** from the ranks.
3. Estimate the **copula parameter**, either by **inverting Kendall's τ** with the relations in Equation (15.3), which is simple and robust, or by **maximum pseudo-likelihood**, maximizing $\sum_i \ln c(u_i, v_i; \theta)$.
4. **Select** among candidate families by information criteria computed from the pseudo-likelihood, by the distance between the fitted and empirical copulas (a Cramér–von Mises statistic, with p-values from a parametric bootstrap), and by plots of the upper tail.

Table 15.1 applies the procedure to 60 years of synthetic annual flood peaks and the corresponding flood volumes, generated with a Gumbel copula. With records this short, several families often fit almost equally well; physical reasoning about tail dependence then guides the choice, and the sensitivity of the design quantity to the choice should be reported.

```{code-cell} python
:tags: [remove-input]
rng = np.random.default_rng(152)
ny = 60; th_true = 2.2
UVt = r_gumbel(th_true, ny, rng)
Mpk = stats.genextreme(-0.08, loc=850, scale=260)                           # flood peak (m3/s)
Mvol = stats.gamma(4.5, scale=22.0)                                         # flood volume (million m3)
peak, vol = Mpk.ppf(UVt[:, 0]), Mvol.ppf(UVt[:, 1])
u = stats.rankdata(peak) / (ny + 1); v = stats.rankdata(vol) / (ny + 1)
tau_hat = stats.kendalltau(peak, vol).statistic
dens = {"Gaussian": c_gauss, "Gumbel": c_gumbel, "Clayton": c_clayton, "Frank": c_frank}
cdfs = {"Gaussian": C_gauss, "Gumbel": C_gumbel, "Clayton": C_clayton, "Frank": C_frank}
emp = np.array([np.mean((u <= u[i]) & (v <= v[i])) for i in range(ny)])
rows, fitted = [], {}
for name in dens:
    th0 = par_from_tau[name](tau_hat)
    nll = lambda t: -np.sum(np.log(np.maximum(dens[name](u, v, t), 1e-300)))
    bnds = {"Gaussian": (-0.99, 0.99), "Gumbel": (1.0001, 20), "Clayton": (1e-4, 30), "Frank": (1e-3, 60)}[name]
    th = optimize.minimize_scalar(nll, bounds=bnds, method="bounded").x
    fitted[name] = th
    cvm = np.sum((cdfs[name](u, v, th) - emp) ** 2)
    rows.append(f"| {name} | {th0:.3f} | {th:.3f} | {2 * nll(th) + 2:.1f} | {cvm:.4f} |")
glue("tau_hat", float(tau_hat), display=False)
Markdown("| Copula | Parameter (τ inversion) | Parameter (pseudo-likelihood) | AIC | Cramér–von Mises distance |\n|---|---|---|---|---|\n" + "\n".join(rows))
```

*Table 15.1. Copulas fitted to 60 synthetic pairs of annual flood peak and volume (generated with a Gumbel copula, θ = 2.2; sample Kendall's τ = {glue:text}`tau_hat:.3f`). Lower AIC and distance indicate a better fit.*

## 15.6 Joint return periods and conditional simulation

### Joint return periods

For annual data, let $u = F(x)$ and $v = G(y)$ be the non-exceedance probabilities of a peak $x$ and a volume $y$. Two joint events are natural:

- **OR**: the peak exceeds $x$ *or* the volume exceeds $y$ (or both). Its annual probability is $1 - C(u, v)$, so
\begin{equation*}
T_{\text{OR}} = \frac{1}{1 - C(u, v)} .
\tag{15.4}
\end{equation*}
- **AND**: both are exceeded. By the addition rule (Equation 2.2), its probability is $1 - u - v + C(u, v)$, so
\begin{equation*}
T_{\text{AND}} = \frac{1}{1 - u - v + C(u, v)} .
\tag{15.5}
\end{equation*}

Always $T_{\text{OR}} \le \min(T_x, T_y) \le \max(T_x, T_y) \le T_{\text{AND}}$, where $T_x = 1/(1 - u)$ and $T_y = 1/(1 - v)$ are the univariate return periods. The relevant definition depends on how the system fails: a structure that fails if either variable is exceeded needs the OR definition, one that fails only under both needs AND. For two 100-year values, the AND return period is 10,000 years under independence but only a few hundred years under strong upper tail dependence (Example 15.2). Figure 15.2 shows return-period contours for the fitted Gumbel copula. Other definitions, such as the **Kendall return period**, based on the distribution of $C$ itself, have been proposed to give a single critical region.

```{code-cell} python
:tags: [hide-input]
th = fitted["Gumbel"]
P = np.linspace(400, 3200, 220); Vv = np.linspace(20, 330, 220); PP, VV = np.meshgrid(P, Vv)
UU, WW = Mpk.cdf(PP), Mvol.cdf(VV)
Cg = C_gumbel(np.clip(UU, 1e-9, 1 - 1e-12), np.clip(WW, 1e-9, 1 - 1e-12), th)
T_or = 1 / (1 - Cg); T_and = 1 / np.maximum(1 - UU - WW + Cg, 1e-12)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
cs1 = ax.contour(PP, VV, T_or, levels=[10, 50, 100], colors=c1, linewidths=1.8)
cs2 = ax.contour(PP, VV, T_and, levels=[10, 50, 100], colors=c2, linewidths=1.8, linestyles="dashed")
ax.clabel(cs1, fmt={10: "10", 50: "50", 100: "100"}, fontsize=8.5); ax.clabel(cs2, fmt={10: "10", 50: "50", 100: "100"}, fontsize=8.5)
ax.plot(peak, vol, "o", color=ink, ms=3.5)
ax.plot([], [], color=c1, lw=1.8, label="$T_{OR}$ (years)"); ax.plot([], [], color=c2, lw=1.8, ls="--", label="$T_{AND}$ (years)")
ax.set_xlabel("Flood peak (m³ s⁻¹)"); ax.set_ylabel("Flood volume (10⁶ m³)"); ax.legend(loc="upper left", fontsize=9.5)
plt.show()
```

*Figure 15.2. Joint return-period contours of annual flood peak and volume for the fitted Gumbel copula with GEV and gamma marginals: OR (solid) and AND (dashed) definitions, with the 60 synthetic observations. A point on the 100-year AND contour lies well inside the 100-year OR contour.*

### Conditional distributions and simulation

The conditional distribution of $V$ given $U = u$ is the partial derivative of the copula,

\begin{equation*}
C_{V \mid U}(v \mid u) = \frac{\partial C(u, v)}{\partial u} ,
\tag{15.6}
\end{equation*}

and conditional distributions given exceedance, such as $P(V > v \mid U > u) = [1 - u - v + C(u, v)]/(1 - u)$, follow from the joint cdf. They answer questions such as "given that this year's peak exceeds the 50-year value, what volume should be expected?" (Example 15.3), and they are the basis of **conditional simulation**: draw $u$, then draw $v$ from Equation (15.6) by inverting it numerically (or with closed forms, as for the Frank copula), and transform both back through the marginal quantile functions. Such simulations generate design hydrographs with realistic combinations of peak and volume, or multisite rainfall fields with realistic joint extremes.

## 15.7 Lab (planned)

A Colab notebook will extract annual flood peaks and volumes from an openly available daily discharge record, fit marginal distributions and several copulas, and compare OR and AND joint return periods for design.

## 15.8 Worked examples

**Example 15.1 — Kendall's τ by counting pairs.** Six floods have peaks (m³ s⁻¹) and volumes (million m³) as below. Count concordant and discordant pairs and compute Kendall's τ, and the Gumbel copula parameter by inversion.

```{code-cell} python
pk = np.array([620, 910, 750, 1340, 1050, 880.0]); vl = np.array([62, 78, 55, 140, 97, 84.0])
nc = nd = 0
for i in range(6):
    for j in range(i + 1, 6):
        s = np.sign(pk[j] - pk[i]) * np.sign(vl[j] - vl[i]); nc += s > 0; nd += s < 0
tau = (nc - nd) / 15
print(f"concordant {nc}, discordant {nd}; tau = {tau:.3f}; Gumbel theta = {1 / (1 - tau):.2f}")
```

**Example 15.2 — AND return period of two 100-year values.** Compare the AND return period of two variables both at their 100-year values under independence, a Gaussian copula and a Gumbel copula, all with Kendall's τ = 0.5 for the dependent cases.

```{code-cell} python
u0 = 0.99
r = par_from_tau["Gaussian"](0.5); thg = par_from_tau["Gumbel"](0.5)
for name, Cv in [("independent", u0 * u0), ("Gaussian", float(C_gauss(np.array([u0]), np.array([u0]), r))), ("Gumbel", C_gumbel(u0, u0, thg))]:
    print(f"{name:12s}: T_AND = {1 / (1 - 2 * u0 + Cv):7.0f} years, T_OR = {1 / (1 - Cv):5.1f} years")
```

**Example 15.3 — A conditional probability.** With the fitted Gumbel copula of Table 15.1, what is the probability that the volume exceeds its 50-year value given that the peak exceeds its 50-year value? Compare with independence.

```{code-cell} python
u50 = v50 = 1 - 1 / 50
pc = (1 - u50 - v50 + C_gumbel(u50, v50, fitted["Gumbel"])) / (1 - u50)
print(f"P(V > v50 | U > u50) = {pc:.3f} (independence: {1 - v50:.3f})")
```

**Example 15.4 — Tail dependence.** Compute the upper tail dependence coefficient of the Gumbel copula for τ = 0.3, 0.5 and 0.7.

```{code-cell} python
for t_ in (0.3, 0.5, 0.7):
    th_ = 1 / (1 - t_); print(f"tau = {t_}: theta = {th_:.2f}, lambda_U = {2 - 2 ** (1 / th_):.3f}")
```

## 15.9 Common mistakes

- **Assuming independence** between variables driven by the same storm, which underestimates joint extremes.
- **Using the bivariate normal for skewed variables**, or a Gaussian copula when extremes clearly cluster.
- **Fitting copulas to raw data** instead of rank-based pseudo-observations, or fitting marginals and copula inconsistently.
- **Reporting a joint return period without stating its definition** (OR, AND, Kendall).
- **Over-interpreting family selection** from short records; report the sensitivity of design values to the choice.

## 15.10 Exercises

```{code-cell} python
:tags: [remove-cell]
glue("z1", float(1 / (1 - 0.6)), display=False)
glue("z2", float(2 * 0.4 / (1 - 0.4)), display=False)
glue("z3", float(1 / (1 - 0.98 * 0.95)), display=False)
glue("z4", float(1 / (1 - 0.98 - 0.95 + 0.98 * 0.95)), display=False)
glue("z5", float(2 - 2 ** (1 / 2.0)), display=False)
```

1. Find the Gumbel copula parameter for Kendall's τ = 0.6.
2. Find the Clayton copula parameter for Kendall's τ = 0.4.
3. Two independent annual variables are at their 50-year and 20-year values. Find the OR joint return period.
4. For the same values, find the AND joint return period.
5. Find the upper tail dependence coefficient of a Gumbel copula with θ = 2.

```{admonition} Answers
:class: dropdown
1. θ = {glue:text}`z1:.2f`.
2. θ = {glue:text}`z2:.2f`.
3. {glue:text}`z3:.1f` years.
4. {glue:text}`z4:.0f` years.
5. {glue:text}`z5:.3f`.
```

## 15.11 Further reading

- Genest, C., and Favre, A.-C. (2007). Everything you always wanted to know about copula modeling but were afraid to ask. *Journal of Hydrologic Engineering*, 12, 347–368.
- Maity, R. (2022). *Statistical Methods in Hydrology and Hydroclimatology* (2nd ed.). Springer.
- Nelsen, R. B. (2006). *An Introduction to Copulas* (2nd ed.). Springer.
- Salvadori, G., and De Michele, C. (2004). Frequency analysis via copulas: theoretical aspects and applications to hydrological events. *Water Resources Research*, 40, W12511.
- Sklar, A. (1959). Fonctions de répartition à n dimensions et leurs marges. *Publications de l'Institut de Statistique de l'Université de Paris*, 8, 229–231.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
