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

# 6. Continuous distributions for hydrological variables

```{admonition} Core ideas
:class: tip
1. A small family of continuous distributions covers most hydrological variables. Each is chosen for its range (unbounded, positive, bounded), its shape (symmetric or skewed) and, where possible, a physical or mathematical argument.
2. The **normal** distribution fits sums and averages, such as annual totals. The **lognormal** fits positive, skewed variables that arise as products.
3. The **gamma** family, with its shifted form, the **Pearson type III**, and its logarithmic form, the **log-Pearson type III**, fits rainfall amounts, droughts and floods. Flood quantiles are often written as mean + **frequency factor** × standard deviation.
4. Maxima of many values tend toward the **generalized extreme value** distribution, whose shape parameter decides whether the tail is light (Gumbel), heavy (Fréchet) or bounded (Weibull). Minima, such as low flows, have their own extreme-value forms.
5. Intermittent variables need **mixed** distributions with a probability mass at zero, and records drawn from different storm types may need **mixtures**.
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

## 6.1 Uniform and exponential

### Uniform

The **uniform distribution** on $[a, b]$ has constant density $1/(b - a)$. It rarely describes a hydrological variable directly, but it is central to simulation because of the **probability integral transform**: if $X$ has a continuous cdf $F$, then $U = F(X)$ is uniform on $[0, 1]$, and conversely $X = F^{-1}(U)$ has cdf $F$ when $U$ is uniform. Any distribution with a computable quantile function can therefore be sampled by transforming uniform random numbers (Chapter 9), and the same idea underlies probability plots (Chapter 8) and copulas (Chapter 15).

### Exponential

The **exponential distribution**

\begin{equation*}
f(x) = \frac{1}{\beta}\,e^{-x/\beta}, \qquad F(x) = 1 - e^{-x/\beta}, \qquad x \ge 0,
\tag{6.1}
\end{equation*}

has mean and standard deviation both equal to $\beta$, so its coefficient of variation is 1 and its skewness 2. It is the distribution of the time between events of a Poisson process (Chapter 5), and it shares the memoryless property of the geometric distribution. It is used for interarrival times of storms, for the duration of dry spells as a first approximation, and for the excesses of a variable above a high threshold in the peaks-over-threshold method (Chapter 8). A shifted version with a lower bound $x_0$ is common.

## 6.2 Normal and lognormal

### Normal

The **normal** (Gaussian) distribution with mean $\mu$ and standard deviation $\sigma$ has density

\begin{equation*}
f(x) = \frac{1}{\sigma\sqrt{2\pi}}\exp\left[-\frac{(x - \mu)^2}{2\sigma^2}\right], \qquad -\infty < x < \infty .
\tag{6.2}
\end{equation*}

It is symmetric, and about 68%, 95% and 99.7% of its probability lie within one, two and three standard deviations of the mean. Any normal variable is converted to the **standard normal** $Z \sim N(0, 1)$ by $Z = (X - \mu)/\sigma$, and probabilities are read from $\Phi(z)$, the standard normal cdf (Appendix A). Sums of independent normal variables are normal. By the central limit theorem (Chapter 4), the normal distribution suits variables that are sums or averages of many small contributions: annual rainfall and runoff in humid climates, annual mean temperature, and errors of measurement. Its unbounded lower tail is a drawback for variables that must be positive and have a large coefficient of variation.

### Lognormal

A positive variable $X$ is **lognormal** if $Y = \ln X$ is normal with mean $\mu_y$ and standard deviation $\sigma_y$. By Equation (3.9), its density is

\begin{equation*}
f(x) = \frac{1}{x\,\sigma_y\sqrt{2\pi}}\exp\left[-\frac{(\ln x - \mu_y)^2}{2\sigma_y^2}\right], \qquad x > 0 .
\tag{6.3}
\end{equation*}

Its moments follow from the mgf of the normal distribution:

\begin{equation*}
\mu_x = \exp\left(\mu_y + \tfrac{1}{2}\sigma_y^2\right), \qquad C_v^2 = e^{\sigma_y^2} - 1, \qquad \gamma = 3C_v + C_v^3 .
\tag{6.4}
\end{equation*}

The median is $e^{\mu_y}$, smaller than the mean. The skewness is tied to the coefficient of variation, so the two-parameter lognormal cannot fit a sample whose skewness differs much from $3C_v + C_v^3$; a three-parameter form, $\ln(X - x_0)$ normal, adds a lower bound. Just as sums tend to normality, products of many positive factors tend to lognormality, which is one argument for its use for hydraulic conductivity, sediment concentration and flood peaks (Figure 6.1).

```{code-cell} python
:tags: [hide-input]
mu, sd = 600.0, 300.0
sy = np.sqrt(np.log(1 + (sd / mu) ** 2)); my = np.log(mu) - sy ** 2 / 2
x = np.linspace(-400, 2200, 600)
fig, ax = plt.subplots(figsize=(6.4, 3.5))
ax.plot(x, stats.norm.pdf(x, mu, sd), color=c1, lw=2.2, label="Normal")
ax.plot(x[x > 0], stats.lognorm.pdf(x[x > 0], sy, scale=np.exp(my)), color=c2, lw=2.2, label="Lognormal")
ax.axvline(0, color=ink2, lw=0.8)
ax.set_xlabel("x (for example, annual maximum discharge, m³ s⁻¹)"); ax.set_ylabel("Density"); ax.legend(loc="upper right", fontsize=9.5)
plt.show()
glue("pneg", float(stats.norm.cdf(0, mu, sd)), display=False)
glue("ln_skew", float(3 * sd / mu + (sd / mu) ** 3), display=False)
```

*Figure 6.1. Normal and lognormal distributions with the same mean (600) and standard deviation (300). The normal assigns a probability of {glue:text}`pneg:.3f` to negative values, which are impossible for discharge; the lognormal is bounded at zero and has skewness {glue:text}`ln_skew:.2f`.*

## 6.3 Gamma, Pearson type III and log-Pearson type III

### Gamma

The **gamma distribution** with shape $\alpha > 0$ and scale $\beta > 0$ has density

\begin{equation*}
f(x) = \frac{x^{\alpha - 1}e^{-x/\beta}}{\beta^\alpha\,\Gamma(\alpha)}, \qquad x > 0,
\tag{6.5}
\end{equation*}

with mean $\alpha\beta$, variance $\alpha\beta^2$, coefficient of variation $1/\sqrt\alpha$ and skewness $2/\sqrt\alpha$. For $\alpha = 1$ it is the exponential distribution; for integer $\alpha$ it is the distribution of a sum of $\alpha$ independent exponentials (Chapter 4), and as $\alpha$ grows it approaches the normal (Figure 6.2). Its flexibility and positivity make it the standard model for wet-day rainfall amounts and for monthly or seasonal rainfall totals; the Standardized Precipitation Index transforms gamma-distributed rainfall totals to standard normal values (Chapter 8). It also describes the impulse response of a cascade of linear reservoirs.

```{code-cell} python
:tags: [hide-input]
x = np.linspace(0.001, 6, 500)
fig, ax = plt.subplots(figsize=(6.4, 3.5))
for a_, col in zip([0.7, 1.0, 2.0, 5.0, 15.0], [ink2, c3, c1, c2, ink]):
    ax.plot(x, stats.gamma.pdf(x, a_, scale=1 / a_), color=col, lw=2.0, label=f"α = {a_:g}, γ = {2 / np.sqrt(a_):.2f}")
ax.set_ylim(0, 1.8); ax.set_xlabel("x / mean"); ax.set_ylabel("Density"); ax.legend(loc="upper right", fontsize=9)
plt.show()
```

*Figure 6.2. Gamma densities with mean 1 and different shape parameters α, with their skewness γ. Small α gives a J-shaped, highly skewed distribution; large α approaches the normal.*

### Pearson type III and frequency factors

Adding a location parameter $x_0$ gives the **Pearson type III** (P3) distribution: $X - x_0$ is gamma-distributed. Allowing the sign of the scale to be negative extends it to negative skewness, with an upper bound. The P3 is conveniently parameterized by its mean, standard deviation and skewness, and its quantiles are then written as

\begin{equation*}
x_T = \mu + K_T\,\sigma ,
\tag{6.6}
\end{equation*}

where the **frequency factor** $K_T$ depends only on the return period $T$ and the skewness $\gamma$. For $\gamma = 0$ it is the standard normal quantile $z$; for other skewness values a good approximation is the Wilson–Hilferty transformation,

\begin{equation*}
K_T \approx \frac{2}{\gamma}\left[\left(1 + \frac{\gamma z}{6} - \frac{\gamma^2}{36}\right)^3 - 1\right], \qquad z = \Phi^{-1}\!\left(1 - \frac{1}{T}\right),
\tag{6.7}
\end{equation*}

accurate for $|\gamma| \lesssim 2$. Software now computes $K_T$ exactly (Appendix A gives a table), but the frequency-factor form remains a useful way to see how a design value depends on the mean, spread and skewness of the data.

### Log-Pearson type III

If $\log X$ follows a Pearson type III distribution, $X$ is **log-Pearson type III** (LP3). Equation (6.6) is then applied to the logarithms, $\log x_T = \bar y + K_T s_y$, with $K_T$ computed from the skewness of the logarithms. The LP3 has been the standard distribution for flood frequency analysis in the United States since the 1960s, with methods for weighting the sample skew with a regional value and for treating historical floods and low outliers.

## 6.4 Extreme value distributions

### Limiting distributions of maxima

Equation (3.10) gave the distribution of the maximum of $n$ independent values, $[F(x)]^n$. As $n$ grows, this distribution drifts to larger values, but after suitable rescaling it converges to one of only three limiting forms, whatever $F$ is (Fisher and Tippett, 1928; Gnedenko, 1943). The three are unified in the **generalized extreme value** (GEV) distribution:

\begin{equation*}
F(x) = \exp\left\{-\left[1 + \xi\,\frac{x - \mu}{\sigma}\right]^{-1/\xi}\right\}, \qquad 1 + \xi\,\frac{x - \mu}{\sigma} > 0,
\tag{6.8}
\end{equation*}

with location $\mu$, scale $\sigma > 0$ and shape $\xi$:

- $\xi = 0$ (the limit of Equation 6.8): the **Gumbel** or extreme value type I distribution, $F(x) = \exp[-e^{-(x - \mu)/\sigma}]$, arising when the parent has an exponential-like tail (normal, gamma, lognormal in practice);
- $\xi > 0$: the **Fréchet** or type II, with a heavy, power-law upper tail and no upper bound;
- $\xi < 0$: the **reversed Weibull** or type III, with an upper bound at $\mu - \sigma/\xi$.

Sign conventions differ: many hydrological texts write the shape as $k = -\xi$, and SciPy's `genextreme` uses $c = -\xi$. Always check which is meant.

Annual maxima of rainfall and floods often have $\xi$ between 0 and 0.2: heavier tails than the Gumbel distribution assumes. Since a small positive $\xi$ raises high quantiles substantially (Figure 6.3), and since $\xi$ is hard to estimate from short records, regional estimates are often used (Chapter 8). The GEV is the recommended distribution for annual maxima in many countries.

### The Gumbel distribution

Because of its simplicity, the Gumbel distribution is still widely used. Its mean is $\mu + 0.5772\,\sigma$, its standard deviation $\pi\sigma/\sqrt6$ and its skewness fixed at 1.14. Its quantile function is

\begin{equation*}
x_T = \mu - \sigma\ln\left[-\ln\left(1 - \frac{1}{T}\right)\right] = \mu + \sigma\,y_T ,
\tag{6.9}
\end{equation*}

where $y_T$ is the **Gumbel reduced variate**. Plotted against $y_T$, Gumbel quantiles fall on a straight line, which is the basis of Gumbel probability paper (Chapter 8).

### Minima

Low flows and droughts are minima. Their limiting distributions are the mirror images of those for maxima; the most used in hydrology is the **Weibull** distribution, $F(x) = 1 - \exp[-((x - x_0)/\beta)^\kappa]$, with a lower bound $x_0$, often zero (Chapter 8).

```{code-cell} python
:tags: [hide-input]
mu_, sg_ = 100.0, 30.0
T = np.logspace(np.log10(1.05), 3, 200); yT = -np.log(-np.log(1 - 1 / T))
x = np.linspace(20, 400, 600)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 3.6))
for xi, col, lab in [(-0.15, c3, "ξ = −0.15 (bounded)"), (0.0, c1, "ξ = 0 (Gumbel)"), (0.15, c2, "ξ = 0.15 (heavy)")]:
    G = stats.genextreme(-xi, loc=mu_, scale=sg_)                     # SciPy shape c = −ξ
    a1.plot(x, G.pdf(x), color=col, lw=2.0, label=lab)
    a2.plot(yT, G.ppf(1 - 1 / T), color=col, lw=2.2)
a1.set_xlabel("x"); a1.set_ylabel("Density"); a1.set_title("(a) Densities"); a1.legend(loc="center right", fontsize=8.5)
a2.set_xlabel("Gumbel reduced variate $y_T$"); a2.set_ylabel("$x_T$")
a2.text(0.03, 0.95, "(b) Return levels", transform=a2.transAxes, fontsize=11, va="top")
top = a2.twiny(); top.set_xlim(a2.get_xlim())
Tm = np.array([2, 10, 100, 1000]); top.set_xticks(-np.log(-np.log(1 - 1 / Tm))); top.set_xticklabels([str(t) for t in Tm])
top.set_xlabel("T (years)", fontsize=9.5); top.grid(False)
plt.show()
G0, G1 = stats.genextreme(0.0, loc=mu_, scale=sg_), stats.genextreme(-0.15, loc=mu_, scale=sg_)
glue("q100_g", float(G0.ppf(0.99)), display=False); glue("q100_h", float(G1.ppf(0.99)), display=False)
```

*Figure 6.3. GEV distributions with location 100, scale 30 and three shape parameters. (a) The densities differ little in their center. (b) Plotted against the Gumbel reduced variate, the Gumbel quantiles form a straight line, the heavy-tailed case curves upward and the bounded case downward. The 100-year value is {glue:text}`q100_g:.0f` for the Gumbel and {glue:text}`q100_h:.0f` for ξ = 0.15.*

## 6.5 Bounded variables and the beta distribution

Some variables are confined to a finite interval: the fraction of a catchment that is saturated, relative soil moisture, the proportion of days that are wet, a runoff coefficient. The **beta distribution** on $[0, 1]$,

\begin{equation*}
f(x) = \frac{x^{a - 1}(1 - x)^{b - 1}}{B(a, b)}, \qquad 0 \le x \le 1,
\tag{6.10}
\end{equation*}

with mean $a/(a + b)$, takes U-shaped, J-shaped, symmetric or skewed forms depending on its two shape parameters, and is rescaled to any interval $[c, d]$ by a linear transformation. It is also used to express subjective probability about a proportion in Bayesian analysis.

## 6.6 Mixed distributions for intermittent variables

### Mass at zero

Daily rainfall is zero on most days. Its distribution is **mixed**: a probability mass $p_0$ at zero plus a continuous distribution $G$ for the positive values,

\begin{equation*}
F(x) = p_0 + (1 - p_0)\,G(x), \qquad x \ge 0 .
\tag{6.11}
\end{equation*}

By the total probability theorem, $P(X > x) = (1 - p_0)[1 - G(x)]$. The same form fits ephemeral streamflow and the annual maximum flow of rivers that dry out in some years. Fitting a continuous distribution to the full series with zeros, or to logarithms after replacing zeros by a small number, gives wrong results.

### Mixtures of populations

When floods are generated by distinct mechanisms, such as typhoons and summer fronts, or snowmelt and rain, the annual maxima may come from a **mixture** $F = wF_1 + (1 - w)F_2$, or, if each year's maximum is the larger of the maxima from each mechanism, from the product $F = F_1F_2$ (Equation 3.10). Either can produce a sharp bend in a probability plot (Figure 6.4), and fitting a single distribution then misrepresents the upper tail.

```{code-cell} python
:tags: [hide-input]
p0 = 0.65; Gw = stats.gamma(0.8, scale=12.0)
x = np.linspace(0, 60, 500)
F1 = stats.gumbel_r(80, 15); F2 = stats.gumbel_r(70, 45)                   # frontal and typhoon annual maxima
T = np.logspace(np.log10(1.02), 3, 300); q = 1 - 1 / T
xs = np.linspace(20, 600, 4000); Fmix = F1.cdf(xs) * F2.cdf(xs)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 3.5))
a1.plot(np.r_[0, 0, x], np.r_[0, p0, p0 + (1 - p0) * Gw.cdf(x)], color=c1, lw=2.2)
a1.set_xlabel("Daily rainfall (mm)"); a1.set_ylabel("F(x)"); a1.set_title("(a) Mass at zero"); a1.set_ylim(0, 1.02)
a1.text(4, p0 / 2, f"$p_0$ = {p0}", color=ink2, fontsize=9.5)
yT = -np.log(-np.log(q))
a2.plot(yT, np.interp(q, Fmix, xs), color=ink, lw=2.2, label="Max of both")
a2.plot(yT, F1.ppf(q), color=c3, lw=1.5, ls="--", label="Fronts only")
a2.plot(yT, F2.ppf(q), color=c2, lw=1.5, ls="--", label="Typhoons only")
a2.set_xlabel("Gumbel reduced variate"); a2.set_ylabel("Annual max. (mm)"); a2.set_title("(b) Two populations")
a2.legend(loc="upper left", fontsize=8.5)
plt.show()
```

*Figure 6.4. (a) Cumulative distribution of daily rainfall with a probability mass of 0.65 at zero and gamma-distributed wet-day amounts. (b) Annual maximum rainfall when each year's maximum is the larger of a frontal-storm maximum and a typhoon maximum (both Gumbel). On Gumbel axes the combined distribution follows the fronts at short return periods and the typhoons at long ones.*

## 6.7 Summary of distributions

```{code-cell} python
:tags: [remove-input]
Markdown("| Distribution | Range | Parameters | Skewness | Typical variables | SciPy |\n|---|---|---|---|---|---|\n"
         "| Uniform | $[a, b]$ | $a, b$ | 0 | random numbers, simulation | `uniform` |\n"
         "| Exponential | $x \\ge 0$ | $\\beta$ | 2 | interarrival times, excesses | `expon` |\n"
         "| Normal | all $x$ | $\\mu, \\sigma$ | 0 | annual totals, errors | `norm` |\n"
         "| Lognormal | $x > 0$ | $\\mu_y, \\sigma_y$ | $3C_v + C_v^3$ | floods, conductivity | `lognorm` |\n"
         "| Gamma | $x > 0$ | $\\alpha, \\beta$ | $2/\\sqrt\\alpha$ | rainfall amounts, totals | `gamma` |\n"
         "| Pearson III | shifted | $\\mu, \\sigma, \\gamma$ | $\\gamma$ | floods | `pearson3` |\n"
         "| Gumbel | all $x$ | $\\mu, \\sigma$ | 1.14 | annual maxima | `gumbel_r` |\n"
         "| GEV | depends on $\\xi$ | $\\mu, \\sigma, \\xi$ | varies | annual maxima | `genextreme` ($c = -\\xi$) |\n"
         "| Weibull | $x \\ge x_0$ | $x_0, \\beta, \\kappa$ | varies | low flows, minima | `weibull_min` |\n"
         "| Beta | $[0, 1]$ | $a, b$ | varies | fractions, proportions | `beta` |")
```

*Table 6.1. Continuous distributions used in hydrology, with the names of their implementations in `scipy.stats`.*

## 6.8 Worked examples

**Example 6.1 — Normal annual rainfall.** Annual rainfall at a station is normal with mean 1250 mm and standard deviation 210 mm. Find the probability of a year with less than 900 mm and the rainfall exceeded in 90% of years.

```{code-cell} python
N = stats.norm(1250, 210)
print(f"P(X < 900) = {N.cdf(900):.4f}; 0.1-quantile = {N.ppf(0.10):.0f} mm")
```

**Example 6.2 — Lognormal from moments.** Annual maximum discharge has a mean of 850 m³ s⁻¹ and a standard deviation of 380 m³ s⁻¹. Assuming a lognormal distribution, find its parameters and the 100-year discharge.

```{code-cell} python
m, s = 850.0, 380.0
sy = np.sqrt(np.log(1 + (s / m) ** 2)); my = np.log(m) - sy ** 2 / 2
print(f"mu_y = {my:.3f}, sigma_y = {sy:.3f}; Q100 = {np.exp(my + sy * stats.norm.ppf(0.99)):.0f} m3/s")
```

**Example 6.3 — Gumbel from moments.** For the same mean and standard deviation, find the Gumbel parameters and the 100-year discharge. Compare with the lognormal result.

```{code-cell} python
sg = s * np.sqrt(6) / np.pi; mu_g = m - 0.5772 * sg
yT = -np.log(-np.log(1 - 1 / 100))
print(f"mu = {mu_g:.0f}, sigma = {sg:.0f}; Q100 = {mu_g + sg * yT:.0f} m3/s")
```

**Example 6.4 — Frequency factors.** Compare the Wilson–Hilferty approximation (6.7) with the exact Pearson type III frequency factor for $T$ = 100 years and skewness 0.5 and 1.5.

```{code-cell} python
z = stats.norm.ppf(0.99)
for g in (0.5, 1.5):
    K_wh = 2 / g * ((1 + g * z / 6 - g ** 2 / 36) ** 3 - 1)
    print(f"skew {g}: K (Wilson-Hilferty) = {K_wh:.3f}, K (exact) = {stats.pearson3.ppf(0.99, g):.3f}")
```

**Example 6.5 — Intermittent daily rainfall.** Daily rainfall is zero with probability 0.62 and otherwise gamma-distributed with shape 0.75 and mean 14 mm. Find the probability that a day has more than 50 mm, and the expected number of such days in a year, assuming the distribution holds all year.

```{code-cell} python
p0, a_, mean_w = 0.62, 0.75, 14.0
p50 = (1 - p0) * stats.gamma.sf(50, a_, scale=mean_w / a_)
print(f"P(X > 50) = {p50:.4f}; expected days per year = {365 * p50:.1f}")
```

## 6.9 Common mistakes

- **Fitting a normal distribution to positive, skewed data.** It can predict negative values and underestimates high quantiles.
- **Mixing up GEV sign conventions.** $\xi$, $k$ and SciPy's $c$ differ in sign.
- **Assuming the Gumbel by habit.** Its fixed skewness of 1.14 may underestimate rare floods when the true tail is heavier.
- **Taking logarithms of data with zeros.** Use a mixed distribution (Equation 6.11).
- **Applying LP3 frequency factors to the original data** instead of to the logarithms.

## 6.10 Exercises

```{code-cell} python
:tags: [remove-cell]
glue("m1", float(stats.norm.sf(1600, 1250, 210)), display=False)
glue("m2", float(np.exp(5.2 + 0.4 ** 2 / 2)), display=False)
glue("m3", float(1 - np.exp(-np.exp(-(300 - 180) / 45))), display=False)
glue("m4a", float(4 * 5.0), display=False); glue("m4b", float(2 / np.sqrt(4)), display=False)
yv = -np.log(-np.log(1 - 1 / 50)); glue("m5", float(250 + 1.0 * 80 * (2 / 1.0 * ((1 + 1.0 * stats.norm.ppf(0.98) / 6 - 1 / 36) ** 3 - 1))), display=False)
```

1. For the rainfall of Example 6.1, what is the probability of a year with more than 1600 mm?
2. $\ln X$ is normal with mean 5.2 and standard deviation 0.4. Find the mean of $X$.
3. Annual maximum rainfall is Gumbel with $\mu$ = 180 mm and $\sigma$ = 45 mm. What is the annual probability of exceeding 300 mm?
4. A gamma distribution has $\alpha$ = 4 and $\beta$ = 5 mm. Find its mean and skewness.
5. A Pearson type III flood distribution has mean 250 m³ s⁻¹, standard deviation 80 m³ s⁻¹ and skewness 1.0. Estimate the 50-year flood with Equation (6.7).

```{admonition} Answers
:class: dropdown
1. {glue:text}`m1:.4f`.
2. {glue:text}`m2:.0f`.
3. {glue:text}`m3:.4f`.
4. Mean {glue:text}`m4a:.0f` mm, skewness {glue:text}`m4b:.1f`.
5. About {glue:text}`m5:.0f` m³ s⁻¹.
```

## 6.11 Further reading

- Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme Values*. Springer.
- Haan, C. T. (2002). *Statistical Methods in Hydrology* (2nd ed.). Iowa State Press.
- Maity, R. (2022). *Statistical Methods in Hydrology and Hydroclimatology* (2nd ed.). Springer.
- Rao, A. R., and Hamed, K. H. (2000). *Flood Frequency Analysis*. CRC Press.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
