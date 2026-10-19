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

# 4. Moments and descriptive statistics

```{admonition} Core ideas
:class: tip
1. A distribution or a sample is summarized by measures of **location** (mean, median), **spread** (standard deviation, coefficient of variation, interquartile range), **asymmetry** (skewness) and **tail weight** (kurtosis).
2. These are estimates of the **moments** of the distribution: expectations of powers of the variable. Expectation is linear, which makes moments of sums and linear transformations easy to compute.
3. **Covariance** and **correlation** measure linear association between two variables. Correlation is not the same as dependence, and it is sensitive to outliers.
4. The variance of a sum depends on the covariances of its terms: correlated inputs add more variability than independent ones.
5. **Chebyshev's inequality** bounds tail probabilities from the variance alone; the **law of large numbers** says sample means converge to the expectation; the **central limit theorem** says they become normally distributed.
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

## 4.1 Location, spread, asymmetry and tails

Given a sample $x_1, \dots, x_n$, the first step is to describe it with a few numbers.

**Location.** The **sample mean** $\bar x = \frac{1}{n}\sum x_i$ is the center of mass of the data. The **median** is the middle value of the sorted sample (the average of the two middle values when $n$ is even); half the data lie on each side. The **mode** is the most frequent value or, for continuous data, the peak of the histogram. For symmetric data the three coincide; for right-skewed data, typical of hydrology, mode < median < mean, because a few large values pull the mean up.

**Spread.** The **sample variance** and **standard deviation** are

\begin{equation*}
s^2 = \frac{1}{n - 1}\sum_{i=1}^{n} (x_i - \bar x)^2, \qquad s = \sqrt{s^2} .
\tag{4.1}
\end{equation*}

The divisor $n - 1$ rather than $n$ makes $s^2$ an unbiased estimate of the population variance (Chapter 7). The **coefficient of variation** $C_v = s/\bar x$ measures spread relative to the mean and is dimensionless, which allows records of different sizes to be compared. Robust alternatives, less affected by extreme values, are the **interquartile range** (IQR), the difference between the 0.75 and 0.25 quantiles, and the **range**, the difference between the largest and smallest values.

**Asymmetry.** The **sample skewness coefficient** is

\begin{equation*}
g = \frac{n}{(n - 1)(n - 2)}\,\frac{\sum_{i=1}^{n} (x_i - \bar x)^3}{s^3} .
\tag{4.2}
\end{equation*}

Positive values indicate a long right tail. The factor in front reduces the bias of the estimate for small samples, but sample skewness is still very uncertain unless $n$ is large: with 30 values its standard error is about 0.4 even for normal data.

**Tails.** The **kurtosis**, the standardized fourth moment, measures how heavy the tails are relative to the normal distribution, whose kurtosis is 3. The **excess kurtosis** subtracts 3. It is even more uncertain than the skewness and is used mainly for large samples.

Figure 4.1 compares three variables derived from the synthetic daily rainfall of Chapter 1. Aggregation reduces both the relative spread and the skewness, while taking maxima produces a moderately skewed variable of its own.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(7)
ny = 40; doy = np.tile(np.arange(365), ny); nd = len(doy)
season = np.exp(-0.5 * ((doy - 200) / 35) ** 2)
p01 = 0.15 + 0.35 * season; p11 = 0.45 + 0.35 * season; mean_amt = 6.0 + 16.0 * season
wet = np.zeros(nd, bool)
for t in range(1, nd):
    wet[t] = rng.random() < (p11[t] if wet[t - 1] else p01[t])
P = np.where(wet, rng.gamma(0.75, mean_amt / 0.75), 0.0)
Pyr = P.reshape(ny, 365)
sets = [("Wet-day rainfall (mm)", P[P > 0], c1), ("Annual maximum daily (mm)", Pyr.max(1), c2), ("Annual total (mm)", Pyr.sum(1), c3)]
fig, axs = plt.subplots(1, 3, figsize=(6.4, 3.2))
rows = []
for ax, (lab, x, col) in zip(axs, sets):
    ax.hist(x, bins=25, color=col, alpha=0.8, edgecolor=bg)
    ax.axvline(np.mean(x), color=ink, lw=1.6, label="mean"); ax.axvline(np.median(x), color=ink, lw=1.6, ls="--", label="median")
    ax.set_xlabel(lab, fontsize=9.5); ax.set_yticks([])
    rows.append((lab, len(x), np.mean(x), np.median(x), np.std(x, ddof=1), np.std(x, ddof=1) / np.mean(x), stats.skew(x, bias=False)))
axs[0].legend(loc="upper right", fontsize=8.5)
plt.show()
```

*Figure 4.1. Histograms of three variables derived from 40 years of synthetic daily rainfall, with their means (solid) and medians (dashed). Statistics are listed in Table 4.1.*

```{code-cell} python
:tags: [remove-input]
Markdown("| Variable | $n$ | Mean | Median | $s$ | $C_v$ | $g$ |\n|---|---|---|---|---|---|---|\n" +
         "".join(f"| {r[0]} | {r[1]} | {r[2]:.1f} | {r[3]:.1f} | {r[4]:.1f} | {r[5]:.2f} | {r[6]:.2f} |\n" for r in rows))
```

*Table 4.1. Descriptive statistics of the variables in Figure 4.1.*

## 4.2 Expectation and moments

### Expectation

The population counterparts of these statistics are defined through the **expectation**, the probability-weighted average of a variable or of a function of it:

\begin{equation*}
E[g(X)] = \sum_x g(x)\,p_X(x) \quad \text{or} \quad E[g(X)] = \int_{-\infty}^{\infty} g(x)\,f_X(x)\,dx .
\tag{4.3}
\end{equation*}

The **mean** is $\mu = E[X]$. Expectation is **linear**: for constants $a$ and $b$ and any random variables $X$ and $Y$, whether independent or not,

\begin{equation*}
E[aX + bY] = a\,E[X] + b\,E[Y] .
\tag{4.4}
\end{equation*}

It is not, in general, true that $E[g(X)] = g(E[X])$ for nonlinear $g$. For example, the mean of a discharge computed from a nonlinear rating curve is not the rating curve evaluated at the mean stage.

### Moments

The $r$-th **moment about the origin** is $\mu_r' = E[X^r]$ and the $r$-th **central moment** is $\mu_r = E[(X - \mu)^r]$. The second central moment is the **variance**,

\begin{equation*}
\sigma^2 = \operatorname{Var}(X) = E[(X - \mu)^2] = E[X^2] - \mu^2 ,
\tag{4.5}
\end{equation*}

the third and fourth, standardized, give the **skewness** $\gamma = \mu_3/\sigma^3$ and **kurtosis** $\kappa = \mu_4/\sigma^4$. For a linear transformation $Y = a + bX$, $E[Y] = a + b\mu$, $\operatorname{Var}(Y) = b^2\sigma^2$, and the skewness keeps its magnitude (changing sign if $b < 0$). Converting rainfall from inches to millimeters, or temperature from °F to °C, therefore leaves the skewness unchanged.

Moments do not always exist: for very heavy-tailed distributions the integral in Equation (4.3) diverges. This happens for some extreme-value distributions fitted to floods and rainfall (Chapter 6), which is one reason estimation methods that do not rely on high moments, such as L-moments, are preferred (Chapter 7).

## 4.3 Generating functions

The **moment generating function** (mgf) packs all moments into a single function:

\begin{equation*}
M_X(t) = E\!\left[e^{tX}\right], \qquad \mu_r' = \left.\frac{d^r M_X}{dt^r}\right|_{t = 0} ,
\tag{4.6}
\end{equation*}

as follows from expanding $e^{tX}$ in a power series. Two properties make it useful. First, the mgf of a sum of independent variables is the product of their mgfs, so the distribution of a sum can often be identified from its mgf. Second, two distributions with the same mgf (where it exists) are identical. For the exponential distribution with mean $\beta$, for example, $M(t) = (1 - \beta t)^{-1}$ for $t < 1/\beta$, so a sum of $k$ independent exponentials has $M(t) = (1 - \beta t)^{-k}$, the mgf of a gamma distribution (Chapter 6).

When the mgf does not exist, the **characteristic function** $\varphi_X(t) = E[e^{itX}]$, which always exists, plays the same role. These tools are mostly used in derivations; numerical work rarely needs them.

## 4.4 Covariance and correlation

The **covariance** of two variables measures how they vary together:

\begin{equation*}
\operatorname{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)] = E[XY] - \mu_X\mu_Y ,
\tag{4.7}
\end{equation*}

and the **correlation coefficient** standardizes it to the range −1 to 1:

\begin{equation*}
\rho_{XY} = \frac{\operatorname{Cov}(X, Y)}{\sigma_X\sigma_Y} .
\tag{4.8}
\end{equation*}

The sample version $r$ replaces expectations by sample averages. Independent variables have zero covariance, but zero correlation does not imply independence: $\rho$ measures only *linear* association. Figure 4.2 shows why a single number can mislead: a strong nonlinear relation can have a correlation near zero, and a single outlier can create or destroy an apparent correlation. Rank-based measures (Spearman's and Kendall's, Chapters 10 and 15) are more robust.

Covariance determines the variance of sums, which is how variability accumulates in water systems:

\begin{equation*}
\operatorname{Var}(aX + bY) = a^2\sigma_X^2 + b^2\sigma_Y^2 + 2ab\operatorname{Cov}(X, Y) .
\tag{4.9}
\end{equation*}

If two tributaries have positively correlated flows, the main river varies more than if they were independent (Example 4.2). For $n$ independent variables with equal variance, the variance of their mean is $\sigma^2/n$, the basis of the standard error used in Chapter 1.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(4)
n = 60
fig, axs = plt.subplots(1, 3, figsize=(6.4, 2.6), sharey=False)
x = rng.normal(0, 1, n); y = 0.8 * x + 0.6 * rng.normal(0, 1, n)
cases = [("(a) Linear", x, y)]
x2 = np.linspace(-2, 2, n); cases.append(("(b) Nonlinear", x2, x2 ** 2 + 0.2 * rng.normal(0, 1, n)))
x3 = rng.normal(0, 1, n); y3 = rng.normal(0, 1, n); x3[0], y3[0] = 6, 6; cases.append(("(c) One outlier", x3, y3))
for ax, (t, a, b) in zip(axs, cases):
    ax.plot(a, b, "o", color=c1, ms=3.5)
    if t.startswith("(c)"): ax.plot(a[0], b[0], "o", color=c2, ms=6)
    ax.set_title(f"{t}\nr = {np.corrcoef(a, b)[0, 1]:.2f}", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
plt.show()
glue("r_out_wo", float(np.corrcoef(x3[1:], y3[1:])[0, 1]), display=False)
```

*Figure 4.2. Sample correlation coefficients of three synthetic data sets. (a) A linear relation. (b) A strong nonlinear relation with almost zero correlation. (c) Independent data with one outlier (orange), which alone creates the correlation; without it, $r$ = {glue:text}`r_out_wo:.2f`.*

## 4.5 Chebyshev's inequality, the law of large numbers and the central limit theorem

**Chebyshev's inequality** bounds the probability of large deviations using only the mean and variance, whatever the distribution:

\begin{equation*}
P(|X - \mu| \ge k\sigma) \le \frac{1}{k^2} .
\tag{4.10}
\end{equation*}

The bound is often loose (for a normal variable, $P(|X - \mu| \ge 2\sigma)$ is 0.046, against the bound 0.25), but it guarantees that values far from the mean in units of standard deviations are rare.

Applied to the sample mean, whose variance is $\sigma^2/n$, Chebyshev's inequality gives the **(weak) law of large numbers**: for any tolerance $\epsilon > 0$, $P(|\bar X - \mu| \ge \epsilon) \le \sigma^2/(n\epsilon^2) \to 0$ as $n \to \infty$. Sample means converge to the expectation, which is why the relative-frequency interpretation of probability works (Figure 2.1).

The **central limit theorem** goes further: the standardized mean of $n$ independent, identically distributed variables with finite variance approaches a standard normal distribution as $n$ grows,

\begin{equation*}
\frac{\bar X - \mu}{\sigma/\sqrt n} \;\longrightarrow\; N(0, 1) ,
\tag{4.11}
\end{equation*}

whatever the distribution of the individual variables. Figure 4.3 shows how quickly this happens for a strongly skewed variable. The theorem explains why annual totals of rainfall and runoff are often nearly normal, even though daily values are not, and it justifies the normal-based confidence intervals of Chapter 10. Its conditions matter: for strongly dependent daily values, the effective number of independent terms is much smaller than the number of days, and convergence is slower.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(11)
fig, axs = plt.subplots(1, 3, figsize=(6.4, 2.8), sharey=True)
z = np.linspace(-4, 4, 200)
for ax, n, col in zip(axs, [1, 5, 30], [c2, c1, c3]):
    m = rng.exponential(1.0, size=(20000, n)).mean(1)
    ax.hist((m - 1) / (1 / np.sqrt(n)), bins=np.linspace(-4, 6, 50), density=True, color=col, alpha=0.75, edgecolor=bg)
    ax.plot(z, stats.norm.pdf(z), color=ink, lw=1.5)
    ax.set_title(f"n = {n}"); ax.set_xlabel("Standardized mean")
axs[0].set_ylabel("Density")
plt.show()
```

*Figure 4.3. The central limit theorem at work: histograms of the standardized mean of $n$ independent exponential variables, compared with the standard normal density (black). Even for this strongly skewed variable, the mean of 30 values is close to normal.*

## 4.6 Worked examples

**Example 4.1 — Describing a short record.** The annual maximum daily rainfalls (mm) of a station over 15 years are listed below. Compute the mean, median, standard deviation, coefficient of variation and skewness.

```{code-cell} python
x = np.array([88, 132, 74, 96, 210, 105, 81, 119, 93, 147, 70, 101, 163, 90, 112.0])
print(f"mean = {x.mean():.1f}, median = {np.median(x):.1f}, s = {x.std(ddof=1):.1f}, "
      f"Cv = {x.std(ddof=1) / x.mean():.2f}, g = {stats.skew(x, bias=False):.2f}")
```

The largest value, 210 mm, contributes most of the skewness: without it, $g$ falls to {glue:text}`g_wo:.2f`.

```{code-cell} python
:tags: [remove-cell]
glue("g_wo", float(stats.skew(np.delete(x, np.argmax(x)), bias=False)), display=False)
```

**Example 4.2 — Two tributaries.** The annual flows of two tributaries have means of 18 and 12 m³ s⁻¹ and standard deviations of 6 and 5 m³ s⁻¹. Find the mean and standard deviation of their sum if the correlation between them is 0, and if it is 0.7.

```{code-cell} python
m1, m2, s1, s2 = 18, 12, 6, 5
for rho in (0.0, 0.7):
    print(f"rho = {rho}: mean = {m1 + m2} m3/s, sd = {np.sqrt(s1**2 + s2**2 + 2 * rho * s1 * s2):.1f} m3/s")
```

**Example 4.3 — Chebyshev against the normal.** Annual runoff has mean 650 mm and standard deviation 160 mm. Bound the probability that a year departs from the mean by more than 400 mm, and compare with the normal distribution.

```{code-cell} python
k = 400 / 160
print(f"Chebyshev: P <= {1 / k**2:.3f}; normal: P = {2 * stats.norm.sf(k):.4f}")
```

**Example 4.4 — Moments from the mgf.** The time between storms is exponential with mean $\beta$ = 5 days, so its mgf is $M(t) = (1 - \beta t)^{-1}$. Differentiating, $M'(0) = \beta$ and $M''(0) = 2\beta^2$, so the variance is $2\beta^2 - \beta^2 = \beta^2$. Check the derivatives numerically.

```{code-cell} python
beta, h = 5.0, 1e-4
M = lambda t: 1 / (1 - beta * t)
m1 = (M(h) - M(-h)) / (2 * h); m2 = (M(h) - 2 * M(0) + M(-h)) / h ** 2
print(f"E[X] = {m1:.3f} d, E[X^2] = {m2:.2f} d^2, Var = {m2 - m1**2:.2f} d^2 (beta^2 = {beta**2:.0f})")
```

## 4.7 Common mistakes

- **Using the population formula with divisor $n$** for the sample variance of short records. Use $n - 1$.
- **Trusting the skewness of a short record.** With 20–30 values it is highly uncertain; Chapter 8 discusses regional skew.
- **Reading zero correlation as independence.** It excludes only linear association.
- **Adding variances of correlated quantities.** Include the covariance term (Equation 4.9).
- **Assuming $E[g(X)] = g(E[X])$** for a nonlinear function such as a rating curve or a runoff coefficient.

## 4.8 Exercises

```{code-cell} python
:tags: [remove-cell]
xe = np.array([420, 515, 380, 610, 455, 700, 390, 480.0])
glue("h1a", float(xe.mean()), display=False); glue("h1b", float(xe.std(ddof=1)), display=False); glue("h1c", float(xe.std(ddof=1) / xe.mean()), display=False)
glue("h2", float(np.sqrt(4**2 + 3**2 - 2 * 0.5 * 4 * 3)), display=False)
glue("h3", float(1 / 3 ** 2), display=False)
glue("h4", float(3.2 / np.sqrt(16)), display=False)
glue("h5", float(1.5 * 25.4), display=False); glue("h5b", float(1.5 / 4.2), display=False)
```

1. Monthly rainfalls (mm) for eight Julys are 420, 515, 380, 610, 455, 700, 390 and 480. Find the mean, standard deviation and coefficient of variation.
2. $X$ and $Y$ have standard deviations 4 and 3 and correlation 0.5. Find the standard deviation of $X - Y$.
3. What upper bound does Chebyshev's inequality give for the probability of a value more than three standard deviations from the mean?
4. Daily temperature anomalies have a standard deviation of 3.2 °C. What is the standard error of the mean of 16 independent values?
5. An old record gives monthly rainfall in inches, with mean 4.2 in and standard deviation 1.5 in. Convert the standard deviation to millimeters (1 in = 25.4 mm). What is the coefficient of variation in each unit?

```{admonition} Answers
:class: dropdown
1. Mean {glue:text}`h1a:.1f` mm, $s$ = {glue:text}`h1b:.1f` mm, $C_v$ = {glue:text}`h1c:.2f`.
2. {glue:text}`h2:.2f`.
3. {glue:text}`h3:.3f`.
4. {glue:text}`h4:.2f` °C.
5. {glue:text}`h5:.1f` mm; $C_v$ = {glue:text}`h5b:.2f` in both units, since it is dimensionless.
```

## 4.9 Further reading

- Haan, C. T. (2002). *Statistical Methods in Hydrology* (2nd ed.). Iowa State Press.
- Helsel, D. R., Hirsch, R. M., Ryberg, K. R., Archfield, S. A., and Gilroy, E. J. (2020). *Statistical Methods in Water Resources*. U.S. Geological Survey Techniques and Methods, book 4, chapter A3.
- Maity, R. (2022). *Statistical Methods in Hydrology and Hydroclimatology* (2nd ed.). Springer.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
