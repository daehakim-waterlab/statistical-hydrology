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

# 7. Estimating parameters

```{admonition} Core ideas
:class: tip
1. An **estimator** is a rule that turns a sample into a parameter value. Its quality is judged by its **bias**, its **variance** and their combination, the **mean squared error**.
2. The **method of moments** equates sample moments to theoretical ones. It is simple, but higher sample moments, especially the skewness, are biased and unstable for short records.
3. **Maximum likelihood** chooses the parameters that make the observed sample most probable. It is efficient for large samples and gives standard errors, but it can be unstable for small samples and heavy tails.
4. **L-moments**, linear combinations of ordered data, describe location, scale, skewness and kurtosis robustly. They are the standard method for fitting extreme-value distributions and for regional analysis.
5. Every estimate carries **sampling uncertainty**. The **bootstrap**, resampling the data, estimates it when no formula is available.
```

```{code-cell} python
:tags: [remove-cell]
# Plot style shared by all figures in this chapter.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, special
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

## 7.1 What makes a good estimator

A distribution chosen for a variable has unknown **parameters** $\theta$ (a mean, a scale, a shape), and a sample $x_1, \dots, x_n$ is used to estimate them. An **estimator** $\hat\theta$ is a function of the sample. Because the sample is random, so is the estimator: it has a **sampling distribution**, and its quality is judged by the properties of that distribution.

- **Bias**: $b(\hat\theta) = E[\hat\theta] - \theta$. An unbiased estimator is right on average.
- **Variance**: $\operatorname{Var}(\hat\theta)$, the spread of the estimates from sample to sample. Its square root is the **standard error**.
- **Mean squared error**: the overall accuracy,

\begin{equation*}
\operatorname{MSE}(\hat\theta) = E\left[(\hat\theta - \theta)^2\right] = \operatorname{Var}(\hat\theta) + b(\hat\theta)^2 .
\tag{7.1}
\end{equation*}

A slightly biased estimator with a small variance can be more accurate than an unbiased one with a large variance.

- **Consistency**: $\hat\theta$ converges to $\theta$ as $n \to \infty$.
- **Efficiency**: among unbiased estimators, the one with the smallest variance is the most efficient. No unbiased estimator can have a variance below the Cramér–Rao bound, $1/[n I(\theta)]$, where $I(\theta)$ is the Fisher information of one observation.
- **Sufficiency**: a statistic is sufficient if it contains all the information in the sample about $\theta$; for a normal mean, the sample mean is sufficient.

For hydrological samples of 20–60 values, small-sample behavior matters more than asymptotic properties. Figure 7.1 shows a classic problem: the sample skewness of short records is biased toward zero and widely scattered, even with the bias correction of Equation (4.2). This is why the skewness of a single record is an unreliable basis for a three-parameter distribution.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(71)
true_skew = 1.5; a_ = (2 / true_skew) ** 2
ns = np.array([10, 20, 30, 50, 100, 200])
res = []
for n in ns:
    g = stats.skew(rng.gamma(a_, 1.0, size=(4000, n)), axis=1, bias=False)
    res.append(np.percentile(g, [5, 25, 50, 75, 95]))
    if n == 30: frac_above = np.mean(g > true_skew)
res = np.array(res)
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.fill_between(ns, res[:, 0], res[:, 4], color=c1, alpha=0.2, lw=0, label="5–95%")
ax.fill_between(ns, res[:, 1], res[:, 3], color=c1, alpha=0.35, lw=0, label="25–75%")
ax.plot(ns, res[:, 2], color=c1, lw=2.2, label="Median")
ax.axhline(true_skew, color=c2, lw=1.6, ls="--", label="True skewness")
ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels(ns)
ax.set_xlabel("Sample size n"); ax.set_ylabel("Sample skewness g"); ax.legend(loc="lower right", fontsize=9)
plt.show()
glue("g30med", float(res[2, 2]), display=False); glue("g30above", float(100 * frac_above), display=False)
```

*Figure 7.1. Sampling distribution of the bias-corrected sample skewness for samples from a gamma distribution with skewness 1.5. With 30 values the median estimate is only {glue:text}`g30med:.2f`, and only {glue:text}`g30above:.0f`% of the estimates exceed the true value.*

## 7.2 Method of moments

The **method of moments** (MoM) equates as many sample moments as there are parameters to their theoretical expressions and solves for the parameters. For the gamma distribution, with mean $\alpha\beta$ and variance $\alpha\beta^2$,

\begin{equation*}
\hat\alpha = \left(\frac{\bar x}{s}\right)^2, \qquad \hat\beta = \frac{s^2}{\bar x} .
\tag{7.2}
\end{equation*}

For the Gumbel distribution, $\hat\sigma = \sqrt6\,s/\pi$ and $\hat\mu = \bar x - 0.5772\,\hat\sigma$ (Example 6.3); for the lognormal, the moments of the logarithms give $\mu_y$ and $\sigma_y$ directly; for the Pearson type III and log-Pearson type III, the sample mean, standard deviation and skewness give the frequency-factor form (Equation 6.6).

The method is simple and always gives an answer, but it is generally less efficient than maximum likelihood, and three-parameter fits rely on the sample skewness, whose problems Figure 7.1 shows. In practice, the skewness of a short record is often replaced or weighted with a regional value (Chapter 8).

## 7.3 Maximum likelihood

### The likelihood

For independent observations from a density $f(x; \theta)$, the joint density of the sample, viewed as a function of the parameters, is the **likelihood**:

\begin{equation*}
L(\theta) = \prod_{i=1}^{n} f(x_i; \theta), \qquad \ell(\theta) = \ln L(\theta) = \sum_{i=1}^{n} \ln f(x_i; \theta) .
\tag{7.3}
\end{equation*}

The **maximum likelihood estimate** (MLE) is the $\theta$ that maximizes $\ell$: the parameter value under which the observed sample is most probable. When the maximum is in the interior, it solves $\partial\ell/\partial\theta = 0$. For the exponential distribution, $\ell(\beta) = -n\ln\beta - \sum x_i/\beta$, and setting the derivative to zero gives $\hat\beta = \bar x$. For most other distributions, such as the gamma, Gumbel and GEV, the equations are solved numerically; `scipy.stats` distributions have a `fit` method that does this.

### Properties

Under regularity conditions, MLEs are consistent and, for large samples, approximately normal and unbiased with variance equal to the Cramér–Rao bound: they are asymptotically efficient. Their standard errors are estimated from the curvature of the log-likelihood at its maximum, the **observed information**:

\begin{equation*}
\operatorname{SE}(\hat\theta) \approx \left[-\frac{\partial^2\ell}{\partial\theta^2}\Big|_{\hat\theta}\right]^{-1/2} .
\tag{7.4}
\end{equation*}

A sharply peaked log-likelihood means a precise estimate (Figure 7.2). MLEs are also **invariant**: the MLE of a function of the parameters, such as a quantile, is that function of the MLEs. The likelihood extends naturally to censored data, historical floods and covariates (Chapters 8 and 9).

For small samples, however, MLEs of shape parameters can be badly biased or unstable. For the GEV with fewer than about 50 values, the estimate of $\xi$ occasionally takes absurd values; restricting its range or using L-moments is common.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(72)
xg = stats.gumbel_r.rvs(loc=120, scale=35, size=40, random_state=rng)
mu_hat, sg_hat = stats.gumbel_r.fit(xg)
M, S = np.meshgrid(np.linspace(95, 145, 200), np.linspace(22, 55, 200))
LL = np.array([[stats.gumbel_r.logpdf(xg, m, s).sum() for m, s in zip(rm, rs)] for rm, rs in zip(M, S)])
fig, ax = plt.subplots(figsize=(6.4, 3.8))
cs = ax.contour(M, S, LL - LL.max(), levels=[-6, -4, -3, -2, -1, -0.5], colors=c1, linewidths=1.2, linestyles="solid")
ax.clabel(cs, fontsize=8, fmt="%g")
ax.plot(mu_hat, sg_hat, "o", color=c2, ms=7, label="MLE")
ax.plot(120, 35, "x", color=ink, ms=8, mew=2, label="True parameters")
ax.set_xlabel("Location μ"); ax.set_ylabel("Scale σ"); ax.legend(loc="upper right", fontsize=9.5)
plt.show()
glue("mle_mu", float(mu_hat), display=False); glue("mle_sg", float(sg_hat), display=False)
```

*Figure 7.2. Contours of the Gumbel log-likelihood, relative to its maximum, for a synthetic sample of 40 annual maxima drawn with μ = 120 and σ = 35. The MLE is μ̂ = {glue:text}`mle_mu:.1f` and σ̂ = {glue:text}`mle_sg:.1f`. The contour 2 units below the maximum encloses an approximate 86% joint confidence region.*

## 7.4 L-moments

### Probability-weighted moments and L-moments

Conventional moments raise deviations to powers, so a single extreme value can dominate the sample skewness and kurtosis. **L-moments** (Hosking, 1990) are instead linear combinations of the ordered sample $x_{(1)} \le \dots \le x_{(n)}$. They are computed from the **probability-weighted moments**

\begin{equation*}
b_r = \frac{1}{n}\sum_{i=1}^{n} \frac{(i - 1)(i - 2)\cdots(i - r)}{(n - 1)(n - 2)\cdots(n - r)}\,x_{(i)}, \qquad r = 0, 1, 2, 3,
\tag{7.5}
\end{equation*}

as

\begin{equation*}
\ell_1 = b_0, \quad \ell_2 = 2b_1 - b_0, \quad \ell_3 = 6b_2 - 6b_1 + b_0, \quad \ell_4 = 20b_3 - 30b_2 + 12b_1 - b_0 .
\tag{7.6}
\end{equation*}

$\ell_1$ is the mean and $\ell_2$ a measure of scale (half the expected difference between two randomly drawn values). The **L-moment ratios** are the **L-CV** $\tau_2 = \ell_2/\ell_1$, the **L-skewness** $\tau_3 = \ell_3/\ell_2$ and the **L-kurtosis** $\tau_4 = \ell_4/\ell_2$. Unlike the conventional skewness, $|\tau_3| < 1$ and $\tau_4$ is bounded, and the sample estimates are nearly unbiased even for small samples.

### Fitting distributions

Parameters are estimated by equating sample and theoretical L-moments. For the Gumbel distribution, $\sigma = \ell_2/\ln 2$ and $\mu = \ell_1 - 0.5772\,\sigma$. For the GEV, with $k = -\xi$, the L-skewness depends on $k$ alone, $\tau_3 = 2(1 - 3^{-k})/(1 - 2^{-k}) - 3$, and Hosking et al. (1985) gave the accurate approximation

\begin{equation*}
k \approx 7.8590\,c + 2.9554\,c^2, \qquad c = \frac{2}{3 + \tau_3} - \frac{\ln 2}{\ln 3},
\tag{7.7}
\end{equation*}

after which $\sigma = \ell_2 k/[(1 - 2^{-k})\Gamma(1 + k)]$ and $\mu = \ell_1 - \sigma[1 - \Gamma(1 + k)]/k$.

### The L-moment ratio diagram

Every distribution has a characteristic relation between its L-skewness and L-kurtosis: a single point for two-parameter distributions (normal, Gumbel, exponential), a curve for three-parameter families (GEV, Pearson type III). Plotting sample $(\tau_3, \tau_4)$ pairs on this **L-moment ratio diagram** shows which distribution suits a set of records, usually the records of a region (Chapter 8). Figure 7.3 shows the diagram with samples drawn from a GEV.

```{code-cell} python
:tags: [hide-input]
def lmoments(x):
    """Sample L-moments l1, l2 and ratios t3, t4 (unbiased PWM estimators)."""
    x = np.sort(np.asarray(x, float)); n = len(x); i = np.arange(1, n + 1)
    b0 = x.mean()
    b1 = np.sum((i - 1) / (n - 1) * x) / n
    b2 = np.sum((i - 1) * (i - 2) / ((n - 1) * (n - 2)) * x) / n
    b3 = np.sum((i - 1) * (i - 2) * (i - 3) / ((n - 1) * (n - 2) * (n - 3)) * x) / n
    l1, l2, l3, l4 = b0, 2 * b1 - b0, 6 * b2 - 6 * b1 + b0, 20 * b3 - 30 * b2 + 12 * b1 - b0
    return l1, l2, l3 / l2, l4 / l2
def gev_lmom_fit(x):
    """GEV parameters (mu, sigma, xi) from L-moments; xi = -k."""
    l1, l2, t3, _ = lmoments(x)
    c = 2 / (3 + t3) - np.log(2) / np.log(3); k = 7.8590 * c + 2.9554 * c ** 2
    sg = l2 * k / ((1 - 2 ** (-k)) * special.gamma(1 + k)); mu = l1 - sg * (1 - special.gamma(1 + k)) / k
    return mu, sg, -k
kk = np.linspace(-0.45, 0.6, 300)
t3g = 2 * (1 - 3 ** (-kk)) / (1 - 2 ** (-kk)) - 3
t4g = (5 * (1 - 4 ** (-kk)) - 10 * (1 - 3 ** (-kk)) + 6 * (1 - 2 ** (-kk))) / (1 - 2 ** (-kk))
rng = np.random.default_rng(73)
samp = np.array([lmoments(stats.genextreme.rvs(-0.1, loc=100, scale=30, size=40, random_state=rng))[2:] for _ in range(60)])
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.plot(t3g, t4g, color=c1, lw=2.2, label="GEV")
ax.plot(samp[:, 0], samp[:, 1], ".", color=c1, alpha=0.45, ms=6, label="Samples of 40 from GEV (ξ = 0.1)")
for lab, pt, col in [("Normal", (0, 0.1226), ink), ("Gumbel", (0.1699, 0.1504), c2), ("Exponential", (1 / 3, 1 / 6), c3)]:
    ax.plot(*pt, "s", color=col, ms=7); ax.text(pt[0] + 0.015, pt[1] - 0.012, lab, color=col, fontsize=9.5)
t3_true = 2 * (1 - 3 ** 0.1) / (1 - 2 ** 0.1) - 3
ax.set_xlim(-0.1, 0.5); ax.set_ylim(0.0, 0.4)
ax.set_xlabel("L-skewness τ₃"); ax.set_ylabel("L-kurtosis τ₄"); ax.legend(loc="upper left", fontsize=9)
plt.show()
glue("t3_true", float(t3_true), display=False)
```

*Figure 7.3. L-moment ratio diagram with the GEV curve and the points of three two-parameter distributions. The light points are 60 synthetic samples of 40 values from a GEV with ξ = 0.1 (true τ₃ = {glue:text}`t3_true:.3f`). The spread of the points shows the sampling variability of L-moment ratios for typical record lengths, which is why regional averages are used.*

## 7.5 Sampling uncertainty and the bootstrap

### Comparing methods

Which method is best depends on the distribution, the sample size and the quantity of interest. The quantity that matters most in design is usually a high quantile. Figure 7.4 compares three ways of estimating the 100-year value from 30 years of data drawn from a GEV with a moderately heavy tail ($\xi$ = 0.1): a Gumbel distribution fitted by moments, and a GEV fitted by maximum likelihood and by L-moments. The two-parameter Gumbel is stable but biased low, because it cannot represent the heavy tail; the GEV fits are nearly unbiased but much more variable, the MLE more so. This trade-off between bias and variance is a central theme of frequency analysis.

```{code-cell} python
:tags: [hide-input]
import warnings
rng = np.random.default_rng(74)
xi_t, mu_t, sg_t = 0.1, 100.0, 30.0
q_true = stats.genextreme.ppf(0.99, -xi_t, loc=mu_t, scale=sg_t)
yT = -np.log(-np.log(0.99)); est = {"Gumbel (MoM)": [], "GEV (MLE)": [], "GEV (L-moments)": []}
for _ in range(400):
    x = stats.genextreme.rvs(-xi_t, loc=mu_t, scale=sg_t, size=30, random_state=rng)
    sg = np.sqrt(6) * x.std(ddof=1) / np.pi; est["Gumbel (MoM)"].append(x.mean() - 0.5772 * sg + sg * yT)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c_, l_, s_ = stats.genextreme.fit(x)
    est["GEV (MLE)"].append(stats.genextreme.ppf(0.99, c_, loc=l_, scale=s_))
    m_, s2_, x2_ = gev_lmom_fit(x); est["GEV (L-moments)"].append(stats.genextreme.ppf(0.99, -x2_, loc=m_, scale=s2_))
fig, ax = plt.subplots(figsize=(6.4, 3.8))
data = [np.array(v) for v in est.values()]
bp = ax.boxplot(data, whis=(5, 95), showfliers=False, patch_artist=True, widths=0.5)
for patch, col in zip(bp["boxes"], [c3, c2, c1]): patch.set_facecolor(col); patch.set_alpha(0.5)
for med in bp["medians"]: med.set_color(ink)
ax.set_xticks([1, 2, 3]); ax.set_xticklabels(list(est.keys()))
ax.axhline(q_true, color=ink, lw=1.4, ls="--"); ax.text(3.35, q_true + 5, "true", color=ink, fontsize=9.5)
ax.set_ylabel("Estimated 100-year value")
plt.show()
rows = []
for kname, v in zip(est.keys(), data):
    q25, q50, q75 = np.percentile(v, [25, 50, 75])
    rows.append(f"| {kname} | {100 * (q50 / q_true - 1):+.1f}% | {100 * (q75 - q25) / q_true:.1f}% | {100 * np.mean(v > 2 * q_true):.1f}% |")
glue("qtrue7", float(q_true), display=False)
```

*Figure 7.4. Estimates of the 100-year value from 400 synthetic samples of 30 annual maxima drawn from a GEV with ξ = 0.1 (true value {glue:text}`qtrue7:.0f`). Boxes show the 25–75% range and whiskers the 5–95% range. Table 7.1 summarizes bias and error.*

```{code-cell} python
:tags: [remove-input]
Markdown("| Method | Median error | Interquartile range | Estimates above twice the true value |\n|---|---|---|---|\n" + "\n".join(rows))
```

*Table 7.1. Performance of three estimators of the 100-year value in Figure 7.4, relative to the true value. Robust summaries are used because a few maximum likelihood fits produce absurdly large values when the estimated shape parameter is extreme; the last column counts them.*

### The bootstrap

For complicated estimators, formulas for the standard error are unavailable or unreliable. The **bootstrap** (Efron, 1979) estimates the sampling distribution by treating the sample as a stand-in for the population:

1. draw $n$ values from the sample *with replacement*;
2. compute the estimate for this resample;
3. repeat many times (1000 or more) and use the spread of the resampled estimates as the sampling distribution.

The 2.5% and 97.5% percentiles of the bootstrap estimates give a simple **percentile confidence interval**. The bootstrap assumes independent observations; for time series, blocks of consecutive values must be resampled together. It also cannot create information that is not in the sample: with 30 years of data, it cannot reveal the behavior of floods much larger than the largest observed.

## 7.6 Worked examples

**Example 7.1 — Gamma by moments.** July rainfall totals (mm) for 12 years are listed below. Fit a gamma distribution by the method of moments and estimate the probability of a July with less than 150 mm.

```{code-cell} python
x = np.array([312, 198, 455, 260, 141, 388, 230, 297, 512, 176, 344, 265.0])
a_hat = (x.mean() / x.std(ddof=1)) ** 2; b_hat = x.var(ddof=1) / x.mean()
print(f"alpha = {a_hat:.2f}, beta = {b_hat:.1f} mm; P(X < 150) = {stats.gamma.cdf(150, a_hat, scale=b_hat):.3f}")
```

**Example 7.2 — Maximum likelihood for the exponential.** Interstorm periods (days) at a station are 3.1, 7.4, 1.2, 5.8, 2.6, 11.0, 4.3, 0.8, 6.2 and 3.9. Find the MLE of the mean and its standard error. For the exponential, $-\partial^2\ell/\partial\beta^2 = n/\hat\beta^2$ at the maximum.

```{code-cell} python
t = np.array([3.1, 7.4, 1.2, 5.8, 2.6, 11.0, 4.3, 0.8, 6.2, 3.9])
b = t.mean(); se = b / np.sqrt(len(t))
print(f"beta_hat = {b:.2f} d, SE = {se:.2f} d")
```

**Example 7.3 — L-moments and a GEV.** For the 15 annual maxima of Example 4.1, compute the sample L-moments and fit a GEV.

```{code-cell} python
x = np.array([88, 132, 74, 96, 210, 105, 81, 119, 93, 147, 70, 101, 163, 90, 112.0])
l1, l2, t3, t4 = lmoments(x); mu_, sg_, xi_ = gev_lmom_fit(x)
print(f"l1 = {l1:.1f}, l2 = {l2:.1f}, t3 = {t3:.3f}, t4 = {t4:.3f}")
print(f"GEV: mu = {mu_:.1f}, sigma = {sg_:.1f}, xi = {xi_:.3f}; 50-year value = {stats.genextreme.ppf(0.98, -xi_, loc=mu_, scale=sg_):.0f} mm")
```

**Example 7.4 — A bootstrap interval.** Find a 90% bootstrap confidence interval for the 50-year value of Example 7.3, using the Gumbel distribution fitted by L-moments for stability.

```{code-cell} python
rng = np.random.default_rng(5)
y50 = -np.log(-np.log(0.98))
def gumbel_q(x):
    l1, l2, _, _ = lmoments(x); s = l2 / np.log(2); return l1 - 0.5772 * s + s * y50
boot = np.array([gumbel_q(rng.choice(x, len(x), replace=True)) for _ in range(2000)])
print(f"estimate = {gumbel_q(x):.0f} mm; 90% interval = {np.percentile(boot, 5):.0f}–{np.percentile(boot, 95):.0f} mm")
```

## 7.7 Common mistakes

- **Relying on the skewness of a short record** for a three-parameter fit.
- **Reporting an estimate without its uncertainty.** Give a standard error or a confidence interval.
- **Accepting extreme MLE shape parameters** from small samples without checking them against regional information.
- **Mixing parameterizations** between textbooks and software (scale versus rate, $\xi$ versus $k$).
- **Bootstrapping dependent data** value by value instead of in blocks.

## 7.8 Exercises

```{code-cell} python
:tags: [remove-cell]
glue("n1a", float((60 / 25) ** 2), display=False); glue("n1b", float(25 ** 2 / 60), display=False)
glue("n2a", float(np.sqrt(6) * 45 / np.pi), display=False); glue("n2b", float(160 - 0.5772 * np.sqrt(6) * 45 / np.pi), display=False)
glue("n3", float(28 / np.log(2)), display=False)
glue("n4", float(4.0 / np.sqrt(25)), display=False)
glue("n5", float(np.mean([3, 5, 9])), display=False)
```

1. A sample has mean 60 mm and standard deviation 25 mm. Fit a gamma distribution by moments.
2. Annual maxima have mean 160 mm and standard deviation 45 mm. Fit a Gumbel distribution by moments.
3. A sample has L-moments $\ell_1$ = 150 mm and $\ell_2$ = 28 mm. Find the Gumbel scale parameter by L-moments.
4. The MLE of an exponential mean from 25 values is 4.0 days. What is its approximate standard error?
5. Compute $\ell_1$ for the sample 9, 3, 5, and explain why it does not depend on the ordering.

```{admonition} Answers
:class: dropdown
1. $\hat\alpha$ = {glue:text}`n1a:.2f`, $\hat\beta$ = {glue:text}`n1b:.1f` mm.
2. $\hat\sigma$ = {glue:text}`n2a:.1f` mm, $\hat\mu$ = {glue:text}`n2b:.1f` mm.
3. $\hat\sigma$ = {glue:text}`n3:.1f` mm.
4. {glue:text}`n4:.2f` days.
5. $\ell_1$ = {glue:text}`n5:.2f`, the sample mean; $b_0$ gives every ordered value the same weight.
```

## 7.9 Further reading

- Efron, B., and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Hosking, J. R. M. (1990). L-moments: analysis and estimation of distributions using linear combinations of order statistics. *Journal of the Royal Statistical Society B*, 52, 105–124.
- Hosking, J. R. M., Wallis, J. R., and Wood, E. F. (1985). Estimation of the generalized extreme-value distribution by the method of probability-weighted moments. *Technometrics*, 27, 251–261.
- Maity, R. (2022). *Statistical Methods in Hydrology and Hydroclimatology* (2nd ed.). Springer.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
