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

# Appendix B. Python for statistical hydrology

Every computation in this book uses three Python libraries: **NumPy** for arrays and random numbers, **SciPy** (`scipy.stats`, `scipy.optimize`) for distributions, tests and fitting, and **Matplotlib** for figures. They are preinstalled in Google Colab. This appendix summarizes the tools used most often and the conventions that most often cause errors.

```{code-cell} python
:tags: [remove-cell]
import numpy as np
from scipy import stats
from IPython.display import Markdown
```

## B.1 Working with distributions in SciPy

### Distribution objects

Each distribution in `scipy.stats` is an object with the same methods. "Freezing" it with its parameters gives an object that represents one particular distribution:

```{code-cell} python
G = stats.gamma(a=2.5, scale=12.0)                 # shape and scale
print(f"mean = {G.mean():.2f}, sd = {G.std():.2f}, skewness = {float(G.stats(moments='s')):.3f}")
print(f"pdf(30) = {G.pdf(30):.5f}, cdf(30) = {G.cdf(30):.4f}, sf(30) = {G.sf(30):.4f}")
print(f"0.99-quantile = {G.ppf(0.99):.1f}; quantile with exceedance 0.01 = {G.isf(0.01):.1f}")
rng = np.random.default_rng(1)
print("five random values:", np.round(G.rvs(size=5, random_state=rng), 1))
```

The methods are `pdf` (or `pmf` for discrete distributions), `cdf`, `sf` (the exceedance probability $1 - F$, more accurate than `1 - cdf` in the far tail), `ppf` (the quantile function), `isf` (the quantile for a given exceedance probability), `rvs` (random values), `stats` and `moment`.

### Parameterizations

All continuous distributions accept `loc` and `scale`, which shift and stretch a standard form: $X = \text{loc} + \text{scale} \times Y$. Shape parameters come first and have distribution-specific names. Table B.1 relates the SciPy parameters to the notation of this book; mismatched conventions are the most common source of errors.

```{code-cell} python
:tags: [remove-input]
Markdown("| Distribution (book notation) | SciPy call | Notes |\n|---|---|---|\n"
         "| Normal $(\\mu, \\sigma)$ | `norm(loc=mu, scale=sigma)` | |\n"
         "| Lognormal $(\\mu_y, \\sigma_y)$ of $\\ln X$ | `lognorm(s=sigma_y, scale=np.exp(mu_y))` | `loc` adds a lower bound (3-parameter form) |\n"
         "| Exponential, mean $\\beta$ | `expon(scale=beta)` | scale is the mean, not the rate |\n"
         "| Gamma $(\\alpha, \\beta)$ | `gamma(a=alpha, scale=beta)` | scale, not rate |\n"
         "| Pearson III $(\\mu, \\sigma, \\gamma)$ | `pearson3(skew=g, loc=mu, scale=sigma)` | `loc` and `scale` are the mean and standard deviation |\n"
         "| Gumbel $(\\mu, \\sigma)$ | `gumbel_r(loc=mu, scale=sigma)` | `gumbel_l` is for minima |\n"
         "| GEV $(\\mu, \\sigma, \\xi)$ | `genextreme(c=-xi, loc=mu, scale=sigma)` | **c = −ξ**: positive c is the bounded tail |\n"
         "| Generalized Pareto $(\\sigma_u, \\xi)$ | `genpareto(c=xi, scale=sigma_u)` | here c = +ξ |\n"
         "| Weibull $(\\kappa, \\beta, x_0)$ | `weibull_min(c=kappa, loc=x0, scale=beta)` | |\n"
         "| Beta $(a, b)$ | `beta(a, b)` | on $[0, 1]$ unless `loc`, `scale` given |\n"
         "| Binomial $(n, p)$, Poisson $(\\lambda)$ | `binom(n, p)`, `poisson(mu=lam)` | discrete: `pmf` instead of `pdf` |\n"
         "| Geometric $(p)$ | `geom(p)` | counts trials, starting at 1 |")
```

*Table B.1. SciPy parameterizations of the distributions used in the book.*

## B.2 Fitting, testing and resampling

### Fitting

`dist.fit(data)` estimates parameters by maximum likelihood. Parameters can be fixed with `f`-prefixed arguments, most often `floc=0` to fix the location of a two-parameter distribution:

```{code-cell} python
rng = np.random.default_rng(2)
x = stats.genextreme.rvs(-0.1, loc=300, scale=90, size=50, random_state=rng)
c, loc, scale = stats.genextreme.fit(x)
print(f"GEV by ML: xi = {-c:.3f}, mu = {loc:.1f}, sigma = {scale:.1f}")
a, loc0, sc = stats.gamma.fit(x, floc=0)
print(f"two-parameter gamma: alpha = {a:.2f}, beta = {sc:.1f}")
```

SciPy has no built-in L-moment estimators; the short functions of Chapter 7 compute sample L-moments and the GEV parameters from them, and the `lmoments3` package provides a complete implementation.

### Tests

Table B.2 lists the SciPy functions for the tests of Chapters 10–13. Each returns a result object with a `statistic` and, usually, a `pvalue`.

```{code-cell} python
:tags: [remove-input]
Markdown("| Purpose | Function |\n|---|---|\n"
         "| One- and two-sample, paired t tests | `ttest_1samp`, `ttest_ind(equal_var=False)`, `ttest_rel` |\n"
         "| Rank tests for two samples | `mannwhitneyu`, `wilcoxon` |\n"
         "| One-way ANOVA and its rank version | `f_oneway`, `kruskal` |\n"
         "| Goodness of fit | `kstest`, `anderson(method='interpolate')`, `cramervonmises`, `chisquare` |\n"
         "| Normality | `shapiro`, `normaltest`, `probplot` |\n"
         "| Correlation | `pearsonr`, `spearmanr`, `kendalltau` |\n"
         "| Linear regression | `linregress` (simple); `numpy.linalg.lstsq` (multiple) |\n"
         "| Transformations | `boxcox`, `rankdata` (for plotting positions and pseudo-observations) |\n"
         "| Resampling | `bootstrap`, `permutation_test` |")
```

*Table B.2. SciPy functions for statistical tests and related tasks.*

The Mann–Kendall and Pettitt tests are not in SciPy; Chapter 13 gives short implementations, and `kendalltau(time, x)` gives the Mann–Kendall test without ties directly.

### Random numbers and reproducibility

Create one random number generator per analysis with a fixed seed, and pass it to every function that draws random values, so that results can be reproduced exactly:

```{code-cell} python
rng = np.random.default_rng(2024)
boot = [np.mean(rng.choice(x, len(x), replace=True)) for _ in range(2000)]
print(f"bootstrap 90% interval for the mean: {np.percentile(boot, 5):.1f}–{np.percentile(boot, 95):.1f}")
res = stats.bootstrap((x,), np.mean, confidence_level=0.90, random_state=rng)
print(f"scipy.stats.bootstrap (BCa): {res.confidence_interval.low:.1f}–{res.confidence_interval.high:.1f}")
```

## B.3 Reading and cleaning hydrological records

Real records arrive as text or spreadsheet files with dates, values, flags and gaps. The **pandas** library (also preinstalled in Colab) handles them well. The following pattern, not executed here, reads a daily discharge file, checks for gaps, and extracts the annual maximum series by water year:

```python
import pandas as pd

q = pd.read_csv("discharge.csv", parse_dates=["date"], index_col="date")["Q"]   # m3/s
q = q.asfreq("D")                                   # make missing days explicit (NaN)
print("missing days per year:\n", q.isna().groupby(q.index.year).sum())

# Water year starting 1 October: shift the index so each water year is one calendar label
wy = (q.index + pd.DateOffset(months=3)).year
complete = q.groupby(wy).apply(lambda s: s.notna().mean() > 0.95)     # keep years with < 5% gaps
ams = q.groupby(wy).max()[complete]                                   # annual maximum series
date_of_max = q.groupby(wy).idxmax()[complete]
```

Good practice when preparing records:

- **Make gaps explicit** (`asfreq`) and decide, before analysis, how many missing values a year may have; a year whose flood season is missing cannot give its annual maximum.
- **Check units and time stamps**: local time versus UTC, daily values recorded at the end of the day, values in different units in different periods.
- **Keep quality flags** (estimated, ice-affected, above rating) and examine flagged extremes individually.
- **Look for inhomogeneities** with plots and the change-point tests of Chapter 13 before fitting distributions.
- **Record provenance**: source, version, download date and processing steps, in the notebook itself.

## B.4 Further reading

- McKinney, W. (2022). *Python for Data Analysis* (3rd ed.). O'Reilly.
- SciPy documentation, `scipy.stats`: <https://docs.scipy.org/doc/scipy/reference/stats.html>.
- VanderPlas, J. (2016). *Python Data Science Handbook*. O'Reilly.

---

*Text: © Daeha Kim, CC BY-NC 4.0.*
