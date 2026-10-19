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

# 10. Sampling distributions and hypothesis tests

```{admonition} Core ideas
:class: tip
1. Statistics computed from samples have **sampling distributions**. For normal data, the mean, variance and their ratios follow the normal, **chi-square**, **t** and **F** distributions.
2. A **confidence interval** is a range computed from the data that covers the true parameter in a stated fraction of repeated samples.
3. A **hypothesis test** asks whether the data are consistent with a null hypothesis. Its result is a **p-value**. Two errors are possible: rejecting a true hypothesis (type I) or failing to reject a false one (type II). A test's **power** is its ability to detect a real effect, and with short hydrological records power is often low.
4. **Goodness-of-fit tests** (chi-square, Kolmogorov–Smirnov, Anderson–Darling, probability plot correlation) check whether a distribution is consistent with the data, but with small samples they rarely reject anything.
5. **Nonparametric tests** based on ranks need no distributional assumption and are robust to the skewness and outliers typical of hydrological data.
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

## 10.1 Samples and sampling distributions

A **random sample** is a set of independent observations from the same population. Any statistic computed from it, such as the sample mean, has its own probability distribution over repeated samples: its **sampling distribution**. Chapter 4 gave two key results for a sample of size $n$ from a population with mean $\mu$ and variance $\sigma^2$: $E[\bar X] = \mu$ and $\operatorname{Var}(\bar X) = \sigma^2/n$, and, by the central limit theorem, $\bar X$ is approximately normal for large $n$.

For a **normal** population, exact results hold for any $n$:

\begin{equation*}
\frac{\bar X - \mu}{\sigma/\sqrt n} \sim N(0, 1), \qquad \frac{(n - 1)S^2}{\sigma^2} \sim \chi^2_{n-1}, \qquad \frac{\bar X - \mu}{S/\sqrt n} \sim t_{n-1} ,
\tag{10.1}
\end{equation*}

where $S^2$ is the sample variance. The last result, due to W. S. Gosset ("Student", 1908), is what makes inference about a mean possible when $\sigma$ is unknown.

## 10.2 Chi-square, t and F distributions

These three distributions are built from independent standard normal variables $Z_i$:

- **Chi-square** with $\nu$ degrees of freedom: $\chi^2_\nu = Z_1^2 + \dots + Z_\nu^2$. It is a gamma distribution with shape $\nu/2$ and scale 2, with mean $\nu$ and variance $2\nu$.
- **Student's t** with $\nu$ degrees of freedom: $t_\nu = Z/\sqrt{\chi^2_\nu/\nu}$. It is symmetric like the standard normal but has heavier tails, which account for the extra uncertainty from estimating $\sigma$; it approaches $N(0, 1)$ as $\nu$ grows.
- **F** with $\nu_1$ and $\nu_2$ degrees of freedom: the ratio of two independent chi-square variables, each divided by its degrees of freedom, $F = (\chi^2_{\nu_1}/\nu_1)/(\chi^2_{\nu_2}/\nu_2)$. It is the distribution of the ratio of two sample variances from normal populations with equal variances.

Their quantiles, once read from printed tables, are now computed directly (Appendix A).

```{code-cell} python
:tags: [hide-input]
fig, axs = plt.subplots(1, 3, figsize=(6.4, 3.0))
z = np.linspace(-4, 4, 300)
axs[0].plot(z, stats.norm.pdf(z), color=ink, lw=1.6, label="N(0, 1)")
for nu, col in zip([2, 5, 30], [c2, c1, c3]):
    axs[0].plot(z, stats.t.pdf(z, nu), color=col, lw=1.6, label=f"ν = {nu}")
axs[0].set_title("(a) t"); axs[0].legend(loc="upper right", fontsize=7.5)
x = np.linspace(0.01, 30, 300)
for nu, col in zip([2, 5, 10], [c2, c1, c3]):
    axs[1].plot(x, stats.chi2.pdf(x, nu), color=col, lw=1.6, label=f"ν = {nu}")
axs[1].set_title("(b) Chi-square"); axs[1].set_ylim(0, 0.3); axs[1].legend(loc="upper right", fontsize=7.5)
f = np.linspace(0.01, 5, 300)
for (n1, n2), col in zip([(5, 10), (10, 30), (30, 30)], [c2, c1, c3]):
    axs[2].plot(f, stats.f.pdf(f, n1, n2), color=col, lw=1.6, label=f"({n1}, {n2})")
axs[2].set_title("(c) F"); axs[2].legend(loc="upper right", fontsize=7.5)
for ax in axs: ax.set_yticks([])
plt.show()
```

*Figure 10.1. Densities of (a) Student's t, compared with the standard normal, (b) chi-square and (c) F distributions for several degrees of freedom.*

## 10.3 Confidence intervals

A **confidence interval** at level $1 - \alpha$ is an interval computed from the sample by a rule that, over repeated samples, covers the true parameter with probability $1 - \alpha$. From Equation (10.1), for a normal population,

\begin{equation*}
\bar x \pm t_{1-\alpha/2,\,n-1}\,\frac{s}{\sqrt n}
\tag{10.2}
\end{equation*}

is a $(1 - \alpha)$ interval for the mean, and

\begin{equation*}
\left[\frac{(n - 1)s^2}{\chi^2_{1-\alpha/2,\,n-1}},\ \frac{(n - 1)s^2}{\chi^2_{\alpha/2,\,n-1}}\right]
\tag{10.3}
\end{equation*}

is one for the variance. For non-normal data, Equation (10.2) is still approximately right for moderate $n$ by the central limit theorem, but Equation (10.3) is sensitive to non-normality; the bootstrap (Chapter 7) is a general alternative.

The probability statement refers to the procedure, not to one computed interval: once computed, an interval either contains $\mu$ or not. Figure 10.2 shows what "95% confidence" means: about 95% of intervals computed from repeated samples contain the true mean.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(102)
mu, sg, n = 1200.0, 220.0, 20
fig, ax = plt.subplots(figsize=(6.4, 3.6))
miss = 0
for k in range(50):
    x = rng.normal(mu, sg, n); h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    ok = abs(x.mean() - mu) <= h; miss += not ok
    ax.plot([k, k], [x.mean() - h, x.mean() + h], color=c1 if ok else c2, lw=1.8)
    ax.plot(k, x.mean(), "o", color=c1 if ok else c2, ms=3)
ax.axhline(mu, color=ink, lw=1.2, ls="--")
ax.set_xlabel("Sample number"); ax.set_ylabel("95% interval for the mean (mm)")
plt.show()
glue("ci_miss", miss, display=False)
```

*Figure 10.2. 95% confidence intervals for the mean annual rainfall from 50 synthetic samples of 20 years (true mean 1200 mm, dashed). {glue:text}`ci_miss` of the 50 intervals (orange) miss the true value.*

## 10.4 Hypothesis tests: logic, errors and power

### The procedure

A **hypothesis test** weighs the evidence in the data against a **null hypothesis** $H_0$, a default statement such as "the mean annual flow has not changed", in favor of an **alternative** $H_1$. The steps are:

1. State $H_0$ and $H_1$, and whether $H_1$ is one-sided or two-sided.
2. Choose a **test statistic** whose distribution under $H_0$ is known.
3. Compute the statistic from the data and its **p-value**: the probability, under $H_0$, of a statistic at least as extreme as the one observed.
4. Reject $H_0$ if the p-value is smaller than the chosen **significance level** $\alpha$ (often 0.05).

A small p-value says the data would be surprising if $H_0$ were true. It is not the probability that $H_0$ is true, and a large p-value does not prove $H_0$: it may only mean that the sample is too small to detect a difference.

### Common parametric tests

For normal data:

- **One-sample t test** of $H_0: \mu = \mu_0$: $t = (\bar x - \mu_0)/(s/\sqrt n)$ with $n - 1$ degrees of freedom.
- **Two-sample t test** of equal means in two independent samples, for example annual flows before and after a dam. Welch's version, which does not assume equal variances, uses $t = (\bar x_1 - \bar x_2)/\sqrt{s_1^2/n_1 + s_2^2/n_2}$ with approximate degrees of freedom.
- **Paired t test** for paired observations, such as two rain gauges read on the same days: a one-sample test on the differences.
- **F test** of equal variances: $F = s_1^2/s_2^2$ with $(n_1 - 1, n_2 - 1)$ degrees of freedom. It is very sensitive to non-normality.

### Errors and power

Two kinds of error are possible. A **type I error** rejects a true $H_0$; its probability is $\alpha$. A **type II error** fails to reject a false $H_0$; its probability $\beta$ depends on how false $H_0$ is. The **power** $1 - \beta$ is the probability of detecting a real effect. Power grows with the size of the effect relative to the variability and with the sample size. Hydrological records are short and variable, so power is often low: Figure 10.3 shows that detecting a 10% change in mean annual flow with a coefficient of variation of 0.3 needs about {glue:text}`n80:.0f` years on each side of the change for a power of 0.8. Failing to detect a change is therefore weak evidence that there is none.

```{code-cell} python
:tags: [hide-input]
cv = 0.30
n = np.arange(5, 201)
fig, ax = plt.subplots(figsize=(6.4, 3.6))
for d, col in zip([0.05, 0.10, 0.20, 0.30], [ink2, c1, c2, c3]):
    delta = d / (cv * np.sqrt(2 / n)); dfree = 2 * n - 2; tc = stats.t.ppf(0.975, dfree)
    power = stats.nct.sf(tc, dfree, delta) + stats.nct.cdf(-tc, dfree, delta)
    ax.plot(n, power, color=col, lw=2.2, label=f"{100 * d:.0f}% change")
    if d == 0.10: glue("n80", float(n[np.argmax(power >= 0.8)]), display=False)
ax.axhline(0.8, color=ink, lw=1, ls=":")
ax.set_xlabel("Years of record in each period"); ax.set_ylabel("Power"); ax.set_ylim(0, 1)
ax.legend(loc="lower right", fontsize=9.5)
plt.show()
```

*Figure 10.3. Power of a two-sided two-sample t test at α = 0.05 for detecting a change in mean annual flow, as a function of the number of years before and after the change, for flows with a coefficient of variation of 0.3. The dotted line marks the conventional power of 0.8.*

Testing many hypotheses at once, for example a trend test at every station in a country, inflates the number of false rejections: at $\alpha$ = 0.05, one test in twenty rejects a true null by chance. Corrections that control the **false discovery rate**, or tests of **field significance**, are needed (Chapter 13).

## 10.5 Goodness-of-fit tests

Goodness-of-fit tests ask whether a sample is consistent with a hypothesized distribution $F_0$:

- **Chi-square test**: divide the range into $k$ classes, compare observed counts $O_i$ with expected counts $E_i$ under $F_0$, and compute $\chi^2 = \sum (O_i - E_i)^2/E_i$, approximately chi-square with $k - 1 - m$ degrees of freedom when $m$ parameters were estimated. It suits discrete data and large samples; each class should have $E_i \gtrsim 5$.
- **Kolmogorov–Smirnov (KS) test**: the largest vertical distance between the empirical cdf and $F_0$. It is distribution-free when $F_0$ is fully specified, but too lenient when the parameters were estimated from the same data (the Lilliefors correction addresses this for the normal).
- **Anderson–Darling (AD) test**: a weighted squared distance between the empirical cdf and $F_0$ that gives more weight to the tails, which is what matters for extremes.
- **Probability plot correlation coefficient (PPCC) test**: the correlation between the ordered data and the fitted quantiles at the plotting positions (Chapter 8), compared with critical values obtained by simulation.

With 30–50 values, none of these tests has much power to discriminate between plausible distributions, such as GEV and log-Pearson type III, which differ mainly in the tail beyond the data. Goodness-of-fit tests can reject clearly unsuitable distributions; they cannot pick the right one.

## 10.6 Nonparametric tests

Many hydrological variables are skewed, contain outliers, or are censored, so the normality assumed by t and F tests is doubtful. **Nonparametric** tests use the ranks of the data and make no assumption about the distribution:

- **Mann–Whitney (Wilcoxon rank-sum) test**: are two independent samples from the same distribution, or is one shifted relative to the other? All values are ranked together, and the sum of the ranks of one sample is compared with its distribution under $H_0$.
- **Wilcoxon signed-rank test**: the paired counterpart, applied to the ranks of the absolute differences with their signs.
- **Kruskal–Wallis test**: extends the rank-sum test to more than two groups (Chapter 12).
- **Runs test** for randomness: counts runs of values above and below the median; too few runs indicate persistence or trend, too many indicate alternation.
- **Rank correlation tests**: Spearman's $\rho_s$ (the correlation of the ranks) and Kendall's $\tau$ (based on concordant and discordant pairs); Kendall's $\tau$ against time is the Mann–Kendall trend test (Chapter 13).

When the data are normal, rank tests lose little power compared with t tests; when they are skewed, they are often more powerful and always more reliable.

## 10.7 Worked examples

**Example 10.1 — Confidence interval for a mean.** Annual rainfall over 18 years has mean 1335 mm and standard deviation 240 mm. Find a 95% confidence interval for the long-term mean and a 90% interval for the standard deviation.

```{code-cell} python
n, xbar, s = 18, 1335.0, 240.0
h = stats.t.ppf(0.975, n - 1) * s / np.sqrt(n)
lo_v = (n - 1) * s**2 / stats.chi2.ppf(0.95, n - 1); hi_v = (n - 1) * s**2 / stats.chi2.ppf(0.05, n - 1)
print(f"mean: {xbar - h:.0f}–{xbar + h:.0f} mm; standard deviation: {np.sqrt(lo_v):.0f}–{np.sqrt(hi_v):.0f} mm")
```

**Example 10.2 — Before and after.** The annual runoff (mm) of a catchment for 12 years before and 10 years after extensive urbanization is given below. Test whether the mean changed, with Welch's t test and with the Mann–Whitney test.

```{code-cell} python
before = np.array([512, 640, 455, 598, 702, 530, 488, 615, 570, 660, 505, 548.0])
after = np.array([655, 720, 590, 810, 690, 745, 612, 880, 702, 668.0])
t_stat, p_t = stats.ttest_ind(after, before, equal_var=False)
u_stat, p_u = stats.mannwhitneyu(after, before, alternative="two-sided")
print(f"Welch t = {t_stat:.2f}, p = {p_t:.4f}; Mann-Whitney U = {u_stat:.0f}, p = {p_u:.4f}")
```

**Example 10.3 — Goodness of fit.** Test whether the 45 annual maxima of Chapter 8 are consistent with a Gumbel distribution, using the Kolmogorov–Smirnov and Anderson–Darling statistics.

```{code-cell} python
rng = np.random.default_rng(82)
x_am = stats.genextreme.rvs(-0.12, loc=300, scale=95, size=45, random_state=rng)
loc, scale = stats.gumbel_r.fit(x_am)
ks = stats.kstest(x_am, "gumbel_r", args=(loc, scale))
ad = stats.anderson(x_am, dist="gumbel_r", method="interpolate")    # p-value interpolated from tables
print(f"KS D = {ks.statistic:.3f}, p = {ks.pvalue:.3f} (lenient: parameters were estimated)")
print(f"AD A2 = {ad.statistic:.3f}, p = {ad.pvalue:.2f} (the tables stop at 0.25)")
```

**Example 10.4 — A chi-square test for counts.** The numbers of years with 0, 1, 2, 3 and 4 or more heavy-rain days at a station over 60 years are 9, 17, 18, 10 and 6. Test whether the counts follow a Poisson distribution.

```{code-cell} python
obs = np.array([9, 17, 18, 10, 6]); k = np.arange(5)
lam = (obs[:4] @ k[:4] + 4.5 * obs[4]) / obs.sum()                     # approximate mean (4+ counted as 4.5)
p = stats.poisson.pmf(k, lam); p[4] = stats.poisson.sf(3, lam)
E = obs.sum() * p; chi2 = np.sum((obs - E) ** 2 / E)
print(f"lambda = {lam:.2f}; expected = {np.round(E, 1)}; chi2 = {chi2:.2f}, df = {5 - 1 - 1}, p = {stats.chi2.sf(chi2, 3):.3f}")
```

## 10.8 Common mistakes

- **Reading a p-value as the probability that $H_0$ is true**, or a non-significant result as proof of no effect.
- **Ignoring power.** Short records can hide real changes.
- **Using t and F tests on strongly skewed data.** Use rank tests or transform the data.
- **Testing a distribution with parameters estimated from the same data** using standard KS critical values.
- **Testing many stations or many hypotheses** without accounting for multiple testing.

## 10.9 Exercises

```{code-cell} python
:tags: [remove-cell]
glue("q1", float(stats.t.ppf(0.975, 24)), display=False)
glue("q2a", float(52 - stats.t.ppf(0.975, 15) * 12 / 4), display=False); glue("q2b", float(52 + stats.t.ppf(0.975, 15) * 12 / 4), display=False)
tq = (930 - 1000) / (150 / np.sqrt(25)); glue("q3t", float(tq), display=False); glue("q3p", float(2 * stats.t.cdf(tq, 24)), display=False)
glue("q4", float(stats.f.sf(2.4, 14, 19)), display=False)
glue("q5", float(1 - 0.95 ** 40), display=False)
```

1. Find the critical value $t_{0.975}$ for a 95% interval from 25 values.
2. Sixteen measurements of a reach's mean velocity have mean 52 cm s⁻¹ and standard deviation 12 cm s⁻¹. Find a 95% confidence interval for the mean.
3. A sample of 25 annual runoffs has mean 930 mm and standard deviation 150 mm. Test $H_0: \mu$ = 1000 mm (two-sided). Find $t$ and the p-value.
4. Two samples of 15 and 20 values have a variance ratio of 2.4. What is the one-sided p-value of the F test?
5. Forty independent stations without any real trend are tested at α = 0.05. What is the probability that at least one shows a significant trend?

```{admonition} Answers
:class: dropdown
1. {glue:text}`q1:.3f`.
2. {glue:text}`q2a:.1f`–{glue:text}`q2b:.1f` cm s⁻¹.
3. $t$ = {glue:text}`q3t:.2f`, p = {glue:text}`q3p:.3f`.
4. {glue:text}`q4:.3f`.
5. {glue:text}`q5:.2f`.
```

## 10.10 Further reading

- Helsel, D. R., Hirsch, R. M., Ryberg, K. R., Archfield, S. A., and Gilroy, E. J. (2020). *Statistical Methods in Water Resources*. U.S. Geological Survey Techniques and Methods, book 4, chapter A3.
- Maity, R. (2022). *Statistical Methods in Hydrology and Hydroclimatology* (2nd ed.). Springer.
- Wilks, D. S. (2019). *Statistical Methods in the Atmospheric Sciences* (4th ed.). Elsevier.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
