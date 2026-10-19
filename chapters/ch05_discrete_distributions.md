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

# 5. Discrete distributions for counts and occurrences

```{admonition} Core ideas
:class: tip
1. Many hydrological questions are about **counts**: how many years with a flood above the design level, how many storms in a season, how many dry years in a decade.
2. The **binomial** distribution counts occurrences in a fixed number of independent trials, such as exceedances of a design flood in a design life.
3. The **geometric** distribution gives the waiting time to the first occurrence. Its mean is the reciprocal of the annual probability, which is the definition of the **return period**.
4. The **Poisson** distribution counts events occurring at a constant rate in continuous time, such as storms or peaks over a threshold. Its mean equals its variance; counts with larger variance call for the **negative binomial**.
5. The **multinomial** extends the binomial to several categories, and the **hypergeometric** describes sampling without replacement, for example how the largest floods of a record are spread over its years.
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

## 5.1 Bernoulli trials and the binomial distribution

A **Bernoulli trial** is an experiment with two outcomes, "success" with probability $p$ and "failure" with probability $1 - p$. Whether the annual maximum flood exceeds a given level is a Bernoulli trial; so is whether a day is wet. A **Bernoulli process** is a sequence of independent trials with the same $p$, such as successive years under a stationary climate.

The number $X$ of successes in $n$ trials of a Bernoulli process follows the **binomial distribution**. A particular sequence with $x$ successes and $n - x$ failures has probability $p^x(1 - p)^{n - x}$, and there are $\binom{n}{x} = \frac{n!}{x!(n - x)!}$ such sequences, so

\begin{equation*}
P(X = x) = \binom{n}{x} p^x (1 - p)^{n - x}, \qquad x = 0, 1, \dots, n .
\tag{5.1}
\end{equation*}

Since $X$ is a sum of $n$ independent indicator variables, each with mean $p$ and variance $p(1 - p)$, its mean and variance are

\begin{equation*}
E[X] = np, \qquad \operatorname{Var}(X) = np(1 - p) .
\tag{5.2}
\end{equation*}

The binomial distribution answers design questions such as: if a culvert is designed for the 10-year flood, how many times will it be overtopped in 30 years? On average 3, but with a considerable chance of none or of six or more (Figure 5.1). The probability of at least one occurrence, $1 - (1 - p)^n$, met in Chapters 1 and 2, is $1 - P(X = 0)$.

```{code-cell} python
:tags: [hide-input]
fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.4), sharey=True)
for ax, (n, p, col) in zip(axs, [(30, 0.10, c1), (50, 0.02, c2)]):
    k = np.arange(0, 11)
    ax.bar(k, stats.binom.pmf(k, n, p), color=col, width=0.65)
    ax.set_title(f"n = {n}, p = {p} (T = {1 / p:.0f} yr)", fontsize=10); ax.set_xlabel("Number of exceedances")
axs[0].set_ylabel("Probability")
plt.show()
glue("b0", float(stats.binom.pmf(0, 30, 0.1)), display=False); glue("b6", float(stats.binom.sf(5, 30, 0.1)), display=False)
```

*Figure 5.1. Binomial distributions of the number of exceedances: (left) of the 10-year flood in 30 years, (right) of the 50-year flood in 50 years. In the first case the probability of no exceedance is {glue:text}`b0:.3f` and that of six or more is {glue:text}`b6:.3f`.*

## 5.2 Waiting times: geometric and negative binomial

### The geometric distribution and the return period

How long must we wait for the first exceedance? The first success occurs at trial $N = k$ if the first $k - 1$ trials fail and the $k$-th succeeds:

\begin{equation*}
P(N = k) = (1 - p)^{k - 1}\,p, \qquad k = 1, 2, \dots
\tag{5.3}
\end{equation*}

This is the **geometric distribution**. Its mean follows from summing the series $\sum_k k(1 - p)^{k - 1}p$:

\begin{equation*}
E[N] = \frac{1}{p}, \qquad \operatorname{Var}(N) = \frac{1 - p}{p^2} .
\tag{5.4}
\end{equation*}

The mean waiting time between exceedances of a level with annual exceedance probability $p$ is therefore $1/p$ years. This is the **return period** $T = 1/p$: the 100-year flood is the level whose exceedances are, on average, 100 years apart. The geometric distribution shows how misleading the average can be. Its standard deviation is almost as large as its mean, its median is only about $0.69T$ for small $p$, and the first exceedance after construction is more likely in year 1 than in any other single year (Figure 5.2).

The geometric distribution is **memoryless**: $P(N > m + k \mid N > m) = P(N > k)$. Having gone 80 years without a 100-year flood does not make one "due"; under independence, the chance in the next year is still 0.01.

```{code-cell} python
:tags: [hide-input]
p = 0.01; k = np.arange(1, 401)
pmf = stats.geom.pmf(k, p)
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.fill_between(k, 0, pmf, step="mid", color=c1, alpha=0.35, lw=0)
ax.plot(k, pmf, color=c1, lw=1.5)
med = stats.geom.median(p)
ax.axvline(1 / p, color=c2, lw=1.6, ls="--"); ax.text(1 / p + 5, 0.0092, f"mean = {1 / p:.0f}", color=c2, fontsize=9.5)
ax.axvline(med, color=c3, lw=1.6, ls=":"); ax.text(med - 4, 0.0078, f"median = {med:.0f}", color=c3, fontsize=9.5, ha="right")
ax.set_xlabel("Year of first exceedance"); ax.set_ylabel("Probability"); ax.set_ylim(0, 0.0105)
plt.show()
```

*Figure 5.2. Geometric distribution of the year in which the 100-year flood is first exceeded. The mean waiting time is the return period, 100 years, but half of all first exceedances come within 69 years, and the single most likely year is the first.*

### The negative binomial distribution

The number of trials $N_r$ needed to obtain the $r$-th success has the **negative binomial** (Pascal) distribution:

\begin{equation*}
P(N_r = k) = \binom{k - 1}{r - 1} p^r (1 - p)^{k - r}, \qquad k = r, r + 1, \dots,
\tag{5.5}
\end{equation*}

since the $r$-th success must occur at trial $k$, with $r - 1$ successes among the previous $k - 1$ trials. Its mean is $r/p$. A reparameterized form, counting failures rather than trials, is widely used as a model for counts that are more variable than the Poisson distribution allows (Section 5.5).

## 5.3 Rare events: the Poisson distribution

### From binomial to Poisson

Many hydrological events do not occur in discrete trials but at random moments in continuous time: storms, floods above a threshold, earthquakes affecting a dam. Divide a period into $n$ short intervals, each containing an event with small probability $p$, with $\lambda = np$ fixed. As $n \to \infty$, the binomial probabilities converge to

\begin{equation*}
P(X = x) = \frac{\lambda^x e^{-\lambda}}{x!}, \qquad x = 0, 1, 2, \dots,
\tag{5.6}
\end{equation*}

the **Poisson distribution**, with

\begin{equation*}
E[X] = \operatorname{Var}(X) = \lambda .
\tag{5.7}
\end{equation*}

Events that occur independently at a constant average rate form a **Poisson process**. The number of events in a period of length $t$ is Poisson with mean $\lambda t$, and the times between events are exponentially distributed with mean $1/\lambda$ (Chapter 6). The Poisson process is the basis of the peaks-over-threshold method of frequency analysis (Chapter 8).

### Checking the Poisson assumption

The equality of mean and variance gives a simple check: the **dispersion index** $s^2/\bar x$ of observed counts should be close to 1. Values well above 1 (**overdispersion**) are common, because the rate itself varies from year to year with the climate, or because events cluster. Figure 5.3 compares synthetic annual counts of typhoons affecting a coast with a Poisson distribution of the same mean.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(5)
lam_year = rng.gamma(3.0, 2.6 / 3.0, 60)                           # year-to-year variation of the rate
counts = rng.poisson(lam_year)
k = np.arange(0, 10)
obs = np.array([np.mean(counts == j) for j in k])
m, v = counts.mean(), counts.var(ddof=1)
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.bar(k - 0.18, obs, width=0.36, color=c1, label="Observed (60 synthetic years)")
ax.bar(k + 0.18, stats.poisson.pmf(k, m), width=0.36, color=c2, label=f"Poisson, λ = {m:.2f}")
ax.set_xlabel("Typhoons per year"); ax.set_ylabel("Relative frequency"); ax.legend(loc="upper right", fontsize=9.5)
plt.show()
glue("cnt_m", float(m), display=False); glue("cnt_v", float(v), display=False); glue("cnt_d", float(v / m), display=False)
```

*Figure 5.3. Synthetic annual numbers of typhoons over 60 years and the Poisson distribution with the same mean. The sample mean is {glue:text}`cnt_m:.2f` and the variance {glue:text}`cnt_v:.2f`, a dispersion index of {glue:text}`cnt_d:.2f`; the rate was made to vary between years, which makes the counts overdispersed: more years with no typhoons and with many than the Poisson model predicts.*

## 5.4 Multinomial and hypergeometric distributions

### Multinomial

When each trial has $k$ possible outcomes with probabilities $p_1, \dots, p_k$, the numbers of trials falling in each category, $X_1, \dots, X_k$ with $\sum X_i = n$, follow the **multinomial distribution**:

\begin{equation*}
P(X_1 = x_1, \dots, X_k = x_k) = \frac{n!}{x_1!\cdots x_k!}\,p_1^{x_1}\cdots p_k^{x_k} .
\tag{5.8}
\end{equation*}

It describes, for example, how many of the next ten years will be dry, normal or wet when each class has a known probability (Example 5.4). Each $X_i$ alone is binomial with parameters $n$ and $p_i$.

### Hypergeometric

The binomial distribution assumes independent trials with constant probability, as in sampling *with* replacement. When $n$ items are drawn *without* replacement from a finite population of $N$ items, $K$ of which are of a special kind, the number $X$ of special items drawn is **hypergeometric**:

\begin{equation*}
P(X = x) = \frac{\binom{K}{x}\binom{N - K}{n - x}}{\binom{N}{n}} .
\tag{5.9}
\end{equation*}

A hydrological use: in a 50-year record, how surprising is it that four of the five largest floods occurred in the last 15 years? If the record were stationary, the years of the five largest floods would be a random selection of 5 out of 50, and Equation (5.9) gives the probability of such clustering (Example 5.5). This is a simple, assumption-light check before the formal trend tests of Chapter 13.

## 5.5 Choosing a discrete model

Table 5.1 summarizes the distributions of this chapter.

```{code-cell} python
:tags: [remove-input]
Markdown("| Distribution | Describes | Mean | Variance | Typical use |\n|---|---|---|---|---|\n"
         "| Binomial $(n, p)$ | successes in $n$ independent trials | $np$ | $np(1-p)$ | exceedances in a design life |\n"
         "| Geometric $(p)$ | trials to the first success | $1/p$ | $(1-p)/p^2$ | waiting time; return period |\n"
         "| Negative binomial $(r, p)$ | trials to the $r$-th success | $r/p$ | $r(1-p)/p^2$ | waiting for several events; overdispersed counts |\n"
         "| Poisson $(\\lambda)$ | events at a constant rate | $\\lambda$ | $\\lambda$ | storms, peaks over threshold |\n"
         "| Multinomial | counts in $k$ categories | $np_i$ | $np_i(1-p_i)$ | dry, normal and wet years |\n"
         "| Hypergeometric | special items drawn without replacement | $nK/N$ | see text | ranks of extremes in a record |")
```

*Table 5.1. Discrete distributions used in hydrology.*

The choice follows from the structure of the problem: a fixed number of trials suggests the binomial, events in continuous time the Poisson, and waiting times the geometric or negative binomial. When the data suggest that the counts vary more than the model allows, the model's independence or constant-rate assumption is violated and should be revisited. Goodness-of-fit tests for discrete data are given in Chapter 10.

## 5.6 Worked examples

**Example 5.1 — Exceedances in a design life.** A levee protects against the 50-year flood. Find the probabilities of zero, exactly one and two or more exceedances in 20 years.

```{code-cell} python
n, p = 20, 1 / 50
print(f"P(0) = {stats.binom.pmf(0, n, p):.3f}, P(1) = {stats.binom.pmf(1, n, p):.3f}, P(>=2) = {stats.binom.sf(1, n, p):.3f}")
```

**Example 5.2 — Waiting for the first exceedance.** For a structure designed for the 25-year flood, what is the probability that the first exceedance occurs in the first five years? What is the median waiting time?

```{code-cell} python
p = 1 / 25
print(f"P(N <= 5) = {stats.geom.cdf(5, p):.3f}; median waiting time = {stats.geom.median(p):.0f} years")
```

**Example 5.3 — Storm counts.** Heavy storms (more than 100 mm in a day) occur at a station on average 1.6 times per year, as a Poisson process. Find the probability of a year with none, the probability of four or more, and the probability that two years in a row have none.

```{code-cell} python
lam = 1.6
p0 = stats.poisson.pmf(0, lam)
print(f"P(0) = {p0:.3f}; P(>=4) = {stats.poisson.sf(3, lam):.3f}; two years with none: {stats.poisson.pmf(0, 2 * lam):.4f} (= P(0)^2 = {p0**2:.4f})")
```

**Example 5.4 — Dry, normal and wet years.** Years are classified as dry, normal or wet with probabilities 0.25, 0.50 and 0.25. What is the probability that a decade has exactly 4 dry, 4 normal and 2 wet years? What is the probability of at least 5 dry years?

```{code-cell} python
print(f"P(4, 4, 2) = {stats.multinomial.pmf([4, 4, 2], 10, [0.25, 0.5, 0.25]):.4f}; P(dry >= 5) = {stats.binom.sf(4, 10, 0.25):.4f}")
```

**Example 5.5 — Clustering of the largest floods.** In a 50-year record, four of the five largest floods fall in the last 15 years. If the record were stationary, what is the probability of four or more of the five largest floods falling in any given 15-year window?

```{code-cell} python
N, K, n = 50, 15, 5                                                # 50 years, 15 "recent" years, 5 largest floods
print(f"P(X >= 4) = {stats.hypergeom.sf(3, N, K, n):.4f}")
```

The probability is small, which suggests that the recent years differ from the earlier ones. The window was chosen after looking at the data, though, and that makes the result look more significant than it is (Chapter 13).

## 5.7 Common mistakes

- **Thinking a flood is "due".** Under independence, exceedances have no memory.
- **Equating the return period with a typical waiting time.** Waiting times are highly variable; their median is about 0.7 of the return period.
- **Using the binomial for dependent trials.** Consecutive days are not independent; aggregate to independent events first.
- **Assuming Poisson counts without checking dispersion.** Overdispersed counts make Poisson-based intervals too narrow.
- **Using the binomial when sampling without replacement** from a small, finite set, such as the years of a record.

## 5.8 Exercises

```{code-cell} python
:tags: [remove-cell]
glue("k1", float(stats.binom.sf(0, 40, 0.02)), display=False)
glue("k2a", float(25 * 0.2), display=False); glue("k2b", float(np.sqrt(25 * 0.2 * 0.8)), display=False)
glue("k3", float(stats.geom.sf(30, 1 / 50)), display=False)
glue("k4", float(stats.poisson.pmf(2, 0.8 * 3)), display=False)
glue("k5", float(stats.hypergeom.sf(2, 40, 10, 3)), display=False)
```

1. What is the probability that the 50-year flood is exceeded at least once in 40 years?
2. Find the mean and standard deviation of the number of years, out of 25, in which the 5-year drought occurs.
3. What is the probability of waiting more than 30 years for the first exceedance of the 50-year flood?
4. Floods above a threshold occur as a Poisson process at 0.8 per year. What is the probability of exactly two in three years?
5. In a 40-year record, the three largest floods all occurred in the last 10 years. What is this probability under stationarity?

```{admonition} Answers
:class: dropdown
1. {glue:text}`k1:.3f`.
2. Mean {glue:text}`k2a:.1f` years, standard deviation {glue:text}`k2b:.1f` years.
3. {glue:text}`k3:.3f`.
4. {glue:text}`k4:.3f`.
5. {glue:text}`k5:.4f`.
```

## 5.9 Further reading

- Benjamin, J. R., and Cornell, C. A. (1970). *Probability, Statistics, and Decision for Civil Engineers*. McGraw-Hill.
- Haan, C. T. (2002). *Statistical Methods in Hydrology* (2nd ed.). Iowa State Press.
- Maity, R. (2022). *Statistical Methods in Hydrology and Hydroclimatology* (2nd ed.). Springer.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
