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

# 1. Statistics and hydrological variability

```{admonition} Core ideas
:class: tip
1. Hydrological variables such as rainfall, streamflow and groundwater level vary in ways that cannot be predicted in detail. Statistics describes this variability and lets us make decisions in spite of it.
2. Two kinds of uncertainty matter: **natural variability**, which more data cannot remove, and **knowledge uncertainty**, which comes from short records, measurement error and imperfect models.
3. A record is a **sample** from a population we never observe completely. Design values estimated from 30 or 50 years of data carry sampling uncertainty that must be stated.
4. Hydrological data come in several forms: continuous series, event series, counts, intermittent and censored values. The form decides which methods apply.
5. Before any analysis, plot the record and look at it: seasonality, intermittency, skewness, persistence and changes are often visible at a glance.
```

```{code-cell} python
:tags: [remove-cell]
# Plot style shared by all figures in this chapter.
import numpy as np
import matplotlib.pyplot as plt
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

## 1.1 Variability, uncertainty and risk in water problems

A culvert must pass the floods of the next fifty years; a reservoir must supply a city through droughts no one has yet seen; a levee must be high enough, but not so high that it wastes money. None of these quantities can be predicted. Next year's largest flood depends on weather that has not yet happened, and on the state of the catchment when it arrives. What hydrology *can* do is describe how often floods of different sizes occur, and how confident we are in that description. That is the subject of statistical hydrology.

It helps to separate two kinds of uncertainty:

- **Natural variability** (also called aleatory uncertainty) is the randomness of the process itself. Even with a perfect model and an infinitely long record, we could not say whether next year's maximum flow will be above or below the median. Probability distributions describe it.
- **Knowledge uncertainty** (epistemic uncertainty) comes from what we do not know: short records, measurement errors, the choice of distribution, changes in land use or climate. More data and better models can reduce it, and confidence intervals describe it.

Both enter design. The **risk** that a structure fails during its life depends on natural variability, through the probability of exceedance; our confidence in that risk depends on knowledge uncertainty. Chapter 9 combines them.

Statistical hydrology is old. Allen Hazen plotted flood records on probability paper in 1914, Emil Gumbel built the theory of extreme values in the 1940s and 1950s, and Harold Hurst's analysis of Nile records in the 1950s revealed the long memory of hydrological series (see [*A Brief History*](https://hydroclimatol.com/water-history/history-of-hydrology/#the-statistics-of-long-records)). The methods of this book grew from those beginnings.

## 1.2 Types of hydrological data

Hydrological observations take several forms, and each calls for its own methods:

- **Continuous series** are measured at regular intervals: hourly water levels, daily discharge, monthly evaporation. Successive values are usually correlated (Chapter 14).
- **Event series** are extracted from continuous series: the largest flow in each year (**annual maxima**), or all peaks above a threshold (**peaks over threshold**). Their values are usually close to independent, which makes them the raw material of frequency analysis (Chapter 8).
- **Counts** record how many times something happened: rainy days in a month, floods in a decade. They follow discrete distributions (Chapter 5).
- **Intermittent variables** are exactly zero for part of the time: daily rainfall in most climates, streamflow in ephemeral rivers. They need mixed distributions with a probability mass at zero (Chapter 6).
- **Censored values** are known only to lie above or below a limit: a flood that overtopped the gauge, a concentration below the detection limit, a historical flood known only from a flood mark.

Data also differ in scale. Discharge and rainfall are measured on a **ratio scale**, with a true zero, so that "twice as much" is meaningful; temperature in °C is on an **interval scale**; drought categories ("moderate", "severe") are **ordinal**; land-use classes are **nominal**. Most methods in this book assume ratio or interval data; ordinal data call for rank-based methods (Chapters 10 and 13).

## 1.3 Population, sample and record length

Statistics distinguishes the **population**, the full set of values the process could produce, from the **sample**, the values actually observed. A 40-year record of annual maximum floods is a sample of 40 values from a population that, for a stationary climate and catchment, is conceptually infinite. Every quantity computed from the record, its mean, its standard deviation, its estimated 100-year flood, is a **statistic**: it would take a different value if a different 40 years had been observed.

This **sampling variability** is the central practical problem of statistical hydrology, because design return periods (100, 200, 1000 years) far exceed typical record lengths (20 to 60 years). Figure 1.1 shows how it shrinks as records lengthen. Many synthetic records are drawn from a known population of annual maxima, and the 100-year flood is estimated from each by the method of moments (Chapter 7). With 20 years of data the estimates scatter widely about the true value; even with 100 years the scatter remains appreciable.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(1)
loc, scale = 100.0, 35.0                                         # true Gumbel population (mm or m3/s)
yT = -np.log(-np.log(1 - 1 / 100))                               # Gumbel reduced variate for T = 100
q_true = loc + scale * yT
lengths = np.array([10, 15, 20, 30, 40, 50, 70, 100, 150, 200])
pct = []
for n in lengths:
    x = rng.gumbel(loc, scale, size=(5000, n))
    s = x.std(axis=1, ddof=1); m = x.mean(axis=1)
    a_hat = s * np.sqrt(6) / np.pi; u_hat = m - 0.5772 * a_hat    # method of moments
    pct.append(np.percentile(u_hat + a_hat * yT, [5, 50, 95]))
pct = np.array(pct)
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.fill_between(lengths, pct[:, 0], pct[:, 2], color=c1, alpha=0.25, lw=0, label="5–95% of estimates")
ax.plot(lengths, pct[:, 1], color=c1, lw=2.2, label="Median estimate")
ax.axhline(q_true, color=c2, lw=1.6, ls="--", label="True 100-year value")
ax.set_xscale("log"); ax.set_xticks(lengths); ax.set_xticklabels(lengths)
ax.set_xlabel("Record length (years)"); ax.set_ylabel("Estimated 100-year value")
ax.legend(loc="upper right", fontsize=9.5)
plt.show()
i30 = list(lengths).index(30)
glue("q_true", float(q_true), display=False)
glue("q30_lo", float(pct[i30, 0]), display=False); glue("q30_hi", float(pct[i30, 2]), display=False)
```

*Figure 1.1. Sampling variability of the estimated 100-year value from records of different lengths, drawn from a Gumbel population whose true 100-year value is {glue:text}`q_true:.0f`. With 30 years of data, 90% of the estimates fall between {glue:text}`q30_lo:.0f` and {glue:text}`q30_hi:.0f`.*

Two further assumptions underlie most methods: that the sample values are **independent** (one year's flood tells nothing about the next), and that they are **identically distributed** (the population does not change over the record). Together they are abbreviated **i.i.d.** Annual maxima are often close to i.i.d.; daily flows are not, because a wet day tends to follow a wet day. Climate change, urbanization and river regulation can violate the second assumption; Chapters 9 and 13 discuss how to detect and handle that.

## 1.4 A first look at a record

Before computing anything, plot the data. Figure 1.2 shows 40 years of synthetic daily rainfall at a station with a wet summer monsoon, generated by a simple stochastic model (wet and dry days follow a Markov chain, and wet-day amounts a gamma distribution, Chapters 5 and 6). Several features typical of real records are visible:

- **Intermittency**: most days are dry, so the daily series is a mixture of zeros and positive amounts.
- **Seasonality**: rain is concentrated in summer.
- **Skewness**: wet-day amounts are strongly right-skewed: many small values and a few very large ones.
- **Aggregation**: annual totals are much less skewed and less variable, relative to their mean, than daily amounts; annual maxima form a new variable with a distribution of its own.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(7)
ny = 40; doy = np.tile(np.arange(365), ny); nd = len(doy)
season = np.exp(-0.5 * ((doy - 200) / 35) ** 2)                 # summer monsoon
p01 = 0.15 + 0.35 * season; p11 = 0.45 + 0.35 * season          # P(wet | dry), P(wet | wet)
mean_amt = 6.0 + 16.0 * season                                   # mean wet-day amount (mm)
wet = np.zeros(nd, bool)
for t in range(1, nd):
    wet[t] = rng.random() < (p11[t] if wet[t - 1] else p01[t])
P = np.where(wet, rng.gamma(0.75, mean_amt / 0.75), 0.0)
Pyr = P.reshape(ny, 365)
annual, amax = Pyr.sum(1), Pyr.max(1)
fig, axs = plt.subplots(2, 2, figsize=(6.4, 5.4))
axs[0, 0].bar(np.arange(365), Pyr[0], width=1.0, color=c1)
axs[0, 0].set_xlabel("Day of year"); axs[0, 0].set_ylabel("P (mm)"); axs[0, 0].set_title("(a) One year of daily rainfall")
axs[0, 1].hist(P[P > 0], bins=np.arange(0, 160, 5), color=c1, edgecolor=bg)
axs[0, 1].set_yscale("log"); axs[0, 1].set_xlabel("Wet-day rainfall (mm)"); axs[0, 1].set_ylabel("Number of days")
axs[0, 1].set_title("(b) Wet-day amounts")
yrs = np.arange(1, ny + 1)
axs[1, 0].plot(yrs, annual, "o-", color=c3, ms=3.5, lw=1)
axs[1, 0].set_xlabel("Year"); axs[1, 0].set_ylabel("Annual total (mm)"); axs[1, 0].set_title("(c) Annual totals")
axs[1, 1].plot(yrs, amax, "o-", color=c2, ms=3.5, lw=1)
axs[1, 1].set_xlabel("Year"); axs[1, 1].set_ylabel("Annual max. daily (mm)"); axs[1, 1].set_title("(d) Annual maxima")
plt.show()
wetfrac = float(wet.mean()); cv_d = float(P[P > 0].std() / P[P > 0].mean()); cv_a = float(annual.std() / annual.mean())
glue("wetfrac", 100 * wetfrac, display=False); glue("cv_d", cv_d, display=False); glue("cv_a", cv_a, display=False)
glue("ann_mean", float(annual.mean()), display=False)
```

*Figure 1.2. Forty years of synthetic daily rainfall. (a) One year, showing intermittency and the summer wet season. (b) Wet-day amounts on a logarithmic count scale: strongly right-skewed. (c) Annual totals, mean {glue:text}`ann_mean:.0f` mm. (d) Annual maximum daily rainfall. Rain falls on {glue:text}`wetfrac:.0f`% of days; the coefficient of variation is {glue:text}`cv_d:.2f` for wet-day amounts but only {glue:text}`cv_a:.2f` for annual totals.*

Plots also reveal problems that summary numbers hide: gaps, sudden shifts caused by a gauge relocation, values recorded in the wrong units, or a slow trend. Checking for them is part of every analysis, and Chapter 13 provides formal tests.

## 1.5 Roadmap of the book

The book has four parts.

- **Part I** (Chapters 2–4) builds the language of probability: events, random variables, distributions and moments.
- **Part II** (Chapters 5–9) develops the distributions used in hydrology, the methods for estimating their parameters, and their main application: frequency analysis of floods, rainfall and droughts, and the evaluation of risk in design.
- **Part III** (Chapters 10–12) covers statistical inference: confidence intervals and hypothesis tests, regression, and multivariate methods.
- **Part IV** (Chapters 13–15) turns to dependence in time and between variables: trends and seasonality, stochastic time-series models, and copulas for joint extremes.

Every chapter computes its examples in Python with NumPy, SciPy and Matplotlib; Appendix B summarizes the tools, and Appendix A gives the statistical tables, computed rather than printed.

## 1.6 Worked examples

**Example 1.1 — The chance of a rare flood.** A bridge is designed for the 100-year flood, which has a probability of 0.01 of being exceeded in any year. Assuming independent years, what is the probability that it is exceeded at least once in 30 years?

```{code-cell} python
p, n = 0.01, 30
print(f"P(at least one exceedance in {n} years) = {1 - (1 - p) ** n:.3f}")
```

A "100-year" flood is far from unlikely within a human lifetime. Chapter 9 develops this idea of risk.

**Example 1.2 — Sampling error of a mean.** The annual rainfall of Figure 1.2 has the sample mean and standard deviation computed below. How precisely does a 40-year record determine the long-term mean? For independent values, the standard error of the mean is $s/\sqrt{n}$ (Chapter 4).

```{code-cell} python
m, s, n = annual.mean(), annual.std(ddof=1), len(annual)
print(f"mean = {m:.0f} mm, s = {s:.0f} mm, standard error = {s / np.sqrt(n):.0f} mm ({100 * s / np.sqrt(n) / m:.1f}% of the mean)")
```

**Example 1.3 — Intermittency.** From the synthetic record, estimate the probability that a day is wet, and the probability that a summer day (day 170–230) is wet.

```{code-cell} python
summer = (doy >= 170) & (doy <= 230)
print(f"P(wet) overall = {np.mean(P > 0):.2f}; P(wet) in summer = {np.mean(P[summer] > 0):.2f}")
```

## 1.7 Common mistakes

- **Treating a design value as exact.** A 100-year flood estimated from 30 years of data may be in error by tens of percent (Figure 1.1).
- **Reading "100-year flood" as "once every 100 years".** It is a probability per year, not a schedule.
- **Applying methods for independent data to daily series.** Daily flows and rainfall are correlated in time; use event series or time-series methods.
- **Skipping the plot.** Gaps, unit errors and shifts are easy to see and easy to miss in a table.
- **Ignoring zeros.** Averaging only wet days, or taking logarithms of series that contain zeros, gives wrong results.

## 1.8 Exercises

```{code-cell} python
:tags: [remove-cell]
glue("e1", float(1 - 0.98 ** 25), display=False)
glue("e2", float(0.99 ** 50), display=False)
glue("e3", float(220 / np.sqrt(36)), display=False)
glue("e4", float(220 / np.sqrt(144)), display=False)
```

1. What is the probability that the 50-year flood is exceeded at least once in 25 years?
2. What is the probability that the 100-year flood is **not** exceeded in 50 years?
3. Annual rainfall has a standard deviation of 220 mm. What is the standard error of a mean computed from 36 years of independent data?
4. How long a record would halve the standard error of Exercise 3?
5. Classify each as a continuous series, event series, count, intermittent or censored variable: (a) annual maximum hourly rainfall; (b) daily rainfall; (c) number of days with more than 80 mm of rain in a year; (d) a historical flood known only to have exceeded the level of a bridge deck.

```{admonition} Answers
:class: dropdown
1. {glue:text}`e1:.3f`.
2. {glue:text}`e2:.3f`.
3. About {glue:text}`e3:.0f` mm.
4. Four times as long, 144 years; the standard error becomes {glue:text}`e4:.0f` mm.
5. (a) Event series; (b) intermittent continuous series; (c) count; (d) censored.
```

## 1.9 Further reading

- Haan, C. T. (2002). *Statistical Methods in Hydrology* (2nd ed.). Iowa State Press.
- Helsel, D. R., Hirsch, R. M., Ryberg, K. R., Archfield, S. A., and Gilroy, E. J. (2020). *Statistical Methods in Water Resources*. U.S. Geological Survey Techniques and Methods, book 4, chapter A3.
- Maity, R. (2022). *Statistical Methods in Hydrology and Hydroclimatology* (2nd ed.). Springer.
- Naghettini, M. (ed.) (2017). *Fundamentals of Statistical Hydrology*. Springer.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
