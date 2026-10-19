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

# 8. Frequency analysis of hydrological extremes

```{admonition} Core ideas
:class: tip
1. **Frequency analysis** relates the magnitude of an extreme event to its probability, so that structures can be designed for a chosen return period.
2. It starts from an **event series**: annual maxima, or all independent peaks above a threshold. The two are linked through the Poisson process.
3. Data are drawn on **probability plots** using plotting positions; a candidate distribution is fitted, checked against the plot and goodness-of-fit measures, and used to compute **design quantiles** with their confidence intervals.
4. Rainfall frequency for many durations is summarized in **intensity–duration–frequency** curves; low flows and droughts use minima, durations and deficits.
5. Records are short. **Regional frequency analysis** pools information from many sites to reduce the uncertainty of rare quantiles.
```

```{code-cell} python
:tags: [remove-cell]
# Plot style shared by all figures in this chapter.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, special
from scipy.optimize import curve_fit
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
def lmoments(x):
    """Sample L-moments l1, l2 and ratios t3, t4 (Chapter 7)."""
    x = np.sort(np.asarray(x, float)); n = len(x); i = np.arange(1, n + 1)
    b0 = x.mean(); b1 = np.sum((i - 1) / (n - 1) * x) / n
    b2 = np.sum((i - 1) * (i - 2) / ((n - 1) * (n - 2)) * x) / n
    b3 = np.sum((i - 1) * (i - 2) * (i - 3) / ((n - 1) * (n - 2) * (n - 3)) * x) / n
    l2 = 2 * b1 - b0
    return b0, l2, (6 * b2 - 6 * b1 + b0) / l2, (20 * b3 - 30 * b2 + 12 * b1 - b0) / l2
def gev_from_lmom(l1, l2, t3):
    """GEV (mu, sigma, xi) from L-moments, Hosking's approximation (Chapter 7)."""
    c = 2 / (3 + t3) - np.log(2) / np.log(3); k = 7.8590 * c + 2.9554 * c ** 2
    sg = l2 * k / ((1 - 2 ** (-k)) * special.gamma(1 + k)); return l1 - sg * (1 - special.gamma(1 + k)) / k, sg, -k
gev_q = lambda F, mu, sg, xi: stats.genextreme.ppf(F, -xi, loc=mu, scale=sg)
yred = lambda F: -np.log(-np.log(F))                                      # Gumbel reduced variate
```

## 8.1 Annual maxima and peaks over threshold

### Two kinds of event series

Frequency analysis needs independent, identically distributed values (Chapter 1). Continuous records of discharge or rainfall are neither, so an **event series** is extracted first.

- The **annual maximum series** (AMS) takes the largest value in each year (often a water year that starts in a dry season, so that one wet season is not split). Annual maxima are usually close to independent, but some years contribute a minor event while others have several large ones.
- The **peaks-over-threshold** or **partial duration series** (POT, PDS) takes every independent peak above a threshold $u$, chosen so that there are on average one to three peaks per year. Peaks must be separated by enough time, typically several times the catchment's response time, and the flow must drop substantially between them, to be counted as independent.

The POT series uses more of the information in a record, which helps when records are short or when frequent events (return periods of a few years) matter.

### Linking the two series

If peaks above $u$ occur as a Poisson process with rate $\lambda$ per year, and their excesses $Y = X - u$ have cdf $G$, the annual maximum is at most $x$ when no peak exceeds $x$, which has probability $\exp\{-\lambda[1 - G(x - u)]\}$. The excesses over a high threshold are well described by the **generalized Pareto** distribution,

\begin{equation*}
G(y) = 1 - \left(1 + \xi\,\frac{y}{\sigma_u}\right)^{-1/\xi}, \qquad y > 0,
\tag{8.1}
\end{equation*}

(the exponential distribution for $\xi = 0$), and with it the annual maxima follow a GEV with the same shape $\xi$: the two approaches are consistent. In terms of return periods, an event with average recurrence interval $T_P$ in the POT series (one occurrence every $T_P$ years on average) has an annual exceedance probability $1 - e^{-1/T_P}$, so

\begin{equation*}
T_A = \frac{1}{1 - e^{-1/T_P}} .
\tag{8.2}
\end{equation*}

The difference matters only for frequent events: $T_P$ = 1 year corresponds to $T_A$ = 1.58 years, but $T_P$ = 10 years to $T_A$ = 10.5 years.

## 8.2 Return period and exceedance probability

For annual maxima with cdf $F$, the **annual exceedance probability** of a level $x$ is $p = 1 - F(x)$ and its **return period** is $T = 1/p$, the mean number of years between exceedances (Chapter 5). The **T-year event** or design quantile is

\begin{equation*}
x_T = F^{-1}\!\left(1 - \frac{1}{T}\right) .
\tag{8.3}
\end{equation*}

Return period is a statement about probability, not about timing: a 100-year flood can occur two years in a row, and its chance of occurring at least once in 30 years is 26% (Example 1.1). Chapter 9 treats the risk over a design life. Increasingly, design guidance states the annual exceedance probability ("1% AEP flood") instead of the return period, to avoid that misunderstanding.

## 8.3 Plotting positions and probability plots

### Plotting positions

To compare data with a fitted distribution, each observation is assigned an empirical non-exceedance probability. Sorting the $n$ values in ascending order, the $i$-th smallest value $x_{(i)}$ is given a **plotting position**

\begin{equation*}
p_i = \frac{i - a}{n + 1 - 2a} ,
\tag{8.4}
\end{equation*}

where the constant $a$ distinguishes several formulas: $a = 0$ gives the **Weibull** formula $i/(n + 1)$, which estimates the expected non-exceedance probability of the $i$-th value for any distribution; $a = 0.44$ the **Gringorten** formula, nearly unbiased for Gumbel quantiles; $a = 0.4$ the **Cunnane** formula, a compromise for many distributions; and $a = 0.5$ the **Hazen** formula, the oldest. The differences matter mainly for the largest values: in a 50-year record, the largest flood is plotted at a return period of 51 years by Weibull and 89 by Gringorten.

### Probability plots

A **probability plot** transforms the probability axis so that a chosen distribution plots as a straight line. For the Gumbel distribution, the reduced variate $y = -\ln(-\ln p)$ is linear in $x$ (Equation 6.9); for the normal, the standard normal quantile $z = \Phi^{-1}(p)$; for the lognormal, $z$ against $\ln x$. Plotting the sorted data against the transformed plotting positions, and the fitted distributions as lines, shows at a glance how well each fits, especially in the upper tail that matters for design (Figure 8.1). Printed probability paper has been replaced by software, but the principle is the same.

## 8.4 Fitting and choosing a distribution

Figure 8.1 shows a synthetic record of 45 annual maximum discharges fitted with three common distributions: the Gumbel and the GEV fitted by L-moments, and the log-Pearson type III fitted by the moments of the logarithms. The Gumbel, whose skewness is fixed, cannot follow the largest values; the GEV and the log-Pearson type III fit the record equally well but diverge beyond it, and the three 1000-year estimates range from {glue:text}`q1000_lo:.0f` to {glue:text}`q1000_hi:.0f` m³ s⁻¹. Choosing among them is the hardest step of frequency analysis, and several kinds of evidence are used:

- **Visual fit** on the probability plot, especially of the largest values.
- **Goodness-of-fit statistics**: the probability plot correlation coefficient (PPCC), the correlation between the sorted data and the fitted quantiles at the plotting positions; the Anderson–Darling and Kolmogorov–Smirnov tests (Chapter 10); and information criteria such as AIC for likelihood fits. With 30–50 values these tests rarely reject any reasonable distribution, so they are a weak guide.
- **Regional evidence**: the L-moment ratio diagram of many sites in a region (Chapter 7) shows which family fits them collectively.
- **Physical reasoning** and **national guidance**: many countries prescribe a distribution and method (GEV with L-moments, log-Pearson type III with weighted skew, and so on) so that designs are consistent.

Two data problems need special care. **Outliers**, unusually high or low values, should be checked against the original records before being removed; a high outlier is often the most informative flood in the record. Low outliers (years with very small maxima) can distort a fitted distribution's upper tail and are often censored. **Historical and paleoflood information**, such as flood marks and documents from before systematic gauging, can extend the effective record by centuries when treated as censored data in a likelihood framework.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(82)
x_am = stats.genextreme.rvs(-0.12, loc=300, scale=95, size=45, random_state=rng)
n = len(x_am); xs = np.sort(x_am); i = np.arange(1, n + 1)
pp = (i - 0.44) / (n + 0.12)                                            # Gringorten
l1, l2, t3, t4 = lmoments(x_am)
sg_gum = l2 / np.log(2); mu_gum = l1 - 0.5772 * sg_gum
mu_gev, sg_gev, xi_gev = gev_from_lmom(l1, l2, t3)
ly = np.log10(x_am); my_, sy_, gy_ = ly.mean(), ly.std(ddof=1), stats.skew(ly, bias=False)
models = {
    "Gumbel (L-moments)": (lambda F: mu_gum + sg_gum * yred(F), c3),
    "GEV (L-moments)": (lambda F: gev_q(F, mu_gev, sg_gev, xi_gev), c1),
    "Log-Pearson III (moments)": (lambda F: 10 ** (my_ + sy_ * stats.pearson3.ppf(F, gy_)), c2),
}
Fg = 1 - 1 / np.logspace(np.log10(1.01), 3, 200)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for lab, (q, col) in models.items():
    ax.plot(yred(Fg), q(Fg), color=col, lw=2.2, label=lab)
ax.plot(yred(pp), xs, "o", color=ink, ms=4.5, label="Annual maxima (Gringorten)")
top = ax.twiny(); top.set_xlim(ax.get_xlim())
Tt = np.array([2, 5, 10, 50, 100, 1000]); top.set_xticks(yred(1 - 1 / Tt)); top.set_xticklabels([str(t) for t in Tt])
top.set_xlabel("Return period (years)", fontsize=9.5); top.grid(False)
ax.set_xlabel("Gumbel reduced variate"); ax.set_ylabel("Annual maximum discharge (m³ s⁻¹)")
ax.legend(loc="upper left", fontsize=9)
plt.show()
rows = []
for lab, (q, col) in models.items():
    r = np.corrcoef(xs, q(pp))[0, 1]
    rows.append(f"| {lab} | {r:.4f} | {q(1 - 1 / 10):.0f} | {q(1 - 1 / 100):.0f} | {q(1 - 1 / 1000):.0f} |")
glue("xi_gev", float(xi_gev), display=False)
q1000s = [q(1 - 1 / 1000) for q, _ in models.values()]
glue("q1000_lo", float(min(q1000s)), display=False); glue("q1000_hi", float(max(q1000s)), display=False)
```

*Figure 8.1. A synthetic record of 45 annual maximum discharges on Gumbel probability axes, with three fitted distributions. The fitted GEV shape is ξ = {glue:text}`xi_gev:.2f`. Table 8.1 gives the goodness of fit and the design quantiles.*

```{code-cell} python
:tags: [remove-input]
Markdown("| Distribution | PPCC | $Q_{10}$ | $Q_{100}$ | $Q_{1000}$ |\n|---|---|---|---|---|\n" + "\n".join(rows))
```

*Table 8.1. Probability plot correlation coefficients and design quantiles (m³ s⁻¹) of the three fits in Figure 8.1. The Gumbel fits the largest values poorly; the other two fit similarly, and their quantiles differ increasingly beyond the record.*

## 8.5 Design quantiles and their confidence intervals

A design quantile estimated from a sample is uncertain (Chapter 7), and the uncertainty grows rapidly with the return period. For the Gumbel distribution fitted by moments, a classical approximation for the standard error of the quantile $x_T = \bar x + K_T s$ is

\begin{equation*}
\operatorname{SE}(\hat x_T) \approx \frac{s}{\sqrt n}\sqrt{1 + 1.1396\,K_T + 1.1000\,K_T^2} ,
\tag{8.5}
\end{equation*}

where $K_T = (\sqrt6/\pi)(y_T - 0.5772)$ is the Gumbel frequency factor. For other distributions and methods, the bootstrap (Chapter 7) or the curvature of the likelihood (profile likelihood) gives confidence intervals. Figure 8.2 shows a 90% bootstrap band for the GEV fit of Figure 8.1: at 100 years it extends from about 40% below the estimate to 40% above it, and at 1000 years it is wider still.

These intervals reflect sampling uncertainty only. They assume the chosen distribution is correct and the record stationary; the divergence of the fits in Figure 8.1 shows that the choice of distribution adds an uncertainty of similar size.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(83)
Fq = 1 - 1 / np.logspace(np.log10(1.05), 3, 120)
boot = []
for _ in range(1000):
    xb = rng.choice(x_am, n, replace=True); a = lmoments(xb)
    boot.append(gev_q(Fq, *gev_from_lmom(a[0], a[1], a[2])))
lo, hi = np.percentile(boot, [5, 95], axis=0)
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.fill_between(yred(Fq), lo, hi, color=c1, alpha=0.22, lw=0, label="90% bootstrap band")
ax.plot(yred(Fq), gev_q(Fq, mu_gev, sg_gev, xi_gev), color=c1, lw=2.2, label="GEV (L-moments)")
ax.plot(yred(pp), xs, "o", color=ink, ms=4.5, label="Annual maxima")
top = ax.twiny(); top.set_xlim(ax.get_xlim()); top.set_xticks(yred(1 - 1 / Tt)); top.set_xticklabels([str(t) for t in Tt])
top.set_xlabel("Return period (years)", fontsize=9.5); top.grid(False)
ax.set_xlabel("Gumbel reduced variate"); ax.set_ylabel("Discharge (m³ s⁻¹)"); ax.legend(loc="upper left", fontsize=9)
plt.show()
j100 = np.argmin(np.abs(Fq - 0.99))
glue("ci_lo", float(lo[j100]), display=False); glue("ci_hi", float(hi[j100]), display=False)
glue("q100", float(gev_q(0.99, mu_gev, sg_gev, xi_gev)), display=False)
```

*Figure 8.2. The GEV fit of Figure 8.1 with a 90% bootstrap confidence band from 1000 resamples. The 100-year estimate is {glue:text}`q100:.0f` m³ s⁻¹, with a 90% interval from {glue:text}`ci_lo:.0f` to {glue:text}`ci_hi:.0f` m³ s⁻¹.*

## 8.6 Rainfall intensity–duration–frequency relations

The design of drains, culverts and small catchments needs rainfall intensities for many durations. For each duration $d$ (for example 1, 2, 3, 6, 12 and 24 hours), the maximum $d$-hour rainfall total is extracted from each year of a continuous record using a moving window, a distribution is fitted to these annual maxima, and the quantiles for chosen return periods are divided by $d$ to give intensities. Plotting intensity against duration, one curve per return period, gives the **intensity–duration–frequency** (IDF) curves (Figure 8.3), used for example with the rational method in the [Engineering Hydrology](https://hydroclimatol.com/engineering-hydrology/chapters/ch11_design_rainfall.html) book.

For convenience the curves are summarized by empirical formulas, such as

\begin{equation*}
i = \frac{a}{(d + b)^c} ,
\tag{8.6}
\end{equation*}

fitted separately for each return period, or by formulas that include $T$ explicitly. Intensities decrease with duration because the most intense rain lasts only a short time. Fitting each duration separately can produce curves that cross for long return periods; fitting all durations together, with parameters that vary smoothly with duration, avoids this.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(81); ny = 40; Hh = ny * 8760
hr = np.arange(Hh); doy = (hr // 24) % 365
season = np.exp(-0.5 * ((doy - 205) / 30) ** 2)                          # summer rainy season
starts = np.nonzero(rng.random(Hh) < 0.003 + 0.012 * season)[0]          # storm starts
Ph = np.zeros(Hh + 400)
for s in starts:                                                          # storms with hourly variability and convective bursts
    d = int(rng.exponential(8)) + 1
    prof = rng.exponential(0.8 + 2.6 * season[s]) * rng.gamma(1.2, 1 / 1.2, d)
    if rng.random() < 0.25: prof[rng.integers(0, d)] += rng.exponential(14 * season[s] + 2)
    Ph[s:s + d] += prof
Ph = Ph[:Hh]; cs = np.r_[0, np.cumsum(Ph)]
durs = np.array([1, 2, 3, 6, 12, 24]); Ts = [2, 10, 50, 100]
idf = np.zeros((len(Ts), len(durs)))
for j, d in enumerate(durs):
    roll = cs[d:] - cs[:-d]
    am = np.array([roll[y * 8760:(y + 1) * 8760 - d + 1].max() for y in range(ny)])
    a = lmoments(am); sg_ = a[1] / np.log(2); mu_ = a[0] - 0.5772 * sg_   # Gumbel by L-moments
    idf[:, j] = [(mu_ + sg_ * yred(1 - 1 / T)) / d for T in Ts]
dd = np.linspace(1, 24, 100)
fig, ax = plt.subplots(figsize=(6.4, 4.0))
fits = {}
for k, (T, col) in enumerate(zip(Ts, [c3, c1, c2, ink])):
    pr, _ = curve_fit(lambda d, a, b, c: a / (d + b) ** c, durs, idf[k], p0=(100, 0.5, 0.8), bounds=([0, 0, 0], [np.inf, 10, 2]))
    fits[T] = pr
    ax.plot(durs, idf[k], "o", color=col, ms=5)
    ax.plot(dd, pr[0] / (dd + pr[1]) ** pr[2], color=col, lw=2.0, label=f"T = {T} years")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(durs); ax.set_xticklabels(durs)
ax.set_xlabel("Duration (h)"); ax.set_ylabel("Intensity (mm h⁻¹)"); ax.legend(loc="upper right", fontsize=9.5)
plt.show()
glue("idf_1_100", float(idf[3, 0]), display=False); glue("idf_24_100", float(idf[3, -1] * 24), display=False)
glue("idf_a", float(fits[100][0]), display=False); glue("idf_b", float(fits[100][1]), display=False); glue("idf_c", float(fits[100][2]), display=False)
```

*Figure 8.3. IDF curves derived from 40 years of synthetic hourly rainfall: points are Gumbel quantiles of the annual maximum intensity for each duration, lines are fits of Equation (8.6). The 100-year intensity is {glue:text}`idf_1_100:.0f` mm h⁻¹ for 1 hour and the 100-year 24-hour depth {glue:text}`idf_24_100:.0f` mm. For T = 100 years the fit gives a = {glue:text}`idf_a:.0f`, b = {glue:text}`idf_b:.2f` and c = {glue:text}`idf_c:.2f` (i in mm h⁻¹, d in hours).*

## 8.7 Low flows and droughts

### Low-flow frequency

Water supply, water quality permits and ecological flows depend on low flows. The standard statistic is the annual minimum of the $d$-day average flow, for example the 7-day minimum, and its quantiles: the **7Q10** is the annual minimum 7-day flow with a 10-year return period, that is, with non-exceedance probability 0.1 in any year. Minima are bounded below by zero, and the Weibull distribution (Chapter 6) or the log-Pearson type III are common choices. Rivers that cease to flow in some years need a mixed distribution with a mass at zero.

### Droughts and run theory

Droughts are defined by duration and deficit as well as magnitude. In **run theory**, a threshold $Q_0$ (for example the flow exceeded 80% of the time) is set, and a **drought event** is a run of consecutive time steps below it. Each event has a **duration** (number of steps), a **deficit volume** or severity (the accumulated shortfall below the threshold) and an **intensity** (deficit divided by duration) (Figure 8.4). Frequency analysis is then applied to the annual maximum deficit or duration, or jointly to both (Chapter 15).

For precipitation, the **Standardized Precipitation Index** (SPI) fits a distribution, usually the gamma, to rainfall totals over a chosen window (1, 3, 6 or 12 months) for each calendar month, and transforms each total to the standard normal value with the same cumulative probability. SPI values below −1 indicate moderate drought, below −1.5 severe and below −2 extreme, which allows comparison across climates and seasons.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(84)
nm = 30 * 12; mon = np.arange(nm) % 12
seas = np.exp(-0.5 * ((mon - 6.5) / 1.6) ** 2)
z = np.zeros(nm)
for t in range(1, nm): z[t] = 0.6 * z[t - 1] + np.sqrt(1 - 0.36) * rng.normal()
Qm = (8 + 60 * seas) * np.exp(0.5 * z - 0.125)                          # monthly flow (m3/s)
Q0 = np.percentile(Qm, 20)                                              # flow exceeded 80% of the time
below = Qm < Q0
runs = []; t = 0
while t < nm:
    if below[t]:
        s = t
        while t < nm and below[t]: t += 1
        runs.append((s, t - s, np.sum(Q0 - Qm[s:t]) * 2.6298e6 / 1e6))   # start, months, deficit (million m3)
    else: t += 1
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 3.6), width_ratios=[1.7, 1])
w = slice(0, 120); tt = np.arange(nm)[w] / 12
a1.plot(tt, Qm[w], color=c1, lw=1.4)
a1.fill_between(tt, Qm[w], Q0, where=below[w], color=c2, alpha=0.5, lw=0, step=None)
a1.axhline(Q0, color=ink2, lw=1, ls="--"); a1.set_yscale("log")
a1.set_xlabel("Year"); a1.set_ylabel("Monthly flow (m³ s⁻¹)"); a1.set_title("(a) Deficits below $Q_{80}$")
dur = np.array([r[1] for r in runs]); dfc = np.array([r[2] for r in runs])
a2.plot(dur + rng.uniform(-0.15, 0.15, len(dur)), dfc, "o", color=c2, ms=5)
a2.set_xlabel("Duration (months)"); a2.set_ylabel("Deficit (10⁶ m³)"); a2.set_title("(b) Events")
plt.show()
glue("n_runs", len(runs), display=False); glue("max_def", float(dfc.max()), display=False); glue("Q0", float(Q0), display=False)
```

*Figure 8.4. Run theory applied to 30 years of synthetic monthly flow with a threshold of {glue:text}`Q0:.1f` m³ s⁻¹, the flow exceeded 80% of the time. (a) The first ten years, with deficits shaded. (b) Duration and deficit volume of all {glue:text}`n_runs` events; the largest deficit is {glue:text}`max_def:.0f` million m³. Longer droughts tend to have larger deficits, so the two must be analyzed jointly.*

## 8.8 Regional frequency analysis

### The index-flood method

A 40-year record says little about the 500-year flood. But a region may have twenty gauges with 40 years each, and if their floods behave similarly once scaled by their size, the records can be pooled. The **index-flood method** (Dalrymple, 1960) assumes that within a **homogeneous region**, the distributions at all sites are identical apart from a site-specific scale, the **index flood** $\mu_j$, usually the mean annual flood:

\begin{equation*}
Q_T^{(j)} = \mu_j\,q_T ,
\tag{8.7}
\end{equation*}

where $q_T$ is the regional **growth curve**, common to all sites. The procedure, in its L-moment form (Hosking and Wallis, 1997), is:

1. **Screen** the data and form a region of sites with similar physiographic and climatic characteristics.
2. **Test homogeneity**: compare the dispersion of the sites' L-CV and L-skewness with what sampling variability alone would produce (the heterogeneity measure $H$).
3. **Choose a distribution** from the regional average L-moment ratios on the L-moment ratio diagram.
4. **Estimate the growth curve** from the regional L-moment ratios, averaged with weights proportional to record length.
5. **Estimate the index flood** at each site from its own record or, at ungauged sites, by regression on catchment area, rainfall and other characteristics (Chapter 11).

Figure 8.5 shows the gain. Synthetic sites share the same growth curve; their at-site estimates of the 100-year growth factor scatter widely because of short records, while regional estimates are much less variable.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(85)
xi_r, mu_r, sg_r = 0.10, 1.0, 0.32
mean_r = stats.genextreme.mean(-xi_r, loc=mu_r, scale=sg_r)
q100_true = stats.genextreme.ppf(0.99, -xi_r, loc=mu_r, scale=sg_r) / mean_r
def simulate_region(nsite=12):
    nrec = rng.integers(18, 51, nsite); at_site, L = [], []
    for j in range(nsite):
        x = rng.uniform(80, 900) * stats.genextreme.rvs(-xi_r, loc=mu_r, scale=sg_r, size=nrec[j], random_state=rng) / mean_r
        a = lmoments(x / x.mean()); L.append(a); at_site.append(gev_q(0.99, *gev_from_lmom(1.0, a[1], a[2])))
    L = np.array(L); w = nrec / nrec.sum()
    return nrec, np.array(at_site), gev_q(0.99, *gev_from_lmom(1.0, np.sum(w * L[:, 1]), np.sum(w * L[:, 2])))
nrec, at_site, q100_reg = simulate_region()
all_site, all_reg = [], []
for _ in range(300):
    _, a_s, r_ = simulate_region(); all_site.extend(a_s); all_reg.append(r_)
all_site, all_reg = np.array(all_site), np.array(all_reg)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 3.7), width_ratios=[1.5, 1])
a1.plot(nrec, at_site, "o", color=c1, ms=6, label="At-site")
a1.axhline(q100_reg, color=c2, lw=2.2, label="Regional")
a1.axhline(q100_true, color=ink, lw=1.4, ls="--", label="True")
a1.set_xlabel("Record length (years)"); a1.set_ylabel("100-year growth factor $q_{100}$"); a1.set_title("(a) One region")
a1.legend(loc="lower right", fontsize=8.5)
bp = a2.boxplot([all_site, all_reg], whis=(5, 95), showfliers=False, patch_artist=True, widths=0.55)
for patch, col in zip(bp["boxes"], [c1, c2]): patch.set_facecolor(col); patch.set_alpha(0.5)
for med in bp["medians"]: med.set_color(ink)
a2.set_xticks([1, 2]); a2.set_xticklabels(["At-site", "Regional"]); a2.axhline(q100_true, color=ink, lw=1.4, ls="--")
a2.set_title("(b) 300 regions")
plt.show()
rmse = lambda v: 100 * np.sqrt(np.mean((v - q100_true) ** 2)) / q100_true
glue("rmse_site", float(rmse(all_site)), display=False); glue("rmse_reg", float(rmse(all_reg)), display=False)
glue("as_lo", float(at_site.min()), display=False); glue("as_hi", float(at_site.max()), display=False)
```

*Figure 8.5. Regional frequency analysis with synthetic regions of 12 sites that share the same GEV growth curve but differ in size and record length (18–50 years). (a) In one region, at-site estimates of the 100-year growth factor range from {glue:text}`as_lo:.2f` to {glue:text}`as_hi:.2f`; the regional estimate uses record-length-weighted L-moment ratios. (b) Over 300 simulated regions, the root mean squared error is {glue:text}`rmse_site:.0f`% for at-site estimates and {glue:text}`rmse_reg:.0f`% for regional ones.*

The price is the homogeneity assumption: pooling dissimilar sites introduces bias. In practice the bias from moderate heterogeneity is usually smaller than the sampling error it removes, which is why regional methods are recommended in national flood guidance in many countries.

## 8.9 Labs (planned)

Two Colab notebooks are planned: a flood frequency analysis of an openly available annual maximum discharge record, comparing distributions, estimation methods and confidence intervals; and the derivation of IDF curves from a long hourly rainfall record.

## 8.10 Worked examples

**Example 8.1 — Plotting positions.** For a 30-year record, find the return periods assigned to the largest and the fifth-largest values by the Weibull and Gringorten formulas.

```{code-cell} python
n = 30
for rank in (1, 5):
    i = n + 1 - rank
    for name, a in (("Weibull", 0.0), ("Gringorten", 0.44)):
        p = (i - a) / (n + 1 - 2 * a)
        print(f"rank {rank}, {name}: T = {1 / (1 - p):.1f} years")
```

**Example 8.2 — A Gumbel quantile and its standard error.** Annual maximum daily rainfall over 35 years has mean 142 mm and standard deviation 48 mm. Estimate the 50-year value by the Gumbel method of moments and its standard error with Equation (8.5).

```{code-cell} python
xbar, s, n, T = 142.0, 48.0, 35, 50
K = np.sqrt(6) / np.pi * (yred(1 - 1 / T) - 0.5772)
xT = xbar + K * s; se = s / np.sqrt(n) * np.sqrt(1 + 1.1396 * K + 1.1 * K ** 2)
print(f"K = {K:.3f}; x50 = {xT:.0f} mm; SE = {se:.0f} mm; approx. 90% interval {xT - 1.645 * se:.0f}–{xT + 1.645 * se:.0f} mm")
```

**Example 8.3 — Peaks over threshold.** Over 25 years, 58 independent peaks exceeded a threshold of 400 m³ s⁻¹, and their excesses have mean 160 m³ s⁻¹. Assuming exponential excesses ($\xi$ = 0), estimate the 100-year annual-maximum flood.

```{code-cell} python
lam, u, beta = 58 / 25, 400.0, 160.0
# annual max cdf: exp(-lam * exp(-(x - u)/beta)) = 1 - 1/T  ->  x = u + beta * ln(lam / -ln(1 - 1/T))
T = 100
print(f"lambda = {lam:.2f}/yr; Q100 = {u + beta * np.log(lam / -np.log(1 - 1 / T)):.0f} m3/s")
```

**Example 8.4 — 7Q10.** The annual minimum 7-day flows (m³ s⁻¹) of a river over 20 years are listed below. Fit a two-parameter Weibull distribution by maximum likelihood and estimate the 7Q10.

```{code-cell} python
q7 = np.array([3.8, 2.6, 5.1, 1.9, 4.4, 3.2, 2.2, 6.0, 3.5, 2.9, 1.6, 4.0, 3.1, 2.4, 5.5, 2.8, 3.6, 2.0, 4.7, 3.3])
kappa, loc0, beta_w = stats.weibull_min.fit(q7, floc=0)
print(f"shape = {kappa:.2f}, scale = {beta_w:.2f} m3/s; 7Q10 = {stats.weibull_min.ppf(0.1, kappa, scale=beta_w):.2f} m3/s")
```

**Example 8.5 — Regional estimate at an ungauged site.** A regional growth curve gives $q_{100}$ = 2.45. At an ungauged catchment, a regional regression predicts a mean annual flood of 310 m³ s⁻¹. Estimate the 100-year flood.

```{code-cell} python
print(f"Q100 = {2.45 * 310:.0f} m3/s")
```

## 8.11 Common mistakes

- **Mixing annual-maximum and peaks-over-threshold return periods** for frequent events (Equation 8.2).
- **Counting dependent peaks** from the same flood as separate events in a POT series.
- **Extrapolating far beyond the record** without stating the uncertainty, or reporting a single "best" distribution when several fit equally well.
- **Removing high outliers** because they look inconvenient; they usually carry the most information about the tail.
- **Assuming stationarity** without checking for trends or regime changes (Chapter 13).

## 8.12 Exercises

```{code-cell} python
:tags: [remove-cell]
glue("o1", float(1 / (1 - np.exp(-1 / 2))), display=False)
glue("o2", float(1 / (1 - (40 + 1 - 1 - 0.44) / (40 + 0.12))), display=False)
Kq = np.sqrt(6) / np.pi * (yred(0.99) - 0.5772); glue("o3", float(380 + Kq * 120), display=False)
glue("o4", float(180 * 2.2), display=False)
lam4 = 2.0; glue("o5", float(250 + 90 * np.log(lam4 / -np.log(1 - 1 / 50))), display=False)
```

1. An event has an average recurrence interval of 2 years in a POT series. What is its annual-maximum return period?
2. In a 40-year record, what return period does the Gringorten formula assign to the largest value?
3. Annual maximum floods have mean 380 m³ s⁻¹ and standard deviation 120 m³ s⁻¹. Find the 100-year flood by the Gumbel method of moments.
4. A site's mean annual flood is 180 m³ s⁻¹ and the regional 50-year growth factor is 2.2. Find the 50-year flood.
5. Peaks above 250 m³ s⁻¹ occur at 2.0 per year with exponential excesses of mean 90 m³ s⁻¹. Find the 50-year annual-maximum flood.

```{admonition} Answers
:class: dropdown
1. {glue:text}`o1:.2f` years.
2. {glue:text}`o2:.0f` years.
3. {glue:text}`o3:.0f` m³ s⁻¹.
4. {glue:text}`o4:.0f` m³ s⁻¹.
5. {glue:text}`o5:.0f` m³ s⁻¹.
```

## 8.13 Further reading

- England, J. F., Jr., et al. (2019). *Guidelines for Determining Flood Flow Frequency—Bulletin 17C*. U.S. Geological Survey Techniques and Methods, book 4, chapter B5.
- Hosking, J. R. M., and Wallis, J. R. (1997). *Regional Frequency Analysis: An Approach Based on L-Moments*. Cambridge University Press.
- Maity, R. (2022). *Statistical Methods in Hydrology and Hydroclimatology* (2nd ed.). Springer.
- McKee, T. B., Doesken, N. J., and Kleist, J. (1993). The relationship of drought frequency and duration to time scales. *Proceedings of the 8th Conference on Applied Climatology*, American Meteorological Society, 179–184.
- Rao, A. R., and Hamed, K. H. (2000). *Flood Frequency Analysis*. CRC Press.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
