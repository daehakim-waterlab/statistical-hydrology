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

# 13. Trend, seasonality and stationarity

```{admonition} Core ideas
:class: tip
1. A time series is **stationary** if its statistical properties do not change with time. Most methods of this book assume it, and hydrological series violate it through **trends**, **abrupt changes** and **seasonality**.
2. The **Mann–Kendall** test detects monotonic trends using ranks, and the **Theil–Sen** slope estimates their size. Positive **autocorrelation** inflates the false-detection rate unless it is accounted for.
3. **Change-point** tests such as Pettitt's locate abrupt shifts, for example after a dam or a gauge relocation.
4. **Seasonality** is described by monthly statistics or by a few **harmonics**; removing it leaves anomalies for further analysis.
5. **Spectral analysis** finds dominant periodicities; **wavelet analysis** shows how they change over time.
6. **Transformations** (logarithm, Box–Cox, normal scores) bring skewed data closer to normality before methods that assume it.
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
def mann_kendall(x):
    """Mann-Kendall S, Z and two-sided p-value (no tie correction), and the Theil-Sen slope."""
    x = np.asarray(x, float); n = len(x)
    d = x[None, :] - x[:, None]; iu = np.triu_indices(n, 1)
    S = np.sum(np.sign(d[iu])); varS = n * (n - 1) * (2 * n + 5) / 18
    Z = (S - np.sign(S)) / np.sqrt(varS) if S != 0 else 0.0
    slope = np.median(d[iu] / (iu[1] - iu[0]))
    return S, Z, 2 * stats.norm.sf(abs(Z)), slope
```

## 13.1 Stationarity and ergodicity

A **time series** is a sequence of observations ordered in time, $x_1, x_2, \dots, x_n$, regarded as one **realization** of a **stochastic process** $\{X_t\}$. The set of all realizations the process could have produced is its **ensemble**. Statistical properties are defined over the ensemble: the mean $E[X_t]$ at each time, the variance, and the covariance between values at different times.

A process is **strictly stationary** if its joint distributions are unchanged by a shift in time. It is **weakly** (second-order) **stationary** if its mean and variance are constant and the covariance between $X_t$ and $X_{t+k}$ depends only on the lag $k$. Hydrological series are rarely stationary in this sense at the daily or monthly scale, because of seasonality, but annual series, or seasonal series after the seasonal cycle has been removed, often are approximately stationary.

We observe only one realization. To estimate ensemble properties from it, we need **ergodicity**: time averages over a long record converge to ensemble averages. A stationary process with correlations that die out with increasing lag is ergodic for the mean. The assumption underlies every estimate of a long-term mean, variance or distribution from a single record, and it is why a trend or a shift matters: it means the record is not a sample from one fixed population.

## 13.2 Tests for randomness and trend

### Randomness

Before fitting a distribution, it is worth checking that a series has no structure in time. The **runs test** counts runs of consecutive values above or below the median; too few runs indicate persistence or a trend, too many alternation. The **turning point test** counts local peaks and troughs, whose expected number in a random series of length $n$ is $2(n - 2)/3$. The lag-1 autocorrelation coefficient (Chapter 14) is another simple check.

### The Mann–Kendall test

The most widely used trend test in hydrology is the rank-based **Mann–Kendall test** (Mann, 1945; Kendall, 1975). It counts, over all pairs of observations, how often the later value is larger than the earlier one:

\begin{equation*}
S = \sum_{i < j} \operatorname{sgn}(x_j - x_i), \qquad \operatorname{Var}(S) = \frac{n(n - 1)(2n + 5)}{18},
\tag{13.1}
\end{equation*}

where the variance applies to an independent series without ties. Under the null hypothesis of no trend, the standardized statistic $Z = (S - \operatorname{sgn} S)/\sqrt{\operatorname{Var}(S)}$ is approximately standard normal for $n \gtrsim 10$. The test detects any monotonic trend, linear or not, and is insensitive to outliers and to the distribution of the data. It is equivalent to testing Kendall's rank correlation between the series and time.

The magnitude of the trend is estimated by the **Theil–Sen slope** (Sen, 1968), the median of the slopes between all pairs of points:

\begin{equation*}
\hat\beta = \operatorname{median}_{i < j}\left(\frac{x_j - x_i}{j - i}\right) .
\tag{13.2}
\end{equation*}

It is far more robust than the least-squares slope when the data are skewed or contain outliers (Figure 13.1). A parametric alternative tests the slope of a linear regression on time (Chapter 11), which is more powerful for normal data with a linear trend but sensitive to outliers.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(131)
yrs = np.arange(1971, 2026); n = len(yrs)
x = 180 + 1.1 * (yrs - yrs[0]) + stats.gumbel_r.rvs(0, 22, n, random_state=rng) - 0.5772 * 22
x[3] += 220                                                               # an exceptional early flood
S, Z, p, sen = mann_kendall(x)
b_ls, a_ls = np.polyfit(yrs, x, 1)
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.plot(yrs, x, "o-", color=c1, ms=3.5, lw=1)
ax.plot(yrs, np.median(x - sen * (yrs - yrs.mean())) + sen * (yrs - yrs.mean()), color=c2, lw=2.2, label=f"Theil–Sen: {sen:.2f} per year")
ax.plot(yrs, a_ls + b_ls * yrs, color=ink2, lw=1.6, ls="--", label=f"Least squares: {b_ls:.2f} per year")
ax.set_xlabel("Year"); ax.set_ylabel("Annual maximum daily rainfall (mm)"); ax.legend(loc="upper left", fontsize=9.5)
plt.show()
glue("mkZ", float(Z), display=False); glue("mkp", float(p), display=False); glue("sen13", float(sen), display=False); glue("ls13", float(b_ls), display=False)
```

*Figure 13.1. A synthetic series of 55 annual maxima with a true trend of 1.1 mm per year and one exceptional early value. The Mann–Kendall test gives Z = {glue:text}`mkZ:.2f` (p = {glue:text}`mkp:.4f`). The Theil–Sen slope is {glue:text}`sen13:.2f` mm per year; the least-squares slope, {glue:text}`ls13:.2f` mm per year, is pulled further from the true value by the single outlier. Both carry sampling error: with this much year-to-year variability, even 55 years pin down a trend only roughly.*

### Autocorrelation and field significance

The variance in Equation (13.1) assumes independent values. With positive autocorrelation, common in annual streamflow because of groundwater and lake storage, the true variance of $S$ is larger, and the test rejects too often: a series with no trend but persistent fluctuations looks trended (Figure 13.2). Remedies include **prewhitening** (removing the estimated lag-1 autocorrelation before testing), and correcting the variance of $S$ using the autocorrelation of the ranks (Hamed and Rao, 1998). Both have limitations: prewhitening can remove part of a real trend, and long-term persistence is hard to distinguish from trend in a short record.

When many stations are tested, some will show significant trends by chance (Chapter 10), and nearby stations are correlated, so their tests are not independent. **Field significance** methods, often based on resampling the whole network together, test whether the number of significant stations exceeds what chance and cross-correlation would produce.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(132)
rhos = [0.0, 0.2, 0.4, 0.6]; n = 50; reps = 2000
rej_raw, rej_pw = [], []
for rho in rhos:
    rr, rp = 0, 0
    for _ in range(reps):
        e = rng.normal(size=n); y = np.zeros(n); y[0] = e[0] / np.sqrt(1 - rho ** 2) if rho < 1 else e[0]
        for t in range(1, n): y[t] = rho * y[t - 1] + e[t]
        rr += mann_kendall(y)[2] < 0.05
        r1 = np.corrcoef(y[:-1], y[1:])[0, 1]; rp += mann_kendall(y[1:] - r1 * y[:-1])[2] < 0.05
    rej_raw.append(rr / reps); rej_pw.append(rp / reps)
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.plot(rhos, rej_raw, "o-", color=c2, lw=2.2, label="Mann–Kendall on the series")
ax.plot(rhos, rej_pw, "s-", color=c1, lw=2.2, label="After prewhitening")
ax.axhline(0.05, color=ink, lw=1, ls="--"); ax.text(0.02, 0.06, "nominal 5%", color=ink2, fontsize=9.5)
ax.set_xlabel("Lag-1 autocorrelation of a trend-free series"); ax.set_ylabel("Rejection rate at α = 0.05")
ax.legend(loc="upper left", fontsize=9.5)
plt.show()
glue("rej06", float(100 * rej_raw[-1]), display=False)
```

*Figure 13.2. Fraction of 2000 trend-free synthetic series of 50 years in which the Mann–Kendall test finds a "significant" trend at the 5% level, as a function of lag-1 autocorrelation. With autocorrelation 0.6 the false-detection rate reaches {glue:text}`rej06:.0f`%; prewhitening restores it to near 5%.*

### Removing a trend

If a trend is confirmed and judged real (not an artifact of instrument change or data handling), it can be removed before further analysis by subtracting a fitted line or smooth curve, or the trend can be built into the model, for example as a time-varying location parameter in a nonstationary frequency analysis (Chapter 9).

## 13.3 Change points

Some changes are abrupt: a dam begins operation, a gauge is moved, a land-use change happens quickly. **Change-point** tests look for a single time at which the distribution shifts. The nonparametric **Pettitt test** (Pettitt, 1979) computes, for each possible change time $t$, the Mann–Whitney statistic comparing $x_1, \dots, x_t$ with $x_{t+1}, \dots, x_n$:

\begin{equation*}
U_t = \sum_{i=1}^{t}\sum_{j=t+1}^{n} \operatorname{sgn}(x_j - x_i), \qquad K = \max_t |U_t|, \qquad p \approx 2\exp\left(\frac{-6K^2}{n^3 + n^2}\right) .
\tag{13.3}
\end{equation*}

The time that maximizes $|U_t|$ is the most probable change point (Figure 13.3). Cumulative-sum (CUSUM) plots of deviations from the mean, and double-mass curves against a reference station (Chapter 16 of the [Physical Hydrology](https://hydroclimatol.com/physical-hydrology/chapters/ch16_models_uncertainty_and_change.html) book), are useful graphical companions. A trend test and a change-point test can both respond to either kind of change; the plot usually shows which is more plausible, and station metadata often explain abrupt shifts.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(133)
yrs2 = np.arange(1976, 2026); n2 = len(yrs2)
q = np.where(yrs2 < 2003, 42.0, 31.0) * np.exp(rng.normal(0, 0.18, n2))   # annual mean flow, drop after 2003
U = np.array([np.sum(np.sign(q[t + 1:][None, :] - q[:t + 1][:, None])) for t in range(n2 - 1)])
K = np.max(np.abs(U)); tcp = np.argmax(np.abs(U)); p_pet = min(1.0, 2 * np.exp(-6 * K ** 2 / (n2 ** 3 + n2 ** 2)))
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 4.6), sharex=True)
a1.plot(yrs2, q, "o-", color=c1, ms=3.5, lw=1)
a1.hlines([q[:tcp + 1].mean(), q[tcp + 1:].mean()], [yrs2[0], yrs2[tcp + 1]], [yrs2[tcp], yrs2[-1]], color=c2, lw=2.2)
a1.set_ylabel("Annual mean flow (m³ s⁻¹)")
a2.plot(yrs2[:-1], U, color=c3, lw=2.0); a2.axvline(yrs2[tcp], color=c2, lw=1.2, ls="--")
a2.set_xlabel("Year"); a2.set_ylabel("Pettitt $U_t$")
plt.show()
glue("cp_year", int(yrs2[tcp]), display=False); glue("cp_p", "< 0.001" if p_pet < 0.001 else f"= {p_pet:.3f}", display=False)
```

*Figure 13.3. Pettitt test on a synthetic series of annual mean flow that drops after 2003, for example because of upstream abstraction. The statistic $|U_t|$ peaks at {glue:text}`cp_year`, the last year before the drop (p {glue:text}`cp_p`); the orange lines are the means before and after.*

## 13.4 Seasonality and harmonic analysis

Within-year variation is the strongest signal in most daily and monthly hydrological series. The simplest description uses the 12 monthly means and standard deviations. A smoother and more economical description, useful for daily data, represents the seasonal mean by a few **harmonics**:

\begin{equation*}
\mu(t) = \bar x + \sum_{k=1}^{h}\left[A_k\cos\left(\frac{2\pi k t}{P}\right) + B_k\sin\left(\frac{2\pi k t}{P}\right)\right],
\tag{13.4}
\end{equation*}

with period $P$ = 365.25 days or 12 months. Because the sine and cosine terms are orthogonal over a full cycle, the coefficients are simple averages, $A_k = \frac{2}{N}\sum x_t\cos(2\pi kt/P)$ and $B_k = \frac{2}{N}\sum x_t\sin(2\pi kt/P)$, and each harmonic has amplitude $C_k = \sqrt{A_k^2 + B_k^2}$ and explains a share $C_k^2/(2s^2)$ of the variance. Two to four harmonics usually suffice for temperature and evaporation; rainfall and streamflow, with sharper seasonal peaks, may need more (Figure 13.4).

Seasonality is removed by **standardizing** each value by the seasonal mean and standard deviation, $z_t = [x_t - \mu(t)]/\sigma(t)$. The resulting anomalies are the starting point for trend tests on seasonal data and for the time-series models of Chapter 14.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(134)
days = np.arange(365 * 10); Pd = 365.25
true_mean = 2.2 + 1.8 * np.exp(-0.5 * ((days % 365 - 200) / 30) ** 2) * 3.0     # sharp summer peak in daily flow (mm/d)
z = np.zeros(len(days))
for t in range(1, len(days)): z[t] = 0.9 * z[t - 1] + np.sqrt(1 - 0.81) * rng.normal()
xq = true_mean * np.exp(0.35 * z - 0.06)
fig, ax = plt.subplots(figsize=(6.4, 3.6))
d1 = days < 365 * 3
ax.plot(days[d1] / 365, xq[d1], color=ink2, lw=0.8, label="Daily flow")
for h, col in zip([1, 4], [c2, c1]):
    fit = np.full(len(days), xq.mean())
    for k in range(1, h + 1):
        A = 2 / len(days) * np.sum(xq * np.cos(2 * np.pi * k * days / Pd)); B = 2 / len(days) * np.sum(xq * np.sin(2 * np.pi * k * days / Pd))
        fit += A * np.cos(2 * np.pi * k * days / Pd) + B * np.sin(2 * np.pi * k * days / Pd)
    ax.plot(days[d1] / 365, fit[d1], color=col, lw=2.2, label=f"{h} harmonic{'s' if h > 1 else ''}")
ax.set_xlabel("Year"); ax.set_ylabel("Flow (mm d⁻¹)"); ax.legend(loc="upper right", fontsize=9)
plt.show()
```

*Figure 13.4. Three years of synthetic daily streamflow with a sharp summer peak, and the seasonal mean represented by one and by four harmonics. A single harmonic cannot capture the peak's sharpness.*

## 13.5 Spectral and wavelet analysis

### The periodogram

**Spectral analysis** decomposes the variance of a series by frequency. The **periodogram** of a series of length $n$ at the Fourier frequencies $f_j = j/n$ is

\begin{equation*}
I(f_j) = \frac{1}{n}\left|\sum_{t=1}^{n} x_t\,e^{-2\pi i f_j t}\right|^2 ,
\tag{13.5}
\end{equation*}

computed with the fast Fourier transform. Peaks indicate periodicities, such as the annual cycle, or the 2–7-year band associated with El Niño. The raw periodogram is very noisy, so it is smoothed over neighboring frequencies or averaged over segments (Welch's method). Peaks must be judged against a suitable **null spectrum**: hydrological series often behave like **red noise**, an autoregressive process whose spectrum rises smoothly toward low frequencies, so a peak at low frequency is significant only if it rises above the red-noise level, not above a flat white-noise level.

### Wavelets

The periodogram assumes the periodicities persist through the whole record. **Wavelet analysis** shows how they change over time by correlating the series with a short, oscillating function, the **wavelet**, shifted along the record and stretched to different scales. The **Morlet wavelet**, a complex sinusoid within a Gaussian envelope, is commonly used. The squared magnitude of the transform, the **wavelet power**, is displayed as a function of time and period (Figure 13.5; Torrence and Compo, 1998). Near the ends of the record the wavelet extends beyond the data, so results within the **cone of influence** are unreliable. Simpler multiscale decompositions, such as the Haar wavelet, split a series into averages and differences at successively longer scales.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(135)
nm = 600; tt = np.arange(nm) / 12                                           # 50 years of monthly anomalies
sig = np.where(tt < 25, np.sin(2 * np.pi * tt / 3.5), 0.9 * np.sin(2 * np.pi * tt / 8.0))
e = np.zeros(nm)
for t in range(1, nm): e[t] = 0.5 * e[t - 1] + rng.normal(0, 0.6)
xw = sig + e
f = np.fft.rfftfreq(nm, d=1 / 12)[1:]; I = (np.abs(np.fft.rfft(xw - xw.mean())) ** 2 / nm)[1:]
periods = np.geomspace(1, 16, 60); w0 = 6.0
fx = np.fft.fft(xw - xw.mean()); om = 2 * np.pi * np.fft.fftfreq(nm, d=1 / 12)
power = np.empty((len(periods), nm))
for k, Tp in enumerate(periods):
    s_ = Tp * (w0 + np.sqrt(2 + w0 ** 2)) / (4 * np.pi)                     # scale for the Morlet period
    psi = np.pi ** -0.25 * np.exp(-0.5 * (s_ * om - w0) ** 2) * (om > 0) * np.sqrt(2 * np.pi * s_ * 12)
    power[k] = np.abs(np.fft.ifft(fx * psi)) ** 2 / s_
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 3.8), width_ratios=[1, 1.6])
a1.semilogx(1 / f, I, color=c1, lw=1.0); a1.set_xlim(0.5, 30); a1.set_xticks([1, 2, 4, 8, 16]); a1.set_xticklabels([1, 2, 4, 8, 16])
a1.set_xlabel("Period (years)"); a1.set_ylabel("Periodogram"); a1.set_title("(a) Periodogram")
pcm = a2.pcolormesh(tt, periods, power, cmap="Blues", shading="auto", vmax=np.percentile(power, 99.5))
coi = 1.033 / np.sqrt(2) * np.minimum(tt - tt[0] + 1 / 12, tt[-1] - tt + 1 / 12)   # Morlet cone of influence (period)
a2.fill_between(tt, np.clip(coi, 1, 16), 16, color=bg, alpha=0.65, lw=0)
a2.set_yscale("log"); a2.set_ylim(1, 16); a2.set_yticks([1, 2, 4, 8, 16]); a2.set_yticklabels([1, 2, 4, 8, 16])
a2.set_xlabel("Year"); a2.set_ylabel("Period (years)"); a2.set_title("(b) Wavelet power"); a2.grid(False)
plt.show()
```

*Figure 13.5. A synthetic 50-year monthly series with a 3.5-year oscillation in the first half and an 8-year oscillation in the second, plus red noise. (a) The periodogram shows both periods but not when they occur. (b) The Morlet wavelet power (darker is stronger) shows the switch at year 25. The pale area marks the approximate cone of influence, where edge effects matter.*

## 13.6 Transformations toward normality

Many methods (t tests, regression intervals, ARMA models) assume normality, while hydrological variables are usually skewed. Common transformations are:

- the **logarithm**, for positive, right-skewed data;
- the **Box–Cox** family (Box and Cox, 1964), $y = (x^\lambda - 1)/\lambda$ for $\lambda \ne 0$ and $\ln x$ for $\lambda = 0$, with $\lambda$ chosen by maximum likelihood; $\lambda = 0.5$ (square root) and $\lambda = 1/3$ (cube root) are often near-optimal for rainfall;
- the **normal score** transform, which maps each value through its empirical non-exceedance probability to the standard normal quantile, guaranteeing normal marginals.

Normality is checked with a normal probability plot and with tests such as **Shapiro–Wilk** or tests on the skewness and kurtosis. Back-transforming results requires care: the mean of transformed values does not back-transform to the mean of the original values (Chapter 11).

## 13.7 Lab (planned)

A Colab notebook will apply the Mann–Kendall and Pettitt tests to openly available streamflow and rainfall records, compare results with and without accounting for autocorrelation, and assess field significance across a network of stations.

## 13.8 Worked examples

**Example 13.1 — A Mann–Kendall test by hand.** Test the series 12, 15, 11, 18, 20, 17, 23, 22 for a monotonic trend.

```{code-cell} python
xs = np.array([12, 15, 11, 18, 20, 17, 23, 22.0])
S, Z, p, sen = mann_kendall(xs)
print(f"S = {S:.0f}, Var(S) = {len(xs) * (len(xs) - 1) * (2 * len(xs) + 5) / 18:.1f}, Z = {Z:.2f}, p = {p:.4f}, Sen slope = {sen:.2f} per step")
```

**Example 13.2 — Turning points.** How many turning points are expected in a random series of 40 values, and with what standard deviation, if the variance is $(16n - 29)/90$?

```{code-cell} python
n = 40
print(f"expected = {2 * (n - 2) / 3:.1f}, sd = {np.sqrt((16 * n - 29) / 90):.2f}")
```

**Example 13.3 — Harmonic coefficients.** Monthly mean temperatures (°C) for January to December are given below. Compute the amplitude of the first harmonic and the share of variance it explains.

```{code-cell} python
Tm = np.array([-1.5, 1.0, 6.2, 12.8, 18.0, 22.1, 25.4, 25.9, 21.0, 14.3, 7.4, 1.0])
mth = np.arange(1, 13)
A1 = 2 / 12 * np.sum(Tm * np.cos(2 * np.pi * mth / 12)); B1 = 2 / 12 * np.sum(Tm * np.sin(2 * np.pi * mth / 12))
C1 = np.hypot(A1, B1)
print(f"amplitude = {C1:.2f} °C; variance explained = {100 * C1 ** 2 / 2 / Tm.var():.1f}%")
```

**Example 13.4 — Box–Cox transformation.** Find the maximum-likelihood Box–Cox exponent for the annual maxima of Example 4.1 and compare the skewness before and after.

```{code-cell} python
xb = np.array([88, 132, 74, 96, 210, 105, 81, 119, 93, 147, 70, 101, 163, 90, 112.0])
yb, lam = stats.boxcox(xb)
print(f"lambda = {lam:.2f}; skewness {stats.skew(xb, bias=False):.2f} -> {stats.skew(yb, bias=False):.2f}")
```

## 13.9 Common mistakes

- **Testing autocorrelated series as if independent**, which exaggerates significance.
- **Reporting many significant stations** without assessing field significance.
- **Calling a step change a trend**, or a trend a step, without plotting the data and checking station history.
- **Comparing periodogram peaks with white noise** when the series is red.
- **Interpreting wavelet power inside the cone of influence.**

## 13.10 Exercises

```{code-cell} python
:tags: [remove-cell]
glue("y1", float(30 * 29 * 65 / 18), display=False)
glue("y2", float(2 * stats.norm.sf(2.1)), display=False)
glue("y3", float(np.median([2.0, 3.5, 1.0, -0.5, 2.5])), display=False)
glue("y4", float(2 * (25 - 2) / 3), display=False)
glue("y5", float(min(1, 2 * np.exp(-6 * 180 ** 2 / (40 ** 3 + 40 ** 2)))), display=False)
```

1. Find $\operatorname{Var}(S)$ of the Mann–Kendall statistic for 30 independent values without ties.
2. A Mann–Kendall test gives Z = 2.1. What is the two-sided p-value?
3. The pairwise slopes of a short series are 2.0, 3.5, 1.0, −0.5 and 2.5. What is the Theil–Sen slope?
4. How many turning points are expected in a random series of 25 values?
5. A Pettitt test on 40 years gives K = 180. Find the approximate p-value.

```{admonition} Answers
:class: dropdown
1. {glue:text}`y1:.1f`.
2. {glue:text}`y2:.3f`.
3. {glue:text}`y3:.1f`.
4. {glue:text}`y4:.1f`.
5. {glue:text}`y5:.3f`.
```

## 13.11 Further reading

- Hamed, K. H., and Rao, A. R. (1998). A modified Mann-Kendall trend test for autocorrelated data. *Journal of Hydrology*, 204, 182–196.
- Helsel, D. R., Hirsch, R. M., Ryberg, K. R., Archfield, S. A., and Gilroy, E. J. (2020). *Statistical Methods in Water Resources*. U.S. Geological Survey Techniques and Methods, book 4, chapter A3.
- Maity, R. (2022). *Statistical Methods in Hydrology and Hydroclimatology* (2nd ed.). Springer.
- Pettitt, A. N. (1979). A non-parametric approach to the change-point problem. *Applied Statistics*, 28, 126–135.
- Torrence, C., and Compo, G. P. (1998). A practical guide to wavelet analysis. *Bulletin of the American Meteorological Society*, 79, 61–78.
- Yue, S., Pilon, P., Phinney, B., and Cavadias, G. (2002). The influence of autocorrelation on the ability to detect trend in hydrological series. *Hydrological Processes*, 16, 1807–1829.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
