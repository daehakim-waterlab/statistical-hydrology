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

# 14. Stochastic models of hydrological time series

```{admonition} Core ideas
:class: tip
1. Successive values of hydrological series are correlated. The **autocorrelation** and **partial autocorrelation** functions describe this memory and guide the choice of model.
2. **Autoregressive** (AR) models express the present value as a weighted sum of past values plus noise; **moving-average** (MA) models as a weighted sum of past noise. **ARMA** combines them, **ARIMA** adds differencing for nonstationary series, and **ARMAX** adds external inputs such as rainfall.
3. Models are built by the Box–Jenkins cycle: **identify** a candidate from the correlation functions, **estimate** its parameters, and **check** that its residuals are white noise, preferring the most parsimonious adequate model.
4. ARMA models give **forecasts** whose uncertainty grows with lead time toward the unconditional variance.
5. Seasonal models such as **Thomas–Fiering** generate long **synthetic sequences** that preserve the statistics of the record, which are used to evaluate reservoirs and water systems under droughts not seen in the record.
```

```{code-cell} python
:tags: [remove-cell]
# Plot style shared by all figures in this chapter.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize
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
def acf(x, nlag):
    """Sample autocorrelation r_0..r_nlag."""
    x = np.asarray(x, float) - np.mean(x); c0 = x @ x / len(x)
    return np.array([1.0] + [(x[:-k] @ x[k:]) / len(x) / c0 for k in range(1, nlag + 1)])
def pacf(x, nlag):
    """Sample partial autocorrelation by the Durbin-Levinson recursion."""
    r = acf(x, nlag); phi = np.zeros((nlag + 1, nlag + 1)); out = [1.0]
    for k in range(1, nlag + 1):
        phi[k, k] = (r[k] - phi[k - 1, 1:k] @ r[1:k][::-1]) / (1 - phi[k - 1, 1:k] @ r[1:k])
        phi[k, 1:k] = phi[k - 1, 1:k] - phi[k, k] * phi[k - 1, 1:k][::-1]
        out.append(phi[k, k])
    return np.array(out)
def simulate_arma(phi, theta, n, rng, sigma=1.0, burn=200):
    """Simulate x_t = sum phi_i x_{t-i} + e_t + sum theta_j e_{t-j}."""
    p, q = len(phi), len(theta); e = rng.normal(0, sigma, n + burn); x = np.zeros(n + burn)
    for t in range(max(p, q), n + burn):
        x[t] = sum(phi[i] * x[t - 1 - i] for i in range(p)) + e[t] + sum(theta[j] * e[t - 1 - j] for j in range(q))
    return x[burn:]
```

## 14.1 Autocorrelation and partial autocorrelation

The memory of a stationary series is described by its **autocorrelation function** (ACF), the correlation between values $k$ steps apart:

\begin{equation*}
\rho_k = \frac{\operatorname{Cov}(X_t, X_{t+k})}{\operatorname{Var}(X_t)}, \qquad r_k = \frac{\sum_{t=1}^{n-k}(x_t - \bar x)(x_{t+k} - \bar x)}{\sum_{t=1}^{n}(x_t - \bar x)^2} .
\tag{14.1}
\end{equation*}

For an independent series, the sample autocorrelations $r_k$ are approximately normal with mean zero and standard deviation $1/\sqrt n$, so values outside $\pm 1.96/\sqrt n$ suggest real correlation. Daily streamflow has autocorrelations that decay slowly over weeks; annual flow often has a lag-1 autocorrelation of 0.1–0.4 in catchments with large groundwater or lake storage, and near zero elsewhere.

The **partial autocorrelation function** (PACF) $\phi_{kk}$ is the correlation between $X_t$ and $X_{t+k}$ after removing the linear effect of the intermediate values. It is computed from the ACF by the Durbin–Levinson recursion. The two functions together identify simple models (Figure 14.1 and Table 14.1): an AR($p$) process has a PACF that cuts off after lag $p$ and an ACF that decays; an MA($q$) process has an ACF that cuts off after lag $q$ and a PACF that decays.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(141)
cases = [("AR(1), φ = 0.7", [0.7], []), ("AR(2), φ = (0.5, 0.3)", [0.5, 0.3], []), ("MA(1), θ = 0.7", [], [0.7])]
n, L = 500, 12; lim = 1.96 / np.sqrt(n)
fig, axs = plt.subplots(2, 3, figsize=(6.4, 4.2), sharey=True)
for j, (lab, ph, th) in enumerate(cases):
    x = simulate_arma(ph, th, n, rng)
    for i, (fn, name) in enumerate([(acf, "ACF"), (pacf, "PACF")]):
        v = fn(x, L); ax = axs[i, j]
        ax.vlines(np.arange(1, L + 1), 0, v[1:], color=c1 if i == 0 else c2, lw=2.5)
        ax.axhspan(-lim, lim, color=ink2, alpha=0.12, lw=0); ax.axhline(0, color=ink2, lw=0.8)
        ax.set_ylim(-0.4, 1.0); ax.set_xticks([1, 4, 8, 12])
        if i == 0: ax.set_title(lab, fontsize=9.5)
        if j == 0: ax.set_ylabel(name)
        if i == 1: ax.set_xlabel("Lag")
plt.show()
```

*Figure 14.1. Sample ACF (top) and PACF (bottom) of 500 simulated values from three models. The gray band is $\pm 1.96/\sqrt n$. AR models have a decaying ACF and a PACF that cuts off after lag $p$; the MA(1) model has the opposite pattern.*

## 14.2 Autoregressive and moving-average models

### Autoregressive models

The **first-order autoregressive** model, AR(1), for a series of deviations from the mean, $z_t = x_t - \mu$, is

\begin{equation*}
z_t = \phi\,z_{t-1} + \varepsilon_t ,
\tag{14.2}
\end{equation*}

where $\varepsilon_t$ is **white noise**: independent, with mean zero and variance $\sigma_\varepsilon^2$. It is stationary if $|\phi| < 1$. Multiplying by $z_{t-k}$ and taking expectations shows that its ACF decays geometrically, $\rho_k = \phi^k$, and its variance is $\sigma_z^2 = \sigma_\varepsilon^2/(1 - \phi^2)$. The AR(1) model is the simplest description of storage-driven persistence: it is the discrete form of a linear reservoir driven by random inputs.

The AR($p$) model, $z_t = \phi_1 z_{t-1} + \dots + \phi_p z_{t-p} + \varepsilon_t$, has parameters related to the ACF by the **Yule–Walker equations** $\rho_k = \sum_{i=1}^{p}\phi_i\rho_{k-i}$ for $k = 1, \dots, p$, which give moment estimates of the $\phi_i$ from the sample ACF.

### Moving-average models

The **moving-average** model MA($q$) writes the series as a weighted sum of current and past shocks:

\begin{equation*}
z_t = \varepsilon_t + \theta_1\varepsilon_{t-1} + \dots + \theta_q\varepsilon_{t-q} .
\tag{14.3}
\end{equation*}

It is always stationary, its memory is exactly $q$ steps, and for MA(1), $\rho_1 = \theta/(1 + \theta^2)$, which can never exceed 0.5 in magnitude. An MA model is **invertible**, expressible as an infinite AR model, if $|\theta| < 1$ for MA(1); invertibility makes the parameterization unique. Aggregation and some physical routing processes produce MA-like behavior.

With the **backshift operator** $B z_t = z_{t-1}$, models are written compactly: AR(1) is $(1 - \phi B)z_t = \varepsilon_t$ and MA(1) is $z_t = (1 + \theta B)\varepsilon_t$.

## 14.3 ARMA, ARIMA and models with exogenous inputs

The **ARMA**($p$, $q$) model combines both parts:

\begin{equation*}
\phi(B)\,z_t = \theta(B)\,\varepsilon_t, \qquad \phi(B) = 1 - \phi_1 B - \dots - \phi_p B^p, \quad \theta(B) = 1 + \theta_1 B + \dots + \theta_q B^q .
\tag{14.4}
\end{equation*}

ARMA(1,1) is often adequate for annual streamflow: it can reproduce a small lag-1 correlation followed by slow decay, a pattern that reflects both a fast and a slow storage component and that an AR(1) model with the same lag-1 correlation cannot.

Nonstationary series with a stochastic trend are made stationary by **differencing**, $w_t = (1 - B)^d x_t$; an ARMA model for $w_t$ is an **ARIMA**($p$, $d$, $q$) model of $x_t$. **Seasonal** ARIMA models add AR and MA terms at the seasonal lag (12 months) and seasonal differencing, though in hydrology it is usually better to remove seasonality by standardizing each month (Chapter 13) and model the anomalies.

When an external driver $u_t$ is known, such as rainfall driving streamflow, the **ARMAX** model adds a distributed-lag term:

\begin{equation*}
\phi(B)\,z_t = \omega(B)\,u_t + \theta(B)\,\varepsilon_t ,
\tag{14.5}
\end{equation*}

a transfer-function model closely related to the unit hydrograph. The noise model then describes the errors of the input–output relation rather than the flow itself.

## 14.4 Identification, estimation and diagnostic checking

Box and Jenkins (1970) proposed an iterative cycle:

1. **Identification**: transform the series to stationarity (remove seasonality, difference if necessary, transform toward normality) and choose candidate orders from the ACF and PACF (Table 14.1).
2. **Estimation**: estimate the parameters by least squares or maximum likelihood. AR models can be fitted by ordinary regression on lagged values; models with MA terms require numerical optimization.
3. **Diagnostic checking**: the residuals of an adequate model should be white noise. Their ACF should lie within the confidence band, and the **Ljung–Box** statistic
\begin{equation*}
Q = n(n + 2)\sum_{k=1}^{m}\frac{r_k^2(\hat\varepsilon)}{n - k}
\tag{14.6}
\end{equation*}
should be consistent with a chi-square distribution with $m - p - q$ degrees of freedom (Ljung and Box, 1978).

Among adequate models, the most **parsimonious** is preferred: extra parameters fit the record better but forecast and simulate worse. Information criteria formalize the trade-off, $\text{AIC} = n\ln\hat\sigma_\varepsilon^2 + 2(p + q)$ and the stricter $\text{BIC} = n\ln\hat\sigma_\varepsilon^2 + (p + q)\ln n$.

```{code-cell} python
:tags: [remove-input]
Markdown("| Model | ACF | PACF |\n|---|---|---|\n"
         "| AR($p$) | decays (exponential or damped oscillation) | cuts off after lag $p$ |\n"
         "| MA($q$) | cuts off after lag $q$ | decays |\n"
         "| ARMA($p$, $q$) | decays after lag $q - p$ | decays after lag $p - q$ |\n"
         "| White noise | no significant values | no significant values |\n"
         "| Nonstationary (trend, random walk) | decays very slowly | large at lag 1 |")
```

*Table 14.1. Patterns of the ACF and PACF used to identify ARMA models.*

## 14.5 Forecasting

The minimum mean-squared-error forecast of a future value is its conditional expectation given the past. For AR(1), the forecast $h$ steps ahead from time $t$ and its error variance are

\begin{equation*}
\hat z_{t+h} = \phi^h z_t, \qquad \operatorname{Var}(z_{t+h} - \hat z_{t+h}) = \sigma_\varepsilon^2\,\frac{1 - \phi^{2h}}{1 - \phi^2} .
\tag{14.7}
\end{equation*}

Forecasts relax toward the mean, and the error variance grows toward the unconditional variance $\sigma_z^2$: beyond a few times the memory of the series, the forecast is simply the climatological mean. For a transformed variable, such as log flow, the forecast and its interval are back-transformed with the bias considerations of Chapter 11. Figure 14.2 forecasts monthly flow anomalies with a fitted AR(1) model.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(142)
z = simulate_arma([0.65], [], 300, rng, sigma=0.75)
x_lag, x_now = z[:-1], z[1:]
phi_hat = (x_lag @ x_now) / (x_lag @ x_lag); res = x_now - phi_hat * x_lag; s_e = res.std(ddof=1)
H = 12; t0 = 240; h = np.arange(1, H + 1)
fc = phi_hat ** h * z[t0]; se = s_e * np.sqrt((1 - phi_hat ** (2 * h)) / (1 - phi_hat ** 2))
fig, ax = plt.subplots(figsize=(6.4, 3.9))
ax.plot(np.arange(t0 - 36, t0 + 1), z[t0 - 36:t0 + 1], color=c1, lw=1.6, label="Observed anomalies")
ax.plot(t0 + h, z[t0 + 1:t0 + H + 1], "o", color=ink2, ms=3.5, label="What happened next")
ax.fill_between(t0 + h, fc - 1.96 * se, fc + 1.96 * se, color=c2, alpha=0.2, lw=0, label="95% interval")
ax.plot(np.r_[t0, t0 + h], np.r_[z[t0], fc], color=c2, lw=2.2, label="AR(1) forecast")
ax.axhline(0, color=ink2, lw=0.8)
ax.set_xlabel("Month"); ax.set_ylabel("Standardized anomaly")
fig.legend(loc="outside lower center", ncol=4, fontsize=8.5, handlelength=1.5, columnspacing=1.0)
plt.show()
lb_m = 10; rr = acf(res, lb_m)[1:]; nres = len(res)
Qlb = nres * (nres + 2) * np.sum(rr ** 2 / (nres - np.arange(1, lb_m + 1)))
glue("phi_hat", float(phi_hat), display=False); glue("Qlb", float(Qlb), display=False); glue("Qlb_p", float(stats.chi2.sf(Qlb, lb_m - 1)), display=False)
```

*Figure 14.2. Forecasts of monthly flow anomalies up to 12 months ahead from an AR(1) model fitted to 300 synthetic months ($\hat\phi$ = {glue:text}`phi_hat:.2f`). The forecast decays toward zero and the interval widens toward the unconditional spread. The Ljung–Box test on the residuals (10 lags) gives Q = {glue:text}`Qlb:.1f`, p = {glue:text}`Qlb_p:.2f`, consistent with white noise.*

## 14.6 Synthetic streamflow generation

Reservoirs and water supply systems must be designed for droughts longer or more severe than any in the short historical record. **Synthetic sequences**, many long series generated by a stochastic model that reproduces the statistics of the record, show the range of droughts the system could face.

For monthly flows, the classic **Thomas–Fiering** model (Thomas and Fiering, 1962) is a seasonal AR(1) model with parameters that vary by month. Working with log flows $y$ to handle skewness, the value in month $m$ is generated from the previous month as

\begin{equation*}
y_m = \mu_m + b_m\,(y_{m-1} - \mu_{m-1}) + \sigma_m\sqrt{1 - r_m^2}\;\varepsilon, \qquad b_m = r_m\frac{\sigma_m}{\sigma_{m-1}},
\tag{14.8}
\end{equation*}

where $\mu_m$ and $\sigma_m$ are the mean and standard deviation of month $m$, $r_m$ the correlation between months $m - 1$ and $m$, and $\varepsilon$ standard normal. The model preserves the monthly means, standard deviations and month-to-month correlations. It does not, by construction, preserve longer-term persistence or the variability of annual totals, for which disaggregation models and models with long memory have been developed.

Figure 14.3 fits the model to 40 years of synthetic "historical" monthly flow and generates 200 sequences of the same length. For each, the storage needed to supply a constant demand is computed by the **sequent peak** algorithm, the running maximum of accumulated deficits. The historical record gives one storage value; the synthetic sequences give a distribution, from which a storage with a chosen reliability can be read.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(143)
ny = 40; nm = ny * 12; mon = np.arange(nm) % 12
mu_true = np.log(20 + 120 * np.exp(-0.5 * ((np.arange(12) - 6.5) / 1.3) ** 2)); sd_true = 0.35 + 0.25 * np.exp(-0.5 * ((np.arange(12) - 6.5) / 1.5) ** 2)
yh = np.zeros(nm); zz = 0.0
for t in range(nm):
    zz = 0.55 * zz + np.sqrt(1 - 0.55 ** 2) * rng.normal(); yh[t] = mu_true[mon[t]] + sd_true[mon[t]] * zz
Qh = np.exp(yh)                                                            # "historical" monthly flow (million m3)
Y = yh.reshape(ny, 12); mu_m, sd_m = Y.mean(0), Y.std(0, ddof=1)
r_m = np.array([np.corrcoef(Y[1:, 0], Y[:-1, 11])[0, 1]] + [np.corrcoef(Y[:, m], Y[:, m - 1])[0, 1] for m in range(1, 12)])
def thomas_fiering(nyears, rng):
    y = np.zeros(nyears * 12); last = mu_m[11]
    for t in range(nyears * 12):
        m = t % 12; pm = (m - 1) % 12
        b = r_m[m] * sd_m[m] / sd_m[pm]
        y[t] = mu_m[m] + b * (last - mu_m[pm]) + sd_m[m] * np.sqrt(1 - r_m[m] ** 2) * rng.normal(); last = y[t]
    return np.exp(y)
def sequent_peak(Q, D):
    K = 0.0; Kmax = 0.0
    for q in Q: K = max(0.0, K + D - q); Kmax = max(Kmax, K)
    return Kmax
D = 0.75 * Qh.mean()
K_hist = sequent_peak(Qh, D)
syn = [thomas_fiering(ny, rng) for _ in range(200)]
K_syn = np.array([sequent_peak(s, D) for s in syn])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 3.5))
Qs_m = np.array([s.reshape(ny, 12).mean(0) for s in syn])
a1.fill_between(np.arange(1, 13), np.percentile(Qs_m, 5, 0), np.percentile(Qs_m, 95, 0), color=c1, alpha=0.25, lw=0, label="Synthetic 5–95%")
a1.plot(np.arange(1, 13), Qh.reshape(ny, 12).mean(0), "o-", color=ink, ms=4, lw=1.4, label="Historical")
a1.set_xticks([1, 3, 5, 7, 9, 11]); a1.set_xlabel("Month"); a1.set_ylabel("Mean flow (10⁶ m³)"); a1.set_title("(a) Monthly means"); a1.legend(loc="upper left", fontsize=8.5)
a2.hist(K_syn, bins=25, color=c1, alpha=0.7, edgecolor=bg)
a2.axvline(K_hist, color=c2, lw=2.2, label="Historical record")
a2.axvline(np.percentile(K_syn, 90), color=ink, lw=1.4, ls="--", label="90th percentile")
a2.set_xlabel("Required storage (10⁶ m³)"); a2.set_title("(b) Storage for 75% of mean flow"); a2.legend(loc="upper right", fontsize=8.5)
a2.set_yticks([])
plt.show()
glue("K_hist", float(K_hist), display=False); glue("K_90", float(np.percentile(K_syn, 90)), display=False)
glue("K_exc", float(100 * np.mean(K_syn > K_hist)), display=False)
```

*Figure 14.3. Thomas–Fiering generation of monthly flows. (a) The 200 synthetic 40-year sequences reproduce the monthly means of the historical record. (b) The storage needed to supply 75% of the mean flow, from the sequent-peak algorithm: the historical record requires {glue:text}`K_hist:.0f` million m³, but {glue:text}`K_exc:.0f`% of the synthetic sequences need more, and the 90th percentile is {glue:text}`K_90:.0f` million m³.*

## 14.7 Lab (planned)

A Colab notebook will identify, fit and check ARMA models for an openly available annual and monthly streamflow record, generate synthetic sequences with a seasonal model, and evaluate the storage–yield relation of a reservoir.

## 14.8 Worked examples

**Example 14.1 — AR(1) properties.** Annual flow anomalies follow an AR(1) model with φ = 0.35 and noise standard deviation 120 mm. Find the standard deviation of annual flow and the correlation between flows three years apart.

```{code-cell} python
phi, se_ = 0.35, 120.0
print(f"sigma_z = {se_ / np.sqrt(1 - phi ** 2):.0f} mm; rho_3 = {phi ** 3:.3f}")
```

**Example 14.2 — Yule–Walker for AR(2).** A series has $r_1$ = 0.62 and $r_2$ = 0.45. Estimate the AR(2) coefficients.

```{code-cell} python
r1, r2 = 0.62, 0.45
A = np.array([[1, r1], [r1, 1]]); phi12 = np.linalg.solve(A, [r1, r2])
print(f"phi1 = {phi12[0]:.3f}, phi2 = {phi12[1]:.3f}")
```

**Example 14.3 — Forecast intervals.** For the model of Example 14.1 with a current anomaly of +250 mm, give the forecasts and 95% intervals for one and five years ahead.

```{code-cell} python
for h in (1, 5):
    f_ = phi ** h * 250; s_ = se_ * np.sqrt((1 - phi ** (2 * h)) / (1 - phi ** 2))
    print(f"h = {h}: forecast {f_:+.0f} mm, 95% interval {f_ - 1.96 * s_:+.0f} to {f_ + 1.96 * s_:+.0f} mm")
```

**Example 14.4 — Sequent peak.** Monthly inflows (million m³) over one dry period are listed below. What storage is needed to supply 10 million m³ per month without shortage?

```{code-cell} python
q = np.array([14, 9, 6, 4, 3, 5, 8, 12, 18, 7, 5, 9.0])
print(f"required storage = {sequent_peak(q, 10.0):.0f} million m3")
```

## 14.9 Common mistakes

- **Fitting ARMA models to seasonal or trending series** without first removing seasonality or trend.
- **Choosing high-order models** that fit the record but forecast poorly; check parsimony with AIC or BIC.
- **Skipping residual checks**; correlated residuals mean the model has missed structure.
- **Expecting skill far beyond the memory of the series**; long-range forecasts tend to the mean.
- **Designing storage from the historical record alone**; it is one sample of possible droughts.

## 14.10 Exercises

```{code-cell} python
:tags: [remove-cell]
glue("x1", float(0.6 ** 2), display=False)
glue("x2", float(0.5 / (1 + 0.5 ** 2)), display=False)
glue("x3", float(1.5 / np.sqrt(1 - 0.8 ** 2)), display=False)
glue("x4", float(1.96 / np.sqrt(64)), display=False)
qq = np.array([6, 4, 3, 8, 12.0]); glue("x5", float(sequent_peak(qq, 7.0)), display=False)
```

1. An AR(1) process has φ = 0.6. What is its lag-2 autocorrelation?
2. What is the lag-1 autocorrelation of an MA(1) process with θ = 0.5?
3. An AR(1) process has φ = 0.8 and noise standard deviation 1.5. Find the standard deviation of the process.
4. What are the approximate 95% limits for the sample ACF of 64 independent values?
5. Inflows of 6, 4, 3, 8 and 12 units must meet a demand of 7 units per step. Find the required storage.

```{admonition} Answers
:class: dropdown
1. {glue:text}`x1:.2f`.
2. {glue:text}`x2:.2f`.
3. {glue:text}`x3:.2f`.
4. ±{glue:text}`x4:.3f`.
5. {glue:text}`x5:.0f` units.
```

## 14.11 Further reading

- Box, G. E. P., and Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control*. Holden-Day.
- Ljung, G. M., and Box, G. E. P. (1978). On a measure of lack of fit in time series models. *Biometrika*, 65, 297–303.
- Maity, R. (2022). *Statistical Methods in Hydrology and Hydroclimatology* (2nd ed.). Springer.
- Salas, J. D., Delleur, J. W., Yevjevich, V., and Lane, W. L. (1980). *Applied Modeling of Hydrologic Time Series*. Water Resources Publications.
- Thomas, H. A., and Fiering, M. B. (1962). Mathematical synthesis of streamflow sequences for the analysis of river basins by simulation. In Maass, A., et al., *Design of Water-Resource Systems*, Harvard University Press, 459–493.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
