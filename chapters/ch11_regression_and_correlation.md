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

# 11. Regression and correlation

```{admonition} Core ideas
:class: tip
1. **Regression** describes how the mean of one variable changes with others. Least squares gives the line, or plane, that minimizes the squared errors.
2. Many hydrological relations are power laws, which become linear after taking **logarithms**. Back-transforming predictions requires a bias correction.
3. **Multiple regression** uses several explanatory variables; it is the standard method for estimating flood statistics at ungauged sites from catchment characteristics.
4. A regression is not finished until its **residuals** have been examined for curvature, unequal spread, non-normality and influential points.
5. **Confidence intervals** describe the uncertainty of the fitted mean; **prediction intervals**, much wider, describe the uncertainty of a new observation.
6. Correlation does not imply causation. Confounding variables, common trends and variables built from shared terms can produce **spurious correlation**.
```

```{code-cell} python
:tags: [remove-cell]
# Plot style shared by all figures in this chapter.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
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
# Synthetic regional data set used throughout the chapter: 40 gauged catchments
rng = np.random.default_rng(111)
ns = 40
area = np.exp(rng.uniform(np.log(15), np.log(3000), ns))                    # km2
rain = rng.uniform(1100, 1700, ns)                                          # mean annual precipitation (mm)
slope = np.exp(rng.normal(np.log(0.08), 0.5, ns)) * (area / 300) ** -0.15   # mean slope (m/m), smaller in large basins
lnQ = np.log(0.9) + 0.72 * np.log(area) + 1.1 * np.log(rain / 1400) + 0.25 * np.log(slope / 0.08) + rng.normal(0, 0.30, ns)
Qbar = np.exp(lnQ)                                                          # mean annual flood (m3/s)
```

## 11.1 Simple linear regression

### Least squares

Regression models the relation between a **response** $y$ and an **explanatory variable** $x$ as

\begin{equation*}
y_i = a + b\,x_i + \varepsilon_i ,
\tag{11.1}
\end{equation*}

where the errors $\varepsilon_i$ are independent, with mean zero and constant variance $\sigma^2$. The **least-squares** estimates minimize the sum of squared residuals $\sum (y_i - a - bx_i)^2$. Setting its derivatives with respect to $a$ and $b$ to zero gives

\begin{equation*}
\hat b = \frac{S_{xy}}{S_{xx}} = \frac{\sum (x_i - \bar x)(y_i - \bar y)}{\sum (x_i - \bar x)^2}, \qquad \hat a = \bar y - \hat b\,\bar x ,
\tag{11.2}
\end{equation*}

so the line passes through the point of means. The residual variance is estimated by $s^2 = \sum e_i^2/(n - 2)$, with residuals $e_i = y_i - \hat y_i$, and the **coefficient of determination**

\begin{equation*}
R^2 = 1 - \frac{\sum e_i^2}{\sum (y_i - \bar y)^2}
\tag{11.3}
\end{equation*}

is the fraction of the variance of $y$ explained by the regression; in simple regression it equals the squared correlation coefficient $r^2$.

### Inference on the slope

If the errors are also normal, $(\hat b - b)/\operatorname{SE}(\hat b)$ follows a t distribution with $n - 2$ degrees of freedom, where $\operatorname{SE}(\hat b) = s/\sqrt{S_{xx}}$. This gives confidence intervals for the slope and a test of $H_0: b = 0$, that is, of whether $y$ depends linearly on $x$ at all.

Note that regressing $y$ on $x$ and $x$ on $y$ give different lines, because each minimizes errors in a different direction. The regression of $y$ on $x$ is the right one for predicting $y$ from $x$.

## 11.2 Transformations and nonlinear models

### Log transformations

Hydrological relations are often power laws: the mean annual flood grows roughly as a power of catchment area, sediment load as a power of discharge, discharge as a power of stage. Taking logarithms of $y = \alpha x^\beta$ gives the linear model $\ln y = \ln\alpha + \beta\ln x$, which least squares can fit; the errors become multiplicative, which usually matches the data better, since larger catchments have larger absolute scatter. Figure 11.1 fits the mean annual flood of 40 synthetic catchments against their area on logarithmic axes.

Back-transforming a prediction made in log space gives an estimate of the **median**, not the mean, of $y$. If the log-space residuals are normal with variance $s^2$, the mean is larger by the factor $e^{s^2/2}$; for non-normal residuals the **smearing estimator** (Duan, 1983), the average of $e^{e_i}$, is used. Ignoring this correction underestimates mean loads and volumes systematically.

### Nonlinear least squares

Some models cannot be linearized, such as a rating curve $Q = a(h - h_0)^b$ with an unknown zero-flow stage $h_0$. They are fitted by **nonlinear least squares**, minimizing the squared errors numerically from a starting guess (`scipy.optimize.curve_fit`; Example 11.4). The errors are then additive in the original units unless the model is fitted to logarithms.

```{code-cell} python
:tags: [hide-input]
x, y = np.log(area), np.log(Qbar)
n = len(x); xb = x.mean(); Sxx = np.sum((x - xb) ** 2)
b = np.sum((x - xb) * (y - y.mean())) / Sxx; a = y.mean() - b * xb
e = y - (a + b * x); s = np.sqrt(np.sum(e ** 2) / (n - 2)); R2 = 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)
x0 = np.linspace(np.log(10), np.log(4000), 200); tq = stats.t.ppf(0.95, n - 2)
ci = tq * s * np.sqrt(1 / n + (x0 - xb) ** 2 / Sxx); pi = tq * s * np.sqrt(1 + 1 / n + (x0 - xb) ** 2 / Sxx)
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.fill_between(np.exp(x0), np.exp(a + b * x0 - pi), np.exp(a + b * x0 + pi), color=c1, alpha=0.12, lw=0, label="90% prediction interval")
ax.fill_between(np.exp(x0), np.exp(a + b * x0 - ci), np.exp(a + b * x0 + ci), color=c1, alpha=0.35, lw=0, label="90% confidence interval")
ax.plot(np.exp(x0), np.exp(a + b * x0), color=c1, lw=2.2, label="Fitted median")
ax.plot(area, Qbar, "o", color=ink, ms=4.5, label="Gauged catchments (synthetic)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Catchment area (km²)"); ax.set_ylabel("Mean annual flood (m³ s⁻¹)"); ax.legend(loc="upper left", fontsize=9)
plt.show()
se_b = s / np.sqrt(Sxx)
glue("b11", float(b), display=False); glue("seb11", float(se_b), display=False); glue("R211", float(R2), display=False)
glue("alpha11", float(np.exp(a)), display=False); glue("s11", float(s), display=False)
glue("pi_factor", float(np.exp(tq * s * np.sqrt(1 + 1 / n))), display=False)
```

*Figure 11.1. Mean annual flood against catchment area for 40 synthetic catchments, fitted in log space: $\hat Q$ = {glue:text}`alpha11:.2f` $A^{\hat b}$ with $\hat b$ = {glue:text}`b11:.3f` ± {glue:text}`seb11:.3f` (standard error) and $R^2$ = {glue:text}`R211:.2f`. The prediction interval for a new catchment spans a factor of about {glue:text}`pi_factor:.1f` on either side of the fitted value.*

## 11.3 Multiple linear regression

With several explanatory variables, the model is $y_i = \beta_0 + \beta_1 x_{i1} + \dots + \beta_p x_{ip} + \varepsilon_i$, or in matrix form $\mathbf y = \mathbf X\boldsymbol\beta + \boldsymbol\varepsilon$, where the first column of $\mathbf X$ is ones. The least-squares estimate is

\begin{equation*}
\hat{\boldsymbol\beta} = (\mathbf X^\mathsf{T}\mathbf X)^{-1}\mathbf X^\mathsf{T}\mathbf y, \qquad \operatorname{Cov}(\hat{\boldsymbol\beta}) = s^2(\mathbf X^\mathsf{T}\mathbf X)^{-1},
\tag{11.4}
\end{equation*}

with $s^2 = \sum e_i^2/(n - p - 1)$. Each coefficient measures the effect of its variable with the others held fixed. Adding variables always increases $R^2$, so models are compared with the **adjusted** $R^2 = 1 - (1 - R^2)(n - 1)/(n - p - 1)$, with information criteria, or, best, by **cross-validation**: refitting the model without each catchment in turn and predicting it.

When explanatory variables are strongly correlated with each other (**multicollinearity**), their coefficients become unstable and can take implausible signs, although predictions may remain good. The **variance inflation factor** $\text{VIF}_j = 1/(1 - R_j^2)$, where $R_j^2$ is from regressing $x_j$ on the other variables, measures it; values above about 5–10 signal trouble.

Regional regression of flood statistics on catchment characteristics (area, mean annual precipitation, slope, soils, land cover) is a standard way to estimate the index flood at ungauged sites (Chapter 8). Table 11.1 fits such a model to the synthetic catchments.

```{code-cell} python
:tags: [remove-input]
X = np.column_stack([np.ones(ns), np.log(area), np.log(rain), np.log(slope)])
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
res = y - X @ beta; p = X.shape[1] - 1
s2 = res @ res / (ns - p - 1); cov = s2 * np.linalg.inv(X.T @ X); se = np.sqrt(np.diag(cov))
R2m = 1 - res @ res / np.sum((y - y.mean()) ** 2); R2adj = 1 - (1 - R2m) * (ns - 1) / (ns - p - 1)
def vif(j):
    Z = np.delete(X[:, 1:], j - 1, axis=1); Z = np.column_stack([np.ones(ns), Z])
    g, *_ = np.linalg.lstsq(Z, X[:, j], rcond=None); r = X[:, j] - Z @ g
    return 1 / (r @ r / np.sum((X[:, j] - X[:, j].mean()) ** 2))
names = ["Intercept", "ln(area)", "ln(precipitation)", "ln(slope)"]
rows = [f"| {names[j]} | {beta[j]:.3f} | {se[j]:.3f} | {beta[j] / se[j]:.2f} | {2 * stats.t.sf(abs(beta[j] / se[j]), ns - p - 1):.4f} | {'' if j == 0 else f'{vif(j):.2f}'} |" for j in range(4)]
glue("R2m", float(R2m), display=False); glue("R2adj", float(R2adj), display=False); glue("sm", float(np.sqrt(s2)), display=False)
Markdown("| Variable | Coefficient | Standard error | t | p | VIF |\n|---|---|---|---|---|---|\n" + "\n".join(rows))
```

*Table 11.1. Multiple regression of the logarithm of the mean annual flood on the logarithms of area, mean annual precipitation and mean slope for the 40 synthetic catchments. $R^2$ = {glue:text}`R2m:.3f`, adjusted $R^2$ = {glue:text}`R2adj:.3f`, residual standard deviation {glue:text}`sm:.3f` in natural-log units.*

## 11.4 Diagnostics and model evaluation

The least-squares formulas can be applied to any data; whether the results mean anything depends on the assumptions. The **residuals** are the main diagnostic tool (Figure 11.2):

- **Residuals against fitted values** should show no pattern. Curvature suggests a missing transformation or variable; a funnel shape (spread growing with the fitted value) indicates **heteroscedasticity**, which often disappears after a log transformation.
- A **normal probability plot** of residuals checks the normality needed for t-based intervals.
- **Residuals against time or space** reveal autocorrelation or regional patterns that violate independence.
- **Leverage** measures how far a point's explanatory values lie from the others; **Cook's distance** combines leverage and residual size to measure a point's **influence** on the fitted coefficients. Influential points should be checked, not automatically deleted.

Performance should be judged on data not used for fitting. **Leave-one-out cross-validation** gives the prediction error that a new, ungauged site would face, and it is usually larger than the residual standard deviation.

```{code-cell} python
:tags: [hide-input]
fitted = X @ beta
H = X @ np.linalg.inv(X.T @ X) @ X.T; h = np.diag(H)
cook = res ** 2 / ((p + 1) * s2) * h / (1 - h) ** 2
loo = res / (1 - h)                                                        # leave-one-out prediction errors
fig, axs = plt.subplots(1, 3, figsize=(6.4, 3.0))
axs[0].plot(fitted, res, "o", color=c1, ms=4); axs[0].axhline(0, color=ink2, lw=1)
axs[0].set_xlabel("Fitted ln Q"); axs[0].set_ylabel("Residual"); axs[0].set_title("(a) Residuals")
osm, osr = stats.probplot(res, fit=False)
axs[1].plot(osm, np.sort(res), "o", color=c1, ms=4); axs[1].plot(osm, osm * np.sqrt(s2), color=c2, lw=1.5)
axs[1].set_xlabel("Normal quantile"); axs[1].set_title("(b) Normal plot")
axs[2].plot(h, cook, "o", color=c1, ms=4); axs[2].set_xlabel("Leverage"); axs[2].set_ylabel("Cook's distance"); axs[2].set_title("(c) Influence")
plt.show()
glue("loo_rmse", float(np.sqrt(np.mean(loo ** 2))), display=False)
```

*Figure 11.2. Diagnostics for the regression of Table 11.1: (a) residuals against fitted values, (b) normal probability plot of residuals and (c) Cook's distance against leverage. No pattern or highly influential point is apparent. The leave-one-out cross-validated error is {glue:text}`loo_rmse:.3f` in natural-log units, larger than the residual standard deviation.*

## 11.5 Confidence and prediction intervals

At a new value $x_0$, two different intervals can be computed. The **confidence interval** for the mean response,

\begin{equation*}
\hat y_0 \pm t_{1-\alpha/2,\,n-2}\;s\sqrt{\frac{1}{n} + \frac{(x_0 - \bar x)^2}{S_{xx}}},
\tag{11.5}
\end{equation*}

reflects only the uncertainty of the fitted line, which shrinks as $n$ grows. The **prediction interval** for a single new observation,

\begin{equation*}
\hat y_0 \pm t_{1-\alpha/2,\,n-2}\;s\sqrt{1 + \frac{1}{n} + \frac{(x_0 - \bar x)^2}{S_{xx}}},
\tag{11.6}
\end{equation*}

adds the scatter of individual points about the line, which does not shrink with more data. For estimating a flood statistic at an ungauged catchment, the prediction interval is the relevant one (Figure 11.1). Both widen away from $\bar x$, warning against extrapolation beyond the range of the data, where the form of the relation itself is unverified.

## 11.6 Correlation, causation and spurious relations

A strong correlation can arise without any causal link from $x$ to $y$:

- **Confounding**: both variables depend on a third. Catchments with steep slopes may have high floods because they also receive more rain.
- **Common trends**: two series that both trend over time (urban area and flood peaks, or anything and the year) are correlated whether or not one causes the other; detrending or differencing is needed before correlating them.
- **Spurious ratio correlation**: variables that share a term are correlated even when the underlying quantities are independent. Plotting runoff per unit area against $1/\text{area}$, or two ratios with the same denominator, creates correlation from the shared term alone (Figure 11.3).
- **Selection and aggregation**: correlations among averages over large areas or long periods are typically much stronger than among individual values.

Physical reasoning, designed experiments (such as paired catchments) or careful modeling of alternative explanations are needed to support causal claims.

```{code-cell} python
:tags: [hide-input]
rng2 = np.random.default_rng(113)
V1, V2, A = rng2.lognormal(3, 0.3, 200), rng2.lognormal(3, 0.3, 200), rng2.lognormal(4, 0.6, 200)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 3.3))
a1.plot(V1, V2, "o", color=c1, ms=3.5); a1.set_xlabel("$V_1$"); a1.set_ylabel("$V_2$")
a1.set_title(f"(a) Independent, r = {np.corrcoef(V1, V2)[0, 1]:.2f}")
a2.plot(V1 / A, V2 / A, "o", color=c2, ms=3.5); a2.set_xlabel("$V_1 / A$"); a2.set_ylabel("$V_2 / A$")
a2.set_title(f"(b) Divided by A, r = {np.corrcoef(V1 / A, V2 / A)[0, 1]:.2f}")
plt.show()
```

*Figure 11.3. Spurious ratio correlation. (a) Two independent variables. (b) The same variables divided by a third, independent variable $A$ appear strongly correlated, because both ratios share the random denominator.*

## 11.7 Lab (planned)

A Colab notebook will build a regional regression of the mean annual flood and mean annual runoff on catchment attributes from an openly available large-sample dataset, with cross-validation and residual diagnostics.

## 11.8 Worked examples

**Example 11.1 — A regression by hand.** Six stations have mean annual precipitation $P$ (mm) and mean annual runoff $Q$ (mm) as below. Fit $Q = a + bP$, compute $R^2$, and predict $Q$ for $P$ = 1400 mm.

```{code-cell} python
P = np.array([1050, 1180, 1260, 1330, 1450, 1590.0]); Q = np.array([420, 515, 560, 640, 720, 830.0])
bh = np.sum((P - P.mean()) * (Q - Q.mean())) / np.sum((P - P.mean()) ** 2); ah = Q.mean() - bh * P.mean()
r2 = np.corrcoef(P, Q)[0, 1] ** 2
print(f"Q = {ah:.0f} + {bh:.3f} P; R2 = {r2:.3f}; Q(1400) = {ah + bh * 1400:.0f} mm")
```

**Example 11.2 — Back-transformation bias.** In Figure 11.1, the log-space residual standard deviation is {glue:text}`s11:.3f`. By what factor does the median prediction underestimate the mean flood of a catchment, assuming normal residuals? Compare with the smearing estimator.

```{code-cell} python
print(f"normal-theory factor = {np.exp(s ** 2 / 2):.3f}; smearing factor = {np.mean(np.exp(e)):.3f}")
```

**Example 11.3 — Prediction at an ungauged site.** Use the multiple regression of Table 11.1 to estimate the mean annual flood of an ungauged catchment with area 420 km², mean annual precipitation 1520 mm and slope 0.06, with a 90% prediction interval.

```{code-cell} python
x0v = np.array([1, np.log(420), np.log(1520), np.log(0.06)])
yhat = x0v @ beta; se_pred = np.sqrt(s2 * (1 + x0v @ np.linalg.inv(X.T @ X) @ x0v)); tq = stats.t.ppf(0.95, ns - p - 1)
print(f"median estimate = {np.exp(yhat):.0f} m3/s; 90% PI = {np.exp(yhat - tq * se_pred):.0f}–{np.exp(yhat + tq * se_pred):.0f} m3/s")
```

**Example 11.4 — Fitting a rating curve.** Fit $Q = a(h - h_0)^b$ to the gaugings below by nonlinear least squares on the logarithms of discharge.

```{code-cell} python
h_g = np.array([0.62, 0.80, 1.05, 1.31, 1.58, 1.92, 2.30, 2.75])
Q_g = np.array([1.9, 6.0, 14.5, 26.0, 41.0, 64.0, 95.0, 140.0])
f_log = lambda h, a, h0, b: np.log(a) + b * np.log(np.clip(h - h0, 1e-6, None))
(aa, h0, bb), _ = curve_fit(f_log, h_g, np.log(Q_g), p0=(20, 0.4, 1.7), bounds=([0.1, 0, 0.5], [500, 0.6, 4]))
print(f"Q = {aa:.1f} (h - {h0:.2f})^{bb:.2f}")
```

## 11.9 Common mistakes

- **Back-transforming log predictions without a bias correction** when the mean is needed.
- **Reporting the confidence interval** when a prediction for a single new site is needed.
- **Extrapolating** beyond the range of the explanatory variables.
- **Interpreting coefficients under multicollinearity** as physical effects.
- **Correlating ratios with shared terms**, or trending series, and calling the result a relation.

## 11.10 Exercises

```{code-cell} python
:tags: [remove-cell]
glue("w1", float(0.82 ** 2), display=False)
glue("w2", float(3.1 * 250 ** 0.7), display=False)
glue("w3", float(np.exp(0.35 ** 2 / 2)), display=False)
glue("w4", float(1 / (1 - 0.9)), display=False)
glue("w5", float(1 - (1 - 0.64) * (30 - 1) / (30 - 3 - 1)), display=False)
```

1. A simple regression has correlation coefficient 0.82. What fraction of the variance of $y$ does it explain?
2. A regional relation gives $\bar Q = 3.1A^{0.7}$ (m³ s⁻¹, $A$ in km²). Estimate $\bar Q$ for 250 km².
3. A log-space regression has residual standard deviation 0.35. By what factor should the back-transformed median be multiplied to estimate the mean, under normality?
4. An explanatory variable has $R_j^2$ = 0.9 when regressed on the others. What is its VIF?
5. A regression with three explanatory variables and 30 observations has $R^2$ = 0.64. Find the adjusted $R^2$.

```{admonition} Answers
:class: dropdown
1. {glue:text}`w1:.2f`.
2. {glue:text}`w2:.0f` m³ s⁻¹.
3. {glue:text}`w3:.3f`.
4. {glue:text}`w4:.0f`.
5. {glue:text}`w5:.3f`.
```

## 11.11 Further reading

- Duan, N. (1983). Smearing estimate: a nonparametric retransformation method. *Journal of the American Statistical Association*, 78, 605–610.
- Helsel, D. R., Hirsch, R. M., Ryberg, K. R., Archfield, S. A., and Gilroy, E. J. (2020). *Statistical Methods in Water Resources*. U.S. Geological Survey Techniques and Methods, book 4, chapter A3.
- Maity, R. (2022). *Statistical Methods in Hydrology and Hydroclimatology* (2nd ed.). Springer.
- Montgomery, D. C., Peck, E. A., and Vining, G. G. (2021). *Introduction to Linear Regression Analysis* (6th ed.). Wiley.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
