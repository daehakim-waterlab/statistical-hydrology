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

# Appendix A. Statistical tables

Printed statistical tables were once indispensable. Today every value can be computed directly, as the code behind each table shows, and the code is more accurate than interpolating in a table. The tables are kept for quick reference and for examinations. All are computed with `scipy.stats` when the book is built.

```{code-cell} python
:tags: [remove-cell]
import numpy as np
from scipy import stats
from IPython.display import Markdown
def md_table(header, rows):
    return Markdown("| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n" + "\n".join("| " + " | ".join(r) + " |" for r in rows))
```

## A.1 Standard normal distribution

Table A.1 gives $\Phi(z) = P(Z \le z)$ for the standard normal variable. The row gives $z$ to one decimal and the column the second decimal. For negative $z$, use $\Phi(-z) = 1 - \Phi(z)$. In Python: `stats.norm.cdf(z)` and, for quantiles, `stats.norm.ppf(p)`.

```{code-cell} python
:tags: [hide-input]
rows = [[f"**{z0:.1f}**"] + [f"{stats.norm.cdf(z0 + d):.4f}" for d in np.arange(10) / 100] for z0 in np.arange(0, 3.5, 0.1)]
md_table(["z"] + [f"{d:.2f}" for d in np.arange(10) / 100], rows)
```

*Table A.1. Cumulative probabilities $\Phi(z)$ of the standard normal distribution.*

```{code-cell} python
:tags: [hide-input]
ps = [0.5, 0.8, 0.9, 0.95, 0.975, 0.98, 0.99, 0.995, 0.998, 0.999]
md_table(["$p$"] + [f"{p}" for p in ps], [["$z_p$"] + [f"{stats.norm.ppf(p):.3f}" for p in ps],
                                          ["$T = 1/(1-p)$"] + [f"{1 / (1 - p):.0f}" for p in ps]])
```

*Table A.2. Standard normal quantiles $z_p = \Phi^{-1}(p)$, with the corresponding return periods for annual data.*

## A.2 t, chi-square and F critical values

```{code-cell} python
:tags: [hide-input]
alphas = [0.10, 0.05, 0.025, 0.01, 0.005]
dfs = list(range(1, 31)) + [40, 60, 120]
rows = [[f"**{d}**"] + [f"{stats.t.ppf(1 - a, d):.3f}" for a in alphas] for d in dfs]
rows.append(["**∞**"] + [f"{stats.norm.ppf(1 - a):.3f}" for a in alphas])
md_table(["ν"] + [f"{a}" for a in alphas], rows)
```

*Table A.3. Critical values $t_{1-\alpha,\nu}$ of Student's t distribution: the value exceeded with probability α (one tail). For a two-sided test at level α, use the column α/2. In Python: `stats.t.ppf(1 - alpha, nu)`.*

```{code-cell} python
:tags: [hide-input]
qs = [0.005, 0.01, 0.025, 0.05, 0.10, 0.90, 0.95, 0.975, 0.99, 0.995]
dfs = list(range(1, 31)) + [40, 50, 60, 80, 100]
rows = [[f"**{d}**"] + [f"{stats.chi2.ppf(q, d):.3f}" if stats.chi2.ppf(q, d) < 10 else f"{stats.chi2.ppf(q, d):.2f}" for q in qs] for d in dfs]
md_table(["ν"] + [f"{q}" for q in qs], rows)
```

*Table A.4. Quantiles $\chi^2_{q,\nu}$ of the chi-square distribution: the value not exceeded with probability $q$. In Python: `stats.chi2.ppf(q, nu)`.*

```{code-cell} python
:tags: [hide-input]
n1s = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30]
n2s = list(range(1, 11)) + [12, 15, 20, 25, 30, 40, 60, 120]
rows = [[f"**{b}**"] + [f"{stats.f.ppf(0.95, a, b):.2f}" for a in n1s] for b in n2s]
md_table(["$\\nu_2$ \\ $\\nu_1$"] + [str(a) for a in n1s], rows)
```

*Table A.5. Critical values $F_{0.95}(\nu_1, \nu_2)$ of the F distribution, exceeded with probability 0.05; $\nu_1$ is the numerator and $\nu_2$ the denominator degrees of freedom. In Python: `stats.f.ppf(0.95, nu1, nu2)`.*

## A.3 Critical values of common tests

```{code-cell} python
:tags: [hide-input]
ns = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
rows = [[f"**{n}**"] + [f"{stats.kstwo.ppf(1 - a, n):.3f}" for a in (0.10, 0.05, 0.01)] + [f"{1.96 / np.sqrt(n):.3f}"] for n in ns]
md_table(["$n$", "KS, α = 0.10", "KS, α = 0.05", "KS, α = 0.01", "ACF limit ±1.96/√n"], rows)
```

*Table A.6. Critical values of the Kolmogorov–Smirnov statistic $D_n$ for a fully specified distribution (they are too large when parameters are estimated from the same data; Chapter 10), and the approximate 95% limits for sample autocorrelations of white noise (Chapter 14).*

Other tests used in the book rely on normal approximations or simple formulas and need no table:

- **Mann–Kendall**: $Z = (S - \operatorname{sgn}S)/\sqrt{n(n - 1)(2n + 5)/18}$, compared with Table A.2 (Chapter 13).
- **Pettitt**: $p \approx 2\exp[-6K^2/(n^3 + n^2)]$ (Chapter 13).
- **Mann–Whitney** for large samples: $Z = [U - n_1n_2/2]/\sqrt{n_1n_2(n_1 + n_2 + 1)/12}$ (Chapter 10).
- **Runs test**: for $n_1$ values above and $n_2$ below the median, the number of runs has mean $1 + 2n_1n_2/(n_1 + n_2)$ and variance $2n_1n_2(2n_1n_2 - n_1 - n_2)/[(n_1 + n_2)^2(n_1 + n_2 - 1)]$.

## A.4 Frequency factors

```{code-cell} python
:tags: [hide-input]
Ts = [2, 5, 10, 25, 50, 100, 200, 500, 1000]
skews = np.round(np.arange(-1.0, 3.01, 0.2), 1)
rows = [[f"**{g:.1f}**"] + [f"{stats.pearson3.ppf(1 - 1 / T, g):.3f}" for T in Ts] for g in skews]
md_table(["Skew \\ T (years)"] + [str(T) for T in Ts], rows)
```

*Table A.7. Frequency factors $K_T$ of the Pearson type III distribution, $x_T = \mu + K_T\sigma$, as a function of the skewness coefficient (Chapter 6). For the log-Pearson type III, apply them to the logarithms. In Python: `stats.pearson3.ppf(1 - 1/T, skew)`.*

```{code-cell} python
:tags: [hide-input]
ns = [10, 15, 20, 25, 30, 40, 50, 60, 80, 100]
rows = []
for n in ns:
    y = -np.log(-np.log(np.arange(1, n + 1) / (n + 1)))                    # reduced variates at Weibull positions
    yn, sn = y.mean(), y.std()
    rows.append([f"**{n}**", f"{yn:.4f}", f"{sn:.4f}"] + [f"{(-np.log(-np.log(1 - 1 / T)) - yn) / sn:.3f}" for T in (10, 50, 100, 500)])
yinf = ["**∞**", "0.5772", f"{np.pi / np.sqrt(6):.4f}"] + [f"{(-np.log(-np.log(1 - 1 / T)) - 0.5772) / (np.pi / np.sqrt(6)):.3f}" for T in (10, 50, 100, 500)]
rows.append(yinf)
md_table(["$n$", "$\\bar y_n$", "$\\sigma_n$", "$K_{10}$", "$K_{50}$", "$K_{100}$", "$K_{500}$"], rows)
```

*Table A.8. Gumbel frequency factors adjusted for sample size, $K_T = (y_T - \bar y_n)/\sigma_n$, where $\bar y_n$ and $\sigma_n$ are the mean and standard deviation of the reduced variates of an ordered sample of size $n$ at Weibull plotting positions, and $y_T = -\ln[-\ln(1 - 1/T)]$. The last row is the method-of-moments limit used in Chapter 6.*

---

*Tables in this appendix are generated by the Python code shown. © Daeha Kim, CC BY-NC 4.0.*
