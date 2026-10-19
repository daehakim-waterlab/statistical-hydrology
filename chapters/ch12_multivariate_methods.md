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

# 12. Multivariate methods

```{admonition} Core ideas
:class: tip
1. Hydrological data often come as **matrices**: many stations by many times, or many catchments by many attributes. Their structure is summarized by the **covariance** or **correlation matrix**.
2. **Principal component analysis** finds the uncorrelated combinations of variables that explain the most variance. Applied to space–time fields it gives **empirical orthogonal functions** (spatial patterns) and their time series.
3. **Canonical correlation analysis** finds the pairs of combinations of two variable sets that are most strongly correlated, for example climate indices and regional rainfall.
4. **Analysis of variance** tests whether the means of several groups differ, by comparing variability between groups with variability within them.
5. Correlated random variables are **generated** by transforming independent ones with a factor of the target covariance matrix, which is the basis of multisite stochastic simulation.
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

## 12.1 Data matrices and covariance

A multivariate data set is arranged as an $n \times p$ **data matrix** $\mathbf X$, with $n$ observations (rows: years, months, catchments) of $p$ variables (columns: stations, attributes). Subtracting each column's mean gives the **anomaly** matrix $\mathbf X_a$, and the sample **covariance matrix** is

\begin{equation*}
\mathbf S = \frac{1}{n - 1}\,\mathbf X_a^\mathsf{T}\mathbf X_a ,
\tag{12.1}
\end{equation*}

a symmetric $p \times p$ matrix with the variances on the diagonal and the covariances off it. Dividing each column also by its standard deviation (**standardizing**) gives the **correlation matrix** $\mathbf R$. Standardization matters when variables have different units or very different variances, such as catchment area and mean slope; without it, the variable with the largest variance dominates every analysis.

The covariance matrix contains all the information that linear methods use. Its **eigenvalues** and **eigenvectors** describe the directions and amounts of variability in the $p$-dimensional cloud of observations, and the methods of this chapter are different ways of exploiting them.

## 12.2 Principal components and empirical orthogonal functions

### Principal components

**Principal component analysis** (PCA) finds new variables, linear combinations of the original ones, that are uncorrelated and successively capture as much variance as possible. The first principal component is $u_1 = \mathbf e_1^\mathsf{T}\mathbf x$, where the unit vector $\mathbf e_1$ maximizes the variance $\mathbf e_1^\mathsf{T}\mathbf S\,\mathbf e_1$. The solution is the eigenvector of $\mathbf S$ with the largest eigenvalue; in general,

\begin{equation*}
\mathbf S\,\mathbf e_k = \lambda_k\,\mathbf e_k, \qquad \operatorname{Var}(u_k) = \lambda_k, \qquad \text{fraction explained} = \frac{\lambda_k}{\sum_j \lambda_j} ,
\tag{12.2}
\end{equation*}

with $\lambda_1 \ge \lambda_2 \ge \dots \ge 0$. The eigenvectors (**loadings**) show how each original variable contributes; the **scores** $\mathbf U = \mathbf X_a\mathbf E$ are the values of the new variables for each observation. Because the eigenvectors are orthogonal, the components are uncorrelated, and keeping the first few gives the best low-dimensional approximation of the data in the least-squares sense. In practice PCA is computed by the singular value decomposition of $\mathbf X_a$.

How many components to keep? A **scree plot** of eigenvalues usually shows a few large values followed by a slowly decaying tail of noise; rules of thumb keep components above the "elbow", components that together explain a chosen share of variance (say 80–90%), or components whose eigenvalues stand out from those of random data.

### Empirical orthogonal functions

In climatology and hydrology, PCA of space–time fields is called **empirical orthogonal function** (EOF) analysis. Each column of $\mathbf X$ is a station or grid point and each row a time. The eigenvectors are spatial patterns (**EOFs**), and the scores are their amplitude time series (**principal components**). Figure 12.1 analyzes synthetic monthly rainfall anomalies at 25 stations generated from two hidden patterns plus noise: a region-wide mode and a north–south dipole. EOF analysis recovers both.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(121)
p, n = 25, 360
lon, lat = rng.uniform(126, 130, p), rng.uniform(34, 38.5, p)
pat1 = 1.0 + 0.1 * (lon - 128)                                            # region-wide pattern
pat2 = (lat - 36.25) / 2.25                                               # north-south dipole
a1_t = rng.normal(0, 1.0, n); a2_t = rng.normal(0, 0.6, n)
X = np.outer(a1_t, pat1) + np.outer(a2_t, pat2) + rng.normal(0, 0.45, (n, p))
Xa = X - X.mean(0)
U, svals, Vt = np.linalg.svd(Xa, full_matrices=False)
lam = svals ** 2 / (n - 1); frac = lam / lam.sum()
eof1, eof2 = Vt[0] * np.sign(Vt[0].sum()), Vt[1] * np.sign(np.corrcoef(Vt[1], pat2)[0, 1])
fig, axs = plt.subplots(1, 3, figsize=(6.4, 3.2), width_ratios=[1, 1, 1])
axs[0].plot(np.arange(1, 11), 100 * frac[:10], "o-", color=c1, ms=4)
axs[0].set_xlabel("Component"); axs[0].set_ylabel("Variance explained (%)"); axs[0].set_title("(a) Scree plot")
for ax, eof, ttl in [(axs[1], eof1, "(b) EOF 1"), (axs[2], eof2, "(c) EOF 2")]:
    sc = ax.scatter(lon, lat, c=eof, cmap="RdBu_r", vmin=-0.4, vmax=0.4, s=40, edgecolor=ink2, linewidth=0.4)
    ax.set_title(ttl); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
fig.colorbar(sc, ax=axs[1:], shrink=0.8, label="Loading")
plt.show()
glue("f1", float(100 * frac[0]), display=False); glue("f2", float(100 * frac[1]), display=False)
glue("r_pc1", float(abs(np.corrcoef(U[:, 0], a1_t)[0, 1])), display=False); glue("r_pc2", float(abs(np.corrcoef(U[:, 1], a2_t)[0, 1])), display=False)
```

*Figure 12.1. EOF analysis of 30 years of synthetic monthly rainfall anomalies at 25 stations. (a) The first two components explain {glue:text}`f1:.0f`% and {glue:text}`f2:.0f`% of the variance; the rest is noise. (b) EOF 1, a region-wide pattern of the same sign everywhere. (c) EOF 2, a north–south dipole. The principal component time series correlate with the hidden generating signals at {glue:text}`r_pc1:.2f` and {glue:text}`r_pc2:.2f`.*

EOFs are mathematical constructs: they are orthogonal by design, so physically distinct patterns that are not orthogonal get mixed. Rotation methods (such as varimax) relax orthogonality to obtain more localized, interpretable patterns. PCA is also used to reduce many correlated catchment attributes to a few uncorrelated predictors before regression (**principal component regression**), which avoids the multicollinearity problems of Chapter 11.

## 12.3 Canonical correlation

**Canonical correlation analysis** (CCA) relates two sets of variables, $\mathbf x$ ($p$ variables) and $\mathbf y$ ($q$ variables). It finds weight vectors $\mathbf a$ and $\mathbf b$ such that the **canonical variates** $v = \mathbf a^\mathsf{T}\mathbf x$ and $w = \mathbf b^\mathsf{T}\mathbf y$ have the largest possible correlation, then further pairs uncorrelated with the first, and so on. The squared canonical correlations are the eigenvalues of $\mathbf S_{xx}^{-1}\mathbf S_{xy}\mathbf S_{yy}^{-1}\mathbf S_{yx}$.

A typical application links large-scale climate predictors, such as sea-surface temperature indices, to rainfall or streamflow at a set of stations for seasonal forecasting. With many variables and short records, CCA overfits badly, so it is usually applied to the leading principal components of each set rather than to the raw variables, and its skill is checked by cross-validation (Example 12.3).

## 12.4 Analysis of variance

### One-way analysis of variance

Do the means of several groups differ? For example, is mean annual runoff different among catchments with forest, agricultural and urban land cover, or is monthly rainfall different among El Niño, neutral and La Niña years? With $k$ groups of sizes $n_j$, group means $\bar y_j$ and grand mean $\bar y$, the total variability splits into a part **between** groups and a part **within** them:

\begin{equation*}
\underbrace{\sum_{j}\sum_{i} (y_{ij} - \bar y)^2}_{SS_T} = \underbrace{\sum_j n_j (\bar y_j - \bar y)^2}_{SS_B} + \underbrace{\sum_j\sum_i (y_{ij} - \bar y_j)^2}_{SS_W} .
\tag{12.3}
\end{equation*}

Under the null hypothesis that all group means are equal, with normal errors of equal variance, the ratio of mean squares

\begin{equation*}
F = \frac{SS_B/(k - 1)}{SS_W/(N - k)}
\tag{12.4}
\end{equation*}

follows an F distribution with $(k - 1, N - k)$ degrees of freedom, where $N = \sum n_j$. A large $F$ means the groups differ more than the scatter within them would explain. Results are reported in an **ANOVA table** (Table 12.1).

**Two-way ANOVA** handles two factors at once, such as land cover and soil type, and can test their **interaction**, that is, whether the effect of one factor depends on the level of the other.

### Multiple comparisons and the nonparametric alternative

A significant $F$ says only that at least one mean differs. To find which, pairs of groups are compared with methods that control the overall error rate, such as **Tukey's honestly significant difference** or the **Bonferroni** correction (testing each of $m$ pairs at $\alpha/m$). When the data are skewed or the variances unequal, the **Kruskal–Wallis** test compares groups using ranks (Chapter 10).

```{code-cell} python
:tags: [remove-input]
rng = np.random.default_rng(124)
groups = {"Forest": rng.normal(560, 70, 9), "Agricultural": rng.normal(610, 70, 11), "Urban": rng.normal(700, 80, 7)}
ally = np.concatenate(list(groups.values())); N = len(ally); k = len(groups); gm = ally.mean()
SSB = sum(len(v) * (v.mean() - gm) ** 2 for v in groups.values()); SSW = sum(np.sum((v - v.mean()) ** 2) for v in groups.values())
F = (SSB / (k - 1)) / (SSW / (N - k)); pF = stats.f.sf(F, k - 1, N - k)
H, pH = stats.kruskal(*groups.values())
glue("anovaF", float(F), display=False); glue("anovaP", float(pF), display=False); glue("kwP", float(pH), display=False)
glue("means", ", ".join(f"{kname.lower()} {v.mean():.0f} mm" for kname, v in groups.items()), display=False)
Markdown("| Source | Sum of squares | Degrees of freedom | Mean square | F | p |\n|---|---|---|---|---|---|\n"
         f"| Between groups | {SSB:.0f} | {k - 1} | {SSB / (k - 1):.0f} | {F:.2f} | {pF:.4f} |\n"
         f"| Within groups | {SSW:.0f} | {N - k} | {SSW / (N - k):.0f} | | |\n"
         f"| Total | {SSB + SSW:.0f} | {N - 1} | | | |")
```

*Table 12.1. One-way ANOVA of mean annual runoff (mm) for 27 synthetic catchments in three land-cover classes, with group means {glue:text}`means`. The F test gives p = {glue:text}`anovaP:.4f`; the Kruskal–Wallis test gives p = {glue:text}`kwP:.4f`.*

## 12.5 Generating correlated variables

Stochastic simulation of water systems often requires correlated inputs: rainfall at several gauges, inflows to several reservoirs. To generate a vector $\mathbf x$ with mean $\boldsymbol\mu$ and covariance $\boldsymbol\Sigma$ from independent standard normal variables $\mathbf z$, factor the covariance as $\boldsymbol\Sigma = \mathbf L\mathbf L^\mathsf{T}$ (for example by the **Cholesky decomposition**, with $\mathbf L$ lower triangular) and set

\begin{equation*}
\mathbf x = \boldsymbol\mu + \mathbf L\,\mathbf z ,
\tag{12.5}
\end{equation*}

since then $\operatorname{Cov}(\mathbf x) = \mathbf L\,\operatorname{Cov}(\mathbf z)\,\mathbf L^\mathsf{T} = \boldsymbol\Sigma$. The covariance matrix must be positive definite; matrices estimated from records with gaps, or assembled from separately estimated correlations, may not be, and need repair.

For non-normal variables, such as skewed monthly flows, a common approach generates correlated normal variables and transforms each to its own marginal distribution through the probability integral transform. The rank correlation is preserved, but the product-moment correlation of the transformed variables is somewhat smaller than that of the normal ones, so the normal correlations must be adjusted upward. This is a Gaussian copula, a special case of the copula methods of Chapter 15. Figure 12.2 generates gamma-distributed monthly rainfall at three gauges with a target correlation structure.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(125)
Rt = np.array([[1.0, 0.8, 0.5], [0.8, 1.0, 0.6], [0.5, 0.6, 1.0]])
L = np.linalg.cholesky(Rt)
Z = rng.standard_normal((5000, 3)) @ L.T
G = stats.gamma(1.6, scale=60.0)                                           # monthly rainfall marginal (mm)
Xg = G.ppf(stats.norm.cdf(Z))
Rs = np.corrcoef(Xg.T); Rsp = stats.spearmanr(Xg).statistic
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 3.3))
a1.plot(Xg[:800, 0], Xg[:800, 1], "o", color=c1, ms=2.5, alpha=0.6)
a1.set_xlabel("Gauge 1 (mm)"); a1.set_ylabel("Gauge 2 (mm)"); a1.set_title("(a) Simulated pairs")
pairs = [(0, 1), (0, 2), (1, 2)]
a2.plot([Rt[i, j] for i, j in pairs], [Rs[i, j] for i, j in pairs], "o", color=c2, ms=7, label="Pearson, gamma")
a2.plot([Rt[i, j] for i, j in pairs], [Rsp[i, j] for i, j in pairs], "s", color=c3, ms=6, label="Spearman, gamma")
a2.plot([0.4, 0.9], [0.4, 0.9], color=ink2, lw=1, ls="--")
a2.set_xlabel("Correlation of normal variables"); a2.set_ylabel("Correlation after transform"); a2.set_title("(b) Correlations")
a2.legend(loc="upper left", fontsize=8.5)
plt.show()
```

*Figure 12.2. Monthly rainfall at three gauges generated with gamma marginals (mean 96 mm) from correlated normal variables via the Cholesky factor of the target correlation matrix. (a) Simulated values at two gauges. (b) The product-moment (Pearson) correlations of the gamma variables are slightly lower than the normal ones, so the normal correlations must be inflated to hit a target; rank correlations change in a known way.*

## 12.6 Worked examples

**Example 12.1 — Principal components of three attributes.** The correlation matrix of three standardized catchment attributes is given below. Find the variance explained by each principal component and the loadings of the first.

```{code-cell} python
R3 = np.array([[1.0, 0.75, 0.40], [0.75, 1.0, 0.55], [0.40, 0.55, 1.0]])
w, V = np.linalg.eigh(R3); order = np.argsort(w)[::-1]; w, V = w[order], V[:, order]
print("variance explained (%):", np.round(100 * w / w.sum(), 1))
print("loadings of PC1:", np.round(V[:, 0] * np.sign(V[:, 0].sum()), 3))
```

**Example 12.2 — One-way ANOVA.** Using the data of Table 12.1, test at α = 0.05 whether mean annual runoff differs among the land-cover classes, and use Bonferroni-corrected t tests to find which pairs differ.

```{code-cell} python
print(f"F = {F:.2f}, p = {pF:.4f}")
names = list(groups)
for i in range(3):
    for j in range(i + 1, 3):
        t_, p_ = stats.ttest_ind(groups[names[i]], groups[names[j]])
        print(f"{names[i]} vs {names[j]}: p = {p_:.4f} -> {'differ' if p_ < 0.05 / 3 else 'no evidence'} at 0.05/3")
```

**Example 12.3 — A first canonical correlation.** Two climate indices $\mathbf x$ and rainfall at three stations $\mathbf y$ are simulated below with a common signal. Compute the first canonical correlation.

```{code-cell} python
rng = np.random.default_rng(126); n = 200
s = rng.normal(size=n)
Xc = np.column_stack([s + rng.normal(0, 0.8, n), 0.5 * s + rng.normal(0, 1.0, n)])
Yc = np.column_stack([0.7 * s + rng.normal(0, 1, n), 0.6 * s + rng.normal(0, 1, n), 0.2 * s + rng.normal(0, 1, n)])
C = np.cov(np.hstack([Xc, Yc]).T); Sxx, Syy, Sxy = C[:2, :2], C[2:, 2:], C[:2, 2:]
M = np.linalg.solve(Sxx, Sxy) @ np.linalg.solve(Syy, Sxy.T)
print(f"first canonical correlation = {np.sqrt(np.max(np.linalg.eigvals(M).real)):.3f}")
```

**Example 12.4 — Cholesky factor.** Find the Cholesky factor of a 2 × 2 correlation matrix with correlation ρ, and generate one correlated pair from $z$ = (0.5, −1.2) for ρ = 0.7.

```{code-cell} python
rho = 0.7; L2 = np.array([[1, 0], [rho, np.sqrt(1 - rho ** 2)]])
print("L =", np.round(L2, 3).tolist(), "; x =", np.round(L2 @ np.array([0.5, -1.2]), 3))
```

## 12.7 Common mistakes

- **Running PCA on unstandardized variables with different units.** The largest variance dominates.
- **Interpreting EOFs as physical modes** without checking their robustness; orthogonality can mix patterns.
- **Overfitting CCA** with many variables and few observations.
- **Using ANOVA with very unequal variances or skewed data** without a nonparametric check.
- **Generating non-normal correlated variables with the target correlation applied directly** to the normal variables.

## 12.8 Exercises

```{code-cell} python
:tags: [remove-cell]
w2 = np.linalg.eigvalsh(np.array([[1, 0.6], [0.6, 1.0]])); glue("v1", float(100 * w2.max() / 2), display=False)
glue("v2", float(stats.f.ppf(0.95, 2, 24)), display=False)
glue("v3", float(np.sqrt(1 - 0.5 ** 2)), display=False)
glue("v4", float(0.05 / 6), display=False)
glue("v5", float(3.2 / 5), display=False)
```

1. Two standardized variables have correlation 0.6. What percentage of total variance does the first principal component explain?
2. Find the 5% critical value of $F$ for a one-way ANOVA with 3 groups and 27 observations.
3. In the 2 × 2 Cholesky factor of a correlation matrix with ρ = 0.5, what is the lower-right element?
4. Six pairwise comparisons are made with a Bonferroni correction at an overall level of 0.05. What level is used for each?
5. The eigenvalues of a 5-variable correlation matrix sum to 5. If the first is 3.2, what fraction of variance does it explain?

```{admonition} Answers
:class: dropdown
1. {glue:text}`v1:.0f`%.
2. {glue:text}`v2:.2f`.
3. {glue:text}`v3:.3f`.
4. {glue:text}`v4:.4f`.
5. {glue:text}`v5:.2f`.
```

## 12.9 Further reading

- Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.
- Maity, R. (2022). *Statistical Methods in Hydrology and Hydroclimatology* (2nd ed.). Springer.
- Wilks, D. S. (2019). *Statistical Methods in the Atmospheric Sciences* (4th ed.). Elsevier.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
