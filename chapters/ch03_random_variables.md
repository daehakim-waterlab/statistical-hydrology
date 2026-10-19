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

# 3. Random variables and their distributions

```{admonition} Core ideas
:class: tip
1. A **random variable** attaches a number to each outcome of an experiment. It is **discrete** if it takes separate values (counts) and **continuous** if it varies over a range (depths, discharges).
2. Its distribution is described by a **probability mass function** or **density function**, by the **cumulative distribution function** $F(x) = P(X \le x)$, or by its inverse, the **quantile function**. Hydrologists often use the **exceedance probability** $1 - F(x)$.
3. Two or more variables are described by a **joint distribution**. From it follow the **marginal** distribution of each variable and the **conditional** distribution of one given the others.
4. Variables are **independent** when their joint distribution is the product of their marginals.
5. The distribution of a **function** of random variables can be derived: by a change of variables for monotone transformations, and by the cumulative distribution for maxima and minima, which is the root of extreme-value theory.
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

## 3.1 Discrete and continuous random variables

A **random variable** $X$ is a rule that assigns a real number to each outcome of an experiment: the number of rainy days in a week, the depth of tomorrow's rain, the maximum discharge of next year. By convention, capital letters denote the random variable and lower-case letters the values it can take, so that $P(X \le x)$ reads "the probability that the annual maximum is at most $x$".

A **discrete** random variable takes a finite or countable set of values, typically counts. Its distribution is given by the **probability mass function** (pmf)

\begin{equation*}
p_X(x) = P(X = x), \qquad \sum_x p_X(x) = 1 .
\tag{3.1}
\end{equation*}

A **continuous** random variable can take any value in an interval. The probability of any single exact value is zero; probabilities are assigned to intervals through the **probability density function** (pdf) $f_X(x)$:

\begin{equation*}
P(a < X \le b) = \int_a^b f_X(x)\,dx, \qquad f_X(x) \ge 0, \qquad \int_{-\infty}^{\infty} f_X(x)\,dx = 1 .
\tag{3.2}
\end{equation*}

The density is not a probability: it has units of 1/x and can exceed 1. Only its integral over an interval is a probability.

## 3.2 Distribution, density and quantile functions

### The cumulative distribution function

Both kinds of variable have a **cumulative distribution function** (cdf)

\begin{equation*}
F_X(x) = P(X \le x) = \begin{cases} \sum_{u \le x} p_X(u) & \text{discrete} \\ \int_{-\infty}^{x} f_X(u)\,du & \text{continuous} \end{cases}
\tag{3.3}
\end{equation*}

It rises from 0 to 1 and never decreases; for a discrete variable it is a staircase with a step of height $p_X(x)$ at each possible value, and for a continuous variable it is smooth, with $f_X = dF_X/dx$ (Figure 3.1).

In hydrology the complement is often more natural: the **exceedance probability** $P(X > x) = 1 - F_X(x)$, the probability that a flood exceeds a given level. For annual maxima, its reciprocal is the return period (Chapter 8).

### Quantiles

The inverse of the cdf is the **quantile function**: the $p$-quantile $x_p$ is the value not exceeded with probability $p$,

\begin{equation*}
x_p = F_X^{-1}(p) .
\tag{3.4}
\end{equation*}

The median is $x_{0.5}$, the quartiles $x_{0.25}$ and $x_{0.75}$. A design flood is a quantile: the 100-year flood is the 0.99-quantile of the annual maximum distribution. For a discrete variable, $x_p$ is defined as the smallest value with $F_X(x) \ge p$.

```{code-cell} python
:tags: [hide-input]
fig, axs = plt.subplots(2, 2, figsize=(6.4, 5.0))
k = np.arange(0, 8); pk = stats.binom.pmf(k, 7, 0.3)                 # rainy days in a week
axs[0, 0].bar(k, pk, color=c1, width=0.6); axs[0, 0].set_title("(a) pmf: rainy days in a week")
axs[0, 0].set_xlabel("x"); axs[0, 0].set_ylabel("$p_X(x)$")
axs[1, 0].step(np.r_[-1, k, 8], np.r_[0, np.cumsum(pk), 1], where="post", color=c1, lw=2)
axs[1, 0].set_title("(c) cdf"); axs[1, 0].set_xlabel("x"); axs[1, 0].set_ylabel("$F_X(x)$"); axs[1, 0].set_xlim(-0.5, 7.5)
G = stats.gamma(a=2.5, scale=12.0)                                  # storm depth (mm)
x = np.linspace(0, 150, 400)
axs[0, 1].plot(x, G.pdf(x), color=c2, lw=2.2)
xx = np.linspace(40, 60, 50); axs[0, 1].fill_between(xx, 0, G.pdf(xx), color=c2, alpha=0.3, lw=0)
axs[0, 1].set_title("(b) pdf: storm depth"); axs[0, 1].set_xlabel("x (mm)"); axs[0, 1].set_ylabel("$f_X(x)$ (mm⁻¹)")
axs[1, 1].plot(x, G.cdf(x), color=c2, lw=2.2)
q90 = G.ppf(0.9); axs[1, 1].plot([0, q90, q90], [0.9, 0.9, 0], color=ink2, lw=1, ls="--")
axs[1, 1].text(q90 + 3, 0.05, f"$x_{{0.9}}$ = {q90:.0f} mm", color=ink2, fontsize=9.5)
axs[1, 1].set_title("(d) cdf"); axs[1, 1].set_xlabel("x (mm)"); axs[1, 1].set_ylabel("$F_X(x)$")
plt.show()
glue("p4060", float(G.cdf(60) - G.cdf(40)), display=False)
```

*Figure 3.1. A discrete variable, the number of rainy days in a week, with (a) its pmf and (c) its staircase cdf, and a continuous variable, the depth of a storm, with (b) its pdf and (d) its cdf. The shaded area in (b), the probability that a storm brings 40–60 mm, is {glue:text}`p4060:.3f`. The dashed line in (d) reads the 0.9-quantile from the cdf.*

### Empirical distribution functions

From a sample $x_1, \dots, x_n$, the **empirical cdf** $\hat F(x)$ is the fraction of sample values not exceeding $x$. Sorting the sample in ascending order, $x_{(1)} \le \dots \le x_{(n)}$, it rises by $1/n$ at each value. It estimates $F_X$ without assuming any distribution and converges to it as $n$ grows. Plotting positions, used to draw data on probability plots, are refinements of it (Chapter 8).

## 3.3 Joint, marginal and conditional distributions

### Joint distributions

Hydrological problems often involve several variables together: the depth and duration of a storm, the peak and volume of a flood, rainfall at two stations. Their **joint distribution** gives probabilities of combinations. For two discrete variables, the joint pmf is $p_{XY}(x, y) = P(X = x, Y = y)$; for continuous variables, the joint density $f_{XY}(x, y)$ gives

\begin{equation*}
P(X \in A, Y \in B) = \int_A \int_B f_{XY}(x, y)\,dy\,dx ,
\tag{3.5}
\end{equation*}

and the joint cdf is $F_{XY}(x, y) = P(X \le x, Y \le y)$.

### Marginal distributions

The distribution of one variable alone, ignoring the other, is its **marginal** distribution, obtained by summing or integrating the joint distribution over the other variable:

\begin{equation*}
p_X(x) = \sum_y p_{XY}(x, y), \qquad f_X(x) = \int_{-\infty}^{\infty} f_{XY}(x, y)\,dy .
\tag{3.6}
\end{equation*}

Table 3.1 shows a joint pmf for the number of heavy-rain days in July at two nearby stations. The row and column totals are the marginal pmfs; they are written in the margins of the table, which gives the concept its name.

```{code-cell} python
:tags: [remove-input]
pj = np.array([[0.30, 0.08, 0.02, 0.00],
               [0.07, 0.17, 0.06, 0.01],
               [0.02, 0.05, 0.09, 0.03],
               [0.00, 0.01, 0.03, 0.06]])
pA, pB = pj.sum(1), pj.sum(0)
lab = ["0", "1", "2", "3+"]
rows = "".join(f"| **{lab[i]}** | " + " | ".join(f"{v:.2f}" for v in pj[i]) + f" | {pA[i]:.2f} |\n" for i in range(4))
Markdown("| $X$ (station A) \\\\ $Y$ (station B) | 0 | 1 | 2 | 3+ | $p_X(x)$ |\n|---|---|---|---|---|---|\n" + rows +
         "| $p_Y(y)$ | " + " | ".join(f"{v:.2f}" for v in pB) + " | 1.00 |")
```

*Table 3.1. Joint pmf of the number of July days with more than 50 mm of rain at two nearby stations (illustrative values). The last column and the last row are the marginal pmfs.*

### Conditional distributions

The **conditional** distribution of $Y$ given $X = x$ follows from Equation (2.3):

\begin{equation*}
p_{Y \mid X}(y \mid x) = \frac{p_{XY}(x, y)}{p_X(x)}, \qquad f_{Y \mid X}(y \mid x) = \frac{f_{XY}(x, y)}{f_X(x)} .
\tag{3.7}
\end{equation*}

It is a slice of the joint distribution, rescaled to integrate to one. Figure 3.2 shows a joint density of storm depth and duration: long storms tend to bring more rain, so the conditional distribution of depth shifts upward as duration increases. This is the kind of relation that regression (Chapter 11) and copulas (Chapter 15) describe.

```{code-cell} python
:tags: [hide-input]
rho = 0.7
mu, sd = np.array([np.log(12.0), np.log(25.0)]), np.array([0.6, 0.7])   # ln duration (h), ln depth (mm)
D, R = np.meshgrid(np.geomspace(1, 80, 240), np.geomspace(2, 250, 240))
L1, L2 = (np.log(D) - mu[0]) / sd[0], (np.log(R) - mu[1]) / sd[1]
f = np.exp(-(L1 ** 2 - 2 * rho * L1 * L2 + L2 ** 2) / (2 * (1 - rho ** 2))) / (2 * np.pi * sd[0] * sd[1] * np.sqrt(1 - rho ** 2) * D * R)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 3.8))
a1.contour(D, R, f * D * R, levels=7, colors=c1, linewidths=1.2)   # density of (ln D, ln R)
a1.set_xscale("log"); a1.set_yscale("log")
for ax_, tk in [(a1.xaxis, [2, 5, 10, 20, 50]), (a1.yaxis, [5, 10, 20, 50, 100, 200])]:
    ax_.set_ticks(tk); ax_.set_ticklabels([str(t) for t in tk]); ax_.set_minor_formatter(plt.NullFormatter())
for d0, col in zip([6, 12, 30], [c3, ink2, c2]):
    a1.axvline(d0, color=col, lw=1.2, ls="--")
a1.set_xlabel("Duration (h)"); a1.set_ylabel("Depth (mm)"); a1.set_title("(a) Joint density")
r = np.linspace(1, 160, 400)
for d0, col in zip([6, 12, 30], [c3, ink2, c2]):
    m_c = mu[1] + rho * sd[1] / sd[0] * (np.log(d0) - mu[0]); s_c = sd[1] * np.sqrt(1 - rho ** 2)
    a2.plot(r, stats.lognorm.pdf(r, s_c, scale=np.exp(m_c)), color=col, lw=2.2, label=f"duration {d0} h")
a2.set_xlabel("Depth (mm)"); a2.set_ylabel("Conditional density"); a2.set_title("(b) Depth given duration")
a2.legend(loc="upper right", fontsize=9)
plt.show()
```

*Figure 3.2. (a) Contours of an illustrative joint distribution of storm duration and depth, on logarithmic axes (a bivariate lognormal with correlation 0.7 between the logarithms). (b) Conditional densities of depth for storms lasting 6, 12 and 30 hours: slices of (a) along the dashed lines, rescaled to unit area.*

## 3.4 Independence

Two random variables are **independent** if every event defined by one is independent of every event defined by the other. Equivalently, their joint distribution factors into the product of the marginals:

\begin{equation*}
p_{XY}(x, y) = p_X(x)\,p_Y(y), \qquad f_{XY}(x, y) = f_X(x)\,f_Y(y) \quad \text{for all } x, y .
\tag{3.8}
\end{equation*}

Then the conditional distribution of one does not depend on the value of the other. In Table 3.1, $p_{XY}(0, 0)$ = 0.30, while $p_X(0)\,p_Y(0)$ = {glue:text}`indep00:.3f`: the stations are dependent, as nearby stations usually are. For $n$ independent variables, the joint distribution is the product of $n$ marginals, which is what makes the likelihood of a sample easy to write (Chapter 7).

```{code-cell} python
:tags: [remove-cell]
glue("indep00", float(pA[0] * pB[0]), display=False)
```

## 3.5 Functions of random variables

### Monotone transformations

If $Y = g(X)$ with $g$ increasing, then $Y \le y$ exactly when $X \le g^{-1}(y)$, so $F_Y(y) = F_X(g^{-1}(y))$. Differentiating gives the **change-of-variables** formula, which holds for decreasing $g$ too when the absolute value is kept:

\begin{equation*}
f_Y(y) = f_X\!\left(g^{-1}(y)\right) \left| \frac{d g^{-1}(y)}{dy} \right| .
\tag{3.9}
\end{equation*}

The derivative accounts for the stretching or compression of the axis. Two applications recur throughout the book. If $\ln X$ is normal, $X$ is **lognormal** (Chapter 6). And if water level $H$ has a known distribution and discharge follows a rating curve $Q = a(H - h_0)^b$, Equation (3.9) gives the distribution of discharge (Example 3.3). Quantiles transform even more simply: for increasing $g$, the $p$-quantile of $Y$ is $g$ of the $p$-quantile of $X$.

### Maxima and minima

The largest of $n$ independent variables with the same cdf $F$ is at most $x$ only if all of them are, so

\begin{equation*}
F_{\max}(x) = P(X_1 \le x, \dots, X_n \le x) = [F(x)]^n .
\tag{3.10}
\end{equation*}

Similarly, the smallest exceeds $x$ only if all do: $F_{\min}(x) = 1 - [1 - F(x)]^n$. Equation (3.10) explains why annual maxima have distributions of their own, shifted toward larger values and with a characteristic shape (Figure 3.3), and it is the starting point of extreme-value theory (Chapter 6). Daily values are not independent, so the real annual maximum of 365 daily values behaves as the maximum of fewer, "effectively independent" values.

### Sums and other functions

The distribution of a sum of independent variables is the **convolution** of their distributions; for many variables it approaches a normal distribution, by the central limit theorem (Chapter 4). For general functions of several variables, closed forms are rare, and **Monte Carlo simulation**, drawing many random samples and computing the function for each, is the practical method (Chapter 9).

```{code-cell} python
:tags: [hide-input]
E = stats.expon(scale=8.0)                                          # daily rainfall on wet days (mm)
x = np.linspace(0, 120, 500)
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.plot(x, E.pdf(x), color=ink2, lw=1.5, ls="--", label="One value, f(x)")
for n, col in zip([10, 50, 120], [c3, c1, c2]):
    ax.plot(x, n * E.cdf(x) ** (n - 1) * E.pdf(x), color=col, lw=2.2, label=f"Maximum of {n}")
ax.set_xlabel("x (mm)"); ax.set_ylabel("Density"); ax.set_ylim(0, 0.13)
ax.legend(loc="upper right", fontsize=9.5)
plt.show()
```

*Figure 3.3. Density of the maximum of $n$ independent exponential values with mean 8 mm, from Equation (3.10). As $n$ grows the distribution of the maximum shifts to the right, while its shape and width change little.*

## 3.6 Worked examples

**Example 3.1 — Probabilities from a cdf.** The time between storms at a station, in days, follows $F(t) = 1 - e^{-t/4}$. Find the probability that the next storm comes within 2 days, the probability of a dry spell longer than 10 days, and the median time between storms.

```{code-cell} python
F = lambda t: 1 - np.exp(-t / 4)
print(f"P(T <= 2) = {F(2):.3f}; P(T > 10) = {1 - F(10):.3f}; median = {4 * np.log(2):.2f} days")
```

**Example 3.2 — Marginal and conditional probabilities.** From Table 3.1, find the probability that station B has no heavy-rain day in July, and the same probability given that station A has two.

```{code-cell} python
print(f"P(Y = 0) = {pB[0]:.2f}; P(Y = 0 | X = 2) = {pj[2, 0] / pA[2]:.3f}")
```

**Example 3.3 — From stage to discharge.** The annual maximum stage at a gauge is normally distributed with mean 3.0 m and standard deviation 0.6 m. The rating curve is $Q = 25(H - 0.5)^{1.6}$. Find the median and the 0.99-quantile of the annual maximum discharge, and its density at 150 m³ s⁻¹.

```{code-cell} python
Hd = stats.norm(3.0, 0.6)
g = lambda h: 25 * (h - 0.5) ** 1.6
ginv = lambda q: 0.5 + (q / 25) ** (1 / 1.6)
dginv = lambda q: (1 / 1.6) * (q / 25) ** (1 / 1.6 - 1) / 25
print(f"median Q = {g(Hd.median()):.0f} m3/s; Q_0.99 = {g(Hd.ppf(0.99)):.0f} m3/s")
print(f"f_Q(150) = {Hd.pdf(ginv(150)) * dginv(150):.2e} per m3/s")
```

**Example 3.4 — The largest of many.** If daily rainfall on wet days is exponential with mean 8 mm, and a season has 60 independent wet days, what is the probability that the season's largest daily amount exceeds 50 mm?

```{code-cell} python
print(f"P(max > 50) = {1 - E.cdf(50) ** 60:.3f} (single day: {1 - E.cdf(50):.4f})")
```

## 3.7 Common mistakes

- **Reading a density as a probability.** $f(x)$ can exceed 1; only areas under it are probabilities.
- **Confusing non-exceedance and exceedance.** Design tables use $P(X > x)$; the cdf gives $P(X \le x)$.
- **Assuming the marginals determine the joint distribution.** They do only under independence.
- **Forgetting the Jacobian** when transforming a density (Equation 3.9). Quantiles, by contrast, transform directly.
- **Treating daily values as independent** when deriving the distribution of annual maxima from Equation (3.10).

## 3.8 Exercises

```{code-cell} python
:tags: [remove-cell]
glue("g1", float(np.exp(-15 / 6) - np.exp(-30 / 6)), display=False)
glue("g2", float(-6 * np.log(0.1)), display=False)
glue("g3", float(pj[:, 1].sum()), display=False)
glue("g4", float(pj[3, 3] / pA[3]), display=False)
glue("g5", float(0.995 ** 30), display=False)
```

1. For $F(x) = 1 - e^{-x/6}$, find $P(15 < X \le 30)$.
2. For the same distribution, find the 0.9-quantile.
3. From Table 3.1, find $P(Y = 1)$.
4. From Table 3.1, find $P(Y = 3{+} \mid X = 3{+})$.
5. Thirty independent variables each have $F(x_0) = 0.995$. What is the probability that their maximum does not exceed $x_0$?

```{admonition} Answers
:class: dropdown
1. {glue:text}`g1:.3f`.
2. {glue:text}`g2:.1f`.
3. {glue:text}`g3:.2f`.
4. {glue:text}`g4:.2f`.
5. {glue:text}`g5:.3f`.
```

## 3.9 Further reading

- Ang, A. H.-S., and Tang, W. H. (2007). *Probability Concepts in Engineering* (2nd ed.). Wiley.
- Haan, C. T. (2002). *Statistical Methods in Hydrology* (2nd ed.). Iowa State Press.
- Maity, R. (2022). *Statistical Methods in Hydrology and Hydroclimatology* (2nd ed.). Springer.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
