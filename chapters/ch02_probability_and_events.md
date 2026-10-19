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

# 2. Probability and events

```{admonition} Core ideas
:class: tip
1. Probability is defined on **events**, sets of possible outcomes of an uncertain experiment, and obeys three axioms from which every other rule follows.
2. The **addition rule** gives the probability that at least one of two events occurs; the **multiplication rule** gives the probability that both occur.
3. **Conditional probability** updates a probability when other information is known. Two events are **independent** when knowing one tells nothing about the other; hydrological events often are not.
4. The **total probability theorem** combines conditional probabilities over a set of exclusive cases; **Bayes' theorem** reverses the direction of conditioning.
5. Most probability statements in design, such as the chance that a levee fails somewhere or that a flood coincides with a high tide, are combinations of these few rules.
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

## 2.1 Experiments, outcomes and events

Probability theory starts with an **experiment** whose result is uncertain: next year's maximum water level at a gauge, tomorrow's rainfall, the number of dry days in July. The set of all possible outcomes is the **sample space** $\Omega$. It may be finite (wet or dry: $\Omega = \{W, D\}$), countably infinite (the number of storms in a year: $\{0, 1, 2, \dots\}$) or continuous (the maximum discharge: all non-negative real numbers).

An **event** is a set of outcomes, a subset of $\Omega$: "the maximum discharge exceeds 500 m³ s⁻¹", "at least three storms occur". Events are combined with the operations of set theory:

- the **union** $A \cup B$ occurs if $A$ or $B$ (or both) occurs;
- the **intersection** $A \cap B$, often written $AB$, occurs if both occur;
- the **complement** $\bar A$ occurs if $A$ does not.

Two events are **mutually exclusive** if they cannot occur together, $A \cap B = \varnothing$, and a set of events is **collectively exhaustive** if one of them must occur. A set of events that is both, such as "dry", "light rain" and "heavy rain" for a day, is a **partition** of the sample space.

## 2.2 Axioms and rules of probability

### The axioms

A probability $P$ assigns to each event a number satisfying three axioms (Kolmogorov, 1933):

1. $P(A) \ge 0$ for every event $A$;
2. $P(\Omega) = 1$;
3. if $A_1, A_2, \dots$ are mutually exclusive, $P(A_1 \cup A_2 \cup \dots) = P(A_1) + P(A_2) + \dots$

Everything else follows. Since $A$ and $\bar A$ are exclusive and together make up $\Omega$,

\begin{equation*}
P(\bar A) = 1 - P(A),
\tag{2.1}
\end{equation*}

and, splitting $A \cup B$ into the exclusive pieces $A$ and $B \cap \bar A$, the **addition rule** is

\begin{equation*}
P(A \cup B) = P(A) + P(B) - P(A \cap B) .
\tag{2.2}
\end{equation*}

The last term removes the double counting of outcomes that belong to both events. For exclusive events it vanishes.

### What a probability means

The axioms say how probabilities combine, not what they are. Three interpretations are used in hydrology:

- **Relative frequency**: $P(A)$ is the long-run fraction of repetitions of the experiment in which $A$ occurs. "The annual exceedance probability of this level is 0.1" means that over many years it is exceeded in about 10% of them. Figure 2.1 shows how the relative frequency settles toward the probability as the number of trials grows, slowly and with large excursions in short records.
- **Classical**: when outcomes are equally likely by symmetry, $P(A)$ is the fraction of outcomes in $A$. It rarely applies directly to natural processes.
- **Subjective** or **Bayesian**: $P(A)$ is a degree of belief, updated with data through Bayes' theorem (Section 2.4). It is natural for one-off questions such as "will this dam fail?" and for combining expert judgment with short records.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(2)
p, nyears = 0.10, 1000
fig, ax = plt.subplots(figsize=(6.4, 3.6))
for k, col in enumerate([c1, c2, c3, ink2]):
    x = rng.random(nyears) < p
    ax.plot(np.arange(1, nyears + 1), np.cumsum(x) / np.arange(1, nyears + 1), color=col, lw=1.4)
ax.axhline(p, color=ink, lw=1.2, ls="--")
ax.set_xscale("log"); ax.set_ylim(0, 0.4)
ax.set_xlabel("Number of years observed"); ax.set_ylabel("Fraction of years with exceedance")
plt.show()
```

*Figure 2.1. Relative frequency of years in which a level with annual exceedance probability 0.10 (dashed line) is exceeded, for four synthetic sequences. After 30 years the fractions still differ widely; after 1000 they agree closely with the probability.*

## 2.3 Conditional probability and independence

### Conditioning

Knowing that one event has occurred changes the probability of another. The **conditional probability** of $A$ given $B$, for $P(B) > 0$, is

\begin{equation*}
P(A \mid B) = \frac{P(A \cap B)}{P(B)} ,
\tag{2.3}
\end{equation*}

the fraction of the outcomes in $B$ that are also in $A$. Rearranged, it gives the **multiplication rule** $P(A \cap B) = P(A \mid B)\,P(B) = P(B \mid A)\,P(A)$.

Conditional probabilities are estimated from data by counting. Table 2.1 classifies 40 years of synthetic daily rainfall (the record of Chapter 1) by whether each day and the day before it were wet. The probability that a day is wet is much higher after a wet day than after a dry one: rain is **persistent**.

```{code-cell} python
:tags: [remove-input]
rng = np.random.default_rng(7)
ny = 40; doy = np.tile(np.arange(365), ny); nd = len(doy)
season = np.exp(-0.5 * ((doy - 200) / 35) ** 2)
p01 = 0.15 + 0.35 * season; p11 = 0.45 + 0.35 * season
wet = np.zeros(nd, bool)
for t in range(1, nd):
    wet[t] = rng.random() < (p11[t] if wet[t - 1] else p01[t])
prev, today = wet[:-1], wet[1:]
n_ww, n_wd = np.sum(prev & today), np.sum(prev & ~today)
n_dw, n_dd = np.sum(~prev & today), np.sum(~prev & ~today)
N = len(today)
pw = today.mean(); pw_w = n_ww / (n_ww + n_wd); pw_d = n_dw / (n_dw + n_dd)
glue("pw", float(pw), display=False); glue("pw_w", float(pw_w), display=False); glue("pw_d", float(pw_d), display=False)
Markdown("| | Today wet | Today dry | Total |\n|---|---|---|---|\n"
         f"| Yesterday wet | {n_ww} | {n_wd} | {n_ww + n_wd} |\n"
         f"| Yesterday dry | {n_dw} | {n_dd} | {n_dw + n_dd} |\n"
         f"| Total | {n_ww + n_dw} | {n_wd + n_dd} | {N} |")
```

*Table 2.1. Pairs of consecutive days in 40 years of synthetic daily rainfall. $P(\text{wet})$ = {glue:text}`pw:.2f`, but $P(\text{wet} \mid \text{wet yesterday})$ = {glue:text}`pw_w:.2f` and $P(\text{wet} \mid \text{dry yesterday})$ = {glue:text}`pw_d:.2f`.*

### Independence

Events $A$ and $B$ are **independent** if knowing that one occurred does not change the probability of the other, $P(A \mid B) = P(A)$, or equivalently

\begin{equation*}
P(A \cap B) = P(A)\,P(B) .
\tag{2.4}
\end{equation*}

Independence is a property of the process, not of the events' definitions, and it must be justified. Annual maximum floods in successive years are usually treated as independent; rainy days, as Table 2.1 shows, are not. Exclusive events with non-zero probabilities are never independent: if one occurs, the other cannot.

## 2.4 Total probability and Bayes' theorem

### Total probability

Suppose the events $B_1, \dots, B_k$ form a partition, for example the types of storm that can produce a flood. Any event $A$ can then be split into the exclusive pieces $A \cap B_i$, and the multiplication rule gives the **total probability theorem**:

\begin{equation*}
P(A) = \sum_{i=1}^{k} P(A \mid B_i)\,P(B_i) .
\tag{2.5}
\end{equation*}

It computes a probability by considering each case separately and weighting by how often the case occurs. In hydrology it is used to combine floods from different mechanisms (snowmelt and rain, typhoons and fronts), to account for a variable that is known only in probability, and to build mixed distributions (Chapter 6).

### Bayes' theorem

Combining the multiplication rule written both ways with Equation (2.5) gives **Bayes' theorem**:

\begin{equation*}
P(B_i \mid A) = \frac{P(A \mid B_i)\,P(B_i)}{\sum_{j} P(A \mid B_j)\,P(B_j)} .
\tag{2.6}
\end{equation*}

It reverses the direction of conditioning: from the probability of the evidence $A$ given each hypothesis $B_i$, it gives the probability of each hypothesis given the evidence. $P(B_i)$ is the **prior** probability and $P(B_i \mid A)$ the **posterior**.

A flood warning system illustrates why this matters. Suppose a system detects 90% of floods (the **hit rate**) but also issues warnings on 5% of days without floods (the **false-alarm rate**). If floods occur on 1% of days, Bayes' theorem shows that most warnings are false alarms (Example 2.3). The rarer the event, the lower the posterior probability for the same detection skill (Figure 2.2). This is one reason warning systems for rare floods need very low false-alarm rates.

```{code-cell} python
:tags: [hide-input]
prior = np.logspace(-4, -0.5, 200)
fig, ax = plt.subplots(figsize=(6.4, 3.7))
for far, col in zip([0.10, 0.05, 0.01, 0.001], [c2, c1, c3, ink]):
    post = 0.9 * prior / (0.9 * prior + far * (1 - prior))
    ax.semilogx(prior, post, color=col, lw=2.2, label=f"False-alarm rate {far:g}")
ax.set_xlabel("Prior probability of a flood day"); ax.set_ylabel("P(flood | warning)")
ax.set_ylim(0, 1); ax.legend(loc="upper left", fontsize=9)
plt.show()
```

*Figure 2.2. Probability that a flood actually occurs when a warning is issued, from Bayes' theorem, for a system with a hit rate of 0.9 and different false-alarm rates.*

## 2.5 Probability of hydrological events

Many design questions reduce to unions and intersections of simple events.

**Systems in series and in parallel.** A flood-defense system made of several reaches fails if *any* reach fails: a **series** system, whose failure is the union of the reach failures. For independent reaches with failure probabilities $p_i$, it is easiest to compute the complement:

\begin{equation*}
P(\text{system fails}) = 1 - \prod_i (1 - p_i) .
\tag{2.7}
\end{equation*}

A water supply with two independent sources fails only if *both* fail: a **parallel** system, with failure probability $\prod_i p_i$. Series systems are less reliable than their weakest component, parallel systems more reliable than their strongest.

**Exceedance in a number of years.** The same complement argument gives the probability that an event with annual probability $p$ occurs at least once in $n$ independent years, $1 - (1 - p)^n$ (Example 1.1); Chapter 5 develops the full distribution of the number of occurrences.

**Coincident events.** Coastal and estuarine flooding often depends on two events together, such as a river flood coinciding with a storm surge. If the two were independent, their joint probability would be the product of their probabilities. But both are often driven by the same storm, so $P(A \cap B)$ is larger, sometimes much larger, than $P(A)P(B)$. Treating dependent events as independent then underestimates the risk; Chapter 15 introduces copulas to model such dependence.

## 2.6 Worked examples

**Example 2.1 — A levee in three reaches.** A levee consists of three reaches whose annual probabilities of failure are 0.004, 0.006 and 0.010. Assuming the reaches fail independently, what is the annual probability that the levee fails somewhere?

```{code-cell} python
p = np.array([0.004, 0.006, 0.010])
print(f"P(failure) = {1 - np.prod(1 - p):.4f} (sum of the three: {p.sum():.4f})")
```

For small probabilities the result is close to the sum, but always slightly less.

**Example 2.2 — Floods from two storm types.** In a basin, 70% of annual maximum floods come from summer frontal storms and 30% from typhoons. The probability that the annual maximum flood exceeds a certain level is 0.02 if it comes from a frontal storm and 0.12 if it comes from a typhoon. What is the annual exceedance probability, and what fraction of the exceedances are caused by typhoons?

```{code-cell} python
P_type = {"frontal": 0.70, "typhoon": 0.30}
P_exc = {"frontal": 0.02, "typhoon": 0.12}
P_A = sum(P_exc[k] * P_type[k] for k in P_type)
print(f"P(exceedance) = {P_A:.3f}; P(typhoon | exceedance) = {P_exc['typhoon'] * P_type['typhoon'] / P_A:.2f}")
```

**Example 2.3 — A flood warning.** For the warning system of Section 2.4 (hit rate 0.90, false-alarm rate 0.05, flood days 1%), what is the probability of a flood given a warning, and given no warning?

```{code-cell} python
hit, far, pf = 0.90, 0.05, 0.01
p_warn = hit * pf + far * (1 - pf)
print(f"P(flood | warning) = {hit * pf / p_warn:.3f}; P(flood | no warning) = {(1 - hit) * pf / (1 - p_warn):.4f}")
```

**Example 2.4 — Flood and surge.** At an estuary, the annual probability of a river flood above a threshold is 0.05 and that of a storm surge above another threshold is 0.04. Records show that both occur together in about 1% of years. Are the events independent, and what is the probability that at least one occurs in a year?

```{code-cell} python
pA, pB, pAB = 0.05, 0.04, 0.01
print(f"P(A)P(B) = {pA * pB:.4f} vs P(AB) = {pAB}: dependent (ratio {pAB / (pA * pB):.0f})")
print(f"P(A or B) = {pA + pB - pAB:.3f}")
```

## 2.7 Common mistakes

- **Adding probabilities of events that are not exclusive.** Subtract the intersection (Equation 2.2).
- **Multiplying probabilities of dependent events.** The product rule needs independence; floods and surges, or consecutive wet days, are often dependent.
- **Confusing $P(A \mid B)$ with $P(B \mid A)$.** The probability of a warning given a flood is not the probability of a flood given a warning (Example 2.3).
- **Treating exclusive events as independent.** If one excludes the other, they are strongly dependent.
- **Trusting relative frequencies from short records.** Thirty years are too few to pin down a probability of 0.1 (Figure 2.1).

## 2.8 Exercises

```{code-cell} python
:tags: [remove-cell]
glue("f1", float(1 - (1 - 0.002) ** 5), display=False)
glue("f2", float(0.1 * 0.15), display=False)
glue("f3", float(0.3 + 0.25 - 0.12), display=False)
glue("f4a", float(0.8 * 0.4 + 0.3 * 0.6), display=False); glue("f4b", float(0.8 * 0.4 / (0.8 * 0.4 + 0.3 * 0.6)), display=False)
glue("f5", float(0.12 / 0.3), display=False)
```

1. A flood-defense system has five independent sections, each with an annual failure probability of 0.002. What is the annual probability that the system fails?
2. A town has two independent water sources whose probabilities of failing in a drought year are 0.10 and 0.15. What is the probability that both fail?
3. In a given month, the probability of at least one day above 50 mm of rain is 0.30 at station A and 0.25 at station B, and 0.12 that it happens at both. What is the probability that it happens at least at one of them?
4. In El Niño years, which occur with probability 0.4, the probability of a dry summer is 0.8; in other years it is 0.3. What is the probability of a dry summer? Given a dry summer, what is the probability that it was an El Niño year?
5. Using the numbers of Exercise 3, what is the probability of a heavy-rain month at B given one at A?

```{admonition} Answers
:class: dropdown
1. {glue:text}`f1:.4f`.
2. {glue:text}`f2:.3f`.
3. {glue:text}`f3:.2f`.
4. {glue:text}`f4a:.2f`; {glue:text}`f4b:.2f`.
5. {glue:text}`f5:.2f`, much higher than 0.25: the stations are dependent.
```

## 2.9 Further reading

- Benjamin, J. R., and Cornell, C. A. (1970). *Probability, Statistics, and Decision for Civil Engineers*. McGraw-Hill.
- Haan, C. T. (2002). *Statistical Methods in Hydrology* (2nd ed.). Iowa State Press.
- Maity, R. (2022). *Statistical Methods in Hydrology and Hydroclimatology* (2nd ed.). Springer.
- Kolmogorov, A. N. (1933). *Grundbegriffe der Wahrscheinlichkeitsrechnung*. Springer.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
