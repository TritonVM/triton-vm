# FRI Soundness

This page bounds the [knowledge error](soundness.md#round-by-round-soundness) of Triton VM's [interactive oracle proof](soundness.md#the-interactive-oracle-proof) (IOP) for the FRI verifier as implemented. That verifier consists of `FriVerifier` in `triton-vm/src/fri.rs` and the check `CombinationCodewordMismatch` in `Verifier::verify`.

The chapter on [Soundness](soundness.md) bounds the knowledge error for a verifier that performs the checks described in its section [FRI](soundness.md#fri). The implemented verifier performs fewer checks, which section [Verifier Checks](soundness.md#verifier-checks) of that chapter lists. Every transcript that the described verifier accepts is also accepted by the implemented one, but not conversely. So the bounds of that chapter do not carry over.

This page uses the notation of the chapter on Soundness. In particular, $D$ is the domain of the committed words, $\mathbb{F} \subset \mathbb{E}$ are the base field and its extension of degree 3, $N$ is the trace length, $\rho = 1/4$ is the rate, $t = 80$ is the number of queries, $\eta$ is the parameter that sets the radius in section [The Proximity Parameters](soundness.md#the-proximity-parameters), and $\mathsf{RS}_d$ is the code of evaluations on $D$ of polynomials over $\mathbb{E}$ of degree at most $d$. Polynomials are written in the indeterminate $X$. A word on a domain *agrees on a set $S$* with a polynomial if the two take the same value at every point of $S$.

The result of this page is that the implemented verifier has slightly larger errors, under the same theorems and, in the conjectured regime, the same assumptions as the chapter on Soundness. Let $R$ be the number of folding rounds, $D_r := \{x^{2^r} : x \in D\}$, and $\Delta := 2^R - 2$. Section [Degree Bounds](#degree-bounds) shows where $\Delta$ comes from. The analysis of the chapter on Soundness goes through with two changes:

- The low-degree test checks proximity to the Reed-Solomon code of polynomials of degree less than $\rho \vert D \vert + \Delta$, instead of degree less than $\rho \vert D \vert$.
- Every folding round is analyzed on the domain $D_1$ of length $\vert D \vert / 2$, instead of on the domain $D_r$ of length $\vert D \vert / 2^r$.

The section [Concrete Numbers](#concrete-numbers) gives the resulting errors, each recomputed by the test `soundness_chapter_agrees_with_code`. The last section, [Is the Larger Code Necessary?](#is-the-larger-code-necessary), explains why the first change may be an artifact of the proof.

## The Implemented Checks

FRI runs on the words $f_0, \ldots, f_R$ and the folding challenges $z_1, \ldots, z_R$ of section [FRI](soundness.md#fri), on the domains $D_r$ defined above. The prover commits to $f_{r-1}$ before the verifier samples $z_r$. After $z_R$, the prover sends $f_R$ in full, together with a polynomial of degree less than $\rho \vert D_R \vert$, which the final round compares with $f_R$.

For $s \in \mathbb{F} \setminus \{0\}$ and $a, b, z \in \mathbb{E}$, define the *line value*

$$ L_s(a, b; z) := \frac{a + b}{2} + z \, \frac{a - b}{2s} . $$

It is the value at $z$ of the line through the points $(s, a)$ and $(-s, b)$. By the definition of the fold in section [FRI](soundness.md#fri), $\mathrm{fold}(f, z)(s^2) = L_s(f(s), f(-s); z)$ for every word $f$ on a domain that contains $s$ and $-s$.

The verifier samples $t$ query indices. Each index determines a point $x \in D_0$, which is uniformly random in $D_0$, as section [Query Round](soundness.md#query-round) shows. For each query, the verifier does the following. Every value it opens is authenticated against the Merkle root of its word.

1. It opens $f_0(x)$ and $f_0(-x)$. It computes $y(x)$, the value at $x$ of the word that FRI tests, defined in [Round 4](soundness.md#round-4-deep-batching), from the opened rows of the three tables. It checks $f_0(x) = y(x)$ (`CombinationCodewordMismatch`).
2. It computes $v_1 := L_x(f_0(x), f_0(-x); z_1)$, which is $\mathrm{fold}(f_0, z_1)(x^2)$.
3. For $r = 1, \ldots, R - 1$, it opens $f_r(-x^{2^r})$ and computes $v_{r+1} := L_{x^{2^r}}(v_r, f_r(-x^{2^r}); z_{r+1})$.
4. It checks $v_R = f_R(x^{2^R})$ (`FriValidationError::LastCodewordMismatch`).

Steps 2 and 3 only compute values: their openings are authenticated, but they compare nothing. The final round is that of section [Final Round](soundness.md#final-round): it checks that the polynomial sent with $f_R$ has degree less than $\rho \vert D_R \vert$ and agrees with the interpolant of $f_R$ at a random point.

The described verifier performs two more checks. It checks $f_0(-x) = y(-x)$. And for $1 \leqslant r < R$, it opens $f_r(x^{2^r})$ and checks that it equals $v_r$.

With the second check, $v_{r+1}$ is the fold of the committed word $f_r$ at $x^{2^{r+1}}$. Without it, $v_r$ need not be a value of $f_r$. The analysis of [Block et al., Theorem 4.2](https://eprint.iacr.org/2023/1071), cited in the chapter on Soundness, assumes this check.

## The Words on $D_1$

Exchanging $x$ and $-x$ leaves $v_1$ unchanged, because $L_{-s}(b, a; z) = L_s(a, b; z)$. So $v_1$ depends on $x$ only through $w := x^2$. For $r \geqslant 1$, the value $v_{r+1}$ depends on $v_r$, on $x^{2^r} = w^{2^{r-1}}$, and on $f_r(-x^{2^r})$. By induction on $r$, every $v_r$ depends on $x$ only through $w$. For $1 \leqslant r \leqslant R$, write $V_r(w)$ for the value $v_r$ of either $x$ with $x^2 = w$. This defines a word $V_r$ on $D_1$.

In the described verifier, the value of round $r$ is a value of a word on $D_r$. In the implemented verifier, $V_r$ is a word on $D_1$. Two points of $D_1$ with the same power $w^{2^{r-1}}$ may have different values of $V_r$, so $V_r$ is in general not a word on $D_r$. This is why the analysis below works on $D_1$ throughout.

To analyze one step at a time, write $V_{r+1}$ as a line in the challenge $z_{r+1}$. For $w \in D_1$ and either square root $x$ of $w$, define

$$ P_0(w) := \frac{f_0(x) + f_0(-x)}{2} , \qquad Q_0(w) := \frac{f_0(x) - f_0(-x)}{2x} . $$

Replacing $x$ by $-x$ leaves both values unchanged; in $Q_0$, it changes the sign of both numerator and denominator. For $1 \leqslant r < R$, let $w_r := w^{2^{r-1}} \in D_r$, and define

$$ P_r(w) := \frac{V_r(w) + f_r(-w_r)}{2} , \qquad Q_r(w) := \frac{V_r(w) - f_r(-w_r)}{2 w_r} . $$

The definition of the line value gives, for $0 \leqslant r < R$,

$$ V_{r+1} = P_r + z_{r+1} Q_r . $$

The words $P_r$ and $Q_r$ depend on $f_0, \ldots, f_r$ and $z_1, \ldots, z_r$. So they are fixed before the verifier samples $z_{r+1}$. Conversely, $P_r$ and $Q_r$ determine the values they are formed from:

$$ f_0(x) = P_0(x^2) + x \, Q_0(x^2) \quad (x \in D_0) , \qquad V_r(w) = P_r(w) + w_r Q_r(w) \quad (1 \leqslant r < R) . $$

## Degree Bounds

For $1 \leqslant r \leqslant R$, define

$$ K_r := \rho \vert D_1 \vert + 2^{R-1} - 2^{r-1} . $$

Then $K_R = \rho \vert D_1 \vert$ and $K_r = K_{r+1} + 2^{r-1}$. Also $2 K_1 = 2 \rho \vert D_1 \vert + 2^R - 2 = \rho \vert D \vert + \Delta$.

The following two statements hold for every set $S \subseteq D_1$. Each follows by substituting the polynomials it names into the last displayed equation of section [The Words on $D_1$](#the-words-on-d_1).

- For $1 \leqslant r < R$: if $P_r$ and $Q_r$ agree on $S$ with polynomials $p$ and $q$ of degree less than $K_{r+1}$, then $V_r$ agrees on $S$ with $p(X) + X^{2^{r-1}} q(X)$, whose degree is less than $K_{r+1} + 2^{r-1} = K_r$.
- If $P_0$ and $Q_0$ agree on $S$ with polynomials $p$ and $q$ of degree less than $K_1$, then $f_0$ agrees with $F(X) := p(X^2) + X q(X^2)$ at every $x \in D_0$ with $x^2 \in S$. The degree of $F$ is less than $2 K_1 = \rho \vert D \vert + \Delta$.

The word with which step 4 compares $V_R$ also has a degree bound. The final round ensures that $f_R$ is the evaluation on $D_R$ of a polynomial $h$ of degree less than $\rho \vert D_R \vert$. Then the word $w \mapsto f_R(w^{2^{R-1}})$ on $D_1$ is the evaluation of $h(X^{2^{R-1}})$, whose degree is at most $2^{R-1} (\rho \vert D_R \vert - 1) < \rho \vert D_1 \vert = K_R$.

## The Proximity Parameters

The radius $\delta$, the list size $\ell$, and the false witness probability $\varepsilon_\mathsf{fw}$ are those of section [The Proximity Parameters](soundness.md#the-proximity-parameters), with the code $\mathsf{RS}_{\rho \vert D \vert + \Delta - 1}$ in place of $\mathsf{RS}_{\rho \vert D \vert - 1}$. That is, in their definitions, $\rho$ is replaced by $(\rho \vert D \vert + \Delta) / \vert D \vert$, and $\rho_-$ by $(\rho \vert D \vert + \Delta - 1)/\vert D \vert$. List members are tuples of polynomials of degree at most $\rho \vert D \vert + \Delta$.

The folding rounds use other codes, on $D_1$, at the same radius $\delta$. For $1 \leqslant r \leqslant R$, let $\varepsilon_{\mathsf{fold},r}$ be the false witness probability for lines of the code of evaluations on $D_1$ of polynomials of degree less than $K_r$, at radius $\delta$. It is the value of $\varepsilon_\mathsf{fw}$ for the length $\vert D_1 \vert$ and the rate $K_r / \vert D_1 \vert$, with $\eta$ replaced by $\eta_r$, the value for which the regime's definition of $\delta$, applied to this code, gives the radius $\delta$:

- In the proven regime, $\eta_r := (1 - \delta) - \sqrt{(K_r - 1) / \vert D_1 \vert}$. In the formulas of [Proven Regime](soundness.md#proven-regime) for the auxiliary parameter $\theta$ and for $\varepsilon_\mathsf{fw}$, it takes the place of $\eta$, and $(K_r - 1) / \vert D_1 \vert$ takes the place of $\rho_-$. With this $\eta_r$, [Dao–Kominers–Thaler, Theorem 5.12](https://eprint.iacr.org/2026/2056), applied on $D_1$, has the agreement threshold $\lceil (1 - \delta) \vert D_1 \vert \rceil$. So it covers every set of at least $(1 - \delta) \vert D_1 \vert$ points, as the arguments below require.
- In the conjectured regime, $\eta_r := 1 - K_r / \vert D_1 \vert - \delta$. It enters $\varepsilon_\mathsf{fw}$ together with the rate $K_r / \vert D_1 \vert$.

Each $\eta_r$ is at least $\eta$. In the proven regime, $(K_r - 1) / \vert D_1 \vert = (2 K_r - 2)/\vert D \vert$ is less than $(\rho \vert D \vert + \Delta - 1)/\vert D \vert$, and $1 - \delta$ exceeds the square root of the latter by at least $\eta$. In the conjectured regime, $K_r / \vert D_1 \vert \leqslant K_1 / \vert D_1 \vert = (\rho \vert D \vert + \Delta) / \vert D \vert$.

The conjectured regime also requires that $\delta$ is at most the [Elias radius](soundness.md#conjectured-regime) of the code. The Elias radius decreases as the rate increases, and each of these codes has rate at most $(\rho \vert D \vert + \Delta) / \vert D \vert$, the rate of $\mathsf{RS}_{\rho \vert D \vert + \Delta - 1}$. So the requirement holds for each of them.

## The State Function

For $S \subseteq D_1$, let

$$ \mathcal{T}(S) := \{ x \in D_0 : x^2 \in S \text{ and } f_0(x) = y(x) \} . $$

Call $S$ *admissible* if $\vert \mathcal{T}(S) \vert \geqslant (1 - \delta) \vert D \vert$. Each point of $D_1$ has two square roots in $D_0$, so an admissible $S$ has at least $(1 - \delta) \vert D_1 \vert$ points.

The state function is that of section [Conditions](soundness.md#conditions), with the conditions after round 4 and after the folding rounds replaced. A partial transcript is *not* doomed in the following cases.

- **After round 4:** the claimed values satisfy the quotient equation, and $y$ agrees with $\mathsf{RS}_{\rho \vert D \vert + \Delta - 1}$ on a set of at least $(1 - \delta) \vert D \vert$ points.
- **After folding round $r$, for $1 \leqslant r \leqslant R$:** the claimed values satisfy the quotient equation, and $V_r$ agrees on some admissible set with a polynomial of degree less than $K_r$.
- **After the query round (unchanged):** the claimed values satisfy the quotient equation, every check at the queried positions passes (steps 1 and 4, and the authentication of the openings), and the polynomial sent together with $f_R$ is the interpolant of $f_R$.

As in the chapter on Soundness, folding round $r$ consists of the commitment to $f_{r-1}$ and the challenge $z_r$. The condition after folding round $r$ depends on $V_r$ and on $\mathcal{T}(S)$. The word $V_r$ depends on $f_0, \ldots, f_{r-1}$ and $z_1, \ldots, z_r$, and the set $\mathcal{T}(S)$ depends on $f_0$ and $y$.

Neither depends on $f_r$, the prover's next message. So that message does not change the state, as condition (3) of the [definition](soundness.md#round-by-round-soundness) requires.

Every condition from round 4 on requires the quotient equation, whose claimed values do not change after round 4. Each argument below assumes that a later state is not doomed, so the quotient equation holds. A doomed earlier state must then fail the other part of its condition.

## Round-by-Round Errors

### Rounds 1 to 4

The analyses of rounds 1 to 4 in the chapter on Soundness hold as written, with the code $\mathsf{RS}_{\rho \vert D \vert + \Delta - 1}$ in place of $\mathsf{RS}_{\rho \vert D \vert - 1}$ and list members of degree at most $\rho \vert D \vert + \Delta$. This changes $\ell$ and $\varepsilon_\mathsf{fw}$, as defined in the previous section, and the degree bound of the quotient equation, which becomes

$$ d_\alpha := N + k (\rho \vert D \vert + \Delta) + k - 1 , $$

where $k = 4$ is the number of quotient segments. The reconstruction in round 4 needs an agreement set with more points than $\rho \vert D \vert + \Delta$, the degree bound of list members. It has at least $(1 - \delta) \vert D \vert$ points, and $1 - \delta > (\rho \vert D \vert + \Delta) / \vert D \vert$ in both regimes. The same inequality keeps the argument of [The State Function](soundness.md#the-state-function) valid: a list member of the main table agrees with words with values in $\mathbb{F}$ on more points than its degree bound $\rho \vert D \vert + \Delta$, so its coefficients lie in $\mathbb{F}$. The extractor of [Knowledge Soundness](soundness.md#knowledge-soundness) decodes the code $\mathsf{RS}_{\rho \vert D \vert + \Delta}$ in place of $\mathsf{RS}_{\rho \vert D \vert}$.

### Folding Round 1

Let a transcript be doomed after round 4, and not doomed after folding round 1. Then $V_1 = P_0 + z_1 Q_0$ agrees on some admissible set $S$ with a polynomial of degree less than $K_1$. The words $P_0$ and $Q_0$ are fixed before $z_1$, and $S$ has at least $(1 - \delta) \vert D_1 \vert$ points. Apply [mutual correlated agreement](soundness.md#proven-regime) for lines to the code on $D_1$ of polynomials of degree less than $K_1$. Its exceptional challenges do not depend on the agreement set, so it applies although $S$ may depend on $z_1$. Except with probability $\varepsilon_{\mathsf{fold},1}$ over $z_1$, the words $P_0$ and $Q_0$ agree on $S$ with polynomials of degree less than $K_1$.

By the second statement of [Degree Bounds](#degree-bounds), $f_0$ then agrees with a polynomial $F$ of degree less than $\rho \vert D \vert + \Delta$ at every $x$ with $x^2 \in S$. On $\mathcal{T}(S)$, also $y = f_0$, so $y$ agrees with $F$ on $\mathcal{T}(S)$. Since $S$ is admissible, this set has at least $(1 - \delta) \vert D \vert$ points. So the transcript is not doomed after round 4, a contradiction. The error of folding round 1 is therefore at most $\varepsilon_{\mathsf{fold},1}$.

### Folding Rounds 2 to $R$

Let $1 \leqslant r < R$. Let a transcript be doomed after folding round $r$, and not doomed after folding round $r + 1$. Then $V_{r+1} = P_r + z_{r+1} Q_r$ agrees on some admissible set $S$ with a polynomial of degree less than $K_{r+1}$. The words $P_r$ and $Q_r$ are fixed before $z_{r+1}$. Except with probability $\varepsilon_{\mathsf{fold},r+1}$ over $z_{r+1}$, mutual correlated agreement for lines gives polynomials of degree less than $K_{r+1}$ with which $P_r$ and $Q_r$ agree on $S$.

By the first statement of [Degree Bounds](#degree-bounds), $V_r$ then agrees on $S$ with a polynomial of degree less than $K_r$. The set $S$ is admissible, so the transcript is not doomed after folding round $r$, a contradiction. The error of folding round $r + 1$ is therefore at most $\varepsilon_{\mathsf{fold},r+1}$.

All folding rounds apply mutual correlated agreement on $D_1$. In the chapter on Soundness, folding round $r$ applies it on $D_r$, and the errors halve from round to round. Here they do not, so their sum is up to $R$ times the error of one round.

### Query Round

Let a transcript be doomed after folding round $R$, and not doomed after the query round. By the condition after the query round, the polynomial sent with $f_R$ is its interpolant, so $f_R$ is the evaluation of a polynomial $h$ of degree less than $\rho \vert D_R \vert$. Let $S_\mathsf{pass}$ be the set of $w \in D_1$ with $V_R(w) = f_R(w^{2^{R-1}})$. By [Degree Bounds](#degree-bounds), $V_R$ agrees on $S_\mathsf{pass}$ with $h(X^{2^{R-1}})$, a polynomial of degree less than $K_R$.

The transcript is doomed after folding round $R$, so $S_\mathsf{pass}$ is not admissible: $\vert \mathcal{T}(S_\mathsf{pass}) \vert < (1 - \delta) \vert D \vert$. In the IOP, the openings are oracle queries, so a query $x$ passes exactly if it passes steps 1 and 4 of [The Implemented Checks](#the-implemented-checks), that is, exactly if $x \in \mathcal{T}(S_\mathsf{pass})$. The prover fixes $f_R$, and with it this set, before the verifier samples the query indices. So each uniformly random query passes with probability less than $1 - \delta$, and all $t$ queries, sampled independently and with replacement, pass with probability at most

$$ \varepsilon_\mathsf{query} := (1 - \delta)^t . $$

### Final Round

The final round is unchanged, with error $\varepsilon_\mathsf{last}$.

### The Knowledge Error of the IOP

The knowledge error of the IOP and the round-by-round error are given by the formulas of section [The Knowledge Error of the IOP](soundness.md#the-knowledge-error-of-the-iop), with the terms of this page. The knowledge error uses the sum of the $\varepsilon_{\mathsf{fold},r}$. The term $\varepsilon_\mathsf{fold}$ in the round-by-round error is the largest of them.

## Concrete Numbers

The protocol constants are those of section [Concrete Numbers](soundness.md#concrete-numbers) of the chapter on Soundness, and $\eta$ is chosen from the same set $\{2^{-j/50} : 50 \leqslant j < 1500\}$, among the values each regime allows, to minimize $\varepsilon_\mathsf{IOP}$. Each quantity has the meaning it has in the results table of that section, with the changes of this page. All errors are given as $\log_2$:

<!-- fri-soundness-results start -->
| Key | Quantity | Proven, $N = 2^{28}$ | Conjectured, $N = 2^{28}$ | Proven, $N = 2^{21}$ | Conjectured, $N = 2^{21}$ |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `num_folding_rounds` | $R$ | 22 | 22 | 15 | 15 |
| `log2_eta` | $\log_2 \eta$ | -21.86 | -4.20 | -23.22 | -4.44 |
| `log2_list_size` | $\log_2 \ell$ | 19.9 | 20.5 | 22.4 | 24.2 |
| `eps_challenges` | $\varepsilon_{\boldsymbol{\chi}}$ | -139.1 | -138.4 | -143.5 | -141.7 |
| `eps_quotient_weights` | $\varepsilon_{\boldsymbol{\kappa}}$ | -172.1 | -171.5 | -169.6 | -167.8 |
| `eps_ood_point` | $\varepsilon_\alpha$ | -141.0 | -140.3 | -145.4 | -143.6 |
| `eps_deep_weights` | $\varepsilon_{\boldsymbol{\mu\nu\gamma}}$ | -97.0 | -139.5 | -100.0 | -142.8 |
| `eps_folding` | $\max_r \varepsilon_{\mathsf{fold},r}$ | -99.0 | -141.5 | -103.9 | -144.8 |
| `eps_folding_sum` | $\sum_r \varepsilon_{\mathsf{fold},r}$ | -96.3 | -137.1 | -103.6 | -141.0 |
| `eps_extractor` | $\varepsilon_\mathsf{Ext}$ | -141.1 | -140.5 | -145.6 | -143.8 |
| `eps_queries` | $\varepsilon_\mathsf{query}$ | -79.6 | -136.5 | -79.6 | -139.7 |
| `eps_last_round` | $\varepsilon_\mathsf{last}$ | -183.0 | -183.0 | -183.0 | -183.0 |
| `eps_rbr` | $\varepsilon_\mathsf{rbr}$ | -79.55 | -136.54 | -79.55 | -139.72 |
| `eps_iop` | $\varepsilon_\mathsf{IOP}$ | -79.55 | -135.39 | -79.55 | -138.78 |
<!-- fri-soundness-results stop -->

The last row is the knowledge error of Triton VM's IOP with the implemented verifier. Compared with the results table of the chapter on Soundness, the proven error is larger by less than half a bit at both trace lengths. The conjectured error is larger by less than one bit.

In the proven regime, the query term again determines the result, and $1 - \delta$ is slightly larger than $\sqrt{(\rho \vert D \vert + \Delta) / \vert D \vert}$. Here $\Delta / \vert D \vert < 2^R / \vert D \vert = 1 / \vert D_R \vert$, where $\vert D_R \vert = 512$ is the length of the last FRI codeword at both trace lengths. So $(\rho \vert D \vert + \Delta) / \vert D \vert = 1/4 + \Delta / \vert D \vert$ is just below $1/4 + 1/512$, and the query term is about $(1/4 + 1/512)^{t/2} \approx 2^{-79.55}$. The chapter on Soundness has about $(1/4)^{t/2} = 2^{-80}$ instead.

## Is the Larger Code Necessary?

The analysis of the folding rounds applies mutual correlated agreement to the code of all polynomials of degree less than $K_{r+1}$ on $D_1$. But $V_{r+1}$ agrees on $S$ with a polynomial of a special form. For $r + 1 = R$, it is $h(X^{2^{R-1}})$, a polynomial in $X^{2^{R-1}}$. Suppose that, in each round, $P_r$ and $Q_r$ agree on $S$ with polynomials in $X^{2^r}$ of degree less than $\rho \vert D_{r+1} \vert$ in that variable. Then $V_r = P_r + X^{2^{r-1}} Q_r$ is a polynomial in $X^{2^{r-1}}$ of degree less than $2 \rho \vert D_{r+1} \vert = \rho \vert D_r \vert$ in that variable. The degree bounds would then not grow, and $\Delta$ would be $0$.

This requires mutual correlated agreement for the code of polynomials in $X^{2^r}$ of degree less than $\rho \vert D_{r+1} \vert$, evaluated on $D_1$. That code is a Reed-Solomon code on $D_{r+1}$ in which every evaluation point appears $2^r$ times. The works cited in the chapter on Soundness prove mutual correlated agreement for Reed-Solomon codes with distinct evaluation points. This page does not assume it for repeated evaluation points. Whether the implemented verifier is as sound as the described one is therefore open.
