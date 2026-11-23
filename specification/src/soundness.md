# Soundness

In a proof system, a *prover* tries to convince a *verifier* that a *claim* is
true. The verifier is a randomized algorithm that accepts or rejects. The proof
system is *sound with soundness error $\varepsilon$* if, for every false claim,
every prover convinces the verifier with probability at most $\varepsilon$.

For Triton VM, the claim consists of the program digest, the version of Triton
VM, the public input, and the public output. A witness for the claim consists of
a program whose digest is the one in the claim, together with the secret input
(the non-determinism) and the initial RAM under which that program, run on the
claim's public input, halts and produces the claim's public output.

Knowledge soundness strengthens soundness: a prover that convinces the verifier
must know a witness. This page establishes it for the interactive protocol, in
its *straight-line* form. An efficient algorithm $\mathsf{Ext}$, the
*extractor*, receives the claim and the transcript of one interaction, including
the prover's oracle messages in full; these are the oracles that the verifier
has access to. The extractor does not rewind the prover or run it again. The
protocol is *knowledge sound with knowledge error $\varepsilon$* if, for every
claim and every prover, the probability that the verifier accepts and the
extractor does not output a witness is at most $\varepsilon$. For a false claim
no witness exists, so knowledge error $\varepsilon$ implies soundness error
$\varepsilon$.

This page computes the knowledge error of the interactive oracle proof (IOP)
behind Triton VM's STARK (scalable transparent argument of knowledge). That
number is the result. The computation has two steps. The first step bounds the
error of each round of the IOP, using the notion of *round-by-round* soundness
defined below. The second step evaluates the resulting formulas for Triton VM's
parameters. Every number in the second step is recomputed by the test
`soundness_chapter_agrees_with_code`, which also checks the constants used here
against the code. A later section sketches how the round-by-round error carries
over to the non-interactive proof that Triton VM actually produces.

All bounds on Reed-Solomon codes that this page relies on are cited, not
derived. They come in two kinds, and this page reports a number for each:

- The *proven* soundness error uses only theorems.
- The *conjectured* soundness error additionally assumes a conjecture about
  Reed-Solomon codes that is not refuted at the time of writing. The section [Status of the Conjecture](#status-of-the-conjecture)
  explains the choice.

The derivations on this page cover only what is specific to Triton VM: the
table-linking arguments, the quotient equation with its randomized segments, the
nested batching of DEEP quotients, the sampling of query indices, and the
final check of the low-degree test. Each of these terms is defined in the
section that analyzes it.

This page reuses the notation of the chapter on
[Zero-Knowledge](zero-knowledge.md). Every symbol is explained where it first
appears, so that chapter need not be read first. Indices such as $i$ and $j$ are
bound anew in each section.

## Round-by-Round Soundness

Consider an interactive protocol in which the verifier's messages are uniformly
random challenges. A *state function* assigns to every partial transcript the
label *doomed* or *not doomed*, such that (1) the empty transcript of a false
claim is doomed, (2) a complete doomed transcript is rejected by the verifier,
and (3) if a partial transcript is doomed and the prover sends its next message,
the transcript remains doomed. The protocol has *round-by-round soundness error*
$\varepsilon_\mathsf{rbr}$ if there is a state function such that, for every
round and every doomed partial transcript ending before that round's challenge,
the probability over that challenge that the transcript stops being doomed is at
most $\varepsilon_\mathsf{rbr}$
([Chiesa–Yogev, ch. 31](https://hash-based-snargs-book.github.io/)).

A *round* consists of a prover message followed by a verifier challenge, and a
*partial transcript* consists of the messages of the first few rounds. This page
bounds, for each round $i$ separately, the probability $\varepsilon_i$ that the
challenge of round $i$ lifts the doom of a doomed partial transcript. An
accepting transcript of a false claim starts doomed, by (1), and ends not
doomed, by (2), so in some round the challenge must lift the doom. The bound
$\varepsilon_i$ holds for every doomed partial transcript, so it also holds for
whichever one the prover's strategy produces. A union bound over the rounds shows
that the soundness error of the interactive protocol is at most the sum of the
$\varepsilon_i$. The maximum of the $\varepsilon_i$ is the round-by-round error
$\varepsilon_\mathsf{rbr}$, which matters only for the non-interactive proof.

The *knowledge* variant
([Chiesa–Yogev, ch. 31](https://hash-based-snargs-book.github.io/)) requires
condition (1) for every claim, true or false, and weakens the requirement on each
round $i$: the probability over the challenge that a doomed transcript stops
being doomed *and* $\mathsf{Ext}$, given the transcript up to that challenge,
fails to output a witness is at most $\varepsilon_i$. Run $\mathsf{Ext}$ on the
transcript up to every challenge, and output the first witness found. By the
same union bound, the probability that the verifier accepts while all these
runs fail is at most the sum of the $\varepsilon_i$. For a true claim, an
honest transcript leaves the doomed state in round 1. That is allowed, because
$\mathsf{Ext}$ succeeds there. $\mathsf{Ext}$ recognizes a witness by running
the program, in time polynomial in $N$. The section
[Knowledge Soundness](#knowledge-soundness) shows that the state function below
satisfies this variant.

## The Interactive Oracle Proof

In an interactive oracle proof (IOP), the prover may send a long message as an
*oracle*: the verifier does not read it in full, but queries it at a few
positions of its choice. In Triton VM, an oracle is a list of function values,
and the prover *commits* to it by sending the root of a Merkle tree over those
values. Opening a position means sending the value together with a proof that it
is in the tree. [Chiesa–Yogev](https://hash-based-snargs-book.github.io/)
explain both concepts in detail.

The base field $\mathbb{F}$ has $p = 2^{64} - 2^{32} + 1$ elements. All
challenges except the query indices of FRI are sampled uniformly from
$\mathbb{E}$, the extension of
$\mathbb{F}$ of degree 3, so $\vert \mathbb{E} \vert = p^3 \approx 2^{192}$.
The execution of the program is recorded in tables, with columns indexed by
registers and rows indexed by points of the *trace domain* $H$, a
multiplicative subgroup of $\mathbb{F}^\times$ whose order $N$ is a power of
2. The *trace columns* are the main and the auxiliary columns. The constraints
that a correct execution satisfies form the *AIR* (algebraic intermediate
representation), described in the chapter on
[arithmetization](arithmetization.md). The prover commits to each column through
its values on a larger domain $D$, a coset of a multiplicative subgroup,
disjoint from $H$, whose order $\vert D \vert$ is a power of 2 (for the trace
lengths on this page, $\vert D \vert = 8N$): it interpolates the column
on $H$, randomizes
the resulting polynomial as described in section
[The Proximity Parameters](#the-proximity-parameters), and evaluates it on $D$.
The *low-degree test* (LDT) later checks that the committed functions on $D$ are
close to such evaluations. Triton VM uses
[FRI](https://eccc.weizmann.ac.il/report/2017/134) as its LDT, applied to one
word that batches *DEEP quotients*, a technique from
[DEEP-FRI](https://eprint.iacr.org/2019/336).

The IOP has the following rounds. Every round consists of a prover message, then
a verifier challenge.

1. The prover sends the claimed $\log_2 N$. It commits to the main table, which
   gives one function $D \to \mathbb{F}$ for each main column. The verifier
   samples $\boldsymbol{\chi}$, the challenges of the table-linking arguments.
2. The prover commits to the auxiliary table: a function $D \to \mathbb{E}$ for
   each auxiliary column, including the
   [batch randomizer](zero-knowledge.md#batch-randomizer), a random column that
   serves zero-knowledge only. The verifier samples
   $\boldsymbol{\kappa} \in \mathbb{E}^\mathsf{c}$, one weight for each of the
   $\mathsf{c}$ AIR constraints.
3. The prover commits to the randomized quotient table: functions
   $\hat{s}_0, \ldots, \hat{s}_k \colon D \to \mathbb{E}$, the evaluations on $D$
   of $k + 1$ segment polynomials $s_0, \ldots, s_k$, with $k = 4$. The section
   [The Quotient and Its Segments](#the-quotient-and-its-segments) explains
   them. The verifier samples the out-of-domain point $\alpha \in \mathbb{E}$.
4. The prover sends the values $\mathsf{a}_i$ and $\mathsf{b}_i$ claimed for
   trace column $i$ at $\alpha$ and at $g\alpha$, where $g$ generates the trace
   domain $H = \langle g \rangle$. It also sends the values $\sigma_i$ claimed
   for $s_i(\alpha^k)$ with $0 \leqslant i < k$, and the values $\tau_i$ claimed
   for $s_i((\zeta\alpha)^k)$ with $1 \leqslant i \leqslant k$, for a fixed
   constant $\zeta \in \mathbb{F}$. The verifier checks the *quotient equation*
   defined below and samples
   $\boldsymbol{\mu}, \boldsymbol{\nu}, \boldsymbol{\gamma}$, the weights that
   batch the DEEP quotients: one per trace column, one per randomized segment,
   and one for each of the four points $\alpha$, $g\alpha$, $\alpha^k$,
   $(\zeta\alpha)^k$.
5. The prover and the verifier run FRI on the batched DEEP word $y$ defined in
   the section [Round 4](#round-4-deep-batching). FRI has a folding round for
   each of $R$ folding challenges, a query round in which the verifier samples
   $t = 80$ indices, and a final round in which the verifier checks the last
   codeword. The section [FRI](#fri) describes these rounds.

The verifier also checks, at each queried index of the first FRI round, that $y$
is consistent with the opened rows of the main, auxiliary, and quotient tables.

> **Note:** The prover chooses $N$, and the verifier rejects $\log_2 N > 28$.
> This page reports the error for $N = 2^{28}$, the largest accepted trace
> length, and for $N = 2^{21}$, a typical one. The error is not smaller at
> $N = 2^{28}$, mainly because $\vert D \vert$ is larger there. This page does
> not prove that the error grows with $N$ in general, so $N = 2^{28}$ is the
> worst case only of the two cases computed.

## Hypotheses on the Arithmetization

The STARK shows that the prover holds tables that satisfy all AIR constraints.
Whether such tables exist only for true claims is a property of the AIR itself,
and the chapters on [arithmetization](arithmetization.md) and
[table linking](table-linking.md) are about that property. This page does not
re-check the AIR. This section states exactly which two properties of the AIR
the rest of the page relies on. Both properties concern the auxiliary table.

**Main and auxiliary columns.** The main table holds the execution trace, and
the prover commits to it before seeing any challenge. The auxiliary table exists
to prove relations *between* tables, for example “the RAM table records the same
memory accesses as the processor table, in a different order”. Each such
relation is a *table-linking argument*. Most arguments use one auxiliary column
in each of two tables. In a lookup argument, the *client* table looks entries
up, and the *server* table stores them; for the other arguments, this page uses
the same two words for the two sides. Each of these columns is an
*accumulator*: row by row, it combines that table's entries with the challenges
$\boldsymbol{\chi}$ into one running value. A final constraint requires the two
running values in the last rows to be equal. A few arguments compare one
table's running value with a value the verifier computes from the claim, or
check an identity within one table. The batch randomizer is the only auxiliary
column that is not an accumulator.

**An example.** A permutation argument shows that the entries $a_1, \ldots, a_n$
of one table are a reordering of the entries $b_1, \ldots, b_n$ of another. With
a challenge $x$, the two accumulators compute $(x - a_1) \cdots (x - a_n)$ and
$(x - b_1) \cdots (x - b_n)$. If the entries are a reordering, the two products
are equal for every $x$. If not, they are two different polynomials in $x$ of
degree $n$, which are equal for at most $n$ values of $x$. So a random $x$
exposes the difference, except with probability at most
$n / \vert \mathbb{E} \vert$. Evaluation arguments use sums of powers of $x$
instead of products, so that the running value is the evaluation at $x$ of a
polynomial whose coefficients are the entries. Lookup arguments use sums of
fractions: the client adds $1/(x - a_i)$ for every entry it looks up, and the
server adds $m/(x - b)$ for every entry $b$ it stores, where the *multiplicity*
$m$ is the number of times the client looks $b$ up. The principle is the same:
equal multisets give equal sums. When an entry spans several columns, further
challenges first compress it into a single element, so in general the running
values depend on several challenges.

**The defect polynomial.** For argument number $j$, the *defect polynomial*
$\Delta_j$ is the difference between the two final running values, viewed as a
polynomial in the challenges. For lookup arguments, first multiply by all
denominators so that the difference is a polynomial. In the example,
$\Delta_j = (x - a_1) \cdots (x - a_n) - (x - b_1) \cdots (x - b_n)$. The
relation that argument $j$ is meant to prove holds exactly when $\Delta_j$ is
the zero polynomial. For an argument that checks an identity within one table,
$\Delta_j$ is the difference of the identity's two sides. Since the running
values are computed from the main table and the challenges only, the main table
and the claim determine $\Delta_j$ before any challenge is sampled.

The page relies on the following two properties.

- **(AIR1) The prover cannot fake an accumulator.** The AIR constraints on an
  accumulator column fix its first value and how each row's value follows from
  the previous one. (AIR1) states that these constraints leave exactly one
  possibility: if an auxiliary table satisfies all AIR constraints, then every
  accumulator column holds the running value computed honestly from the main
  table. There is one exception. To avoid division, the constraint that adds
  $m/(x - e)$ for a row's entry $e$ and multiplicity $m$ (on the client side,
  $m = 1$) is multiplied by $x - e$. If the
  challenge happens to equal the entry, $x = e$, the constraint for that row
  loses its dependence on the next running value and reduces to $-m = 0$. If
  $m \neq 0$, nobody can satisfy it, and the honest prover fails (a
  completeness error). If $m = 0$, the row's next running value is
  unconstrained, and the prover can choose the final value freely. (AIR1) is claimed only for challenges
  where this does not happen; the section
  [Round 1](#round-1-table-linking-challenges) adds the probability that it does
  happen to the soundness error.
- **(AIR2) The AIR describes Triton VM correctly.** Suppose a main table
  satisfies every AIR constraint that does not involve the challenges, and every
  argument's relation holds, that is, $\Delta_j$ is the zero polynomial for
  every $j$. Then the main table records a correct execution: a program whose
  [Tip5](https://eprint.iacr.org/2023/107) digest is the one in the claim, run
  on the claim's public input, some secret input, and some initial RAM, halts
  with the claim's public output.

The section [Round 1](#round-1-table-linking-challenges) uses the two properties
together. If the main table does not record a correct execution, then (AIR2)
gives an argument $j$ whose $\Delta_j$ is not the zero polynomial. A random
challenge is a root of $\Delta_j$ only with small probability. For every other
challenge, (AIR1) forces the two accumulators of argument $j$ to their honest
running values. These differ, and so the final constraint of argument $j$ fails.

> **Note:** A violation of (AIR1) constitutes a soundness bug in the AIR: some
> row leaves an accumulator unconstrained, and the prover can choose the final
> running value freely. Bugs of this kind have been found and fixed in the past.
> Checking (AIR1) and (AIR2) means auditing the AIR, which this page does not do.

> **Note:** (AIR2) treats the program as part of the witness: the claim says only
> that *some* program with the claimed digest was run. In order for this program
> to be different from the one the verifier has in mind, both programs must hash
> to the same Tip5 digest. In other words, the programs are guaranteed to agree
> up to collision resistance of Tip5. This page does not analyze Tip5's
> collision resistance.

## The Proximity Parameters

A *word* is a function $D \to \mathbb{E}$ from a domain $D$ of points to the
field $\mathbb{E}$. The distance between two words is the fraction of points of
$D$ at which they differ (*relative Hamming distance*), and the distance from a
word to a code is its distance to the nearest codeword. A word is
*$\delta$-close* to a code if that distance is at most $\delta$, that is, if it
agrees with some codeword on at least $(1 - \delta) \vert D \vert$ points, and
*$\delta$-far* otherwise.

The radius $\delta$ is the parameter that every bound below depends on. The
section [Query Round](#query-round) shows that the soundness error of the $t$
queries is $(1 - \delta)^t$. So for a given soundness error, a larger $\delta$
means fewer queries. Two other quantities grow with $\delta$:

- the *list size* $\ell$, the maximum number of codewords within distance
  $\delta$ of one word;
- the *false witness probability* $\varepsilon_\mathsf{fw}$, the maximum
  probability, over uniformly random coefficients, that a linear combination of
  words agrees with a codeword on a set of at least $(1 - \delta) \vert D \vert$
  points while the words do not all agree with codewords on that same set. The
  sections below make this precise for each regime.

The code is a Reed-Solomon code with *rate* $\rho$, the ratio of the number of
coefficients of its polynomials to the length $\vert D \vert$. Bounds on $\ell$
and $\varepsilon_\mathsf{fw}$ are known only up to some radius. Theorems reach
the *Johnson radius* $1 - \sqrt{\rho}$. The strongest unrefuted conjecture
reaches the *Elias radius* $r_\mathsf{E}$, defined in section
[Conjectured Regime](#conjectured-regime) and slightly below $1 - \rho$.

These two radii define the two regimes of this page. The *proven* regime uses a
radius just below the Johnson radius, and the *conjectured* regime a radius at
most the Elias radius. In both, a parameter $\eta > 0$ sets the radius, as
defined in each regime's section. The section
[Concrete Numbers](#concrete-numbers) chooses $\eta$ to minimize the knowledge
error of the IOP.

In Triton VM, the trace has $N$ rows. Randomization adds $h$ random
coefficients to each column polynomial, and Triton VM rounds the resulting
length $N + h$ up to the next power of two. This gives $2N$ for every
$N \geqslant 2^9$, which includes all trace lengths considered on this page.
Triton VM uses $\rho = 1/4$: the domain $D$ is larger than that by the factor
$\rho^{-1} = 4$, so
$\rho \vert D \vert = 2N$.

### Proven Regime

Let $\rho_- := (\rho \vert D \vert - 1)/\vert D \vert$ and define

$$
\begin{aligned}
\delta &:= 1 - \frac{\left\lceil \left(\sqrt{\rho_-} + \eta\right) \vert D \vert \right\rceil}{\vert D \vert} , &
\theta &:= \max\left( \left\lceil \frac{\sqrt{\rho_-}}{2\eta} \right\rceil, 3 \right) + \frac{1}{2} , \\
\varepsilon_\mathsf{fw} &:= \frac{8}{3} \cdot \frac{\vert D \vert \, \theta^3}{\rho_- \, (\vert \mathbb{E} \vert - 1)} , &
\ell &:= \frac{1 - \delta - \rho}{(1 - \delta)^2 - \rho} .
\end{aligned}
$$

Only values of $\eta$ with $(1 - \delta)^2 > \rho$ are admissible. Then also
$1 - \delta > (1 - \delta)^2 > \rho$, so numerator and denominator of $\ell$
are positive.

Write $\mathsf{RS}_d$ for the code of evaluations on $D$ of polynomials over
$\mathbb{E}$ of degree at most $d$. The low-degree test checks proximity to
$\mathsf{RS}_{\rho \vert D \vert - 1}$.

[Dao–Kominers–Thaler, Theorem 5.12](https://eprint.iacr.org/2026/2056) proves
that this code has *mutual correlated agreement*. Take any two words $u, v$ and
a uniformly random scalar $z \in \mathbb{E}$. Except with probability at most
$\varepsilon_\mathsf{fw}$ over $z$, the following holds for every set
$\mathcal{A} \subseteq D$ of at least $(1 - \delta) \vert D \vert$ points: if
$u + zv$ agrees with a codeword on $\mathcal{A}$, then $u$ and $v$ each agree
with a codeword on all of $\mathcal{A}$. The word *mutual* refers to this
preservation of the agreement set.

The $w$-fold *interleaved* code consists of $w$-tuples of codewords. A tuple of
$w$ words agrees with it on a set $\mathcal{A}$ if every word of the tuple agrees
with some codeword on $\mathcal{A}$, with the same set $\mathcal{A}$ for all of
them. The tuple is $\delta$-close to the interleaved code if it agrees with it
on a set of at least $(1 - \delta) \vert D \vert$ points. [Corollary 7.7](https://eprint.iacr.org/2026/2056) of the same paper extends
this result, with the same $\varepsilon_\mathsf{fw}$, to interleaved
Reed-Solomon codes of any width. Through
[BCGM25, Lemma 7.1](https://eprint.iacr.org/2025/2051), it extends further to
*affine families* $u_0 + \sum_j z_j u_j$, where the $u_j$ are words or tuples of
words and the $z_j$ are independent and uniformly random. The conclusion is the
same: except with probability $\varepsilon_\mathsf{fw}$, whenever the
combination agrees with a codeword on a set $\mathcal{A}$ of at least
$(1 - \delta) \vert D \vert$ points, every $u_j$, including $u_0$, agrees with a
codeword on $\mathcal{A}$. The bound does
not grow with the number of words. The quantity $\theta$ is an auxiliary
parameter of that theorem. The quantity $\ell$ is the Johnson bound
([Dao–Kominers–Thaler, eq. (67)](https://eprint.iacr.org/2026/2056)) on the
number of codewords of $\mathsf{RS}_{\rho \vert D \vert}$, or of any
interleaving of it, within distance $\delta$ of a fixed word. (Note the degree
bound $\rho \vert D \vert$ instead of $\rho \vert D \vert - 1$; section
[Round 3](#round-3-out-of-domain-point) explains why.)

Dao–Kominers–Thaler state their results in terms of agreement instead of
distance, and with their own symbols. They translate as follows.

 - Their code length $n$ is $\vert D \vert$, and their degree bound $D$ is
   $\rho \vert D \vert - 1$. (Their $D$ is a degree, not the domain.)
 - Their $\rho_- = D/n$ is $\rho_-$ here. Their $\rho = (D+1)/n$ is the rate
   $\rho$ here.
 - Their slack $\eta_0$ is $\eta$ here. Their agreement fraction is
   $a = \sqrt{\rho_-} + \eta_0$, and their agreement threshold is the integer
   $A = \lceil a n \rceil$. The radius here is $\delta = 1 - A/n$, so
   $1 - \delta$ is $a$ rounded up to a multiple of $1/\vert D \vert$, and a word
   agrees with a codeword on at least $A$ points exactly if it is $\delta$-close
   to it.
 - Their $t = m + 1/2$, with
   $m = \max(\lceil \sqrt{\rho_-}/(2\eta_0) \rceil, 3)$, is $\theta$ here. It is
   renamed because $t$ is the number of queries on this page.
 - Their bound $B_\mathrm{cf}$ on the number of challenges $z$ at which mutual
   correlated agreement fails, a count,
   appears here divided by $\vert \mathbb{E} \vert - 1$, as the probability
   $\varepsilon_\mathsf{fw}$. The denominator $\vert \mathbb{E} \vert - 1$
   instead of $\vert \mathbb{E} \vert$ is what the transfer to affine families
   costs; using it for lines too only loosens the bound.
 - Their list bound $n(A - D)/(A^2 - nD)$, eq. (67), gives $\ell$ here. This is
   the one place where their $D$ is *not* $\rho \vert D \vert - 1$: the list
   size is needed for degree bound $\rho \vert D \vert$, so their $D$ is replaced
   by $\rho \vert D \vert$, and numerator and denominator are divided by
   $\vert D \vert^2$.

> **Note:** Dao–Kominers–Thaler is a 2026 preprint with a Lean formalization.
> The earlier, peer-reviewed bound of
> [BCHKS25, Theorem 1.5](https://eprint.iacr.org/2025/2055) for lines has false
> witness probability of order $\vert D \vert/(\eta^5 \vert \mathbb{E} \vert)$
> instead of $\vert D \vert/(\eta^3 \vert \mathbb{E} \vert)$. Replacing
> $\theta^3$ by $\theta^5$ above and choosing $\eta$ anew changes the proven
> result by less than 0.01 bits, at both trace lengths: the optimal $\eta$ grows
> from about $2^{-22}$ to about $2^{-15}$, which keeps the terms with
> $\varepsilon_\mathsf{fw}$ well below the query term.

### Conjectured Regime

The *$p$-ary entropy function* is
$\mathcal{H}_p(x) = x \log_p (p-1) - x \log_p x - (1-x) \log_p (1-x)$, for the
field size $p$. Over an alphabet with $p$ symbols, $p^{\mathcal{H}_p(x) n}$
approximates the number of words within relative distance $x$ of a fixed word
of length $n$.
[Guruswami–Rudra–Sudan, *Essential Coding Theory*, ch. 3](https://cse.buffalo.edu/faculty/atri/courses/coding-theory/book/)
introduces it. Let $r_\mathsf{E}$ denote the *Elias radius* of rate $\rho$ over
$\mathbb{F}$, defined by $\rho = 1 - \mathcal{H}_p(r_\mathsf{E})$. Beyond this
radius, a counting argument shows that every code of rate $\rho$ over
$\mathbb{F}$ has some word with exponentially many codewords within that
distance. The Elias radius over $\mathbb{F}$ is smaller than the one over
$\mathbb{E}$. A conjecture restricted to a smaller radius claims less, so using
$\mathbb{F}$ is the cautious choice.

Let $\mathcal{H}_2(x) := -x \log_2 x - (1-x) \log_2 (1-x)$ be the binary entropy
function. Restrict $\eta$ to $\eta \geqslant 1 - \rho - r_\mathsf{E}$ and define

$$
\delta := 1 - \rho - \eta , \qquad \ell := 2^{c_2 \mathcal{H}_2(\rho) / \eta} , \qquad \varepsilon_\mathsf{fw} := \frac{\ell \cdot \vert D \vert}{\vert \mathbb{E} \vert - 1} ,
$$

so that $\delta \leqslant r_\mathsf{E}$. The constant $c_2$ is that of the
following conjecture. For words $u, v$ of length $n$, a challenge $z$ is
*exceptional* for the line $u + zv$ at distance $\delta$ if $u + zv$ is
$\delta$-close to the code but $u$ and $v$ do not agree with codewords on one
common set of at least $(1 - \delta) n$ points.

**Conjecture** ([S-two whitepaper, Appendix A.5, Conjectures 1 and 2](https://eprint.iacr.org/2026/532)).
There is a small constant $c_2$ such that the following holds for every prime
field $\mathbb{F}_p$, every Reed-Solomon code of length $n$ and rate $\rho$ over
$\mathbb{F}_p$ or an extension of it, and every $\eta$ with
$1 - \rho - \eta \leqslant r_\mathsf{E}$, where $r_\mathsf{E}$ is the Elias
radius over $\mathbb{F}_p$. At distance $\delta = 1 - \rho - \eta$,
every word has at most $2^{c_2 \mathcal{H}_2(\rho) / \eta}$ codewords within
distance $\delta$, and every line $u + zv$ has at most
$n \cdot 2^{c_2 \mathcal{H}_2(\rho) / \eta}$ exceptional challenges $z$.

The whitepaper states the bounds with an additional factor $c_1$; this page
sets $c_1 = 1$. The conjectured regime assumes this conjecture, and further assumes:

- list members of degree at most $\rho \vert D \vert$, one more than the
  code's degree bound, obey the same list-size bound;
- the count of exceptional challenges also bounds the challenges at which mutual
  correlated agreement fails. At an exceptional challenge, $u$ and $v$ agree
  with codewords on *no* common large set; at a failure of mutual correlated
  agreement, they merely fail to agree on the *particular* set on which
  $u + zv$ agrees. Every exceptional challenge is a failure, but not
  conversely, so this assumption is stronger than the conjecture;
- both bounds extend to interleaved codes and to affine families in the same way
  as the proven ones do.

Dividing the count of exceptional challenges by $\vert \mathbb{E} \vert - 1$
gives $\varepsilon_\mathsf{fw}$. This page uses $c_2 = 1.37$. The section
[Status of the Conjecture](#status-of-the-conjecture) derives this value, and
shows how much the result depends on the form of the conjecture.

## The State Function

A *list member* of a set of committed words is a tuple of polynomials over
$\mathbb{E}$ of degree at most $\rho \vert D \vert$, one per word, that agrees
with all of the words on one common set of at least $(1 - \delta) \vert D \vert$
points. There are at most $\ell$ list members. In the conjectured regime, this
bound is one of the assumptions.

A list member of the main table consists of polynomials with coefficients in
$\mathbb{F}$. To see this, write a polynomial of the list member as
$T_0 + \xi T_1 + \xi^2 T_2$ in a basis $1, \xi, \xi^2$ of $\mathbb{E}$ over
$\mathbb{F}$, with $T_0, T_1, T_2$ over $\mathbb{F}$. It agrees with a main word,
which takes values in $\mathbb{F}$, on more than $\rho \vert D \vert$ points.
So $T_1$ and $T_2$ vanish on more points than their degree, and are zero. A
list member of the main table *encodes a valid execution* if its evaluations on
$H$ form a main table that records a correct execution in the sense of (AIR2).

The verifier never sees list members. It sees only the committed words and the
entries it opens. List members connect the two, and union bounds over them
apply, because each word is committed before the next challenge, which fixes its
list before that challenge.

### Conditions

"After round $i$" means after the verifier's challenge of round $i$. The
polynomials $C_j$ and $Z_j$ are defined in [Round 2](#round-2-quotient-weights),
and the quotient equation in [Round 3](#round-3-out-of-domain-point).

The state of a partial transcript depends only on the rounds it has completed.
A partial transcript that has completed round $i$, followed by any part of the
prover's next message, has the state "after round $i$". A partial
transcript that has not completed round 1, including the empty one, is doomed.
A partial transcript is *not* doomed exactly in the following cases.

- **After round 1:** some list member of the main table, together with some
  auxiliary table, satisfies all AIR constraints on $H$.
- **After round 2:** for some list member of the main and auxiliary tables,
  $\sum_j \kappa_j C_j / Z_j$ is a polynomial.
- **After round 3:** $\alpha$ is bad (defined in
  [Round 3](#round-3-out-of-domain-point)), or some list member of the three
  tables satisfies the quotient equation at $\alpha$ with its own values in
  place of the claimed ones.
- **After round 4:** the claimed values satisfy the quotient equation, and $y$
  agrees with $\mathsf{RS}_{\rho\vert D\vert - 1}$ on a set of at least
  $(1 - \delta) \vert D \vert$ points.
- **After folding round $r$, for $1 \leqslant r \leqslant R$:** the claimed
  values satisfy the quotient equation, and the state of FRI after round $r$
  in the sense of
  [Block et al., Theorem 4.2](https://eprint.iacr.org/2023/1071) is not doomed.
  Informally, that state asks whether the latest fold
  $\mathrm{fold}(f_{r-1}, z_r)$ (defined in [FRI](#fri)) agrees with the
  Reed-Solomon code on $D_r$ of degree bound $\rho \vert D_r \vert - 1$ on a
  large set of points $x^{2^r}$, where $x$ ranges over the positions of $D_0$
  that are consistent up to round $r - 1$ (defined in
  [Folding Rounds](#folding-rounds)). Block et al. weigh these positions
  precisely; this page does not restate their bookkeeping.
- **After the query round:** the claimed values satisfy the quotient equation,
  every check at the queried positions passes, and the polynomial sent together
  with the last word $f_R$ is the interpolant of $f_R$.
- **After the final round (complete transcript):** the verifier accepts.

Conditions (1), (2) and (3) of the definition hold. Partial transcripts before
the end of round 1 are doomed, which gives (1) and, for those transcripts, (3).
A prover message does not change the state, because the state depends only on
completed rounds; this gives (3) for all other transcripts. A complete
transcript is doomed exactly if the verifier rejects, which gives (2).

### Intuition

Apart from the clause for a bad $\alpha$, each condition says that some way to
make the verifier accept still exists.
Rounds 1 to 3 show that, for a false claim, no list member passes the checks,
except with small probability. Round 4 and FRI supply the other half: if the
verifier accepts, then, except with small probability, the committed words
agree with a list member whose values at the out-of-domain points are the
claimed ones. Together, the two halves contradict each other.

The condition after round 3 counts a bad $\alpha$ as escaping doom. This is
conservative: it adds the probability of a bad $\alpha$ to the error of round 3,
so the later rounds may assume a good $\alpha$. The quotient equation appears
again in the conditions after round 4, because a failed equation dooms the
transcript for good: the verifier rejects at the end.

The conditions for the FRI rounds track positions, not just words, because the
prover chooses each committed word $f_r$ freely. A prover that commits a
codeword $f_r$ unrelated to the fold of $f_{r-1}$ escapes the folding check
only by creating inconsistencies, which the queries detect. Block et al. make
this precise; a reader who wants the complete formal argument for these rounds
finds it there. The sections below give the error of each round.

## Round-by-Round Errors

### Round 1: Table-Linking Challenges

Fix a list member of the main table that does not encode a valid execution. If
it violates an AIR constraint that does not involve $\boldsymbol{\chi}$, no
auxiliary table can make it satisfy the AIR. Otherwise, by (AIR2), there is an
argument $j$ with $\Delta_j \neq 0$. By (AIR1), if neither
$\Delta_j(\boldsymbol{\chi}) = 0$ nor a denominator of argument $j$ vanishes,
then the accumulators of argument $j$ hold their honest values. Their final
running values then differ, and the final constraint linking them fails.

The *Schwartz–Zippel lemma* states that a non-zero polynomial of total degree
$d$ in $v$ variables vanishes at a uniformly random point of $\mathbb{E}^v$ with
probability at most $d / \vert \mathbb{E} \vert$
([Chiesa–Yogev](https://hash-based-snargs-book.github.io/) state and prove it).
Every compressed entry is a linear combination of main-table cells with
challenges as coefficients, so each denominator $x - (\ldots)$ is a polynomial
of degree 1 in $\boldsymbol{\chi}$, and it is non-zero because of the term $x$.
Apply the lemma to $\Delta_j$ and to each denominator. A union bound over these
events gives

$$
\Pr_{\boldsymbol{\chi}} [\text{argument } j \text{ passes}] \leqslant \frac{\deg \Delta_j + n_j}{\vert \mathbb{E} \vert} ,
$$

where $n_j$ is the number of denominators of argument $j$ (zero for
[permutation](permutation-argument.md) and
[evaluation arguments](evaluation-argument.md)).

The following table bounds both quantities. A *term* is one element entering an
accumulator. Each table has $N$ rows. Each instruction moves at most 10 elements
between the processor and the op stack or RAM, so a processor row contributes at
most 10 terms to those arguments.

| Argument | Kind | Terms per row (client + server) | $\deg \Delta_j + n_j$ at most |
| :--- | :--- | :--- | ---: |
| [Program attestation](program-attestation.md) | evaluation | program table: 1 chunk per 10 rows; hash table: 1 chunk per 6 rows | $N/6 + 9$ |
| Standard input, standard output | evaluation | $5$, claim | $\max(5N, \text{length of the public input or output})$ |
| Instruction lookup | lookup | $1 + 1$ | $4N$ |
| Op stack, RAM | permutation | $10 + 1$ | $10N$ |
| Jump stack | permutation | $1 + 1$ | $N$ |
| Hash input, hash digest, sponge | evaluation | $1 + 1$ | $N$ |
| Hash table to cascade table | lookup | $16 + 1$ | $34N$ |
| Cascade table to lookup table | lookup | $2 + 1$ | $6N$ |
| Processor to U32 table | lookup | $2 + 1$ | $6N$ |
| Clock jump differences | lookup | $3 + 1$ | $8N$ |
| Lookup table contents | evaluation | $1$, public | $N$ |
| Program digest | evaluation | 5 elements, claim | $5$ |
| RAM pointer contiguity | identity in one table | — | $2N$ |

The column "Terms per row" lists the client's terms, then the server's. "Claim"
and "public" mean that the verifier computes that side itself, from the claim or
from a fixed public table.

- **Permutation and evaluation arguments.** Each factor $x - (\ldots)$ has
  degree 1, and in an evaluation argument each compressed entry takes the place
  of one power of $x$. So $\deg \Delta_j$ is at most the larger side's number of
  terms, and the compression challenges add nothing.
- **Lookup arguments.** A [lookup argument](lookup-argument.md) with $n_j$
  denominators, one per term, becomes a polynomial of degree at most $n_j - 1$
  after clearing the denominators, so $\deg \Delta_j + n_j < 2 n_j$.
- **Program attestation.** It compresses the $k$-th chunk of 10 instructions,
  $a_{k,0}, \ldots, a_{k,9}$, into
  $\mathrm{chunk}_k(\beta) := \beta^{10} + \sum_i a_{k,i} \beta^{9-i}$, a
  polynomial of degree 10 in a compression challenge $\beta$. With $K$ chunks,
  the running value is $x^K + \sum_k x^{K-k} \mathrm{chunk}_k(\beta)$, whose
  terms have total degree at most $(K - 1) + 10 = K + 9$.
  The hash table absorbs one chunk per permutation of Tip5, and records each
  permutation in 6 rows, one for the input and one after each of Tip5's 5
  rounds. So there are at most $N/6$
  chunks.
- **RAM pointer contiguity.** This argument checks that the regions of constant
  RAM pointer are contiguous; see [Contiguity of Memory-Pointer
  Regions](contiguity-of-memory-pointer-regions.md). All polynomials here are
  in the argument's challenge $x$. Let $f(x) := \prod_k (x - \pi_k)$, where the
  $\pi_k$ are the pointer values at which a new region starts, and let $f'$ be
  its formal derivative. The regions are contiguous exactly if no $\pi_k$
  repeats,
  that is, if $f$ and $f'$ have no common root, that is, if there are
  polynomials $b_0, b_1$ with $b_0 f + b_1 f' = 1$ (Bézout's identity). The
  prover writes the coefficients of $b_0$ and $b_1$ into main columns, and
  accumulator columns evaluate $f$, $f'$, $b_0$, and $b_1$ at the challenge.
  Here $\Delta_j := b_0 f + b_1 f' - 1$, which the main table determines, and
  whose degree is at most $2N - 2$.

The largest entry is the lookup argument between the
[hash table](hash-table.md), with its 16 cascade clients per row, and the
[cascade table](cascade-table.md).

Let $\varepsilon_{\boldsymbol{\chi}}$ bound the probability over
$\boldsymbol{\chi}$ that a doomed transcript is not doomed after round 1. A list
member of the main table that does not encode a valid execution has a failing
argument $j$, which is fixed before $\boldsymbol{\chi}$ is sampled. The
challenge makes argument $j$ pass with probability at most
$(\deg \Delta_j + n_j)/\vert \mathbb{E} \vert$, which is at most
$34N/\vert \mathbb{E} \vert$, the largest entry of the table. A union bound over
the at most $\ell$ list members gives

$$ \varepsilon_{\boldsymbol{\chi}} := \frac{34 N \ell}{\vert \mathbb{E} \vert} . $$

> **Note:** Let $L$ be the length of the claim's public input or output,
> whichever is longer. A true claim has $L \leqslant 5N$, since `read_io` and
> `write_io` move at most 5 elements per row. A false claim can have any $L$,
> because the verifier does not compare $L$ with $N$. The bound above holds for
> $L \leqslant 34N$. For larger $L$, replace $34N$ by $L$. Each doubling of $L$
> beyond $34N$ costs one bit of security. Nothing in the protocol bounds $L$,
> but the verifier reads the whole public input and output, so $L$ is at most
> what the verifier is willing to process. A verifier that also rejects
> $L > 34N$ keeps the bound above.

### Round 2: Quotient Weights

Row $i$ of a table corresponds to the point $g^i$ of $H$. The *zerofier* of a
set of rows is the polynomial $\prod (X - g^i)$ over those rows; it vanishes
exactly on them. Let $Z_H(X) = X^N - 1$, the zerofier of all rows.

Fix a list member of the main and auxiliary tables that violates some AIR
constraint. For each constraint $j$, let $C_j(X)$ be the polynomial obtained by
substituting this list member's polynomials into constraint $j$. A constraint on
two consecutive rows reads the current row from $T_i(X)$ and the next row from
$T_i(gX)$, where $T_i$ is the list member's polynomial for column $i$. Let
$Z_j(X)$ be the zerofier of the rows on which constraint $j$ must hold: the
first row, all rows, all rows but the last, or the last row. Each $Z_j(X)$
divides $Z_H(X)$, so $Z_H(X) / Z_j(X)$ is a polynomial.

Let $\mathrm{rem}_j(X)$ be the remainder of $(Z_H(X) / Z_j(X)) C_j(X)$ modulo $Z_H(X)$.
Its degree is less than $N$, so its coefficient vector has length $N$. Let $M$
be the $N \times \mathsf{c}$ matrix whose column $j$ is the coefficient vector
of $\mathrm{rem}_j(X)$. Since
$\sum_j \kappa_j C_j(X) / Z_j(X) = \frac{1}{Z_H(X)} \sum_j \kappa_j \frac{Z_H(X)}{Z_j(X)} C_j(X)$,
this sum is a polynomial exactly if $Z_H(X)$ divides
$\sum_j \kappa_j \frac{Z_H(X)}{Z_j(X)} C_j(X)$, that is, exactly if
$M \boldsymbol{\kappa} = 0$.

For a violated constraint $j$, $Z_j(X)$ does not divide $C_j(X)$. Multiplying
both by $Z_H(X)/Z_j(X)$ shows that $Z_H(X)$ does not divide
$(Z_H(X)/Z_j(X)) C_j(X)$, so $\mathrm{rem}_j(X) \neq 0$. Hence $M \neq 0$, and its kernel
is a proper subspace of $\mathbb{E}^\mathsf{c}$. A uniformly random
$\boldsymbol{\kappa}$ lies in that kernel with probability at most
$1/\vert\mathbb{E}\vert$.

In a doomed transcript after round 1, no list member of the main table satisfies
all AIR constraints together with *any* auxiliary table. The main-table part of
a list member of the main and auxiliary tables is a list member of the main
table. So every list member of the main and auxiliary tables violates some
constraint. A union bound over the at most $\ell$ list members shows that the
round-by-round soundness error of round 2 is at most

$$ \varepsilon_{\boldsymbol{\kappa}} := \frac{\ell}{\vert \mathbb{E} \vert} . $$

### Round 3: Out-of-Domain Point

#### The Quotient and Its Segments

The *quotient* is $q(X) := \sum_j \kappa_j C_j(X) / Z_j(X)$, for the honest
tables. It is a polynomial exactly when the condition of round 2 holds, and its
degree is about $\mathsf{d} \rho \vert D \vert$, where $\mathsf{d}$ is the
maximal degree of an AIR constraint. That exceeds the degree bound of the LDT,
so the prover splits $q$ into $k = \mathsf{d}$ *segments* $q_i$ of degree less
than $\rho \vert D \vert$, such that

$$ q(X) = \sum_{i=0}^{k-1} X^i q_i(X^k) . $$

Most STARKs commit to $q_0, \ldots, q_{k-1}$ and let the verifier check
$q(\alpha) = \sum_i \alpha^i q_i(\alpha^k)$ at a random point $\alpha$. Triton
VM differs here for the sake of zero-knowledge, not soundness: revealing the
values $q_i(\alpha^k)$ would leak information about the trace. The section
[Quotient Table Randomization](zero-knowledge.md#quotient-table-randomization)
of the chapter on zero-knowledge describes the remedy. The prover samples a
random polynomial $s_k$ and sets
$s_i(X) := q_i(X) - \zeta^i s_{i+1}(\zeta^k X)$ for $i = k-1, \ldots, 0$, where
$\zeta$ is a fixed public constant chosen in that chapter. Solving this
definition for $q_i$ and substituting it into the segmentation gives

$$ q(X) = \sum_{i=0}^{k-1} X^i s_i(X^k) + \sum_{i=0}^{k-1} (\zeta X)^i s_{i+1}\big((\zeta X)^k\big) . $$

This identity explains the values of round 4: $s_0, \ldots, s_{k-1}$ are needed
at $\alpha^k$, and $s_1, \ldots, s_k$ at $(\zeta\alpha)^k$.

For soundness, only the shape of this identity matters. A cheating prover may
commit to any words in place of the $\hat{s}_i$. The analysis below treats the
segment polynomials of a list member as arbitrary polynomials of degree at most
$\rho \vert D \vert$, and uses neither the randomness of $s_k$ nor any property
of $\zeta$.

#### The Quotient Equation

The verifier evaluates each constraint $j$ on the claimed rows
$(\mathsf{a}_i)_i$ and $(\mathsf{b}_i)_i$, which gives a value
$\bar{C}_j$, divides by $Z_j(\alpha)$, and checks the *quotient equation*

$$
\sum_{j} \kappa_j \frac{\bar{C}_j}{Z_j(\alpha)} = \sum_{i=0}^{k-1} \alpha^i \sigma_i + \sum_{i=0}^{k-1} (\zeta\alpha)^i \tau_{i+1} .
$$

Fix a list member: trace polynomials $T_i$ and segment polynomials
$S_0, \ldots, S_k$, each of degree at most $\rho \vert D \vert$. Let $C_j(X)$
be as in round 2. Define

$$
\begin{aligned}
\Phi(X) &:= Z_H(X) \sum_j \kappa_j \frac{C_j(X)}{Z_j(X)} , \\
\Psi(X) &:= Z_H(X) \left( \sum_{i=0}^{k-1} X^i S_i(X^k) + \sum_{i=0}^{k-1} (\zeta X)^i S_{i+1}\big((\zeta X)^k\big) \right) .
\end{aligned}
$$

Both are polynomials. The factor $Z_H(X)$ clears the denominators of $\Phi(X)$,
and is applied to both sides so that the quotient equation at $\alpha \notin H$
is equivalent to $\Phi(\alpha) = \Psi(\alpha)$. Since $Z_H(X) / Z_j(X)$ has
degree at most $N - 1$, the degree of $\Phi(X)$ is at most
$\mathsf{d} \rho \vert D \vert + N - 1$. The degree of $\Psi(X)$ is at most
$N + k \rho \vert D \vert + k - 1$. Both are therefore at most of degree

$$ d_\alpha := N + \max\big(\mathsf{d} \rho \vert D \vert - 1, \; k \rho \vert D \vert + k - 1\big) . $$

Since $k = \mathsf{d}$, this is $d_\alpha = N + k \rho \vert D \vert + k - 1$.

Suppose the claimed values agree with the list member:
$\sigma_i = S_i(\alpha^k)$, $\tau_i = S_i((\zeta\alpha)^k)$,
$\mathsf{a}_i = T_i(\alpha)$, and $\mathsf{b}_i = T_i(g\alpha)$. Then, for
$\alpha \notin H$, the quotient equation states $\Phi(\alpha) = \Psi(\alpha)$.
If $\Phi(X) \neq \Psi(X)$, this happens with probability at most
$d_\alpha / \vert \mathbb{E} \vert$. If $\Phi(X) = \Psi(X)$, then
$\sum_j \kappa_j C_j(X) / Z_j(X)$ is a polynomial, and the list member passes
round 2's condition.

#### The Error of Round 3

[Round 4](#round-4-deep-batching) divides by $X - \alpha$, $X - g\alpha$,
$X - \alpha^k$, and $X - (\zeta\alpha)^k$ on $D$, so it needs these four
out-of-domain points to lie outside $D$. Evaluating $Z_j(\alpha)$ in the
denominator needs $\alpha \notin H$. Call $\alpha$ *bad* if one of these
conditions fails. The condition $\alpha \in H$ holds for $N$ values of
$\alpha$. The conditions $\alpha \in D$ and $g\alpha \in D$ hold for
$\vert D \vert$ values each, and $\alpha^k \in D$ and $(\zeta\alpha)^k \in D$
for at most $k \vert D \vert$ values each. In total, $\alpha$ is bad for at most
$N + (2k+2)\vert D \vert$ values.

Let $\varepsilon_\alpha$ bound the probability over $\alpha$ that a doomed
transcript is not doomed after round 3. By the state function, this happens in
only two cases. In the first, $\alpha$ is bad. By the count above, this event
has probability at most $(N + (2k+2)\vert D \vert)/\vert\mathbb{E}\vert$, and it
does not depend on the list. In the second, some list member satisfies the
quotient equation at $\alpha$ with its own values. In a doomed transcript, no
list member has $\Phi(X) = \Psi(X)$. Otherwise
$\sum_j \kappa_j C_j(X) / Z_j(X)$ would be a polynomial for its main and
auxiliary part, which is a list member of those two tables, and the transcript
would not be doomed after round 2. So each list member satisfies
the equation with probability at most $d_\alpha/\vert\mathbb{E}\vert$, and a
union bound over the at most $\ell$ list members gives
$\ell \, d_\alpha / \vert\mathbb{E}\vert$. Adding the two cases,

$$
\varepsilon_\alpha := \frac{\ell \, d_\alpha + N + (2k+2) \vert D \vert}{\vert \mathbb{E} \vert} .
$$

> **Note:** The LDT checks for degree at most $\rho \vert D \vert - 1$. An
> honest DEEP quotient (defined in the next section) has degree at most
> $\rho \vert D \vert - 2$, since the trace and segment polynomials have degree
> at most $\rho \vert D \vert - 1$. A polynomial reconstructed from a DEEP
> quotient of degree at most $\rho \vert D \vert - 1$ therefore has degree at
> most $\rho \vert D \vert$, one more than the honest one. This is why list
> members have degree at most $\rho \vert D \vert$. The extra degree enlarges
> $d_\alpha$ by a negligible amount. It also shrinks the denominator of the
> proven list size. Since $1 - \delta \approx \sqrt{\rho_-} + \eta$ and
> $\sqrt{\rho} = 1/2$, the Johnson denominator $(1 - \delta)^2 - \rho_-$ for the
> degree bound $\rho \vert D \vert - 1$ is about $2 \sqrt{\rho}\,\eta = \eta$,
> while the denominator $(1 - \delta)^2 - \rho$ for the degree bound
> $\rho \vert D \vert$ is about $\eta - 1/\vert D \vert$, because
> $\rho - \rho_- = 1/\vert D \vert$. At $N = 2^{21}$, the optimal
> $\eta = 2^{-23}$ equals $2/\vert D \vert$, so the denominator halves and $\ell$
> doubles. The numbers in section [Concrete Numbers](#concrete-numbers) include
> both effects.

### Round 4: DEEP Batching

The verifier computes the word that FRI tests from the committed words and the
claimed values. Let $\hat{t}_i$ be the committed word of trace column $i$, and
define the *elementary DEEP words*

$$
\begin{aligned}
e_{0,i}(X) &:= \frac{\hat{t}_i(X) - \mathsf{a}_i}{X - \alpha} , &
e_{1,i}(X) &:= \frac{\hat{t}_i(X) - \mathsf{b}_i}{X - g\alpha} , \\
e_{2,i}(X) &:= \frac{\hat{s}_i(X) - \sigma_i}{X - \alpha^k} \quad (0 \leqslant i < k) , &
e_{3,i}(X) &:= \frac{\hat{s}_i(X) - \tau_i}{X - (\zeta\alpha)^k} \quad (1 \leqslant i \leqslant k) .
\end{aligned}
$$

Each is a *DEEP quotient*: if the committed word is the evaluation of a
polynomial whose value at the point is the claimed one, the division is exact
and the result is again a low-degree polynomial. Set $e_{2,k} := 0$ and
$e_{3,0} := 0$. The word that FRI tests is

$$
y := \sum_{j=0}^{3} \gamma_j Y_j , \qquad
Y_0 := \sum_i \mu_i e_{0,i} , \quad Y_1 := \sum_i \mu_i e_{1,i} , \quad
Y_2 := \sum_{i=0}^{k} \nu_i e_{2,i} , \quad Y_3 := \sum_{i=0}^{k} \nu_i e_{3,i} .
$$

The batching is nested: the effective weight of $e_{j,i}$ is a product,
$\gamma_j \mu_i$ or $\gamma_j \nu_i$, and the same $\mu_i$ multiplies both DEEP
quotients of trace column $i$. This structure is specific to Triton VM.

The analysis groups the elementary DEEP words by column. For a trace column $i$,
let $\mathbf{e}_i := (e_{0,i}, e_{1,i}, 0, 0)$, and for a segment $i$, let
$\mathbf{e}'_i := (0, 0, e_{2,i}, e_{3,i})$. Each is a 4-tuple of words, to be
compared with the 4-fold interleaved code. Then

$$ (Y_0, Y_1, Y_2, Y_3) = \sum_i \mu_i \mathbf{e}_i + \sum_i \nu_i \mathbf{e}'_i $$

is an affine family over the interleaved code, with $u_0 = 0$ and one
coefficient per trace column and per segment. Likewise,
$y = \sum_j \gamma_j Y_j$ is an affine family over the code
$\mathsf{RS}_{\rho\vert D\vert - 1}$ itself.

Apply mutual correlated agreement for affine families twice.

The weights $\boldsymbol{\mu}, \boldsymbol{\nu}, \boldsymbol{\gamma}$ are
sampled together. The first step holds for every fixed $\boldsymbol{\mu}$ and
$\boldsymbol{\nu}$, so it also holds for random ones.

1. Except with probability $\varepsilon_\mathsf{fw}$ over $\boldsymbol{\gamma}$:
   if $y$ agrees with $\mathsf{RS}_{\rho\vert D\vert - 1}$ on a set
   $\mathcal{A}$ with $\vert \mathcal{A} \vert \geqslant (1 - \delta) \vert D \vert$,
   then the interleaved word $(Y_0, \ldots, Y_3)$ agrees with the interleaved
   code on $\mathcal{A}$.
2. Except with probability $\varepsilon_\mathsf{fw}$ over
   $\boldsymbol{\mu}, \boldsymbol{\nu}$: every $\mathbf{e}_i$ and every
   $\mathbf{e}'_i$ then agrees with the interleaved code on $\mathcal{A}$.

Since mutual correlated agreement preserves the agreement set exactly, the two
steps compose. Outside these two events, every elementary DEEP word agrees on
the one set $\mathcal{A}$ with a polynomial of degree at most
$\rho \vert D \vert - 1$.

The set $\mathcal{A}$ is shared by all columns and both points. This is what
allows reconstruction. Suppose $e_{0,i}$ agrees on $\mathcal{A}$ with a
polynomial $E(X)$. Then $T_i(X) := \mathsf{a}_i + (X - \alpha) E(X)$ has degree
at most $\rho \vert D \vert$, satisfies $T_i(\alpha) = \mathsf{a}_i$, and agrees
with $\hat{t}_i$ on $\mathcal{A}$, because $\alpha \notin D$. In the same way,
$e_{1,i}$ gives $T'_i$ with $T'_i(g\alpha) = \mathsf{b}_i$ that agrees with
$\hat{t}_i$ on $\mathcal{A}$. Since
$\vert \mathcal{A} \vert \geqslant (1 - \delta) \vert D \vert > \rho \vert D \vert$,
two polynomials of degree at most $\rho \vert D \vert$ that agree on
$\mathcal{A}$ are equal, so $T_i = T'_i$. The same holds for the segments; for
$\hat{s}_0$ and $\hat{s}_k$, which have only one DEEP quotient each, there is
nothing to compare. The tuple of these polynomials is a list member whose
values at the out-of-domain points are exactly the claimed values. Its
main-table part has coefficients in $\mathbb{F}$, by the argument in
[The State Function](#the-state-function).

Now combine this with the state. Let the transcript be doomed after round 3,
and suppose it is not doomed after round 4. Then the claimed values satisfy the
quotient equation, and $y$ agrees with the code on some large set
$\mathcal{A}$. Outside the two events above, the reconstructed list member
satisfies the quotient equation with its own values, since these are the
claimed ones. That contradicts the doomed state after round 3. So the
transition happens only inside one of the two events, and a union bound gives

$$ \varepsilon_{\boldsymbol{\mu\nu\gamma}} := 2 \varepsilon_\mathsf{fw} . $$

> **Note:** The batch randomizer is one more auxiliary column with its own
> weight $\mu_i$. It is not constrained by the AIR, and nothing in the above
> depends on its values.

### FRI

FRI tests the word $y$ on $D_0 := D$. Right after round 4, the prover commits
to a word $f_0$ on
$D_0$, which should equal $y$. The verifier cannot compute $y$ everywhere, but
at each queried position it computes $y$ from the opened rows and checks that
$f_0$ agrees.
Then, for $r = 1, \ldots, R$, the verifier samples a folding challenge
$z_r \in \mathbb{E}$, and the prover commits to a word $f_r$ on the domain
$D_r := \{x^2 : x \in D_{r-1}\}$ of half the size. In Triton VM, $D_0$ is a coset
of a multiplicative subgroup of order a power of 2, so each $D_{r-1}$ is closed
under $x \mapsto -x$. The honest $f_r$ is the *fold*

$$
\mathrm{fold}(f_{r-1}, z_r)(x^2) := u_\mathsf{even}(x^2) + z_r \, u_\mathsf{odd}(x^2) , \qquad
u_\mathsf{even}(x^2) := \frac{f_{r-1}(x) + f_{r-1}(-x)}{2} , \quad
u_\mathsf{odd}(x^2) := \frac{f_{r-1}(x) - f_{r-1}(-x)}{2x} .
$$

If $f_{r-1}$ is the evaluation of a polynomial of degree less than $2d$, the fold
is the evaluation of a polynomial of degree less than $d$. The number of folding
rounds $R$ is chosen so that the last word $f_R$ lives on $D_R$ of length
$\vert D_R \vert = \vert D \vert / 2^R = 512$. The prover sends $f_R$ in full,
together with a polynomial of degree less than $\rho \vert D_R \vert = \vert D_R \vert / 4$,
before the query round.

#### Folding Rounds

Call a position $x \in D_0$ *consistent up to round $r$* if, along its path
$x, x^2, x^4, \ldots$, the committed words agree with what they should be:
$f_0(\pm x) = y(\pm x)$, and $f_s(x^{2^s}) = \mathrm{fold}(f_{s-1}, z_s)(x^{2^s})$
for $1 \leqslant s \leqslant r$. The queries check exactly this, at random
positions.

Folding round $r$ consists of the prover's commitment to $f_{r-1}$ (for $r = 1$,
the commitment to $f_0$ right after round 4) and the challenge $z_r$. Suppose
the transcript is doomed before $z_r$. Informally, this means that on the
positions consistent up to round $r - 2$, the fold
$\mathrm{fold}(f_{r-2}, z_{r-1})$ is $\delta$-far from the code. A position
that is also consistent up to round $r - 1$ has $f_{r-1}$ equal to that fold, so
on these positions $f_{r-1}$ is far from the code too. For $r = 1$, read $y$ in
place of the fold: the positions are those where $f_0 = y$, and the doomed state
after round 4 says that $y$ is far. Then the pair
$(u_\mathsf{even}, u_\mathsf{odd})$ formed from $f_{r-1}$ is far from the 2-fold
interleaved code on those positions, because the values of $f_{r-1}$ at $x$ and
$-x$ determine both entries of the pair at $x^2$, and conversely. The fold is
the line $u_\mathsf{even} + z_r u_\mathsf{odd}$. By mutual correlated agreement
for lines, it is close to the code on those positions only for an exceptional
$z_r$.

[Block et al., Theorem 4.2](https://eprint.iacr.org/2023/1071) turns this
sketch into a proof, and uses correlated agreement for lines as its only
coding-theoretic input. The line of folding round $r$ consists of words on
$D_r$, of length $\vert D \vert / 2^r$, and the rate stays $\rho$. With the bound
for lines above, the error of folding round $r$ is
$\varepsilon_{\mathsf{fold},r}$, the value of $\varepsilon_\mathsf{fw}$ with
$\vert D_r \vert$ in place of $\vert D \vert$ and the same $\eta$. In the
proven regime, the radius $\delta$ is rounded to a multiple of
$1/\vert D \vert$; on $D_r$ the same rounding gives a slightly different
radius, a difference of less than $2^r/\vert D \vert$ that this page neglects.
The first folding round has the largest error,

$$ \varepsilon_\mathsf{fold} := \varepsilon_{\mathsf{fold},1} . $$

#### Query Round

The verifier samples $t$ indices into $D_0$. Each index determines one point
$x \in D_0$, and the verifier checks that $x$ is consistent up to round $R$, by
opening $f_{r-1}(\pm x^{2^{r-1}})$ and $f_r(x^{2^r})$ for every $r$.

Suppose the transcript is doomed after the last folding round, and suppose the
last word $f_R$ is the evaluation of a polynomial of low degree; otherwise the
final round catches it. Then at least a $\delta$ fraction of the positions in
$D_0$ is not consistent up to round $R$
([Block et al., Theorem 4.2](https://eprint.iacr.org/2023/1071)). Intuitively,
the doomed state says that the last fold is $\delta$-far from the code, while
$f_R$ is a codeword; every position where they differ, or where an earlier
check fails, is inconsistent. A uniformly random index is consistent with
probability at most $1 - \delta$. With $t$ independent queries, the
probability that all are consistent drops to

$$ \varepsilon_\mathsf{query} := (1 - \delta)^t . $$

The verifier samples the indices independently and with replacement. Repeated
indices do not invalidate this bound, which is stated for independent queries.

The indices must also be uniform. In the non-interactive proof, which the
section [From IOP to Non-Interactive Proof](#from-iop-to-non-interactive-proof)
describes, the verifier computes each index as
$x \bmod \vert D \vert$, where $x$ is a field element produced by the Tip5
sponge, and it discards $x = p - 1$ and draws again. The accepted values
$\{0, \ldots, p - 2\}$ number $p - 1 = 2^{32}(2^{32} - 1)$. Because
$\vert D \vert = 8N$ is a power of 2 not exceeding $2^{31}$, it divides $p - 1$,
so every residue modulo $\vert D \vert$ occurs exactly $(p - 1) / \vert D \vert$
times. The index is therefore uniform.

#### Final Round

The verifier checks the last codeword against its Merkle root and the sent
polynomial's degree, which must be less than $\vert D_R \vert/4$, since each of
the $R$ folds halves the degree bound $\rho \vert D \vert$ and the length
$\vert D \vert$. It then samples a random point of $\mathbb{E}$ and compares the
polynomial with the codeword's interpolant there. Both polynomials have degree
less than $\vert D_R \vert$, so if they differ, they agree at that point with
probability at most

$$ \varepsilon_\mathsf{last} := \frac{\vert D_R \vert - 1}{\vert \mathbb{E} \vert} . $$

### The Knowledge Error of the IOP

The knowledge error of the IOP is the sum of the errors of all rounds, plus
one more term:

$$
\varepsilon_\mathsf{IOP} := \varepsilon_{\boldsymbol{\chi}} + \varepsilon_{\boldsymbol{\kappa}}
+ \varepsilon_\alpha + \varepsilon_{\boldsymbol{\mu\nu\gamma}} + \sum_{r=1}^{R} \varepsilon_{\mathsf{fold},r}
+ \varepsilon_\mathsf{query} + \varepsilon_\mathsf{last} + \varepsilon_\mathsf{Ext} .
$$

The last term is the failure probability of the extractor, defined in section
[Knowledge Soundness](#knowledge-soundness). It is not needed for soundness, but
it is negligible, and including it makes $\varepsilon_\mathsf{IOP}$ a bound on
both errors.

In the conjectured regime, $\varepsilon_\mathsf{fw}$ is proportional to the
domain length, so the folding errors halve from round to round and sum to less
than $2 \varepsilon_\mathsf{fold}$. In the proven regime, $\varepsilon_\mathsf{fw}$
also depends on the domain length through $\rho_-$ and $\theta$, but only
slightly, so the same holds approximately. The test computes every term of the
sum exactly.

The round-by-round error is the largest of them:

$$
\varepsilon_\mathsf{rbr} := \max\left( \varepsilon_{\boldsymbol{\chi}}, \varepsilon_{\boldsymbol{\kappa}},
\varepsilon_\alpha, \varepsilon_{\boldsymbol{\mu\nu\gamma}}, \varepsilon_\mathsf{fold}, \varepsilon_\mathsf{query},
\varepsilon_\mathsf{last} \right) .
$$

## Knowledge Soundness

The extractor $\mathsf{Ext}$ receives the main table's words in full, since they
are oracles of the transcript. It proceeds as follows.

1. Sample uniformly random coefficients $\lambda_i \in \mathbb{E}$ and form the
   word $\omega := \sum_i \lambda_i \hat{t}_i$ over the main columns $i$.
2. Decode $\omega$ with the Guruswami–Sudan algorithm, which lists all codewords
   within distance $\delta$ of the code $\mathsf{RS}_{\rho \vert D \vert}$ in
   polynomial time, up to the Johnson radius
   ([Guruswami–Rudra–Sudan, ch. 17](https://cse.buffalo.edu/faculty/atri/courses/coding-theory/book/)).
   The admissibility condition $(1 - \delta)^2 > \rho$ of the proven regime
   keeps $\delta$ within the Johnson radius of this code.
3. For each codeword $\omega^*$ in that list, let $\mathcal{B}$ be the set of
   points where $\omega$ and $\omega^*$ agree. Interpolate each
   $\hat{t}_i$ on $\mathcal{B}$ to a polynomial $T_i$ of degree at most
   $\rho \vert D \vert$, if possible.
4. Check whether $(T_i)_i$ encodes a valid execution. If so, read the witness
   off its evaluations on $H$:
   - the program, from the program table;
   - the secret input, from the processor rows executing `divine` and from the
     digests consumed by `merkle_step` and `merkle_step_mem`
     ([instructions](instructions.md));
   - the initial RAM, from the first access to each address in the RAM table.

Every list member of the main table appears this way, except with probability
$\ell \vert D \vert / \vert \mathbb{E} \vert$ over the $\lambda_i$. Indeed, a
list member agrees with the words on a common set $\mathcal{A}$, so its linear
combination agrees with $\omega$ on $\mathcal{A}$ and is in the decoded list.
Then $\mathcal{B} \supseteq \mathcal{A}$. A point of $\mathcal{B}$ at which some
$\hat{t}_i$ disagrees with the list member lies in $\mathcal{B}$ only if the
$\lambda_i$ cancel the disagreement there, which has probability at most
$1/\vert \mathbb{E} \vert$. A union bound over the $\vert D \vert$ points and
the at most $\ell$ list members gives the stated probability. Outside this
event, every $\hat{t}_i$ agrees with the list member on all of $\mathcal{B}$,
and interpolation recovers it. Call this failure probability

$$ \varepsilon_\mathsf{Ext} := \frac{\ell \vert D \vert}{\vert \mathbb{E} \vert} . $$

Run $\mathsf{Ext}$ on the transcript up to the challenge $\boldsymbol{\chi}$,
which contains the main table. Its coins $\lambda_i$ are its own, independent of
all challenges. Outside an event of probability
$\varepsilon_\mathsf{Ext}$, it fails only if no list member of the main table
encodes a valid execution. In that case, the analysis of
[Round 1](#round-1-table-linking-challenges) applies word for word, since it
uses only that property and not the falsity of the claim. So the probability
that round 1 lifts the doom while $\mathsf{Ext}$ fails is at most
$\varepsilon_{\boldsymbol{\chi}} + \varepsilon_\mathsf{Ext}$. The analyses of
the later rounds do not use the falsity of the claim at all, so they bound the
transitions with or without the extractor. The knowledge error of the IOP is
therefore at most $\varepsilon_\mathsf{IOP}$, which includes
$\varepsilon_\mathsf{Ext}$ for this reason. For the round-by-round error, the
knowledge variant replaces $\varepsilon_{\boldsymbol{\chi}}$ by
$\varepsilon_{\boldsymbol{\chi}} + \varepsilon_\mathsf{Ext}$; by the results
table, this sum stays below the largest term, so $\varepsilon_\mathsf{rbr}$ is
unchanged. For FRI on its own,
[Block et al., Theorem 5.11](https://eprint.iacr.org/2023/1071) likewise shows
that the round-by-round knowledge error equals the round-by-round soundness
error.

> **Note:** In the conjectured regime, list members lie beyond the Johnson
> radius. No efficient decoder is known up to the Elias radius over fields of
> this size, so the knowledge-soundness claim in that regime also assumes an
> efficient decoder up to the Elias radius.

## From IOP to Non-Interactive Proof

Triton VM compiles the IOP with Merkle trees
([BCS](https://eprint.iacr.org/2016/116.pdf)) and applies the Fiat-Shamir
transform with one Tip5 sponge for the whole transcript. A *sponge* is a hash
function with an internal state of field elements: it *absorbs* input into a
part of the state of size *rate* (10 for Tip5), scrambles the state with a fixed
permutation, and *squeezes* output from the same part; the remaining *capacity*
(6 for Tip5) is never read or written directly
([Chiesa–Yogev](https://hash-based-snargs-book.github.io/) explain sponges and
random oracles). Let $Q$ be the number of hash evaluations a cheating prover
makes. The Fiat-Shamir
transform replaces each challenge by a hash of the transcript so far. A prover
can then re-try a round's challenge by changing its message and hashing again,
and each try succeeds with probability at most $\varepsilon_\mathsf{rbr}$. So
the round-by-round error, not $\varepsilon_\mathsf{IOP}$, governs this step.

The sponge first absorbs the claim. It then absorbs every prover message except
those already bound by an earlier Merkle root: the revealed rows, the
authentication paths, and the last FRI codeword. Modeling Tip5 as a random
oracle is justified for an ideal permutation by
[sponge indifferentiability](https://keccak.team/files/SpongeIndifferentiability.pdf),
at a cost of order $Q^2 / \vert \mathbb{F} \vert^6$ for the capacity of 6 field
elements, which the collision term below dominates.
[Block et al., Theorem 3.15](https://eprint.iacr.org/2023/1071) and the
[ethSTARK documentation, Theorem 8](https://eprint.iacr.org/2021/582) bound the
soundness error of the resulting non-interactive proof by

$$ Q \cdot \varepsilon_\mathsf{rbr} + 3 \cdot \frac{Q^2 + 1}{\vert \mathbb{F} \vert^5} , $$

where $\vert \mathbb{F} \vert^5 \approx 2^{320}$ is the number of Tip5 digests.
[Chiesa–Yogev, Theorems 26.1.1 and 31.3.1](https://hash-based-snargs-book.github.io/)
give the same bound for knowledge soundness, with a straight-line extractor.

The first term dominates. Each factor of 2 in $Q$ costs one bit: for example, a
prover that makes $Q = 2^{64}$ hash evaluations faces a soundness error of
about $2^{64} \cdot 2^{-80} = 2^{-16}$ in the proven regime, and about
$2^{64} \cdot 2^{-137} = 2^{-73}$ in the conjectured regime at $N = 2^{28}$.

## Concrete Numbers

The protocol constants that the computation uses, each checked against the code
by the test `soundness_chapter_agrees_with_code`:

<!-- soundness-constants start -->
| Key | Symbol | Meaning | Value |
| :--- | :--- | :--- | ---: |
| `p` | $p$ | Size of $\mathbb{F}$ | 18446744069414584321 |
| `extension_degree` | | Degree of $\mathbb{E}$ over $\mathbb{F}$ | 3 |
| `num_queries` | $t$ | Number of FRI queries | 80 |
| `fri_expansion_factor` | $\rho^{-1}$ | Inverse rate | 4 |
| `num_trace_randomizers` | $h$ | Trace randomizers per column | 105 |
| `num_quotient_segments` | $k$ | Quotient segments | 4 |
| `air_fan_in` | | Rows per AIR constraint, hence the points $\alpha$ and $g\alpha$ | 2 |
| `constraint_degree` | $\mathsf{d}$ | Maximal constraint degree | 4 |
| `num_constraints` | $\mathsf{c}$ | AIR constraints | 611 |
| `num_cascade_clients` | | Cascade lookups per hash-table row | 16 |
| `max_log2_padded_height` | | Largest $\log_2 N$ the verifier accepts | 28 |
| `last_fri_domain_len` | $\vert D_R \vert$ | Length of the last FRI codeword | 512 |
| `tip5_digest_len` | | Field elements per Tip5 digest | 5 |
| `tip5_capacity` | | Capacity of the Tip5 sponge, in field elements | 6 |
| `attack_unsafe_index` | | Radius numerator of the attack in [Status of the Conjecture](#status-of-the-conjecture) | 122369 |
| `attack_domain_len` | | Code length of that attack | 262144 |
| `attack_rate` | | Rate of that attack | 0.5 |
| `attack_log2_bad_challenges` | | $\log_2$ of its number of bad challenges | 59 |
| `c2` | $c_2$ | Constant of the conjecture | 1.37 |
<!-- soundness-constants stop -->

For each regime and trace length, $\eta$ is chosen from
$\{2^{-j/50} : 50 \leqslant j < 1500\}$, among the admissible values, to
minimize $\varepsilon_\mathsf{IOP}$. All errors are given as $\log_2$:

<!-- soundness-results start -->
| Key | Quantity | Proven, $N = 2^{28}$ | Conjectured, $N = 2^{28}$ | Proven, $N = 2^{21}$ | Conjectured, $N = 2^{21}$ |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `num_folding_rounds` | $R$ | 22 | 22 | 15 | 15 |
| `log2_eta` | $\log_2 \eta$ | -21.94 | -4.20 | -23.00 | -4.44 |
| `log2_list_size` | $\log_2 \ell$ | 19.9 | 20.4 | 22.0 | 24.1 |
| `eps_challenges` | $\varepsilon_{\boldsymbol{\chi}}$ | -139.0 | -138.5 | -143.9 | -141.8 |
| `eps_quotient_weights` | $\varepsilon_{\boldsymbol{\kappa}}$ | -172.1 | -171.6 | -170.0 | -167.9 |
| `eps_ood_point` | $\varepsilon_\alpha$ | -140.9 | -140.4 | -145.8 | -143.7 |
| `eps_deep_weights` | $\varepsilon_{\boldsymbol{\mu\nu\gamma}}$ | -96.8 | -139.6 | -100.6 | -142.9 |
| `eps_folding` | $\varepsilon_\mathsf{fold}$ | -98.8 | -141.6 | -102.6 | -144.9 |
| `eps_folding_sum` | $\sum_r \varepsilon_{\mathsf{fold},r}$ | -97.8 | -140.6 | -101.6 | -143.9 |
| `eps_extractor` | $\varepsilon_\mathsf{Ext}$ | -141.1 | -140.6 | -146.0 | -143.9 |
| `eps_queries` | $\varepsilon_\mathsf{query}$ | -80.0 | -137.3 | -80.0 | -140.5 |
| `eps_last_round` | $\varepsilon_\mathsf{last}$ | -183.0 | -183.0 | -183.0 | -183.0 |
| `eps_rbr` | $\varepsilon_\mathsf{rbr}$ | -80.00 | -137.27 | -80.00 | -140.48 |
| `eps_iop` | $\varepsilon_\mathsf{IOP}$ | -80.00 | -136.31 | -80.00 | -139.56 |
<!-- soundness-results stop -->

The last row is the result of this page: the knowledge error of Triton VM's
IOP. In the conjectured regime, the knowledge claim also assumes an efficient
decoder; see [Knowledge Soundness](#knowledge-soundness).

In the proven regime, the query term bounds the result. It is
$(1 - \delta)^t$, and $1 - \delta$ is slightly larger than $\sqrt{\rho} = 1/2$,
so the query term lies just above $2^{-80}$ for every trace length. In the conjectured regime, the query term and
the terms multiplied by $\ell$ balance at $\eta \approx 2^{-4.2}$ to $2^{-4.4}$;
a smaller $\eta$ improves the query term but increases $\ell$ exponentially in
$1/\eta$.
There, several terms are of similar size, so their sum $\varepsilon_\mathsf{IOP}$
is about one bit larger than the largest of them, $\varepsilon_\mathsf{rbr}$.

> **Note:** The field `Stark::security_level` of the default parameters is 160.
> That value assumes that each query contributes $\log_2 \rho^{-1} = 2$ bits,
> which corresponds to proximity up to the list-decoding capacity
> $1 - \rho$. The counterexamples in the next section,
> [Status of the Conjecture](#status-of-the-conjecture), refute that assumption.
> The concrete parameters fall short of the advertised 160 bits.
> The analysis on this page is meant to guide the selection of parameters. The
> current parameters were selected before it was written.

## Status of the Conjecture

The conjecture of section [Conjectured Regime](#conjectured-regime) states that
Reed-Solomon codes over any prime field $\mathbb{F}_p$ and its extensions have, at
every radius $1 - \rho - \eta$ up to the Elias radius $r_\mathsf{E}$ over
$\mathbb{F}_p$, list size at most $2^{c_2 \mathcal{H}_2(\rho)/\eta}$ and at most
$n \cdot 2^{c_2 \mathcal{H}_2(\rho)/\eta}$ exceptional challenges per line, for
code length $n$.

A *proximity gap* at radius $\delta$ means: for every line $u + zv$, either all
points $z$ give a word $\delta$-close to the code, or only a few do. The
following conjectures, which allow larger radii, are refuted:

- Conjecture 8.4 of [BCIKS20](https://eprint.iacr.org/2020/654) in its
  unrestricted form (proximity gaps up to $1 - \rho - \eta$ with error
  polynomial in $n$), the up-to-capacity conjecture of
  [WHIR](https://eprint.iacr.org/2024/1586), and the list-decoding conjecture of
  [DEEP-FRI](https://eprint.iacr.org/2019/336). Refuted by
  [Diamond–Gruen](https://eprint.iacr.org/2025/2010) and
  [Crites–Stewart](https://eprint.iacr.org/2025/2046), for radii between the
  Elias radius and capacity.
- Proximity gaps at radius $1 - \rho - O(1/\log n)$ over prime fields with
  multiplicative-subgroup domains, the kind Triton VM uses. Refuted by a sketch
  of Krachun and Kazanin,
  which [Kambiré](https://arxiv.org/abs/2604.09724) works out in full. The
  conjectured regime stays at or below the Elias radius. This page does not
  analyze whether the construction reaches below the Elias radius at Triton
  VM's code lengths, where $1/\log_2 \vert D \vert$ is about $2^{-5}$.

The [Proximity Prize](https://github.com/proximity-prize/proximity-prize)
benchmark contains a Lean-checked lower bound for the interleaved Reed-Solomon
code over the sextic extension of the KoalaBear field, the prime field with
$2^{31} - 2^{24} + 1$ elements, with rate $1/2$ and length $2^{18}$
([submission #38](https://github.com/proximity-prize/proximity-prize/pull/38)).
The submission exhibits more than $2^{59}$ bad challenges at radius
$122369/2^{18} \approx 0.4668$. That radius lies below the Elias radius
$r_\mathsf{E} \approx 0.4678$ for KoalaBear at rate $1/2$. With
$\eta = 1/2 - 122369/2^{18}$, the threshold
$2^{18} \cdot 2^{c_2 \mathcal{H}_2(1/2)/\eta}$ is at least $2^{59}$ only if

$$ c_2 \geqslant \frac{(59 - 18)\,\eta}{\mathcal{H}_2(1/2)} \approx 1.361 . $$

This page rounds up to $c_2 = 1.37$. That is the smallest value, to two
decimals, that this lower bound does not refute, and therefore the most
optimistic choice that this one attack allows. Reading the benchmark's bad-challenge
count as a lower bound on the conjecture's threshold assumes that the two count
the same objects. Larger constants remain unrefuted and give fewer bits.

### Sensitivity to the Form of the Conjecture

The attack is a single data point, at one code length $n = 2^{18}$ and one
$\eta$. It fixes one inequality between the constants of a conjecture, but not
how the bound grows with $n$. Triton VM's domains are up to $2^{13}$ times
longer than the attack's, so the form of the conjecture matters as much as its
constants. The following table compares the form used on this page with the
form of [BCIKS20, Conjecture 8.4](https://eprint.iacr.org/2020/654), whose
bound is $n^{e_n} / (\eta \rho)^{e_\eta}$ exceptional challenges per line, with
exponents $e_\eta$ and $e_n$. BCIKS20
does not bound the list size; the table assumes $n^{e_n - 1} / (\eta\rho)^{e_\eta}$,
so that the count of exceptional challenges is the list size times $n$, as in the
S-two form. BCIKS20 suggests
$e_\eta = e_n = 2$, which the attack refutes.

In each row, the free constant is the smallest value, rounded up to two
decimals, that the attack does not refute; the exponential form fixes $c_1 = 1$,
and the polynomial forms fix $e_n$ or set $e_\eta = e_n$. The last two columns give
$\log_2 \varepsilon_\mathsf{IOP}$, with $\eta$ chosen anew for each row.

<!-- soundness-sensitivity start -->
| Key | Form | $c_1$ or $e_\eta$ | $c_2$ or $e_n$ | $N = 2^{28}$ | $N = 2^{21}$ |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `exponential` | S-two, $n \cdot c_1 2^{c_2 \mathcal{H}_2(\rho)/\eta}$ (this page) | 1 | 1.37 | -136.3 | -139.6 |
| `polynomial_linear` | BCIKS20, $n^{e_n} / (\eta\rho)^{e_\eta}$, linear in $n$ | 6.94 | 1 | -120.1 | -125.4 |
| `polynomial_en_1.5` | BCIKS20 | 5.42 | 1.5 | -114.2 | -122.7 |
| `polynomial_quadratic` | BCIKS20, quadratic in $n$ | 3.89 | 2 | -107.6 | -119.7 |
| `polynomial_equal` | BCIKS20, $e_\eta = e_n$ | 2.47 | 2.47 | -100.6 | -116.5 |
<!-- soundness-sensitivity stop -->

Every form agrees with the attack, yet the results differ by up to 36 bits at
$N = 2^{28}$ and 23 bits at $N = 2^{21}$.
Until attacks at other code lengths are known, the conjectured number reflects
the choice of form as much as the evidence.

On the proven side, the same benchmark contains a Lean-checked proof of
correlated agreement beyond the Johnson radius for its one parameter set
([submission #627](https://github.com/proximity-prize/proximity-prize/pull/627)):
radius $81879/2^{18} \approx 0.312$ at rate $1/2$, where the Johnson radius is
$1 - \sqrt{1/2} \approx 0.293$. The proof concerns that field, rate, and length
only. It does not apply to Triton VM's parameters, so the proven regime of this
page ends at the Johnson radius.
