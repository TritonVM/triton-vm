# Index Sampling

Task: given pseudorandomness, sample pseudorandomly $k$ distinct but otherwise uniform indices from the interval $[0:U)$. This index sampling task is needed in several locations, not just in the STARK engine. Outside of the VM, this task can be achieved by rejection sampling until the list of collected samples has the requisite size (namely, $k$). Inside the VM, we like to avoid variable-length execution paths. This note explains how to do it using non-determinism.

## Solution 1: Out-of-Order Divination

We want to sample *exactly* $k$ unique indices, but the sampler in the VM is allowed to divine counter values.

### Outside VM

By rejection sampling until the list of samples has the requisite length I mean something like this:

```python
    function sample_indices(count, upper_bound, randomness):
        list <- []
        counter <- 0
        while length(list) < count:
            integer <- Hash(randomness || counter)
            candidate_index <- integer % upper_bound
            if candidate_index in list:
                continue // reject
            else:
                list <- list || candidate_index
        return list
```

The loop needs to iterate at least `count` number of times. It is possible to parallelize this first step by separating the `count` first iterations from the variable-number follow-ups.

This python-like pseudocode hides important complexity behind the list membership test. To expand this logic, we need sorting and consecutive deduplication.

```python
    function sample_indices(count, upper_bound, randomness):
        list <- []
        counter <- 0
        while length(list) < count:
            integer <- Hash(randomness || counter)
            candidate_index <- integer % upper_bound
            list <- sort(list || candidate_index)
            list.deduplicate()
        return list
```

### Inside VM

In a nutshell, the VM nondeterministically guesses ("divines") the values of the counters out of order, checks that they are within allowed bounds, and then checks that the resulting list of indices is sorted and has no duplicates.

```python
    function sample_indices(count, upper_bound, randomness):
        list <- []
        for i in [0:count):
            counter <- divine()
            assert(counter >= 0)
            assert(counter < count + margin)
            integer <- Hash(randomness || counter)
            index <- integer % upper_bound
            list <- list || index
            if i > 0:
                assert(list[i] < list[i-1])
        return list
```

The Virtual Machine supplies the right counters through `divine()`, having observed them from the out-of-VM execution.

The parameter `margin` should be large enough so that, with overwhelming probability, there is a set of `count`-many counters in `[0:(count+margin))` that map to distinct indices. At the same time, the parameter `margin` should not be too large as it gives the malicious adversary more flexibility in selecting favorable indices.

### Failure Probability

In order for the honest user to fail to find a good counter set, the list `[0:(count+margin))` must contain at least `margin`+1 collisions under `x -> Hash(randomness || x) % upper_bound`. We continue by modeling this map as uniformly random.

The event `margin`+1 collisions or more implies that `margin`+1 samples were drawn from a subset of at most `count`-1 marked indices. The probability of the implied event is
$$\mathrm{Pr}[\# \mathrm{collisions} \geq \mu+1] = \left(\frac{(k-1)}{U}\right)^{\mu + 1} \enspace ,$$
where $U$ denotes `upper_bound`, $k$ denotes `count`, and $\mu$ denotes `margin`.

Given the `count` and the `upper_bound`, the appropriate value for `margin` that pushes this failure probability below $2^{-\lambda}$ is
$$ \left(\frac{(k-1)}{U}\right)^{\mu + 1} \leq 2^{-\lambda} $$
$$ (\mu + 1) \cdot \log_2 \left(\frac{(k-1)}{U}\right) \leq -\lambda $$
$$ \mu \geq \frac{-\lambda}{\log_2 \left(\frac{(k-1)}{U}\right)} - 1 \enspace .$$

For a concrete example, set $k=\lambda=160$ and $U=2^{32}$. Then $\mu$ needs to be at least 6.

### Security Degradation

Suppose the user is malicious and hopes to conceal his fraud by selecting a set of indices that do not expose it. Suppose that the proportion of subsets of $[0:U)$ of size $k$ that are suitable for the adversary is $\rho$. Then clearly with the standard old index sampling method the attacker's success probability is bounded by $\rho \approx 2^{-\lambda}$. The question is whether the improved index sampler enables a higher success probability (and if so, how much higher).

The attacker has access to at most $\binom{k+\mu}{\mu}$ subsets of $[0:U)$ of size $k$. The probability that a given subset is suitable for the attack is $\rho$, and so:
 - The probability that one subset is unsuitable for attack is $1 - \rho$.
 - The probability that all $\binom{k+\mu}{\mu}$ subsets are unsuitable for attack is $\left(1 - \rho\right)^{\binom{k+\mu}{\mu}}$. In this step we assume that whether a subset is suitable for attack, is independent from whether different subset is suitable for attack, even if the intersection of the two subsets is nonzero.
 - The probability that at least one subset is suitable for attack is $1 - \left(1 - \rho\right)^{\binom{k+\mu}{\mu}}$.

The [binomial formula](https://en.wikipedia.org/wiki/Binomial_theorem) expands
$$ \left(1 - \rho\right)^{\binom{k+\mu}{\mu}} = \sum_{i=0}^{\binom{k+\mu}{\mu}} (-1)^i \binom{\binom{k+\mu}{\mu}}{i} \rho^i \enspace . $$
We assume that each subsequent term is smaller in absolute value than the previous because $\rho$ dominates. Indeed, plugging the [well-known upper bound](https://en.wikipedia.org/wiki/Binomial_coefficient#Bounds_and_asymptotic_formulas) on the binomial coefficient $\binom{n}{k} < \left(\frac{n \cdot e}{k}\right)^k$, we get
$$ \binom{\binom{k+\mu}{\mu}}{i} < \binom{\left(\frac{(k+\mu)\cdot e}{\mu}\right)^\mu}{i} < \left( \frac{\left(\frac{(k+\mu)\cdot e}{\mu}\right)^\mu \cdot e}{i} \right)^i $$
and the assumption holds already when $\left(\frac{(k+\mu) \cdot e}{\mu}\right)^\mu < \rho^{-1}$. For a concrete example, set $k=160$ and $\mu = 6$, then the left hand side of this expression is roughly $2^{37.4}$ whereas these parameters target a security level of $\lambda \approx \log_2 \rho^{-1}$.

Using this assumption (on the shrinking absolute value of each successive term) we can derive an expression to quantify the security degradation.
$$ \mathrm{Pr}[\textnormal{attack}] \leq 1 - \left(1 - \rho\right)^{\binom{k+\mu}{\mu}} $$
$$= 1 -  1 + \rho\cdot\binom{k+\mu}{\mu} - \rho^2\cdot\binom{k+\mu}{\mu}\cdot\binom{k+\mu}{\mu}\cdot\frac{1}{2} + \cdots$$
$$ \leq \binom{k+\mu}{\mu} \cdot \rho \enspace .$$

For a concrete example, set $k = 160$ and $\mu = 6$. Then $\binom{k+\mu}{\mu} \approx 2^{34.6}$ and so we lose about $33$ bits of security.

## Solution 2: Stupid Safety Margin

How about we sample $k + \mu$ indices from the start, and use them all no matter what? We only need $k$ for security. The margin parameter $\mu$ is chosen such that finding more than $\mu$ collisions, which is required to undermine the security guarantee, is cryptographically improbable.

The benefit of this solution is that both the index samplers, i.e., inside and outside the VM, have a simple description. Furthermore, there is no longer a nonzero failure probability for honest users. The drawback is that more work is done than necessary, and the proofs are larger too.

So what is the right value of $\mu$? It turns out that this probability matches with the correctness error derived for the previous solution. For redundancy this derivation is repeated here.

Sampling $\mu+1$ or more collisions requires hitting a marked subset of at most $k-1$ indices $\mu + 1$ times. The probability of this event is therefore $\left(\frac{k-1}{U}\right)^{\mu+1}$. This probability is smaller than $2^{-\lambda}$ when
$$ \left(\frac{k-1}{U}\right)^{\mu + 1} \leq 2^{-\lambda} $$
$$(\mu + 1) \cdot \log_2\left(\frac{k-1}{U}\right) \leq -\lambda $$
$$ \mu \geq \frac{-\lambda}{\log_2\left(\frac{k-1}{U}\right)} - 1 \enspace .$$

For a concrete example, set $k=\lambda=160$ and $U=2^{32}$. Then $\mu$ needs to be at least 6.
