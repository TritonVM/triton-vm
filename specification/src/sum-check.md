# Zero-Integral and Sum-Check

Let $T$ be a table of 1 column and $|H|$ rows, and let $\hat{f}(X)$ be any polynomial that agrees with $T$ when evaluated on $H = \left(\omicron^i\right)_ i$, where $\omicron$ is a generator of the subgroup of order $2^k$  $H$. Group theory lets us prove and efficiently verify if $\sum T = 0$.

Decompose $\hat{f}(X)$ into $\hat{f}(X) = \hat{g}(X) + Z_H(X) \cdot \hat{h}(X)$, where $Z_H(X)$ is the zerofier over $H$, and where $\hat{g}(X)$ has degree at most $|H|-1$. The table sums to zero if and only if $\hat{g}(X)$ integrates to zero over $H$ because

$$ \sum_{h \in H} \hat{f}(h) = \sum_{h \in H} (\hat{g}(h) + Z_H(h) \cdot \hat{h}(h)) = \sum_{h \in H} \hat{g}(h) $$

 and this latter proposition is true if and only if the constant term of $\hat{g}(X)$ is zero.

**Theorem.** $\sum_{h \in H} \hat{f}(h) = 0 \, \Leftrightarrow \, X | \hat{g}(X)$ for a subgroup $H \subseteq \mathbb{F}^*$ of order $2^k$.

$Proof.$ Let $K$ be a subgroup of $H$. If $K \neq \{1\}$ then $-1 \in K$ and also $\sum_{k \in K} k = 0$ because the elements of $K$ come in pairs: $k \in K \Leftrightarrow -k \in K$. Therefore $\sum_{h \in H} h = 0$.

The map $H \rightarrow K, h \mapsto h^\alpha$ is a morphism of groups with  $|H| \geq |K| \geq 1$. Therefore we have:

$$ \sum_{h \in H} h^\alpha = \left\{ \begin{matrix}
|H| & \quad \Leftarrow \alpha = 0 \mod |H| \enspace \phantom{.}\\
0 & \quad \Leftarrow \alpha \neq 0 \mod |H| \enspace .
\end{matrix} \right. $$

The polynomial $\hat{g}(X)$ has only one term whose exponent is $0 \mod |H|$, which is the constant term. $\square$

This observation gives rise to the following Polynomial IOP for verifying that a polynomial oracle $[\hat{f}(X)]$ integrates to 0 on a subgroup $H$ of order some power of 2.

**Protocol Zero-Integral**
 - Prover computes $\hat{g}(X) \leftarrow \hat{f}(X) \mod Z_H(X)$ and $\hat{h}(X) \leftarrow \frac{\hat{f}(X) - \hat{g}(X)}{Z_H(X)}$.
 - Prover computes $\hat{g}^\star(X) = \frac{\hat{g}(X)}{X}$.
 - Prover sends $\hat{g}^\star(X)$, of degree at most $|H|-2$, and $\hat{h}(X)$, of degree at most $\mathsf{deg}(\hat{f}(X)) - |H|$ to Verifier.
 - Verifier queries $[\hat{f}(X)]$, $[\hat{g}^\star(X)]$, $[\hat{h}(X)]$ in $z \xleftarrow{\$} \mathbb{F}$ and receives $y_f, y_{g^\star}, y_h$.
 - Verifier tests $y_f \stackrel{?}{=} z \cdot y_{g^\star} + y_h \cdot Z_H(z)$.

This protocol can be adapted to show instead that a given polynomial oracle integrates to $a \neq 0$ on the subgroup $H$, giving rise to the well-known **Sum-Check** protocol. The adaptation follows from the Verifier's capacity to simulate $[\hat{f}^\star(X)]$ from $[\hat{f}(X)]$, where $\hat{f}^\star(X) = \hat{f}(X) - \frac{a}{|H|}$. This simulated polynomial is useful because

$$ \sum_{h \in H} \hat{f}^\star(h) = \sum_{h \in H}\left(\hat{f}(h) - \frac{a}{|H|}\right) = a - \frac{a}{H} \sum_{h \in H} 1 = 0 \enspace .$$

In other words, $\hat{f}(X)$ integrates to $a$ on $H$ iff $\hat{f}^\star(X)$ integrates to $0$ on $H$, and we already a protocol to establish the latter claim.
