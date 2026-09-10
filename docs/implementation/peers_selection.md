# Peer selection

Floresta selects regular outbound peers from a weighted probability distribution. The distribution combines address health, network diversity, and additive service preferences in one sampling path.

The service mix targets:

- 60% unconditioned peers (the "mixed" share);
- 20% compact-filter-capable peers;
- 20% Utreexo-capable peers.

The mixed share may also select peers that provide compact filters, Utreexo, or both. These percentages describe probability mass, not a hard quota for every batch of ten connections.

## Parameters

| Parameter | Value | Purpose |
| --- | ---: | --- |
| Retry time constant $\tau_r$ | 10 minutes | Gradually restores an address after any recent attempt. |
| Failure time constant $\tau_f$ | 6 hours | More slowly restores an address after a failed attempt. |
| Retry floor $\epsilon$ | 0.01 | Keeps recently attempted addresses possible. |
| Failure floor $\rho$ | 0.05 | Heavily penalizes a recent failure without banning it. |
| Tried-peer bonus $b$ | 10 | Favors addresses with a previously successful connection. |
| Represented-group penalty $\eta$ | 0.05 | Penalizes a network group once per connected or in-flight peer. |
| Mixed weight $w_m$ | 0.60 | Unconditioned share of the final distribution. |
| Compact-filter weight $w_c$ | 0.20 | Share conditioned on compact-filter support. |
| Utreexo weight $w_u$ | 0.20 | Share conditioned on Utreexo support. |

## 1. Determine eligibility

An address is excluded from a regular draw when it is:

- banned;
- already connected; or
- reserved by an in-flight connection attempt.

An in-flight address is excluded from the candidates but still counts as a represented network group. This makes sequential draws operate without replacement and pushes later draws toward other groups.

## 2. Compute address health

For an address whose latest attempt happened $\Delta_i$ seconds ago, define the recovery function

$$
R(\Delta, \tau, a)
= a + (1-a)\left(1-e^{-\Delta/\tau}\right).
$$

The retry score is

$$
r_i = R(\Delta_i, \tau_r, \epsilon).
$$

The final health score depends on the address state:

$$
h_i =
\begin{cases}
1, & \text{if the address has never been tried},\\
b r_i, & \text{if the latest attempt succeeded},\\
r_i R(\Delta_i, \tau_f, \rho), & \text{if the latest attempt failed}.
\end{cases}
$$

Connected and banned addresses have no health score because they are ineligible.

All eligible scores are positive. A failed address can therefore recover gradually instead of being blocked by a fixed retry cutoff. At the instant of failure its score is $\epsilon\rho = 0.0005$.

A previously successful address receives a $10\times$ reliability bonus. Its score approaches
$b=10$ as the retry cooldown expires, compared with the baseline score of $1$ for a never-tried
address. The retry term still applies: an address tried at the current instant starts at
$b\\epsilon=0.1$, preventing immediate repeated selection, and exceeds the never-tried baseline
after roughly one minute.

Timestamps in the future are treated as age zero by using saturating subtraction.

Addresses returned by DNS seeds enter the address manager as `Tried`, not `NeverTried`.
Filtered DNS seeds provide known-running endpoints, so these addresses immediately qualify as
previously successful peers while retaining the service flags encoded by the DNS query.

## 3. Form network groups

Addresses are grouped as follows:

- IPv4: the first 16 bits (`/16`);
- IPv6: the first 32 bits (`/32`);
- Tor, I2P, CJDNS, and unknown overlay addresses: one group per address.

For each group $g$:

- $c_g$ is the number of connected or in-flight addresses in that group;
- $H_g$ is the maximum health among its eligible addresses;
- $L_g$ is the sum of health scores among its eligible addresses.

The diversity multiplier and group weight are

$$
d_g = \eta^{c_g},
$$

$$
G_g = d_g H_g.
$$

With $\eta=0.05$, a group has multipliers $1$, $0.05$, and $0.0025$ when it has zero, one, and two represented peers respectively.

Using the maximum health prevents a group from becoming more attractive merely because the address manager contains many entries from it.

## 4. Build the base address distribution

First normalize the group weights:

$$
P(g) = \frac{G_g}{\sum_k G_k}.
$$

Then distribute each group's probability among its addresses according to health:

$$
P(i\mid g) = \frac{h_i}{L_g}.
$$

The base probability for address $i$ is

$$
q_i = P(g(i))P(i\mid g(i)).
$$

This separates network diversity from address reliability:

- group selection prevents a populous subnet from dominating;
- selection within a group favors healthier addresses.

## 5. Add service weights

Let:

- $C$ be the eligible compact-filter-capable addresses;
- $U$ be the eligible Utreexo-capable addresses;
- $Q_C = \sum_{i\in C} q_i$;
- $Q_U = \sum_{i\in U} q_i$.

When both service classes are available, the final probability is

$$
p_i =
0.60q_i
+ 0.20\mathbf{1}_{i\in C}\frac{q_i}{Q_C}
+ 0.20\mathbf{1}_{i\in U}\frac{q_i}{Q_U}.
$$

The terms are additive. A peer that supports both services receives both service contributions. Advertising a desired service never removes or reduces the peer's base contribution.

Each conditional term preserves the base ordering within that service class. A healthy, diverse Utreexo peer therefore remains more likely than a recently failed Utreexo peer.

The compact-filter and Utreexo sets may overlap. The distribution still sums to one because each conditional component contributes exactly its reserved 20% mass.

### Additive example

Assume four equally healthy peers in distinct, unrepresented groups:

1. neither service;
2. compact filters only;
3. Utreexo only;
4. both services.

Each has base probability $q_i=0.25$, while $Q_C=Q_U=0.5$. Their final probabilities are:

| Services | Probability |
| --- | ---: |
| Neither | 0.15 |
| Compact filters only | 0.25 |
| Utreexo only | 0.25 |
| Both | 0.35 |

The dual-service peer receives the base contribution plus both service contributions.

### Missing service classes

If no eligible address provides one of the services, that service's reserved 20% is returned to the unconditioned component:

- no compact-filter candidates: 80% mixed + 20% Utreexo;
- no Utreexo candidates: 80% mixed + 20% compact filters;
- neither class available: 100% mixed.

This preserves a valid distribution and allows peer discovery to continue when advertised service information is missing or stale.

## 6. Explicit service requirements

Some callers request a specific service rather than the normal 60/20/20 mix. For required service set $S$, calculate

$$
Q_S = \sum_{i\in S} q_i.
$$

If $Q_S>0$, condition the base distribution on that service:

$$
p_i =
\begin{cases}
q_i/Q_S, & i\in S,\\
0, & i\notin S.
\end{cases}
$$

If no eligible address advertises the requested service, Floresta logs the condition and falls back to $q$. Service advertisements learned from DNS or gossip may be stale, so the fallback can still discover a suitable peer.

## 7. Sample and reserve

The final weights are sampled with `WeightedIndex`. Once selected, the address is inserted into the address manager's in-flight reservation set.

On the next draw:

- that address is ineligible;
- its network group has an additional represented peer;
- the entire distribution is rebuilt.

The reservation ends when:

- the attempt succeeds and the address becomes connected;
- the attempt fails or times out and the address becomes failed;
- connection setup fails before an actor is started and the reservation is cancelled.

Every address-manager peer whose handshake reaches `Ready` is marked `Connected` before
connection-kind-specific handling. This includes regular, extra, and feeler connections. The
address remains outside the candidate pool until disconnection changes its state to `Tried` or
`Failed`.

This implements sequential weighted sampling without replacement.

## Feeler connections

Feeler connections are intentionally separate. They uniformly select an address that is not banned, connected, or already in flight, without applying service preferences. The selected feeler address is still reserved so another concurrent attempt cannot select it.

## Pseudocode

```text
function select(addresses, in_flight, required_service, now):
    candidates = []
    groups = {}

    for address in addresses:
        group = network_group(address)

        if address.connected or address.id in in_flight:
            groups[group].represented += 1
            continue

        if address.banned:
            continue

        health = health_score(address, now)
        groups[group].health_sum += health
        groups[group].max_health = max(groups[group].max_health, health)
        candidates.push(address, group, health)

    for group in groups with eligible candidates:
        group.weight =
            0.05 ^ group.represented
            * group.max_health

    normalize group weights

    for candidate in candidates:
        q[candidate] =
            group_weight[candidate.group]
            * candidate.health
            / groups[candidate.group].health_sum

    if required_service is set:
        if any candidate advertises required_service:
            p = q conditioned on required_service
        else:
            p = q
    else:
        mixed_weight = 0.60

        if no compact-filter candidate:
            mixed_weight += 0.20
        if no Utreexo candidate:
            mixed_weight += 0.20

        for candidate in candidates:
            p[candidate] = mixed_weight * q[candidate]

            if candidate supports compact filters:
                p[candidate] += 0.20 * q[candidate] / compact_filter_mass

            if candidate supports Utreexo:
                p[candidate] += 0.20 * q[candidate] / utreexo_mass

    selected = weighted_sample(p)
    in_flight.insert(selected.id)
    return selected
```

## Implementation and tests

The implementation is in `crates/floresta-wire/src/p2p_wire/address_man.rs`, primarily in:

- `LocalAddress::address_group`;
- `LocalAddress::selection_health`;
- `AddressMan::build_selection_weights`;
- `AddressMan::get_address_to_connect`.

Unit tests in the same module cover:

- failure penalties and recovery;
- preference for previously successful peers over never-tried addresses;
- subnet multiplicity resistance;
- connected and in-flight group penalties;
- the 60/20/20 service distribution;
- additive weight for dual-service peers;
- required-service conditioning;
- sequential selection without replacement.
