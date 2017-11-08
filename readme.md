# Cryptographically Strong Pseudo Random Number Generation using Generative Adversarial Networks
This project seeks to assess the viability of generative adversarial networks
(GANs) for the implementation of a cryptographically secure pseudo random
number generator (CSPRNG). It draws inspiration from recent work on
bootstrapping encryption schemes using GANs. [Learning to Protect Communications
with Adversarial Neural Cryptography](https://arxiv.org/abs/1610.06918).



## 1 - Generative Models
This section will provide some background on generative models in general,
making reference to [NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/abs/1701.00160).


## 2 - Generative Adversarial Networks
This section will explain the core concepts of generative adversarial networks,
and how their core feature is the adversarial training. This section will also
make reference to the previous link.


## 3 - Cryptographically Secure Pseudo Random Number Generators
A cryptographically secure pseudo-random number generator (CSPRNG) or
cryptographic pseudo-random number generator (CPRNG) is a pseudo-random number
generator (PRNG) with properties that make it suitable for use in cryptography.

### Applications of Random Number Generators
Many aspects of cryptography require random numbers, including key generation,
nonces, one-time pads, and salts in certain signature schemes.

### Security Requirements of CSPRNGs
This section draws on this Wikipedia [page](https://en.wikipedia.org/wiki/Cryptographically_secure_pseudorandom_number_generator#Requirements).

CSPRNG requirements fall into two groups: first, that they pass statistical
randomness tests; and secondly, that they hold up well under serious attack,
even when part of their initial or running state becomes available to an
attacker.

1.  Every CSPRNG should satisfy the **next-bit test**. That is, given the first
    k bits of a random sequence, there is no polynomial-time algorithm that can
    predict the (k+1)th bit with probability of success non-negligibly better
    than 50%. Andrew Yao proved in 1982 that a generator passing the next-bit
    test will pass all other polynomial-time statistical tests for randomness.
2.  Every CSPRNG should withstand "**state compromise extensions**". In the event
    that part or all of its state has been revealed (or guessed correctly),
    it should be impossible to reconstruct the stream of random numbers prior
    to the revelation. Additionally, if there is an entropy input while running,
    it should be infeasible to use knowledge of the input's state to predict
    future conditions of the CSPRNG state.

Example: If the CSPRNG under consideration produces output by computing bits
of π in sequence, starting from some unknown point in the binary expansion,
it may well satisfy the next-bit test and thus be statistically random,
as π appears to be a random sequence. However, this algorithm is not
cryptographically secure; an attacker who determines which bit of pi
(i.e. the state of the algorithm) is currently in use will be able to calculate
all preceding bits as well.

### Random Number Generator Attacks
This section draws on this Wikipedia [page](https://en.wikipedia.org/wiki/Random_number_generator_attack).

Lack of quality in a PRNG generally provides attack vulnerabilities and so
leads to lack of security, even to **complete compromise**, in cryptographic
systems. The RNG process is particularly attractive to attackers because it is
typically a **single isolated hardware or software component** easy to locate.

For **software RNGs**, which this work is concerned with, we can identify these
major types of attacks:

1.  **Direct cryptanalytic attacks**: when an attacker obtains part of the stream
    of random bits and uses this to distinguish the RNG output from a truly
    random stream (i.e. to predict future bits).
2.  **Input-based attacks**: when an attacker manages to modify the input to the
    RNG to attack it, for example by "flushing" existing entropy and putting
    the RNG in a known state.
3.  **State compromise extension attacks**: when the internal secret state of the
    RNG is known at some time, it can be used to predict future output or to
    recover previous output. This can happen when a generator starts up and has
    little or no entropy, so the attacker may be able to guess the initial
    state.

This work is mostly concerned with cryptanalytic attacks.


## 4 - Testing CSPRNGs
Andrew Yao showed that a sequence is random if, and only if, every probabilistic
polynomial-time algorithm fails to predict the next bit of the sequence with a
significant probability. However, using the universal quantifier (every
algorithm) confines the next bit test to being merely theoretical, rather than
a practical test.

### SADEGHIYAN-MOHAJERI TEST
Sadeghiyan and Mohajeri presented a test that measures the randomness of a
sequence based on the predictability of the next bit of an underlying sequence,
given the former bits.

The test algorithm takes advantage of a tree structure, which stores information
on the patterns of subsequences in the overall sequence.

![Pattern Tree](./img/pattern_tree.png)

In the pattern tree, each node in depth `l` represents the number of
occurrences of a binary pattern of length `l` in the underlying sequence.
Each edge connecting two nodes denotes the ratio of the number of child patterns
located in the next later to the number of their parent patterns in the previous
layer that is, a conditional probability `P(child | parent)`. For a large enough
random sequence, it is expected that all the ratios corresponding to the edges
of the pattern tree to be approximately equal to 1/2.

The algorithm for the test is as follows:

``` java
// threshold for whether we consider sequence random
double decision_threshold = (1 + Math.sqrt((x^2)/n)) / 2;
```


## 5 - Architecture of this Project
This section will outline the overall architecture of the implementation.


## 6 - Testing Methodology
This section will outline how the implementation is tested.


## 7 - Evaluation
This section will outline the results of training.
