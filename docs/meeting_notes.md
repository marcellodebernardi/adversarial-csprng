# MEETING NOTES
Discussion points for weekly meeting with supervisors.


## 1 - PAPERS

1.  Passing the **next bit test** is equivalent to passing all polynomial-time
    statistical randomness tests.
1.  **Next bit test is theoretical** (it is expressed in terms of "all possible
    polynomial-time algorithms"). If a CSPRNG can be shown to pass the next-bit
    test we know the output also passes statistical tests for randomness, but
    algorithmically showing proof of passing the theoretical test is unfeasible.
2.  The **Sadeghiyan-Mohajeri test** builds a "pattern tree" of an input, and
    using that tree, computes the probability of predicting the next bit in the
    pattern at any point.
3.  The **practical next-bit test** (PNT) is a better implementation of the SM
    test. This should set basis for discrimination.
