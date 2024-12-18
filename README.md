# minimal_points_in_nsphere
This repo contains code to compute the points required to obtain a ncycle of a given persistence on the nsphere 

# The problem to solve

Given an n-dimensional sphere unit $S^n$, the homology groups are Z in
dimension 0 and n, and 0 otherwise.

Now, suppose we sample points from uniform distribution in $S^n$. If I
am able to sample many many many points, I expect then a persistence
interval [0,1] in dimension n.

The question is, how many points (n) do I need to sample in order to
have, with a high probability, an interval of a length k \iin [0,1] in
n dimensional persistence. We can, at the beginning, set k to 0.5,

Now, the tricky part is what do I mean by high probability? Let us
define it in Monte Carlo fashion - say that, in the collection of N
samples, 95% of cases we observe such a interval.
So, is N = 100, if I sample n points 100 times, 95 times we will
observe a persistence interval of a length greater of equal k in
dimension n.

We want to find n as a function of k and the significance level (95%)
