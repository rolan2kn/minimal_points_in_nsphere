# minimal_points_in_nsphere
This repo contains code to compute the points required to obtain a ncycle of a given persistence on the nsphere 

# The problem to solve

Given an $n$-dimensional sphere unit $\mathbb{S}^n$, the homology groups are $\mathbb{Z}$ in
dimension $0$ and $n$, and $0$ otherwise. Now, suppose we sample points from uniform distribution in $\mathbb{S}^n$. If we are able to sample many many many points, we expect then a persistence interval $[0,1]$ in dimension $n$. \par

The question is, how many points ($nPts$) do we need to sample in order to have, with a high probability, an interval of a length $k \in [0,1]$ in $n$ dimensional persistence. We can, at the beginning, set $k$ to $0.5$,

Now, the tricky part is what do we mean by high probability? Let us define it in Monte Carlo fashion - say that, in the collection of $N$ samples, $95\%$ of cases we observe such a interval.
So, is $N = 100$, if I sample $n$ points $100$ times, $95$ times we will observe a persistence interval of a length greater of equal $k$ in dimension $n$.

Find $n$ as a function of $k$ and the significance level ($95\%$).

Please refer to [Project Reference](dlotko_homework.pdf) for detailed information.
