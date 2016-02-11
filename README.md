This repository contains a "baseline" simple asset-pricing agent class;
many instances of this agent can be used to create markets.

The agent description I use is the simplest possible asset-pricing agent which
captures wealth effects and the effects of heterogenous beliefs. In this first
step example, agents do not form an expectation about prices in the following
period, only dividends. Agents also do not learn.

Price expectations and learning are obvious dynamics to add, but it is important
to note that even without these there are some key features which emerge:

* wth homogenous beliefs and identical preferences, agents who start at different initial cash levels converge quickly to their optimal level of wealth and then behave identically, and zero volume trade is seen after convergence -- a version of no-trade holds.  
* with heterogneous beliefs and identical preferences, there is constant volume of trade between the "wrong" agents and the "right" agents. Prices are continually affected by the presence of the heterogeneous beleifs


The agent problem:

max  u(ct) + beta E[u(ctp1)]
xi
      s.t.
ct   = y0 + (xi0 - xi) * p
ctp1 = xi * d
xi   >= -xi0
xi   < (y+xi0*p) / p   # implied by constraint on ct, budget
ct   >= 0
ctp1 >= 0

The standard Euler:
u'(ct)p = beta E[u'(ctp1)d]
p = beta E[u'(d*xi)d]

The standard CRRA pricing equation:
p = beta E[u'(ctp1) / u'(ct) * d]
p = beta E[u'(d*xi) / u'(y+(xi0-xi)p) * d]

We can invert the pricing equation to back out the demand schedules for assets:

Log utility:
xi = (y+xi0*p) / p * beta/(1+beta)

non-log CRRA utilities:

xi = (y + xi0*p) * (1.0 / ( (beta/p *  np.dot(D**(1.0 - rho), D_prob)) ** (-1.0/rho) + p) )


* Each period unfolds the same:
    * agent is endowed with an asset endowment xi0 and a cash endowment y0
    * agents form their demand schedules demands by solving a two-period problem
    * aggregate excess demand is simply added up for all agents given price p; numerically find the price which sets excess demand to 0
    * after the market-clearing price is realized, appropriate trade occurs between agents
    * in all periods following the initial period, the cash endowment y0 comes from a realization of the stochastic dividend
    * agents repeat each day, but wealth (and thus marginal utility) is always changing.
