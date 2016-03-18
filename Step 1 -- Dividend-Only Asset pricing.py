
# coding: utf-8

# # Dividend-Only Heterogeneous-Agent Asset Pricing 
# 
# 
# Let's define the simplest possible asset-pricing agent. We will strip away as much as possible until we have the most basic agent, and then use that agent to incrementally explore a number of possible feedback mechanisms which may be quite important in "highly heterogeneous"-agent general equilibrium. These will include:
# 
# * differences in wealth levels
# * differences in income processes (due to differing ideosyncratic shocks)
# * differences in beliefs
# * differences in learning mechanisms and/or experience 
# * differences in preferences, including risk aversion and time discounting
# * differences in portfolio allocations or policy functions due to heuristics
# 
# These are roughly in order of "palatability" -- while few people will disagree that differences in wealth levels are important, it becomes harder to measure (and justify) differences further down the list, such as differences in preferences or heuristics. 
# 
# The most basic asset pricing model will allow us to think about the effects of heterogeneity in the above characteristics easily, before moving to more complicated situations. 
# 
# In addition, solving the simplest model first will provide us a foundation against which to test more complex models to look for bugs and coding errors. This second usage will become more apparent as we progress. 
# 
# For simplicity of exposition we will use a two-period model to begin. Under a number of circumstances, the infinite-period model collapses to the two period model for optimization purposes. We will explore this in detail much later. 
# 
# ## The Agent Problem
# 
# We will employ the two-period basic asset pricing problem from the first chapter of Cochrane (2001):
# 
# $$
# \begin{aligned}
# \underset{\xi_2}{\mathrm{max}} & \;\; u(c_{1})  + \beta \mathbb{E}\left[u(c_{2})\right] \\
# c_1 & = \xi_1 d_1 + (\xi_1 - \xi_2)p_1 \\ 
# c_{2} & = \xi_2 d_2 \\
# c_{t} & \ge 0 \forall t \\
# \xi_{t+1} & \ge -\xi_{t} \forall t
# \end{aligned}
# $$
# 
# 
# The agent in question begins period 1 with an endowment of the single risky asset available in the economy, $\xi_1$, which has already realized divided $d_1$ before the period began. Thus the income indowment for the agent at the beginning of period 1 is $\xi_1 d_1$.
# 
# 
# To solve the problem substitute the constraints into the objective, take first derivative and set to zero to find the Euler. We arive at the familiar expression:
# 
# $$
# \begin{aligned}
# u'(c_1)p_1 = \beta \mathbb{E}\left[u'(c_{2}) d_2 \right]
# \end{aligned}
# $$
# 
# For CRRA utility this is straightforward to solve algebraically for $\xi_2$:
# 
# 
# Log CRRA:
# 
# $$
# \begin{aligned}
# \tilde{\xi_2} = \xi_1 \frac{d_1+p_1}{p_1} \frac{\beta}{1 + \beta}
# \end{aligned}
# $$
# 
# 
# Non-Log CRRA:
# 
# $$
# \begin{aligned}
# \tilde{\xi_2}  = \frac{\xi_1 (d_1 + p_1)}{\left( \frac{\beta}{p_1} \mathbb{E}\left[ d_2^{1-\rho} \right] \right)^{-1/\rho} + p_1}
# \end{aligned}
# $$
# 
# Note that the solutions simply tie the choice $\xi_2$ to already known values and the expected values of $d_2$ at time 2.
# 
# The tilde "~" above the the demand indicates that this is unrestricted demand -- we impose the two constraints as follows:
# 
# $$
# \begin{aligned}
# \xi_2 = \min(\xi_1 + \frac{\xi_1 d_1}{p_1}, \max(\tilde{\xi_2}, 0.0))
# \end{aligned}
# $$
# 
# 
# We can examine these two demand functions easily:
# 
# 

# In[1]:

get_ipython().magic(u'matplotlib inline')
from __future__ import division
import pylab as plt
import numpy as np


# Parameters:
beta = 0.99
rho = 2.0
rho3 = 3.0
xi1 = 0.85
d1 = 1.0
y0=1.0
xi0=1.0

D_vals = np.array([0.01, 0.7, 1.2, 1.7, 1.99])
D_probs = np.array([0.5, 3.0, 5.0, 3.0, 0.5])

# Normalize D_probs...
D_probs = D_probs/float(np.sum(D_probs))


# Define two demand functions:
def log_demand_Tm1(xi1, d1, p1, beta):
    xi2 = ((xi1 * d1) / p1 + xi1) * beta/(1.0 + beta)
    return max( min(xi1 + xi1*d1/p1, xi2), 0.0)

def crra_demand_Tm1(xi1, d1, p1, beta, rho, D_vals, D_probs):
    xi2  = xi1 * (d1 + p1) / ( ((beta / p1) * np.dot(D_vals**(1.0-rho), D_probs) )**(-1.0/rho) + p1)
    return max( min(xi1 + xi1*d1/p1, xi2), 0.0)





# Define some prices:
prices = np.linspace(0.01,10,1000)


log_demands = []
crra_demands = []
crra_demands_rho3 = []

# Make list of rho-values to use here (will also use again below):
rho_list = [1.0, rho, rho3]

for p1 in prices:
    log_demands.append( log_demand_Tm1(xi1, d1, p1, beta) )
    crra_demands.append( crra_demand_Tm1(xi1, d1, p1, beta, rho_list[1], D_vals, D_probs))
    crra_demands_rho3.append( crra_demand_Tm1(xi1, d1, p1, beta, rho_list[2], D_vals, D_probs))

# Plot against prices:
plt.plot(log_demands, prices, label="Log demand")
plt.plot(crra_demands, prices, label="CRRA demand\nrho = "+str(rho))
plt.plot(crra_demands_rho3, prices, label="CRRA demand\nrho = "+str(rho3))
plt.xlabel("Quantitiy xi demanded")
plt.ylabel("Price")
plt.xlim([0.0, 8.0])
plt.legend(loc='best', frameon=False)
plt.title("Comparing Log, CRRA rho="+str(rho)+" Demand Fxns")
plt.show()


# If we define multiple agents who have these demand functions, then finding aggregate demand is simple: for ay price, sum across all agents demand at that price. 
# 
# Note that this approach allows us to inject potentially complicated balance sheet dynamics into individual demand, and thus aggregate demand as well. For example, if an agent was borrowing on collateral and the value of collateral dropped, the agent may need to suddenly switch from buying to selling an asset. This might create "kinks" in the aggregate demand curve. We will explore this with a few examples below. 
# 
# 
# Finding price with these agents is now straightforward -- define a supply for every price (could be fixed total supply) and then numerically find the price which clears the market. That is, find the price which sets aggregate demand to zero. 
# 
# We can check that price-finding works by confirming, for example, that the selected price leaves each individual agent with equal marginal utility.
# 
# As a first step in this direction, let's codify the demand functions above in an Asset Pricing Agent object. 

# In[1]:

class AssetPricingAgent(object):


    def __init__(self, beta, rho, xi1, d1, D2_vals, D2_probs):

        # Set basic parameters, beliefs, and state variables:
        self.beta = beta
        self.rho = rho3
        self.D2_vals, self.D2_probs = D2_vals, D2_probs
        self.xi1 = xi1
        self.d1 = d1
        self.y1 = self.xi1 * self.d1

        # Set the utility functions:
        self.u, self.uprime, self.uprime_inverse = self.set_utility(rho=self.rho)

        # Set demand functions:
        self.demand = self.set_demand(xi1=self.xi1, d1=self.d1, beta=self.beta,
                                      rho=self.rho,
                                      D2_vals=self.D2_vals,
                                      D2_probs=self.D2_probs)

    def set_utility(self, rho):

        # First check that rho is positive:
        if rho < 0:
            raise Exception, "rho < 0; rho must be non-negative. Currently rho = "+str(rho)

        # There are three typs of CRRA utility that this could use:
        if rho == 1.0:
            # Then agent has log utility:
            u = np.log
            uprime = lambda c: 1.0/c
            uprime_inverse = lambda z: 1.0/z
        elif rho == 0.0:
            # Then linear utility
            u = lambda c: c
            uprime = lambda c: 1.0
            uprime_inverse = None  # There is literally no inverse for a constant functions
        else:
            # Then non-linear, non-log CRRA:
            one_minus_rho = 1.0 - rho    # Define a couple constants once...
            one_over_rho = 1.0/rho

            u = lambda c, one_m_rho=one_minus_rho: (c**one_m_rho) / one_m_rho
            uprime = lambda c, rho=rho: c**(-rho)
            uprime_inverse = lambda z, one_o_rho=one_over_rho: z**(-one_o_rho)

        # Return values:
        return u, uprime, uprime_inverse


    def set_demand(self, xi1, d1, beta, rho, D2_vals, D2_probs):
        '''
        Simply encapsulate the above values:
        '''
        if rho == 1.0:
            # Then agent has log utility:
            def demand_Tm1(p1, xi1=xi, d1=d1, beta=beta):
                xi2 = ((xi1 * d1) / p1 + xi1) * beta/(1.0 + beta)
                return max( min(xi1 + xi1*d1/p1, xi2), 0.0)
        elif rho == 0.0:
            # Then linear utility
            raise Exception, "Demand will go to limits whenver the price departs from risk-free price. Still to implement."
        else:
            # Then non-linear, non-log CRRA:

            def demand_Tm1(p1, xi1=xi1, d1=d1, beta=beta, rho=rho, D_vals=D2_vals, D_probs=D2_probs):
                xi2  = xi1 * (d1 + p1) / ( ((beta / p1) * np.dot(D_vals**(1.0-rho), D_probs) )**(-1.0/rho) + p1)
                return max( min(xi1 + xi1*d1/p1, xi2), 0.0)

        return demand_Tm1


# We can recreate the demand functions shown above:

# In[ ]:

# Create a list of agents for each of the rho values used above:
agents = []
for crra in rho_list:
    agents.append( AssetPricingAgent(beta=beta, rho=crra, 
                                     xi1=xi1, d1=d1, 
                                     D2_vals=D_vals, D2_probs=D_probs) )

# Now re-create the plots from above:
agent_demands = []
for agent in agents:
    agent_demands.append([])
    for p1 in prices:
        agent_demands[-1].append(agent.demand(p1))



