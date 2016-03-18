import numpy as np
import pylab as plt
from copy import deepcopy
from scipy.optimize import brentq


"""
First, directly lift and modify the DiscreteRV code from Quant-econ.net:

Filename: discrete_rv.py
Authors: Thomas Sargent, John Stachurski
Generates an array of draws from a discrete random variable with a
specified vector of probabilities.
"""
from numpy import cumsum
from numpy.random import uniform


class DiscreteRV(object):
    """
    Generates an array of draws from a discrete random variable with
    vector of probabilities given by q.
    Parameters
    ----------
    q : array_like(float)
        Nonnegative numbers that sum to 1
    Attributes
    ----------
    q : see Parameters
    Q : array_like(float)
        The cumulative sum of q
    """

    def __init__(self, q, vals, seed=None):
        self._q = np.asarray(q)
        self.Q = cumsum(q)
        self.vals = np.array(vals)
        self.RNG = np.random.RandomState(seed)

    def __repr__(self):
        return "DiscreteRV with {n} elements".format(n=self._q.size)

    def __str__(self):
        return self.__repr__()

    @property
    def q(self):
        """
        Getter method for q.
        """
        return self._q

    @q.setter
    def q(self, val):
        """
        Setter method for q.
        """
        self._q = np.asarray(val)
        self.Q = cumsum(val)

    def draw(self, k=1):
        """
        Returns k draws from q.
        For each such draw, the value i is returned with probability
        q[i].
        Parameters
        -----------
        k : scalar(int), optional
            Number of draws to be returned
        Returns
        -------
        array_like(int)
            An array of k independent draws from q
        """
        idraws = self.RNG.uniform(0, 1, size=k)
        return self.vals[self.Q.searchsorted(idraws)]



class AssetPricingAgent(object):
    '''
    A simple asset-pricing agent based on a two-period CRRA model.
    '''

    def __init__(self, beta, rho, D, D_probs, y0, xi0=0.0):
        '''
        Initialize an asset-pricing agent.

        Parameters
        ----------
        beta : float
            Time discounting factor
        rho : float
            Risk aversion paramter
        D : array_like(float, ndim=2 or 1)
            Payoff values for different states of the dividend
        D_probs : array_like(float, ndim=2 or 1)
            Probabilities associated with payoffs for states in payoff_dividend vector
        y0 : float
            Initial cash endowment
        xi0 : float
            Initial endowment of number of shares of the risky asset

        Returns
        -------
        Nothing
        '''

        # Set agent paramters:
        self.beta, self.rho = beta, rho
        self.y = y0
        self.xi0 = xi0

        # Ensure that dividend payoffs are sorted from smallest to largest:
        self.D = np.array(D)
        self.D_probs = np.array(D_probs)
        # Sort:
        ii = np.argsort(self.D)
        self.D = self.D[ii]
        self.D_probs = self.D_probs[ii]

        # Quick check:
        assert np.isclose(np.sum(D_probs), 1.0), "Problem: p_dividend does not sum to 1.0: np.sum(D_probs) = " + str(np.sum(D_probs))

        # Initialize utility and demand functions
        self.u  = None
        self.uprime = None
        self.uprimeprime = None
        self.uprime_inverse = None
        self.demand = None

        # Now actually fill in utility and demand functions
        self.update_utility_demand()


    def renormalize_beliefs(self):
        # Assuming the beliefs have been changed in some way, renormalize them.
        self.D_probs = self.D_probs/sum(self.D_probs)

        # Quick check:
        assert np.isclose(np.sum(D_probs), 1.0), "Problem: p_dividend does not sum to 1.0: np.sum(D_probs) = " + str(np.sum(D_probs))

        # Also automatically reset demand functions appropiriately:
        self.update_utility_demand()


    def update_utility_demand(self):
        # Simply update the utility and demand functions given own current params
        # Initialize utility and demand functions
        self.u, self.uprime, self.uprimeprime, self.uprime_inverse = self.set_utility(self.rho)
        self.demand = self.set_demand(rho=self.rho, y=self.y, xi0=self.xi0,
                                      beta=self.beta, D=self.D,
                                      D_prob=self.D_probs)  #, xi0)

    def set_utility(self, rho):
        '''
        Return the utility functions associated with the risk-aversion parameter
        rho.
        '''
        # Set CRRA utility:
        if rho == 1.0:
            u = np.log
            uprime = lambda c: 1.0/c
            uprimeprime = lambda c: -c**(-2)
            uprime_inverse = lambda z: 1.0/z
        elif rho == 0.0:
            # Then constant utility - risk neutral
            u = lambda c: c
            uprime = lambda c: 1.0
            uprimeprime = lambda c: 0.0
            uprime_inverse = None # Not defined for constant function
        else:
            # The "regular" CRRA
            u = lambda c, oneminusrho=(1.0-rho): c**oneminusrho / oneminusrho
            uprime = lambda c, rho=rho: c**-rho
            uprimeprime = lambda c, rho=rho: -rho*c**(-1.0-rho)
            uprime_inverse = lambda z, rhoinv=(1.0/rho): z**-rhoinv

        return u, uprime, uprimeprime, uprime_inverse


    def set_demand(self, rho, y, xi0, beta, D, D_prob):
        '''
        Set the demand function: xi = f(p)
        '''
        # Set CRRA demand:
        if rho == 1.0:
            # Log demand:
            demand = lambda p,y=y,beta=beta:  max(min( (y+xi0*p) / p * beta/(1+beta), (y+xi0*p) / p), -xi0)
        elif rho == 0.0:
            raise Exception, "Risk Neutral demand not implemented."
            # The problem witn risk-neutral demand: agents will trade to their
            # limits on either side of the risk-neutral price.
            # Confirm
            def demand(p,beta=beta,D=D,D_prob=D_prob):
                risk_free_price = beta * np.dot(D, D_prob)
                if p > risk_free_price:
                    return -xi0
                elif p < risk_free_price:
                    return (y+xi0*p) / p
                else:
                    return xi0
        else:
            # "Regular" CRRA demand:
            def demand(p, y=y, xi0=xi0, beta=beta, D=D, D_prob=D_prob, rho=rho): #, xi0=xi0):
                '''Note: the max imposes the short-selling constraint -- namely,
                the agent cannnot sell more than they have.'''
                xi = (y + xi0*p) * (1.0 / ( (beta/p *  np.dot(D**(1.0 - rho), D_prob) )**(-1.0/rho) + p) )
                return max(min( xi, (y+xi0*p) / p), -xi0)

        return demand


class AssetMarket(object):

    def __init__(self, agents, D, D_probs, aggregate_asset_endowment=1.0, seed=None):
        '''
        Market simply consists of collection of agents.
        '''

        self.seed=seed
        self.agents = agents
        self.price_history = []
        self.volume_history = []
        self.payoff_history = []

        self.total_agent_cash_wealth_history = []
        self.total_agent_initial_asset_endowment = []
        #self.total_agent_asset_volume = []

        self.aggregate_asset_endowment = aggregate_asset_endowment

        # Set up storage for agent values:
        self.agents_history = [] # Store agent variables here.
        for i in range(len(self.agents)):
            temp_storage = {'y':[],'c':[],'xi':[]}
            self.agents_history.append(temp_storage)

        # To find reasonable "first guess" price, find the risk-neutral asset
        # asset price for first-agent's preferences:
        agent0 = self.agents[0]
        self.p0_guess = agent0.beta * np.dot(agent0.D, agent0.D_probs)
        if self.p0_guess <= 0.0:
            self.p0_guess = 1e-5
        self.p_tm1 = self.p0_guess

        # Set up the dividend process:
        # Ensure that dividend payoffs are sorted from smallest to largest:
        self.D = np.array(D)
        self.D_probs = np.array(D_probs)
        ii = np.argsort(self.D)
        self.D = self.D[ii]
        self.D_probs = self.D_probs[ii]

        # Quick check:
        assert np.isclose(np.sum(D_probs), 1.0), "Problem: p_dividend does not sum to 1.0: np.sum(D_probs) = " + str(np.sum(D_probs))

        self.D_process = DiscreteRV(self.D_probs, self.D, seed=self.seed)
        # Done


    def run_markets(self, T):
        '''
        Run market for T periods.
        '''

        # Draw dividend:
        D = self.D_process.draw(T)

        for t in range(T):
            # Clear markets:
            pstar = self.clear_market2()

            # Update
            self.price_history.append(pstar)
            self.payoff_history.append(D[t])

            # Get agent starting wealth, total agent initial asset endowment,
            # and total volume traded.
            # Important to do the following update before updating agent values:
            total_agent_initial_cash = 0.0
            total_agent_initial_asset_endowment = 0.0
            total_agent_asset_volume = 0.0
            for agent in self.agents:
                total_agent_initial_cash += agent.y  # Get wealth they started the period with
                total_agent_initial_asset_endowment += agent.xi0
                total_agent_asset_volume += np.abs(agent.xi0 - agent.demand(pstar))

            self.total_agent_cash_wealth_history.append(total_agent_initial_cash)
            self.total_agent_initial_asset_endowment.append(total_agent_initial_asset_endowment)
            self.volume_history.append(total_agent_asset_volume)

            # Now update agents and histories:
            self.update_agents(pstar=pstar, d=D[t])
        # Done


    def excess_demand(self, p, total_supply, agents):
        '''
        Iterate over all agents and ask for demand given price.
        '''

        '''
        if agents is None:
            agents = self.agents

        if total_supply is None:
            total_supply = self.aggregate_asset_endowment
        '''

        total_demand = 0.0
        for an_agent in agents:
            total_demand += an_agent.demand(p)

        return total_demand - total_supply


    def clear_market2(self, alt_title=""):

        agents_to_use = []
        for agent in self.agents:
            if agent.participate:
                agents_to_use.append(agent)

        supply_to_use = 0.0   #self.aggregate_asset_endowment
        for an_agent in agents_to_use:
            supply_to_use += an_agent.xi0

        pstar, root_results = brentq(f=self.excess_demand, a=1e-6, b=2000, full_output=True, disp=True,
                                     args=(supply_to_use, agents_to_use))

        if not root_results.converged:
            print "WARNING: root_results.converged not True!:  root_results.converged = ", root_results.converged
            print "root_results:", root_results

        print "Market-clearing price for "+ str(len(agents_to_use)) +" agents, total supply "+str(supply_to_use)+":", pstar
        return pstar


    def clear_market(self, p0=None):
        '''
        Given an initial price guess, zero-find on total asset demand.

        p0 is initial price.
        '''

        # Set intial price guess
        if p0 is None:
            p0 = self.p_tm1
            # Note: currently not using first guess....

        # Zero-find to determine price:
        p, root_results = brentq(f=self.excess_demand, a=0.001, b=1000, full_output=True, disp=True)

        if not root_results.converged:
            print "WARNING: root_results.converged not True!:  root_results.converged = ", root_results.converged
            print "root_results:", root_results

        return p


    def update_agents(self, pstar, d):
        '''
        Given pstar, update all agent histories.
        '''
        for i, agent in enumerate(self.agents):
            # Determine and save choices:
            xi = agent.demand(pstar)
            ct = agent.y + pstar*agent.xi0 - pstar*xi
            #print "new c=calc"
            self.agents_history[i]['c'].append(ct)
            self.agents_history[i]['y'].append(agent.y)
            self.agents_history[i]['xi'].append(xi)

            # Update agent value:
            agent.y = xi*d
            agent.update_utility_demand()
        # Done


if __name__ == "__main__":
    import csv

    # Parameters:
    beta = 0.99
    rho = 2.0
    y0 = 1.0

    D = np.array([0.05, 0.7, 1.0, 1.3, 1.95])
    D_probs = np.array([1.0, 3.0, 5.0, 3.0, 1.0])

    D_probs = D_probs/np.sum(D_probs)
    #xi0 = 0.0
    total_supply = 1.0

    seed=123456

    # Define an agent, and then find some prices:
    an_agent = AssetPricingAgent(beta=beta,
                                rho=rho,
                                D=D,
                                D_probs=D_probs,
                                y0=y0)

    # Plot the demand function for various prices:
    prices = np.linspace(0.1,10,100)
    demands = []
    for p in prices:
        demands.append(an_agent.demand(p))

    plt.plot(demands,prices)
    plt.xlabel("Quantity Demanded")
    plt.ylabel("Prices")
    plt.title("One Agent Demand")
    plt.show()
    plt.close()

    # Let's try running the market:
    agent1 = deepcopy(an_agent)  # Average but wrong expectations
    agent2 = deepcopy(an_agent)  # Poor but all else equal
    agent3 = deepcopy(an_agent)  # Rich but all else equal

    total_two_agent_wealth = agent2.y + agent3.y
    agent_2_fraction = (1.0/4.0)
    agent2.y = agent_2_fraction * (total_two_agent_wealth)   # Cut to 1/3 of two agents summed wealth
    agent2.update_utility_demand()

    agent3.y = (1.0-agent_2_fraction) * (total_two_agent_wealth)   # Set to 2/3 of two agents summed wealth
    agent3.update_utility_demand()

    # Give agent 1 wrong expectations:
    INCLUDE_PESSIMIST = True
    if INCLUDE_PESSIMIST:
        agent1.D_probs[0] *= 10.0    # Increase probability of lowest/higest state
        agent1.renormalize_beliefs()
        agent1.update_utility_demand()
        print "Making one agent pessimistic!"


    agents_to_use = [agent1, agent2, agent3]
    supply_to_use = total_supply*len(agents_to_use)

    # Set each agent's initial asset holding endowment:
    for one_agent in agents_to_use:
        one_agent.xi0 = 1.0 #supply_to_use/float(len(agents_to_use))
        one_agent.update_utility_demand()


    # Define the market:
    market = AssetMarket(agents=agents_to_use,
                         D=D, D_probs=D_probs,
                         aggregate_asset_endowment=supply_to_use,
                         seed=seed)

    # Run the market for T periods:
    T=50
    market.run_markets(T=T)


    # Set a flag text:
    extra_descrip = ""
    if INCLUDE_PESSIMIST:
        extra_descrip = "Heterogenous-Beliefs"
    else:
        extra_descrip = "Homogenous-Beliefs"

    # Drop a csv file:
    var_names = ['dividend_eop','price_mid','volume_mid','agent_cash_bop','total_asset_endowment_bop']

    # repeat names as markers for dict:
    var_to_write = {'dividend_eop':market.payoff_history,
                'price_mid':market.price_history,
                'volume_mid':market.volume_history,
                'agent_cash_bop':market.total_agent_cash_wealth_history,
                'total_asset_endowment_bop':market.total_agent_initial_asset_endowment}

    assert (set(var_names) == set(var_to_write.keys())) and (len(var_names) == len(var_to_write.keys())), "List of names var_names not equal to keys in dict var_to_write!"

    # Open csv file
    with open('asset_sim_data-'+ extra_descrip +'.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(var_names)
        # Now construct the "cross section" rows of the data:\
        first_key = var_names[0]
        all_rows = []
        for i in range(len(var_to_write[first_key])):
            temp_row = []
            for key in var_names:
                temp_row.append(var_to_write[key][i])
            all_rows.append(temp_row)
            temp_row = None ## "Disconnect"

        writer.writerows(all_rows)
    # Now close.

    # Plot Prices quickly:
    plt.plot(market.price_history, label="Prices")
    plt.plot(market.payoff_history, label="Dividends")
    plt.plot(market.total_agent_cash_wealth_history, label="Total Agent Cash")
    plt.legend(loc='best',frameon=False)
    plt.title("Prices"+'\n'+extra_descrip)
    plt.savefig("Prices-" + extra_descrip + ".pdf")
    plt.show()

    # Plot Prices vs volume quickly:
    min_y = min(min(market.price_history), min(market.volume_history))
    max_y = max(max(market.price_history), max(market.volume_history))
    plt.plot(market.price_history, label="Prices")
    plt.plot(market.volume_history, label="Volume")
    plt.ylim(min_y - 0.1*(max_y-min_y), max_y)
    plt.legend(loc='best',frameon=False)
    plt.title("Prices vs Volume\n"+ extra_descrip)
    plt.savefig("Prices-vs-Volume-" + extra_descrip + ".pdf")
    plt.show()


    # Plot incomes quickly:
    for i in range(len(market.agents)):
        plt.plot(market.agents_history[i]['y'])
    plt.title("Wealth\n" + extra_descrip)
    plt.savefig("Wealth-" + extra_descrip + ".pdf")
    plt.show()

    # Plot xi quickly:
    for i in range(len(market.agents)):
        plt.plot(market.agents_history[i]['xi'])
    plt.title("Savings\n" + extra_descrip)
    plt.savefig("Savings-" + extra_descrip + ".pdf")

    plt.show()


    # Plot ct quickly:
    for i in range(len(market.agents)):
        plt.plot(market.agents_history[i]['c'])
    plt.title("Consumption\n" + extra_descrip)
    plt.savefig("Consumption-" + extra_descrip + ".pdf")
    plt.show()


    # --------------------------------------------------------------------------
    # Create an economy populated by increasingly more agents, part 1:
    '''Things we need lists of:

    '''
    N_agents = 1
    N_economies = 6  # Create 6 iterations of the economy with increasing agents
    N_agent_list = []
    Max_agent_wealth = 2.5
    Min_agent_wealth = 0.5
    for n in range(N_economies):
        pass
