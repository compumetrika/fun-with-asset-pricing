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
