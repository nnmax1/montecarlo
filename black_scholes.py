
import stdio
import sys
import math

#-----------------------------------------------------------------------

# Return the value of the Gaussian probability function with mean 0.0
# and standard deviation 1.0 at the given x value.

def phi(x):
    return math.exp(-x * x / 2.0) / math.sqrt(2.0 * math.pi)

#-----------------------------------------------------------------------

# Return the value of the Gaussian probability function with mean mu
# and standard deviation sigma at the given x value.

def pdf(x, mu=0.0, sigma=1.0):
    return phi((x - mu) / sigma) / sigma

#-----------------------------------------------------------------------

# Return the value of the cumulative Gaussian distribution function
# with mean 0.0 and standard deviation 1.0 at the given z value.

def Phi(z):
    if z < -8.0: return 0.0
    if z >  8.0: return 1.0
    total = 0.0
    term = z
    i = 3
    while total != total + term:
        total += term
        term *= z * z / float(i)
        i += 2
    return 0.5 + total * phi(z)

#-----------------------------------------------------------------------

# Return standard Gaussian cdf with mean mu and stddev sigma.
# Use Taylor approximation.

def cdf(z, mu=0.0, sigma=1.0):
    return Phi((z - mu) / sigma)

#-----------------------------------------------------------------------

# Black-Scholes formula.


# sigma = volatility 
# t = time to expiration
# rho = risk free rate
def callPrice(stockPrice, exersizePrice, rho, sigma, t):
    a = (math.log(stockPrice/exersizePrice) + (rho + sigma * sigma/2.0) * t) / \
        (sigma * math.sqrt(t))
    b = a - sigma * math.sqrt(t)
    return stockPrice * cdf(a) - exersizePrice * math.exp(-rho * t) * cdf(b)


# example : 
# AMD Call Option w/ $150 strike rho=0.05 10% volatitiy 2 week expiration 
# AMD price was $152.45 at the time
#callp = callPrice(152.45, 150, 0.05, 0.1, 14) 

