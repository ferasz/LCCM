What lccm is

Lccm is a Python package for estimating latent class choice models 
using the Expectation Maximization (EM) algorithm to maximize the likelihood function.


Main Features

Latent Class Choice Models

Supports datasets where the choice set differs across observations

Supports model specifications where the coefficient for a given variable may be generic (same coefficient across all alternatives) 
or alternative specific (coefficients varying across all alternatives or subsets of alternatives) in each latent class

Accounts for sampling weights in case the data you are working with is choice-based 
i.e. Weighted Exogenous Sample Maximum Likelihood (WESML) from (Ben-Akiva and Lerman, 1983) to yield consistent estimates.

Accounts for constraining the choice set across latent classes whereby each latent class can have its own subset of alternatives
in the repective choice set

Accounts for constraining the availability of latent classes to all individuals in the sample whereby it might be the case
that a certain latent class or set of latent classes may be unavailable to certain decision-makers according to the 
modeler.


Where to get it

Available from PyPi::
pip install lccm

https://pypi.python.org/pypi/lccm/0.1.3


For More Information

If the lccm package is useful in your research or work, please cite this package.

License

Modified BSD (3-clause)



Changelog

0.1.3 (April 16th, 2017)

Initial package release for estimating latent class choice models using the Expectation Maximization Algorithm.