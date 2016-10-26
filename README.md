# LCCM = Latent Class Choice Models
Entails the code for estimating latent class choice models using the Expectation Maximization (EM) algorithm in 
addition to constraining the choice set across classes.
The code also accounts for sampling weights in case the data you are working with is choice-based 
i.e. Weighted Exogenous Sample Maximum Likelihood (WESML) from (Moshe and Lerman, 1983) to yield consistent estimates.
This is an optional feature in the code that you account for in case your sample is not random.
