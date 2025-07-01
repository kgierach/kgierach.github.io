---
title: "Hybrid Bayesian CLV Estimation"
mathjax: true
layout: post
---

# Introduction

Customer lifetime value (CLV) is one of the most important — and most elusive — metrics in modern marketing science. Businesses rely on it to guide decisions about acquisition spend, retention strategies, and long-term forecasting. Yet most CLV models in use today reduce customer behavior to simple point estimates, ignoring the uncertainty, heterogeneity, and behavioral nuance that actually drive retention and value.

As a data scientist with a background in marketing analytics and Bayesian modeling, I’ve spent the last few months developing a hybrid probabilistic model that captures customer behavior more faithfully. The goal: to estimate CLV not as a static number, but as a distribution that reflects both the likelihood of churn and the variability of spend over time.

This post outlines the key ideas behind the paper I submitted to an upcoming academic conference, including:

- How a Weibull survival model can predict churn time in a behaviorally grounded way

- How a log-normal hierarchical model can estimate spend with user-level random effects and covariates

- How posterior predictive simulation lets us build full-distribution CLV estimates — not just point forecasts

The model was tested on synthetic data with known structure to validate performance and diagnose error. Despite its simplicity, the framework shows promise: it can be extended, debugged, and scaled with real-world datasets while preserving interpretability.

## Why a Hybrid Bayesian Approach?

Let's examine why a hybrid approach is valuable. Most CLV models fall into one of two camps:

- Purely statistical models that use regression or survival analysis to extrapolate value

- Black-box ML models that optimize for accuracy but sacrifice interpretability and uncertainty modeling

I wanted something different — a middle ground where both behavioral mechanisms and uncertainty could be explicitly modeled. That led to a two-part model:

1. Churn Model:

    A Weibull time-to-event model where the scale parameter is influenced by user covariates (demographics, region, etc.). This captures both early and late churn behaviors while remaining interpretable.

2. Spend Model:

    A hierarchical log-normal model of weekly purchase amount, where users vary randomly around a global intercept, but covariates shift their baseline spend levels. Weekly spend is only simulated up to the churn point.

By combining these via simulation, I could estimate CLV as the sum of expected weekly spend — for each user — across a simulated lifetime horizon.

# Model Architecture

The model fuses two probabilistic components:

- A time-to-churn model, where each user is assigned a latent churn week drawn from a Weibull distribution. However the 
assumption is that churn is affected by advertising exposures.  In this model, we
assume there is an optimal number of exposures at the customer level to prevent churn.  However we
also enable the model to fit a decay parameter that measures advertising fatigue, when too 
many ads are shown to a customer.

- A purchase model, which simulates weekly spend (conditional on survival) using a log-normal distribution with hierarchical structure and covariate effects.  The 
assumptions of the purchase are that fixed effects, such as demographics and region, dictate the amount of purchase, as well
as the individual customers idiosyncratic behavior.

Together, these form a generative model of customer behavior that simulates:

- How long each user remains active, and

- What they spend during that time.

1. Churn Model (Weibull)

Let \( T_u \) represent the latent time of churn for user \( u \), and let \( λ_u \) and \( κ \) be the scale and shape parameters of the Weibull distribution. The scale is modeled as a log-linear function of user co-variates \( x_u \):

$$ \begin{align}
\lambda_u &= \exp\left(\alpha_c + \mathbf{x}_u^\top {\beta}_c \right) \\
T_u &\sim \text{Weibull}(\kappa, \lambda_u)
\end{align} $$

This model captures early and late churn behavior flexibly, depending on $$κ$$, and generalizes across users via partial pooling on $$β_c$$,
 where $$x_u$$ are user co-variates and $β_c$ is the vector of corresponding coefficients.


2. Purchase Model (Hierarchical Log-Normal)

Let $y_{u,t}$ be the purchase amount for user $u$ at week $t$. The purchase is modeled only for $t < T_u$. The log of purchase amount is modeled with both global and user-level effects:

$$ \begin{align}
\mu_{u,t} &= \alpha_u + \alpha_{p} + \mathbf{x}_u^\top {\beta}_t \\
y_{u,t} &\sim \text{LogNormal}(\mu_{u,t}, \sigma_p^2) \quad \text{for } t < T_u \\
\end{align}
$$

Here:

- $$\begin{align} 
    \alpha_{u} \sim \mathcal{N}(0, \sigma_{\alpha_u}^2) 
    \end{align}$$ 
    captures per-user deviation in spending
- $$\alpha_p \sim \mathcal{N}(0, \sigma_{\alpha_p}^2) $$
    serves to capture the global spend intercept
- Covariates (e.g., demographics, region) explain systematic differences in spend behavior
- $$\beta_t$$ is a shared effect coefficient vector for each week in the analysis, and at this time
    each coefficient is modeled independently and drawn from $$N(0, \sigma_\beta )$$

In Figure 1 we can observe a plate diagram that further illustrates the 
combined model structure.  This final, far right shaded circle represents an accumulation 
of purchases prior to the customer churning, whereas the next circle to the left shows 
how purchases are modeled by the LogNormal distribution.  Moving further to the left, we observe how
the Weibull distribution is used to model time to churn, and how each customer effectively 
has their own distribution that characterizes their unique behavior.

<center>
<img src="/assets/img/hybrid_clv_graphviz_labeled.png" width="550" height="180" />


Figure 1: Hybrid Model Architecture Diagram
</center>



# Test Data

The data used in this blog post is stochastically generated data, with baked in ground truth. 
Just as a quick reminder, each customer's purchase patterns were simulated based on fixed effects,
while the churn propensity was simulated based on the same fixed effects, but also
the quantity of advertising exposure which that person received. Typically in real life, one would need to
infer the churn time from a lack of activity as noted in [^2], in the absence of a
deliberate unsubscribe event by the customer.

The following diagram shows the baked-in effects of Emails and Ads:

<center>
<img src="/assets/img/CLV_blog1_data_generaion_parameters_ground_truth.png" width="350" height="350" />


Figure 2: Synthetic Data 3D View of Advertising Effectiveness
</center>

# Posterior Predictive Simulation

After fitting the model with MCMC, we simulate:

- 100 posterior churn times, $$ T_u $$, for each user

$
T_u^{(s)} \sim \mathrm{Weibull}(\kappa_u^{(s)}, \lambda_u^{(s)}) \quad \text{for } s = 1, \dots, S
$


- For each churn time, simulate weekly purchases up to $ T_u $

- Sum these to get 100 estimates of CLV per user, as shown below:

$ \hat{\text{CLV}}_u = \sum_{t=1}^{T_u} \exp\left( \mu_{u,t}  \right) $

This gives us not just a point estimate, but a full posterior distribution of CLV, capturing uncertainty in both duration and monetary value.


## Estimating Churn 

For churn week, the posterior predictive mean of the churn week is shown in the following diagram, which compares the predicted versus actual.
We sample from the Weibull distribution fitted for each user from the MCMC model fitting using Monte-Carlo simulation, more specifically 
inverse transform sampling [^1]:

$$
T_u^{(s)} = \lambda_u^{(s)} \cdot \left( -\log(1 - U^{(s)}) \right)^{1 / \kappa_u^{(s)}}, \quad U^{(s)} \sim \mathrm{Uniform}(0, 1)
$$

The churn time $T_u^{(s)}$ is drawn independently per posterior sample, and purchases are simulated conditionally.

{% highlight python %}

import arviz as az
import numpy as np
import matplotlib.pyplot as plt

idata = az.from_numpyro( mcmc )

\# Sample posterior predictive churn times
lambda_post = idata.posterior["lambda_"].values.reshape(-1, n_customers)
alpha_post = idata.posterior["alpha"].values.reshape(-1, n_customers)    # shape (samples, 1)

\# Inverse CDF sampling (Monte Carlo): Weibull time-to-event
rng = np.random.default_rng(0)
u = rng.uniform( size=lambda_post.shape )
predicted_weeks = lambda_post * (-np.log(1 - u)) ** (1 / alpha_post)

{% endhighlight %}


<center>
<img src="/assets/img/churn_predicted_vs_actual.png" width="250" height="250" />


Figure 3: Predicted vs. Actual Churn
</center>

From Figure 3, we observe that generally, the expected value of the predicted distribution follows the actual churn, within a reasonable margin of error.
However that error compounds when we go to sample for each week for a given user.  

## Estimating CLV based on Churn 

The final step is to take the churn estimates from the previous step
and estimate CLV for each customer.  Recall that the distribution used to 
model each purchase amount is the Log-Normal distribution.  This distribution
often has a great deal of variance, and skew, which makes sampling the 
distribution prone to producing some estimates which are clearly outliers.

That said we recommend a different approach here, and base each CLV 
purchase on the median of the Log-Normal distribution, as fitted for that 
particular customer.  We can still quantify uncertainty, in a Bayesian
fashion by using the samples drawn for each element going into the Log-Normal, 
namely $u$, which quantifies each customer's idiosyncratic behavior, and the partial
pooled estimates for region and demographic, as well as the shared purchase
estimate over time, which is also modeled. 

To generate the point estimates for each customer's purchases over time, we compute
the mean of the Log-Normal distribution, as described previously, for each week up to 
the churn week of the customer:

- $\begin{align}
  \mu_{u,t} &= \alpha_u + \alpha_{p} + \mathbf{x}_u^\top {\beta}_t 
    \end{align}$
- $purchase\_amount_{u,t} = exp( \mu_{u,t} )$  

# Results and Future Work

Overall, our MAPE, for the presented model is shown in the following table:

| Algorithm   | Prediction Type | MAPE  |
|-------------|-----------------|-------|
| Hybrid      | Churn           | 0.249 |
| Hybrid      | Purchase        | 0.49   |
| Lifetimes Gamma-Gamma  | Purchase        | 0.87  |

<center>Table 1: Algorithm Performance Summary</center>

Our model performed reasonably well compared to industry standards, and outperformed the more basic Gamma-Gamma 
model, which does not utilize co-variates.  It is understandable that our model 
would perform better when utilizing more information.

Also, with respect to capturing the over-exposure penalty, our model failed to capture this parameter (it captured it as 0.00),
so this is another area of future work, to refine the model in such a way that
this parameter can be captured, as it is critical to throttling the frequency of advertising
exposures on customers.  That said, our model did capture the effect of Ads and Emails well,
inferring the approximate peak performance of each exposure type.  This is shown 
in Figure 4, below.

<center>
<img src="/assets/img/CLV_blog1_data_generaion_parameters_estimates.png" width="300" height="300" />


Figure 4: Parameter Estimates 3D View of Advertising Effectiveness
</center>


For future work, we will be looking to further improve the model, by tying together user 
purchases over time, potentially modeling satiation for each user, while also by 
partially pooling users together that share common characteristics.  

I hope you found this blog post useful.  Stay tuned for more blog posts in Marketing Science... 

Until next time!

### References

[^1]: [Inverse Transform Sampling — Wikipedia](https://en.wikipedia.org/wiki/Inverse_transform_sampling)

[^2]: E. Baltislam, et. al. "Empirical validation and comparison of models for customer base analysis", 
International Journal of Research in Marketing (2007) 201-209

### Source Code

The full source code, including data generation code, and synthetic data is available 
here on <a href="https://www.github.com/kgierach/Strata-IQ-Blog/tree/main/Bayesian_Hybrid_CLV">GitHub</a>

### Notes

<i>
A version of this work has been submitted for peer review at a scientific conference. This blog post is intended to present the ideas to a broader audience in an informal and educational format. All views and interpretations expressed here are personal and do not imply acceptance or endorsement by any academic venue.
</i>
