from itertools import accumulate
from random import uniform
import scipy as sp  # type: ignore
from math import sqrt


class NormalInverseGamma:
    """Normal-inverse-gamma distribution.
    Utilizes the scipy implementation enhanced with marginal likelohood and bayesian parameter
    estimation."""

    def __init__(
        self,
        mean: float,
        variance: float,
        n_degrees_of_freedom: float,
        n_pseudoobservations: float,
    ) -> None:
        self.distribution = sp.stats.normal_inverse_gamma(
            a=n_degrees_of_freedom / 2,
            b=n_degrees_of_freedom * variance / 2,
            mu=mean,
            lmbda=n_pseudoobservations,
        )
        self.n_dof = n_degrees_of_freedom
        self.n_pseudoobs = n_pseudoobservations
        self.mean = mean
        self.variance = variance
        self.x_marginal_distribution = sp.stats.t(
            df=self.n_dof,
            loc=self.mean,
            scale=sqrt(self.variance),
        )

    def likelihood(self, sample: float) -> float:
        """Calculate sample marginal likelihood.

        Args:
            sample: random sample

        Returns:
            sample marginal likelihood given the distribution parameters (the variance parameter is
            marginalized)
        """

        return self.x_marginal_distribution.pdf(sample)


class discrete_distribution:
    """Simplistic discrete probability distribution."""

    def __init__(self, domain: list[int], probabilities: list[float]) -> None:
        assert len(domain) == len(
            probabilities
        ), "Probability must be defined for each domain element"
        prob_sum = sum(probabilities)
        self.pmf = [probability / prob_sum for probability in probabilities]
        self.cdf = [cdf for cdf in accumulate(self.pmf)]
        self.domain = domain

    def sample(self) -> int:
        variate = uniform(a=0.0, b=1.0)

        for item, cdf in zip(self.domain, self.cdf):
            if cdf >= variate:
                return item

        raise ValueError("Cannot sample the distribution!")
