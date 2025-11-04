from itertools import accumulate
from random import uniform
import scipy as sp  # type: ignore
import numpy as np
import numpy.typing as npt
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
        self.alpha = n_degrees_of_freedom / 2
        self.beta = n_degrees_of_freedom * variance / 2
        self.distribution = sp.stats.normal_inverse_gamma(
            a=self.alpha,
            b=self.beta,
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

    def rvs(
        self, size: tuple[int, ...]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:

        return self.distribution.rvs(size=size)

    def update(self, samples: list[float]) -> None:
        """Update he hyperparameters based on the observed samples.

        Args:
            samples: observed samples
        """

        n_sample = len(samples)
        sample_mean = sum(samples) / n_sample
        sample_squared_residual_sum = sum(
            ((sample - sample_mean) ** 2 for sample in samples)
        )
        updated_mean = (self.n_pseudoobs * self.mean + n_sample * sample_mean) / (
            self.n_pseudoobs + n_sample
        )
        updated_n_pseudoobs = self.n_pseudoobs + n_sample
        updated_n_dof = self.n_dof + n_sample
        updated_variance = (
            self.n_dof * self.variance
            + sample_squared_residual_sum
            + self.n_pseudoobs
            * n_sample
            / (self.n_pseudoobs + n_sample)
            * (self.mean - sample_mean) ** 2
        )
        updated_variance /= updated_n_dof
        self.mean = updated_mean
        self.n_dof = updated_n_dof
        self.n_pseudoobs = updated_n_pseudoobs
        self.variance = updated_variance
        self.alpha = self.n_dof / 2
        self.beta = self.n_dof * self.variance / 2
        self.x_marginal_distribution = sp.stats.t(
            df=self.n_dof,
            loc=self.mean,
            scale=sqrt(self.variance),
        )

    def get_mean(self) -> tuple[float, float]:

        return [self.mean, self.beta / (self.alpha - 1)]

    def get_covariance(self) -> npt.NDArray[np.float64]:

        var_mu = self.beta / (self.alpha - 1) / self.n_pseudoobs
        var_sigma_squared = self.beta**2 / (self.alpha - 1) ** 2 / (self.alpha - 2)
        cov_mu_sigma_squared = 0

        return np.array(
            [[var_mu, cov_mu_sigma_squared], [cov_mu_sigma_squared, var_sigma_squared]]
        )


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
