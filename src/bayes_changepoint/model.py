from typing import Final
from bayes_changepoint.stats import discrete_distribution, NormalInverseGamma


class RunLengthModel:
    """Run length probability model."""

    def __init__(
        self,
        run_lengths: list[int],
        data_distributions: list[NormalInverseGamma],
        probabilities: list[float],
        prior_data_distribution: NormalInverseGamma,
    ) -> None:

        assert len(run_lengths) == len(
            data_distributions
        ), "There must be data distribution for each run length!"
        self.run_length_distribution = discrete_distribution(run_lengths, probabilities)
        """Discrete probability distribution over run lengths."""
        self.data_distributions: list[NormalInverseGamma] = data_distributions
        """Data distributions for each run length."""
        self.prior_data_distribution: NormalInverseGamma = prior_data_distribution
        """Prior data distribution for run length zero."""

    def likelihood(self, sample: float) -> list[float]:
        """Calculates the likelihood of a new sample for each run length in the model.

        Args:
            sample: new sample

        Returns:
            list of likelihood for each run length
        """

        return [
            distribution.likelihood(sample) for distribution in self.data_distributions
        ]

    def grow(self, sample: float):
        """Based on the new sample, recalculate the run length probabilities.

        Args:
            sample: _description_
        """

        LAMBDA: Final = 10
        CHANGEPOINT_HAZARD: Final = 1 / LAMBDA

        run_lengths = self.run_length_distribution.domain
        products = [
            run_length_prob * likelihood
            for run_length_prob, likelihood in zip(
                self.run_length_distribution.pmf, self.likelihood(sample)
            )
        ]

        changepoint_prob = sum(
            [product * CHANGEPOINT_HAZARD for product in products]
        )  # P(r_t=0, x_1:t)
        extended_run_lengt_probs = [
            product * (1 - CHANGEPOINT_HAZARD) for product in products
        ]  # [P(r_t=r_t-1 + 1, x_1:t)]
        run_lengths = [run_length + 1 for run_length in run_lengths]

        # prior distribution is of the run length equal to zero
