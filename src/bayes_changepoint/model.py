from bayes_changepoint.stats import discrete_distribution, NormalInverseGamma


class RunLengthModel:
    """Run length probability model."""

    def __init__(
        self,
        run_lengths: list[int],
        data_distributions: list[NormalInverseGamma],
        probabilities: list[float],
    ) -> None:

        assert len(run_lengths) == len(
            data_distributions
        ), "There must be data distribution for each run length!"
        self.run_length_distribution = discrete_distribution(run_lengths, probabilities)
        """Discrete probability distribution over run lengths."""
        self.data_distributions = data_distributions
        """Data distributions for each run length."""
