import dspy


class Agent(dspy.Module):
    """
    DSPy agent used to optimize the prompt
    """
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.strategy = dspy.Predict(
            "valuation, role: list[str] -> strategy"
        )

    def forward(
        self,
        buyer_valuation: int,
        seller_valuation: int,
        buyer,
        seller,
        other_social_behavior: str,
        learner: str,
        **kwargs
    ):
        with (self.env.pool.session() as env):
            env.init(
                buyer_valuation=buyer_valuation,
                seller_valuation=seller_valuation,
                seller=seller,
                buyer=buyer,
                other_social_behavior=other_social_behavior,
                learner=learner
            )
            valuation = seller_valuation \
                if learner == "seller" else buyer_valuation
            strategy = self.strategy(
                    valuation=valuation,
                    role=learner
            )

            observation, reward = env.run(
                social_behavior=strategy.strategy
            )

        return dspy.Prediction(observations=observation,
                               reward=reward)