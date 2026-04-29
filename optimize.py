import os
import dspy
import random
from typing import List
from envs import BuySellEnv
from core.player import Player
from optimization import Agent

"""
This script performs the prompt optimization using GEPA
"""

def make_examples(
        num_examples: int,
        *,
        seed: int | None = None) -> List[dspy.Example]:

    """
    Generates random environment setting to create the games used during the
    optimization process.
    """

    if seed is not None:
        random.seed(seed)

    examples: List[dspy.Example] = []

    gap_bins = [(0.05, 0.15), (0.15, 0.30), (0.30, 0.45), (0.45, 0.60)]

    for i in range(num_examples):
        learner = random.choice(["seller", "buyer"])
        other_social_behavior = ""

        if learner == "seller":
            seller = Player(model="gpt-4.1-mini-2025-04-12-cde-aia")
            buyer  = Player(model="gemini-2.5-pro")
        else:
            seller = Player(model="gemini-2.5-pro")
            buyer  = Player(model="gpt-4.1-mini-2025-04-12-cde-aia")

        middle = random.uniform(a=50, b=100)
        g_lo, g_hi = gap_bins[i % len(gap_bins)]
        gap_pct = random.uniform(g_lo, g_hi)
        gap = gap_pct * middle

        buyer_val  = round(middle + gap / 2)
        seller_val = round(middle - gap / 2)

        if buyer_val <= seller_val:
            buyer_val = seller_val + 1
        seller_val = max(0, seller_val)

        ex = dspy.Example(
            buyer_valuation=buyer_val,
            seller_valuation=seller_val,
            buyer=buyer,
            seller=seller,
            other_social_behavior=other_social_behavior,
            learner=learner,
            middle=middle,
            gap=gap,
            gap_pct=gap_pct,
            example_id=i,
        ).with_inputs(
            "buyer_valuation", "seller_valuation",
            "buyer", "seller",
            "other_social_behavior", "learner"
        )
        examples.append(ex)

    return examples


def metric_with_feedback(example,
                         pred,
                         trace=None,
                         pred_name=None,
                         pred_trace=None):
    score = pred.reward
    interaction = "\n".join(pred.observations)
    feedback = f"Your strategy led to this interaction\n{interaction}\n"
    if score < 0.5:
        feedback += "The other player won more than you. You were abused."
    else:
        feedback += "You won more than the other player. Good job."

    feedback += f"\n Your Score was {score}"
    if score < 0:
        feedback += " You LOST Money! Very bad."

    return dspy.Prediction(score=score, feedback=feedback)

train_set = make_examples(num_examples=20)
val_set = make_examples(num_examples=10)

gpt41_mini = dspy.LM(
    model='azure/gpt-4.1-mini-2025-04-12-cde-aia',
    api_key=os.getenv("OPENAI_API_KEY_2"),
    api_base=os.getenv("OPENAI_API_BASE_2"),
    api_version=os.getenv("OPENAI_API_VERSION_2")
)

reflection_lm = dspy.LM(
    model='azure/gpt-4.1-mini-2025-04-12-cde-aia',
    api_key=os.getenv("OPENAI_API_KEY_2"),
    api_base=os.getenv("OPENAI_API_BASE_2"),
    api_version=os.getenv("OPENAI_API_VERSION_2"),
    temperature=1.0,
    max_tokens=32000,
)

buy_sell_env = BuySellEnv(logs_dir=".logs/buy-sell-optimization-4")
agent = Agent(env=buy_sell_env)
agent.set_lm(gpt41_mini)

evaluate = dspy.Evaluate(
    devset=val_set,
    metric=metric_with_feedback,
    display_progress=True,
    num_threads=8
)

agent.verbose = False
base_score = evaluate(agent)
print(f"\nBase agent dev success: {float(base_score):.3f}")

optimizer = dspy.GEPA(
    metric=metric_with_feedback,
    auto="light",
    num_threads=8,
    reflection_lm=reflection_lm,
    log_dir=".logs/gepa-buysell-4",
    track_stats=True,
    track_best_outputs=True
)
config = {}
print("\n=== Compiling / optimizing agent ===")
optimized_agent = optimizer.compile(agent,
                                    trainset=train_set,
                                    valset=val_set,
                                    **config)

opt_score = evaluate(optimized_agent)
print(f"\nOptimized agent dev success: {float(opt_score):.3f}")
optimized_agent.save('optimized_005.pkl')