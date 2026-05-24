Overview

This project implements Tabular Q-Learning on the CliffWalking-v1 environment using three exploration strategies:

High Exploration (ϵ=0.5)
Low Exploration (ϵ=0.05)
Decaying Exploration (ϵ decreases over time)

The goal is to compare how exploration affects learning performance.

Requirements

Install dependencies:

`pip install gymnasium numpy matplotlib`
Run the Project
`python main.py`
Environment

Environment used:

CliffWalking-v1

The agent:

starts from the beginning state,
must reach the goal,
and avoid falling into the cliff.
Q-Learning Update Rule
Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]

Where:

α = learning rate
γ = discount factor
Exploration Strategies
1. High Exploration
epsilon = 0.5
2. Low Exploration
epsilon = 0.05
3. Decaying Exploration
epsilon = max(0.01, 1.0 * (0.995 ** episode))
Results
High exploration learns slowly due to excessive randomness.
Low exploration learns faster but may miss the best path.
Decaying exploration gives the best overall performance.
Conclusion

The decaying epsilon strategy performs best because it balances:

exploration in the beginning,
and exploitation later.
