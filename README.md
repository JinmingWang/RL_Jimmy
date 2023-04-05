# RL_Jimmy
This is a repository for my RL projects. I will try to keep it updated as I go along. These are just some basic stuffs I am playing with. (not serious)<p>
这是我的强化学习项目的仓库。我会尽量保持更新。都是些基础的小玩意，边玩边学的。<p>
But I may give up at any time because I am lazy.<p>
但我懒得一，所以可能会随时放弃。<p>
For English speakers, the markdown comments in this project is not very frendly, but the book "Reinforcement Learning: An Introduction" by Richard S. Sutton and Andrew G. Barto is very good, the explanations in the book can be understood by a 5-year-old child English speaker. I recommend it to you.<p>

# 目录 Contents 
### Utility Functions and Support Classes
- utils
  - EnvironmentBasics.py
    - Action class
    - State class
    - Environment class
  - AgentBasics.py
    - Policy class
    - Agent class
  - GridWord.py
    - GridWorld(Environment) class

### Implementation of RL Algorithms
| File | Page in Book | Description | My Comment |
| --- | --- | --- | --- |
| 1_KArmedBandit.ipynb | 25 | k-armed bandit | A little bit messy. |
| 2_BellmanEquation.ipynb | 59 | Bellman equation | Terrible, but ok once you get it. |
| 3_PolicyIteration.ipynb | 74-80 | Policy iteration in Grid World | Good. |
| 4_ValueIteration.ipynb | 82 | Value iteration in Grid World | Nice. |
| 5_MonteCarloBaisc.ipynb | 92 | Monte Carlo in Grid World | Nice. |
| 6_OffPolicyMonteCarlo.ipynb | 110 - 111 | Off-policy Monte Carlo in Grid World | Good. |
| 7_TemporalDifferenceLearning.ipynb | 120 | Temporal difference learning in Grid World | Good. |
| 8_Sarsa.ipynb | 129 | Sarsa in Grid World | Nice. |
| 9_QLearning.ipynb | 131 | Q-learning in Grid World | Nice. |
| a_ExpectedSarsa.ipynb | 133 | Expected Sarsa in Grid World | Nice. |

# 记错本 My Mistakes
| File | Description |
| --- | --- |
| 5_MonteCarloBaisc.ipynb | 把非terminal state的grid也赋予了reward，导致虽然reward巅峰在terminal state，但agent可以通过在其他地方闲逛来疯狂赚reward，天下怎么能有这样的好事，我当场就把非terminal state的reward都改回了0。 |
| 8_Sarsa.ipynb | 忘记了设置terminal state的action-value，导致action-value无法得到来自terminal state的reward，进而导致policy瞎指挥。我甚至看到policy的小箭头在让agent往坑里跳。|


# Future Plans
- [x] TD(0)
- [x] SARSA and Expected SARSA
- [x] Q-Learning
- [ ] Deep RL
- [ ] n-Step Bootstrapping
- [ ] Imitation Learning
- [ ] Reinforcement Learning from Human Feedback (RLHF)