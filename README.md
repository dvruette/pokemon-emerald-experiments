# Train RL agents to play Pokemon Emerald

## Training the Model 🏋️

1. Create a and activate virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r baselines/requirements.txt
```

3. Run training script:
```bash
python baselines/run_baseline_parallel_fast.py
```

## Tracking Training Progress 📈
The current state of each game is rendered to images in the session directory.
You can track the progress in tensorboard by moving into the session directory and running:
```bash
tensorboard --logdir .
```
You can then navigate to `localhost:6006` in your browser to view metrics.
To enable wandb integration, change `use_wandb_logging` in the training script to `True`.


## PPO ELI15 🤔

### Preliminaries

First off, a list of important terms and concepts necessary for understanding the PPO algorithm. Feel free to skim and come back to as needed when the terms come up later in the text.
- **Agent**: In this context, the agent is a learned algorithm that is trained to play the game (actually, to maximize reward, but hopefully that entails playing the game..). It consists of a policy network that predicts which action to take next and of a value network that predicts how much future reward can be gained starting from any given state in the game. Note that the agent does not explicitly try to predict what's going to happen in the game by taking any given action (this is why PPO is referred to as a "model-free" algorithm: the agent doesn't have to try model the world around it), which is different from other AIs such as AlphaGo or AlphaZero.
- **Policy**: The policy is a function (here: a neural net) that takes in a state (a.k.a. observation) and tries to predict the best action for the next step in order to maximize future reward. More specifically, for any state and action, it gives the _probability_ that this action is the best one to take. The policy function is learned, its parameters are updated by the policy loss.
- **Value function**: The value function is a function (here: also a neural net) that takes in a state as input and tries to predict how "good" being in that state is. It tries to answer: "Given that I'm in this state, how much future reward can I possibly get?"
    The parameters of the value function are updated by (you guessed it) the value loss.
- **Policy loss**: The policy loss, or policy objective, measures how much more reward the current policy could gain by taking different actions. The gradient (derivative) of this loss is called the "policy gradient". This is the most complicated part of the algorithm and will be covered in more detail later.
- **Value loss**: The value loss measures the error between the predicted value of any given state (with our learned value function) and the true observed value of that state while playing. This means that if the value loss is close to 0, our learned function predicts with almost perfect accuracy how much future reward the agent will obtain starting from any given state. It is usually undesirable to have a very low value loss, because that means that there is very little left to learn about the environment!
- **Advantage**: The advantage function quantifies how good taking one action is compared to all the other options. It tries to answer the question: "If I take this action, how much better/worse off will I be in the future compared to doing something else?"
    The advantage is computed using the current value function, hence it is only an approximation of the true advantage based on the agent's current knowledge about the world. It is also the quantity that the policy is trained to maximize (i.e. the policy gradient is calculated using this advantage function).


### The Algorithm

So with all of this in mind, how does PPO actually work?
A single PPO iteration consists of the following steps: 
1. Let the agent act in the environment to gather observations and rewards (this is referred to as "trajectories")
2. Select a batch of trajectories
3. Compute the estimated advantage using the current value function
4. Update the policy by using the PPO policy gradient (a.k.a. PPO-clip objective)
5. Update the value function by minimizing its prediction error on the recently observed rewards
6. Repeat 2-5 until all gathered trajectories have been processed

For the sake of efficiency, we might want to reuse the gathered trajectories more than once. This is controlled by the number of epochs and is also referred to as "sample reuse".

Getting to the core of the algorithm: It's time to understand the PPO-clip objective.
Given a state `s` and an action `a`, and assuming that we know the advantage function of that state-action pair, in its most basic form the PPO objective calculates how good a hypothetical new policy `p_new` would be compared to the current policy `p_old`:
```python
def ppo_objective(p_old, p_new, s, a, advantage):
    return p_new(s, a) / p_old(s, a) * advantage(s, a)
```
By computing the gradient with respect to (the parameters of) the new policy, we can do gradient ascent in order to maximize the objective and take a step towards a more optimal policy.

In this most basic form, the update that we get can sometimes be unreasonably large (e.g. if the value function thinks that a certain strategy is very good when in reality it's not), leading the policy to a bad spot where it's hard to recover from. It is crucial that the value function and policy gradually improve in tandem. Since they influence eachother, having one of the two in a bad state will also have ripple effects on the other and can ultimately result in a situation where the agent is stuck and can't improve anymore.

While there's a range of approaches and tricks to prevent this kind of situation from happening, PPO employs a very simple clipping approach that ensures that the new policy exploits at most a certain fraction of the advantage function. So even if the value function thinks that "you can double your reward by using this one simple trick", the policy will remain sceptical and only update in that direction a little bit at a time. In code, this looks as follows:
```python
def ppo_clip_objective(p_old, p_new, s, a, advantage, clip_range=0.2):
    A = advantage(s, a)
    if A >= 0:
        return min(p_new(s, a) / p_old(s, a), 1 + clip_range) * A
    else:
        return max(p_new(s, a) / p_old(s, a), 1 - clip_range) * A
```


### Training an Agent

Now that we understand how the algorithm works, how can we determine if a run is going well and how can we adjust the hyperparameters if it isn't? As alluded to before, the thing we're most afraid of is so-called "collapse", where the agent gets stuck doing a certain action and thinking that this is the best it can do. This is unless, of course, it has actually found the best course of action, in which case training has converged to an optimal solution. But if it happens early on in training, it's very likely that adjustments to the hyperparameters have to be made.

#### How can we spot policy collapse?

The most obvious way to spot collapse is to just watch your agent play the game: Is it stuck doing the same thing over and over again, making no progress and failing to explore different strategies? Then you're probably looking at an over-confident agent stuck in a local optimum.
But beyond manual inspection, there's two metrics in particular to watch out for: The KL-divergence between policy updates (`train/approx_kl`) and the policy entropy (`train/entropy_loss`). While the names are fancy, the intuition behind them is relatively simple.

The KL-divergence quantifies how different the new policy (after one iteration of PPO) is from the old one: If the new one tends to take vastly different actions from the old one, the KL-divergence will be high. If it takes the exact same actions, the KL-divergence will be 0. In a healthy training run, the KL-divergence will be not too high (we're probably updating too much) but also not too low (we're pretty much not updating).

The policy entropy quantifies how sure/unsure the agent is in predicting the best action. A high entropy (-> low entropy loss) means it's very uncertain and a low entropy means it's very certain. Since the policy is initially completely random, at the beginning of training entropy will be high (i.e. entropy loss will be low) and gradually decrease (increase) as the agent learns that some actions are better than others. If entropy decreases too quickly, it likely means that the agent is growing over-confident and is prioritizing exploitation over exploration. If etropy doesn't decrease, it means that the agent isn't becoming more sure of its actions and still exploring more or less aimlessly. Ideally, entropy decreases gradually at a similar rate as reward increases.

#### How can we prevent policy collapse?

There's three parameters that have a pretty direct effect on the metrics discussed above:
- **PPO clip range** (`clip_range`): As outlined above, the clip range affects how far the policy is willing to update its actions in a single iteration. Lowering this value will make it update more cautiously and tend to stick to what it's been doing for longer. If you think of the value function as an advisor, telling the policy "if you're in this state you should really do that", the clip range controls how many iterations it will take for the policy to start doing it. In certain situations, the value function might be like "actually nvm, I'm dumb" a few iterations later, in which case we'll be glad not to have updated all the way.
- **Target KL-divergence** `target_kl`: The target KL-divergence has the same goal as the clip range, namely to prevent the new policy from updating too far from the old one. It achieves this in a more brute-force way by early-stopping the current PPO iteration if this threshold is exceeded. This way it provides a hard upper-bound on how much the policy is allowed to update in any given iteration, whereas the clip range is more of a soft upper-bound.
- **Entropy loss** (`ent_coef`): Another way to prevent collapse is to punish the agent for being too certain in its actions, which can be achieved by adding the entropy as an auxiliary loss function, forcing the agent to trade-off maximizing reward with being too confident in its actions. It's an effective way to encourage exploration, but setting it too high will lead to an agent that is very insecure in its actions. The `ent_coef` parameter controls the fraction of the entropy that is added to the loss.


#### Before we start training

But before we even start training, we have to decide upon a few parameters that affect how the agent interacts with the environment:
- **Environment steps (a.k.a. rollout length)**: This determines the length of trajectories and the frequency at which the agent is updated. Longer trajectories give a higher-quality signal but take longer to simulate, so it's somewhat of a tradeoff.
- **Episode length**: This is the number of steps it takes until the environment is reset (this could be due to Game Over or just a hard-coded step limit). Note that this doesn't have to be equal to the rollout length, it can be larger or smaller (it can also be adaptive or change over time).
- **Gamma (a.k.a. reward horizon)**: This determines the amount of "long-term thinking" in your agent, as future rewards gained after taking a certain action are discounted by this factor for each step that has passed since taking it. For example if gamma is set to 0.99 and the agent receives a reward 20 steps after taking action `a`, the loss function will give `0.99 ** 10 ≈ 0.82` credit for taking that action. Therefore, a value closer to 1 will encourage more long-term planning, whereas a value less close to 1 will incentivise short-term reward maximization.
- **Learning rate and batch size**: These values determine how quickle we descend/ascend the gradient and how many samples we use to estimate the gradient. These values depend a lot on the architecture and size of the model, with larger batches and smaller learning rates generally being required for larger models.

#### Other things to look out for (in decreasing importance)

- **Reward**: Presumably, maximizing reward is what you care about. At least, it's what the agent cares about. In fact, it doesn't care about anything else. Keeping an eye on it and making sure that it 1) goes up and 2) rewards the things that you actually intend to (if there's a way to exploit your reward function, the agent will probably find it) is of course crucial.
- **Value loss** (`train/value loss`): Having an accurate value function is crucial for stable training and learning a good policy, as it's the basis for the advantage estimation underlying all of the policy optimization. However, this doesn't mean that we just want the value loss to be always as close to 0 as possible. Since the value function is trained using the observations made by the agent, it only has limited knowledge of the world and will be a bad approximation of the true value function in the beginning. Ideally, we always want the value loss to be more or less stable around a value greater than 0. If that's the case, it means that we steadily gather new experiences that the value function can't fully explain yet, but also that it's learning to explain those experiences and isn't overwhelmed to the point where it can't make any sense of the signal it's getting.
- **Explained variance** (`train/explained_variance`): This is another signal that helps judge the quality of the value function. If the value is close to one, it means that value function almost perfectly explains all the rewards the agent is getting. A value below 0 on the other hand means that it's doing worse than just predicting nothing (making random predictions). Again, we want this value to be close to 1 but also not stuck at 1, especially in early training, since that would imply that there's no unexplained experiences coming in.
- **Policy gradient loss** (`train/policy_gradient_loss`): This loss quantifies how well the agent is able to adapt to the current value function. A small loss means it's pretty much doing what the value function thinks is best and a high loss means that there's still lots (known!) of ways for the agent to improve. A higher value is generally desirable, but if it's too high it means that the agent is unable to adapt to the value function.
- **Total loss** (`train/loss`): This is probably the least important metric to watch, as it's just the sum of all the auxiliary losses covered above.


### Further Reading
- [Proximal Policy Optimization](https://spinningup.openai.com/en/latest/algorithms/ppo.html#documentation-pytorch-version) by OpenAI, 2018