
# Reading list


## Explainable Deep Reinforcement Learning: State of the Art and Challenges
## Memory-based explainable reinforcement learning
## Wenbo Guo, Xian Wu, Usmann Khan, and Xinyu Xing. 2021. Edge: Explaining deep reinforcement learning policies
input episodes with actions to rnn -> mlp -> deep gaussian process -> reward at time step prediction -> time step importance
timestep importance in losing games -> try random actions -> create lut of winning strategies
## Predicting Future Actions of Reinforcement LearningAgents
explicit planners, implicit planners and non planners, explicit perform the best but probably less interesting for me, implicit planners like deep repeated convLSTM are very interesting for me. non planners did not perform well, looking at difference between implicit and non planners also very interesting
## Thinker: Learning to Plan and Act
## Neural Algorithmic Reasoners are Implicit Planners
## STABILIZING TRANSFORMERS FOR REINFORCEMENT LEARNING
## Decision Transformer: Reinforcement Learning via Sequence Modeling


comparing LSTMS with transformers seems interesting


## Acter
### Relevancy
* Counterfactual sequences
* smallest perturbation to avoid outcome

### Difference
* To avoid reward instead of penalty
* Multi agent

## More on shapley


# Worklog
## Week 39
### New env, simple spread
Pros:

* Easy to train

* Clear cooperation, good explanations shouldnt be hard to extract

* Set local/global reward

Cons:

* Homogeneous agents

* Explanations likely arent too useful because cooperation is too obvious

### Command line interface
Argaparse to set environment, timesteps and tuning paramenters from terminal


## Week 40
### Fixed eval function

* one agent acted randomly

* evaluated as aec instead of parallel

### Counterfactual sequences

Using sequences for action importance gives a temporal aspect

* objective functions made (rewards and action)

* Problem and Evolution objects made, evolution runs


### utils

* added a utils file, currently contains a function to add a seed to a pettingzoo parallelenv

## week 41
### Counterfactual sequences w model

With this new model we only change one timestep directly and then let the policy dictate the environment after. We can use this to find consequences for actions. something something intent

* new method to look for biggest reward change with only changing actions in a single timestep

* pareto optimal points for amount of actions changed

* implemented for discrete, but not functional yet

[Actions changed for a single timestep](./tex/images/best_counterfactuals_with_model.pdf)
## week 42

* fixed issues from week 41

* started implemented random forest surrogate model shapley

* implemented kernel shapley, needs more work


## week 43

* explainer\_step file for shapley api

* now possible to use kernelexplainer to make explanations for important actions in critical states.

[Shapley values for moving up in a critical state in which the agent decided to move up](./tex/images/shap_plot_kernel_move_up.pdf)


## week 44

no progress

## week 45

Integrated gradients plot

* logits dont make sense to explain
* one hot encoding isnt differentiable
* ergo softmax, additional benefit is a confidence measure of action chosen

## week 46

Future predictor

* takes first observation in a sequence
* predicts position of an agent n steps in the future
* takes extra inputs, action as an int helps, action as one-hot helps more
* gets decent results
* shap and ig on these models


## week 47

Future predictor

* takes ig, helps a bit on top of one hot
* should take shap as well
* compare different loss curves and distance measures

# TODO


## sim\_steps

* make sim\_steps accept initial observation (might need to make an env wrapper, will probably be a lot of work)


## sparse rewards

* new methods for critical states in sparse reward environments, probably big q value difference (logits, not softmax)

