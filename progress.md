
# Reading list

## Explainable Deep Reinforcement Learning: State of the Art and Challenges
Explanations for a lot of different earier methods, some i didnt know about, most are not directly relevant for me but section 4.4 might be relevant for background

## Memory-based explainable reinforcement learning
Saving state action transitions to  see how often they occured with success or failure, only makes sense for a reasonably small amount of possible states, ie not cont space as expl.

## Wenbo Guo, Xian Wu, Usmann Khan, and Xinyu Xing. 2021. Edge: Explaining deep reinforcement learning policies
input episodes with actions to rnn -> mlp -> deep gaussian process -> reward at time step prediction -> time step importance
timestep importance in losing games -> try random actions -> create lut of winning strategies

## Predicting Future Actions of Reinforcement Learning Agents
explicit planners, implicit planners and non planners, explicit perform the best but probably less interesting for me, implicit planners like deep repeated convLSTM are very interesting for me. non planners did not perform well, looking at difference between implicit and non planners also very interesting

## An Investigation of Model-Free Planning
Compares lstm with drc, cnn, vin and similar, cnn, vin and drc all work on sequential image input in planning environments, drc performs best, quite a bit better than regular lstms, but are obviously not very usable without sequential image inputs. Also performed better than large non recurrent networks like resnet. drc 3,3 very data efficient though not the best after 1e9 timesteps, which would be drc 9,1, all on sokoban, gridworld or boxworld.

## Value Iteration Networks
somehow planning but i think it just refers to using a value net instead of bellman equation

## Thinker: Learning to Plan and Act
Aug mdp from mdp, making a tree with world model and learning when to prune a branch.

## Neural Algorithmic Reasoners are Implicit Planners
Learning known algorithms instead of unknown ones

## STABILIZING TRANSFORMERS FOR REINFORCEMENT LEARNING
Introducing and comparing GTrXL to human, lstm TrXL and MERLIN@100B, it outperforms humans and lstm by a significant margin and easily beats TrXL, lower data efficiency than lstm but higher stability in general. the GTrXL model has 151M parameters.

## Decision Transformer: Reinforcement Learning via Sequence Modeling
Context instead of frame stacking, outperforms most other methods at the time at least, however paper is from 2019 so might be outdated, sota was CQL, attention could be used as intrinsic explainability and probably useful for prediction nets.

## Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context
paper claims longer memory than rnns (significantly longer than regular transformers), faster on evaluation, uses segment level recurrence and relative positional encoding, data efficiency and stability significantly improved by GTrXL

## Unsupervised Predictive Memory in a Goal-Directed Agent
Memory is not simply enough, correct memory in correct format: MERLIN, improves memory based on 


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


## sparse rewards

* new methods for critical states in sparse reward environments, probably big q value difference (logits, not softmax)

