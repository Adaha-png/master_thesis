
# Reading list
## Acter
### Relevancy
* Counterfactual sequences
* smallest perturbation to avoid outcome

### Difference
* To avoid reward instead of penalty
* Multi agent

## More on shapley

* not intended for temporal


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

## week 42

* fixed issues from week 41


# TODO

## sim\_steps

* [x] make sim\_steps accept seed

* [ ] make sim\_steps accept initial observation (might need to make an env wrapper, will probably be a lot of work)

* [ ] work with continous action space

## feature importance for found actions

* [ ] feature permutation

* [ ] surragate shap and others

## major changes

* environment reset function overhaul

* supersuit reward fix, doesnt respect local reward

## sparse rewards

* new methods for action importance in sparse reward environments

## misc

* try different architectures
