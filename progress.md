
# Reading list
## Acter
### Relevancy
* Counterfactual sequences
* smallest perturbation to avoid outcome

### Difference
* To avoid reward instead of penalty
* Multi agent

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

* objective functions made (rewards and action)

* Problem and Evolution objects made, evolution runs

### utils

* added a utils file, currently contains a function to add a seed to a pettingzoo parallelenv

# TODO

* [x]make sim\_steps accept seed, slight changes to ss need to be made.

* [ ]make sim\_steps accept initial observation (might need to make an env wrapper, will probably be a lot of work)
