# Incentive Q-Flow (IQ-Flow)

This is the code for experiments in the paper IQ-Flow: Mechanism Design for Inducing Cooperative Behavior to Self-Interested Agents in Sequential Social Dilemmas. This implementation benefits from [LIO](https://github.com/011235813/lio/tree/181fecf9e83db03bc9097f5a0b2c8c8d3905cd77) and [JAXRL](https://github.com/ikostrikov/jaxrl).


## Setup
- Python 3.8
- Tensorflow 2.2.1
- Flax 0.3.4
- Jax 0.2.17
- Jaxlib 0.3.0
- Gym 0.26.2
- Follow the setup instructions for the official repository of the paper Learning to Incentivize Other Learning Agents at [LIO](https://github.com/011235813/lio/tree/181fecf9e83db03bc9097f5a0b2c8c8d3905cd77).
- Run `$ pip install -e .` from the root.
- In order to do hyperparameter tuning, install [Optuna](https://optuna.readthedocs.io/en/stable/installation.html).


## Navigation

* `alg/` - Implementation of IQ-Flow and baselines.
* `config/` - Configuration files for experiments.
* `env/` - Implementation of the Escape Room game and wrappers around the SSD environment from [LIO](https://github.com/011235813/lio/tree/181fecf9e83db03bc9097f5a0b2c8c8d3905cd77) and Iterated Matrix Games.
* `eval/` - Evaluation scripts
* `networks/` - Neural network implementations
* `trainer/` - Training scripts
* `utils/` - Utilities


## Examples

### Train IQ-Flow on Iterated Matrix Games

* Set config values in `config/config_img_qflow.py`
* `cd` into the `trainer` folder
* Execute training script `$ python train_multiprocess.py --alg qflow --exp ipd --n_seeds 5 --config config_img_qflow` to run Iterated Prisoner's Dilemma with 5 seeds. You can see the argument options for further configuration details.
* Execute training script `$ python train_multiprocess.py --alg qflow --exp chicken --n_seeds 5 --config config_img_qflow` to run Chicken Game with 5 seeds.
* Execute training script `$ python train_multiprocess.py --alg qflow --exp stag_hunt --n_seeds 5 --config config_img_qflow` to run Stag Hunt with 5 seeds.

### Train IQ-Flow on Escape Room

* Set config values in  `config/config_er_qflow.py`
* `cd` into the `trainer` folder
* Execute training script `$ python train_multiprocess.py --alg qflow --exp er --n_seeds 5 --config config_er_qflow` to run Escape with 5 seeds. You can see the argument options for further configuration details.

### Train IQ-Flow on Cleanup

* Set config values in  `config/config_ssd_qflow.py`
* `cd` into the `trainer` folder
* Execute training script `$ python train_multiprocess.py --alg qflow --exp ssd --n_seeds 5 --config config_ssd_qflow` to run Cleanup with 5 seeds. You can see the argument options for further configuration details.

### Train Incentive Designer(ID) on Escape Room

* Set config values in  `config/config_er_id.py`
* `cd` into the `trainer` folder
* Execute training script `$ python train_multiprocess.py --alg id --exp er --n_seeds 5 --config config_er_id` to run Escape with 5 seeds. You can see the argument options for further configuration details.

### Train Incentive Designer(ID) on Cleanup

* Set config values in  `config/config_ssd_id.py`
* `cd` into the `trainer` folder
* Execute training script `$ python train_multiprocess.py --alg id --exp ssd --n_seeds 5 --config config_ssd_id` to run Cleanup with 5 seeds. You can see the argument options for further configuration details.
