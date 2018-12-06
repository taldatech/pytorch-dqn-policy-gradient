# pytorch-dqn-policy-gradient
PyTorch implementation for Deep Q-Learning and Policy Gradient algorithms on several OpenAI's environemnts

Video:

YouTube - https://youtu.be/Bm1wXu8YBCY 


![taxi](https://github.com/taldatech/pytorch-dqn-policy-gradient/blob/master/imgs/taxi_agent_gif.gif)
![acrobot](https://github.com/taldatech/pytorch-dqn-policy-gradient/blob/master/imgs/AcrobotAgent_edit_1.gif)

- [pytorch-dqn-policy-gradient](#pytorch-dqn-policy-gradient)
  * [Prerequisites](#prerequisites)
  * [Files in the repository](#files-in-the-repository)
  * [Taxi-v2 Environment DQN](#taxi-v2-environment-dqn)
    + [API (`python taxi_main.py --help`)](#api---python-taxi-mainpy---help--)
    + [Playing](#playing)
    + [Training](#training)
  * [Taxi-v2 Environment Policy Gradient](#taxi-v2-environment-policy-gradient)
  * [Acrobot Environment DQN](#acrobot-environment-dqn)
    + [API (`python acrobot_main.py --help`)](#api---python-acrobot-mainpy---help--)
    + [Playing](#playing-1)
    + [Training](#training-1)

## Prerequisites
|Library         | Version |
|----------------------|----|
|`Python`|  `3.5.5 (Anaconda)`|
|`torch`|  `0.4.1`|
|`gym`|  `0.10.9`|
|`IPython`|  `6.4.0`|
|`numpy`|  `1.14.5`|
|`matplotlib`| `3.0.0`|



## Files in the repository

|File name         | Purpsoe |
|----------------------|------|
|`taxi_main.py`| Main application for training/playing a Taxi-v2 agent|
|`acrobot_main.py`| Main application for training/playing a Acrobot-v1 agent|
|`Agent.py`| Classes: TaxiAgent, AcrobotAgent|
|`DQN_model.py`| Classes: DQN_DNN, DQN_CNN (the neural networks architecture)|
|`Helpers.py`| Helper functions (converting states, plotting...)|
|`OneHotGenerator.py`| Class: OneHotGenerator (converts integers to One-Hot-Vectors)|
|`ReplayBuffer.py`| Class: ReplayBuffer (stores memories for the DQN learning)|
|`Schedule.py`| Classes: ExponentialSchedule, LinearSchedule (scheduling of epsilon-greedy policy)|
|`*.pth`| Checkpoint files for the Agents (playing/continual learning)|
|`*_training.status`| Pickle files with the recent training status for a model (episodes seen, total rewards...)|
|`Taxi_Agent.ipynb` | Jupyter Notebook with detailed explanation, derivations and graphs for the Taxi-v2 environemnt| 
|`Acrobot_Agent.ipynb` | Jupyter Notebook with detailed explanation, derivations and graphs for the Acrobot-v1 environemnt| 
|`dqn_pg_writeup_gh.pdf` | Summary of this work| 


## Taxi-v2 Environment DQN

### API (`python taxi_main.py --help`)


You should use the `taxi_main.py` file with the following arguments:

|Argument                 | Description                                 |
|-------------------------|---------------------------------------------|
|-h, --help       | shows arguments description             |
|-t, --train     | train or continue training an agent  |
|-p, --play    | play the environment using an a pretrained agent |
|-n, --name       | model name, for saving and loading, if not set, training will continue from a pretrained checkpoint |
|-m, --mode	| model's mode or state representation ('one-hot', 'location-one-hot'), default: 'one-hot' |
|-e, --episodes| number of episodes to play or train, default: 2 (play), 5000 (train) |
|-x, --exploration| epsilon-greedy scheduling ('exp', 'lin'), default: 'exp'|
|-d, --decay_rate| number of episodes for epsilon decaying, default: 800 |
|-u, --hidden_units| number of neurons in the hidden layer of the DQN, default: 150 |
|-o, --optimizer| optimizing algorithm ('RMSprop', 'Adam'), deafult: 'RMSProp' |
|-r, --learn_rate| learning rate for the optimizer, default: 0.0003 |
|-g, --gamma| gamma parameter for the Q-Learning, default: 0.99 |
|-s, --buffer_size| Replay Buffer size, default: 500000 |
|-b, --batch_size| number of samples in each batch, default: 128 |
|-i, --steps_to_start_learn| number of steps before the agents starts learning, default: 1000 |
|-c, --target_update_freq| number of steps between copying the weights to the target DQN, default: 5000 |
|-a, --clip_grads| use Gradient Clipping regularization (default: False) |
|-z, --batch_norm| use Batch Normalization between DQN's layers (default: False) |
|-y, --dropout| use Dropout regularization on the layers of the DQN (default: False) |
|-q, --dropout_rate| probability for a layer to be dropped when using Dropout, default: 0.4 |

### Playing
Agents checkpoints (files ending with `.pth`) are saved and loaded from the `taxi_agent_ckpt` directory.
Playing a pretrained agent for 2 episodes:

`python taxi_main.py --play`

For more more episodes, e.g 10:

`python taxi_main.py --play -e 10`

Playing a pretrained agent with Location-One-Hot state representation:

`python taxi_main.py --play -m location-one-hot -e 3`

For playing another chekpoint, the `-n` flag must correspond with a `.pth` checkpoint file in the `taxi_agent_ckpt` directory.

### Training

Note: in order to continue training from a pretrained checkpoint you can either:

	1. Name the model with the same name as the saved chekpoint (e.g. if the there exists `taxi_agent_user.pth` the model name should be `user`)
	
	2. Leave out the name (don't use the `-n` flag) and a default pretrained checkpoint will be loaded and a random name will be given (which you can change later)

Examples:

* `python taxi_main.py --train -n my_taxi -m one-hot -e 5000 -x exp -d 800 -u 150 -o RMSprop -r 0.00025 -g 0.99 -s 1000000 -b 128 -i 2000 -c 5000`
* `python taxi_main.py --train -a -m location-one-hot -e 5000 -x lin -d 800 -u 150 -o RMSprop -r 0.00025 -g 0.99 -s 1000000 -b 128 -i 2000 -c 5000`

For full description of the flags, see the full API.

## Taxi-v2 Environment Policy Gradient

if you want to check out Policy gradient methods preformance in the taxi env use PG_taxi_main
there you can use the following parameters to do what you want (train or play)
if no parameters where passed then the default behavior would be to load the best trained agent we have and evaluate it for 3 episodes
(pass -r to see rendering of the evaluation episodes on the screen)

 PG_taxi_main.py [-h] [-t] [-path [AGENT_PATH]] [-save_dir [SAVE_BASE_DIR]]
               [-eps_train [NUM_TRAIN_EPISODES]] [-eps_eval [NUM_EVAL_EPS]]
               [-HL [HL_SIZE]] [-r]

train and play a Policy Gradient Taxi-v2 agent

|Argument                 | Description                                 |
|-------------------------|---------------------------------------------|
|  -h, --help            	|show this help message and exit|
|  -t, --train           	|train a new agent (if not given then we'll load the <br> default agent and evaluate it <br> note: give -r to see screen rendering )|
| -path [AGENT_PATH],<br> --agent_path [AGENT_PATH]| path to a saved pre-trained PGAgent <br> (default agent is taxi_agent_PG\PG_taxi_agent_HL128_trained100000.pt)|
| -save_dir [SAVE_BASE_DIR],<br> --save_base_dir [SAVE_BASE_DIR] | where the agent and training stats will be saved <br> (default is in taxi_agent_PG) |
| -eps_train [NUM_TRAIN_EPISODES],<br> --num_train_episodes [NUM_TRAIN_EPISODES] | number of training episodes (default: 100000) |
| -eps_eval [NUM_EVAL_EPS], --num_eval_eps [NUM_EVAL_EPS]| number of evaluation episodes (default: 3) |
| -HL [HL_SIZE], --HL_size [HL_SIZE] | size of the hidden layer (default: 128) <br> note: if loading an agent make sure to <br> give the same HL_size as the saved agent |
| -r, --render          | if to render evaluation episodes on console screen default is false|

when training the the follwoing stats will be printed on the screen:
rewards (in training episodes), losses (in training episodes),
states (starting states in training episodes), episode_len( in training ...), eval_rewards

they will also be saved in a newly created sub directory SAVE_BASE_DIR\stats as pickle files to be reviewed for later
agent trained name will be saved in this format: PG_taxi_agent_HL[number of hidden nureons]_trained[Number of trained episodes].pt
same format for the stat pickles

we recomend to give a SAVE_BASE_DIR other than the default one as to not override the original agents and stats

examples of use:
`python PG_taxi_main.py`
evaluates the default agent for 3 episodes and prints their stats ( note: on console screen rendering is not turned on by default

`python taxi_main.py --render --num_eval_eps 10`
evaluated the default agent for 10 episodes and renders every episode on the console screen

Enjoy !

## Acrobot Environment DQN

### API (`python acrobot_main.py --help`)


You should use the `acrobot_main.py` file with the following arguments:

|Argument                 | Description                                 |
|-------------------------|---------------------------------------------|
|-h, --help       | shows arguments description             |
|-t, --train     | train or continue training an agent  |
|-p, --play    | play the environment using an a pretrained agent |
|-n, --name       | model name, for saving and loading, if not set, training will continue from a pretrained checkpoint |
|-m, --mode	| model's mode or state representation ('frame_diff', 'frame_seq'), default: 'frame_diff' |
|-e, --episodes| number of episodes to play or train, default: 5 (play), 5000 (train) |
|-x, --exploration| epsilon-greedy scheduling ('exp', 'lin'), default: 'exp'|
|-d, --decay_rate| number of episodes for epsilon decaying, default: 2000 |
|-u, --frame_history_len| number of frames to represent a state in 'frame_seq' mode, default: 4 |
|-o, --optimizer| optimizing algorithm ('RMSprop', 'Adam'), deafult: 'RMSProp' |
|-r, --learn_rate| learning rate for the optimizer, default: 0.0003 |
|-g, --gamma| gamma parameter for the Q-Learning, default: 0.99 |
|-s, --buffer_size| Replay Buffer size, default: 1000000 |
|-b, --batch_size| number of samples in each batch, default: 32 |
|-i, --steps_to_start_learn| number of steps before the agents starts learning, default: 1000 |
|-c, --target_update_freq| number of steps between copying the weights to the target DQN, default: 5000 |
|-a, --clip_grads| use Gradient Clipping regularization (default: False) |
|-z, --batch_norm| use Batch Normalization between DQN's layers (default: False) |
|-y, --dropout| use Dropout regularization on the layers of the DQN (default: False) |
|-q, --dropout_rate| probability for a layer to be dropped when using Dropout, default: 0.4 |

### Playing
Agents checkpoints (files ending with `.pth`) are saved and loaded from the `acrobot_agent_ckpt` directory.
Playing a pretrained agent for 5 episodes:

`python acrobot_main.py --play`

For more more episodes, e.g 10:

`python acrobot_main.py --play -e 10`

Playing a pretrained agent with Frame Sequence state representation:

`python acrobot_main.py --play -m frame_seq -e 3`

For playing another chekpoint, the `-n` flag must correspond with a `.pth` checkpoint file in the `acrobot_agent_ckpt` directory.

### Training

Note: in order to continue training from a pretrained checkpoint you can either:

	1. Name the model with the same name as the saved chekpoint (e.g. if the there exists `acrobot_agent_user.pth` the model name should be `user`)
	
	2. Leave out the name (don't use the `-n` flag) and a default pretrained checkpoint will be loaded and a random name will be given (which you can change later)

Examples:

* `python acrobot_main.py --train -a -n my_acrobot -m frame_diff -e 5000 -x exp -d 1000 -o RMSprop -r 0.00025 -g 0.99 -s 1000000 -b 32 -i 2000 -c 5000`
* `python acrobot_main.py --train -a -z -e 5000 -x exp -d 1000 -o RMSprop -r 0.00025 -g 0.99 -s 1000000 -b 32 -i 2000 -c 5000` (Continual Learning)
* `python acrobot_main.py --train -a -z -n my_acrobot_frame_seq -m frame_seq -u 4 -e 5000 -x lin -d 2000 -o RMSprop -r 0.00025 -g 0.99 -s 1000000 -b 32 -i 2000 -c 5000`

For full description of the flags, see the full API.
