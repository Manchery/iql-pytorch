# IQL Implementation in PyTorch

## IQL

This repo is an unofficial implementation of **Implicit Q-Learning (In-sample Q-Learning)** in PyTorch.

```
@inproceedings{
    kostrikov2022offline,
    title={Offline Reinforcement Learning with Implicit Q-Learning},
    author={Ilya Kostrikov and Ashvin Nair and Sergey Levine},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=68n2s9ZJWF8}
}
```

**Note**: Reward standardization (_We standardize MuJoCo locomotion task rewards by dividing by the difference of returns of the best and worst trajectories in each dataset_) used in [official implementation](https://github.com/ikostrikov/implicit_q_learning/blob/09d700248117881a75cb21f0adb95c6c8a694cb2/train_offline.py#L51C18-L51C18) is missed in this implementation. One can easily add it by itself.

## Train

### Gym-MuJoCo

```
python main_iql.py --env halfcheetah-medium-v2 --expectile 0.7 --temperature 3.0 --eval_freq 5000 --eval_episodes 10 --normalize
```

### AntMaze

```
python main_iql.py --env antmaze-medium-play-v2 --expectile 0.9 --temperature 10.0 --eval_freq 50000 --eval_episodes 100
```

## Results

![mujoco_results](imgs/mujoco_results.png)

![antmaze_results](imgs/antmaze_results.png)

## Acknowledgement

This repo borrows heavily from [sfujim/TD3_BC](https://github.com/sfujim/TD3_BC) and [ikostrikov/implicit_q_learning](https://github.com/ikostrikov/implicit_q_learning).
