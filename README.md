# Mario Gate

---
<div id="installation"></div>Installation
============

You need to install [gym-pull](https://github.com/ppaquette/gym-pull)

```shell
pip install gym-pull
```

 To load and run the environments, run

```python
import gym
import gym_pull
gym_pull.pull('github.com/ppaquette/gym-super-mario')        # Only required once, envs will be loaded with import gym_pull afterwards
env = gym.make('ppaquette/SuperMarioBros-1-1-v0')
```
