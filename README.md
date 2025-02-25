# Legged-Lab
Implementation of legged(currently quadruped) locomotion environment for reinforcement learning.


# Running Video
https://github.com/user-attachments/assets/ce728b39-4f59-4e77-b56a-6e58f545a49d


# Getting Started

1. Prepare all the dependencies.

2. clone the repository
- `git clone https://github.com/asdfGuest/Legged-Lab`
- `cd Legged-Lab`

3. train the model
- `python .\example\run.py Go2 go2 train --headless`

4. play the model
- `python .\example\run.py Go2-Play go2 play`

5. for monitering
- `tensorboard --logdir=.\example\runs`


# Dependencies & Tested On

- Window 11
- Python 3.10.16
- PyTorch 2.5.1+cu118
- Isaac Sim 4.5.0
- Isaac Lab 2.0.0

and our RL library for run examples
- [Simple-RL](https://github.com/asdfGuest/Simple-RL)


# Notes
In our tests, the environment A1, Go2, AnymalC took approximately 2.3, 7.4, 4.9 hours.
