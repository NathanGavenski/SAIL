# SAIL
Official Implementation for [Self-Supervised Adversarial Imitation Learning (SAIL)](https://arxiv.org/pdf/2304.10914.pdf)

---

Behavioural cloning is an imitation learning technique that teaches an agent how to behave via expert demonstrations. Recent approaches use self-supervision of fully-observable
unlabelled snapshots of the states to decode state pairs into
actions. However, the iterative learning scheme employed by these
techniques is prone to get trapped into bad local minima. Previous
work uses goal-aware strategies to solve this issue. However, this
requires manual intervention to verify whether an agent has
reached its goal. We address this limitation by incorporating
a discriminator into the original framework, offering two key
advantages and directly solving a learning problem previous work
had. First, it disposes of the manual intervention requirement.
Second, it helps in learning by guiding function approximation
based on the state transition of the expert’s trajectories. Third,
the discriminator solves a learning issue commonly present in
the policy model, which is to sometimes perform a ‘no action’
within the environment until the agent finally halts.

---
## Dependencies

```bash
conda create --name ENV_NAME python=3.7.2 --y
conda activate ENV_NAME
pip install -r requirements.txt
```

---
## Data

The datasets used in this work are all present in the `./dataset/` folder. 
There are two formats for the same dataset: `<ENV_NAME>.npz` and `Stable<ENV_NAME.npz>`.
Both datasets share the same data, however, the second is in [Stable-Baselines](https://stable-baselines.readthedocs.io/en/master/) format.

---
## Running

All scripts for training the policy are in the `./scripts/adversarial/` folder.
There are also scripts for training BC and GAIL.

If you want to run the scripts using GPU it is enough to change `--gpu -1` to `--gpu 0`.

---
## Results

---
## Citation
```
@INPROCEEDINGS{monteiro2023sail,
    author={Monteiro, Juarez and Gavenski, Nathan and Meneguzzi, Felipe and Barros, Rodrigo C.},
    booktitle={2023 International Joint Conference on Neural Networks (IJCNN)}, 
    title={Self-Supervised Adversarial Imitation Learning}, 
    year={2023},
    pages={1-8},
    doi={10.1109/IJCNN54540.2023.10191197}
}
```
