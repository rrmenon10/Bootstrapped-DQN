# BootstrappedDQN

Implementation of BootstrappedDQN (Osband et. al 2016) using OpenAI baselines (version 0.1.3).

We recommend using the standard OpenAI baselines setup using python3.

To train the network :

```bash
python train.py --env=(Game)
```

Some Atari game scores after training on 40 million frames:

| Game | Score (averaged over last 100 episodes) |
|-------|----------------------------------------|
| Pong | 20.53 |
| Enduro | 501.48 |
| Freeway | 32.81 |
| Seaquest | 4989.4 |
| Kangaroo | 3476 |
| Riverraid | 5307.9 |