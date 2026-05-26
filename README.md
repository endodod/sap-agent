# sap-agent

A reinforcement learning agent that plays Super Auto Pets (Pack 1). The agent is trained with Maskable PPO (stable-baselines3 / sb3-contrib) inside a custom Gymnasium environment that wraps [sap-sim](https://github.com/endodo/sap-sim), a full game simulation. After training, the agent plays 10,000 evaluation games and exports pick-frequency, win-rate, and synergy statistics consumed by [sap-dashboard](https://github.com/endodo/sap-dashboard).

## Related repos

- [`sap-sim`](https://github.com/endodo/sap-sim) — the game simulation (Pack 1, v1.0)
- [`sap-dashboard`](https://github.com/endodo/sap-dashboard) — Next.js dashboard that visualises the exported stats

## Setup

```bash
# 1. Install the sim as a local dep
pip install -e ../sap-sim

# 2. Install agent deps
pip install -r requirements.txt

# 3. Generate the enemy pool (run once)
python scripts/gen_enemy_pool.py

# 4. Train
python agent/train.py

# 5. Evaluate and export stats
python agent/evaluate.py
```

TensorBoard logs are written to `runs/` and model checkpoints to `checkpoints/`.

## Exported data files

| File | Contents |
|------|----------|
| `data/enemy_pool.pkl` | Pre-generated pool of 500 opponent teams (one per round bucket). Required for training. |
| `data/eval_results.json` | Per-game records from 10,000 evaluation episodes: outcome, rounds survived, lives remaining, every buy action, final team composition. |
| `data/stats.json` | Aggregate statistics: overall win rate, mean rounds survived, per-pet pick frequency and win rate, top-20 synergy pairs, and win rate broken down by round reached. |
