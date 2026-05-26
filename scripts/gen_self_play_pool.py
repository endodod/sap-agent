"""
Generate an enemy pool from the current trained model instead of random AI.
Run this between training phases to make opponents match the agent's skill level.

Usage:
    python scripts/gen_self_play_pool.py
    python scripts/gen_self_play_pool.py --checkpoint checkpoints/sap_ppo_500000_steps.zip
    python scripts/gen_self_play_pool.py --n-games 1000
"""
import argparse
import copy
import glob

from sb3_contrib import MaskablePPO
from src.game.data_loader import DataLoader
from src.game.enemy_pool import EnemyPool
from env.sap_env import SAPEnv


def generate(model, data, current_pool, n_games: int) -> dict:
    pool: dict[int, list] = {}
    for i in range(n_games):
        if i % 100 == 0:
            print(f"  game {i}/{n_games}")
        env = SAPEnv(data, current_pool)
        obs, _ = env.reset()
        done = False
        while not done:
            masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=masks, deterministic=False)
            action = int(action)
            # Snapshot the team just before end_turn — same moment AISimulator uses
            if action == 41:
                round_num = env.gs.round
                snapshot = copy.deepcopy(env.gs.player_team)
                pool.setdefault(round_num, []).append(snapshot)
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
    return pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="Path to checkpoint zip. Defaults to latest in checkpoints/.")
    parser.add_argument("--n-games", type=int, default=500)
    parser.add_argument("--output", default="data/enemy_pool.pkl")
    args = parser.parse_args()

    if args.checkpoint:
        ckpt = args.checkpoint
    else:
        candidates = sorted(glob.glob("checkpoints/sap_ppo_*_steps.zip"))
        if not candidates:
            raise FileNotFoundError("No checkpoint found. Run agent/train.py first.")
        ckpt = candidates[-1]
    print(f"Using checkpoint: {ckpt}")

    data = DataLoader().load()
    current_pool = EnemyPool.load(args.output)
    model = MaskablePPO.load(ckpt)

    print(f"Generating self-play pool from {args.n_games} games...")
    pool_dict = generate(model, data, current_pool, args.n_games)

    rounds = sorted(pool_dict.keys())
    print(f"Collected teams for rounds: {rounds}")
    for r in rounds:
        print(f"  round {r}: {len(pool_dict[r])} teams")

    ep = EnemyPool(pool_dict)
    ep.save(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
