import json
import argparse
from collections import defaultdict
from itertools import combinations

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from src.game.data_loader import DataLoader
from src.game.enemy_pool import EnemyPool
from env.sap_env import SAPEnv


def run_evaluation(model_path: str, n_games: int = 10_000):
    data = DataLoader().load()
    enemy_pool = EnemyPool.load("data/enemy_pool.pkl")
    model = MaskablePPO.load(model_path)

    records = []

    for _ in range(n_games):
        env = SAPEnv(data, enemy_pool)
        obs, _ = env.reset()
        pet_picks = []
        done = False

        while not done:
            masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            action = int(action)

            # Record buy_pet actions before stepping
            if 0 <= action <= 24:
                s, t = divmod(action, 5)
                slot = env.gs.shop.pet_slots[s] if s < len(env.gs.shop.pet_slots) else None
                if slot and not slot.is_empty():
                    pet_picks.append({
                        "round": env.gs.round,
                        "pet": slot.item.name,
                        "slot": t,
                    })

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        final_team = [
            (env.gs.player_team.slots[i].name if env.gs.player_team.slots[i] is not None else None)
            for i in range(5)
        ]

        records.append({
            "won": info["wins"] >= 10,
            "rounds_survived": info["round"],
            "lives_remaining": info["lives"],
            "pet_picks": pet_picks,
            "final_team": final_team,
        })

    return records


def compute_stats(records):
    won_records = [r for r in records if r["won"]]
    win_rate = len(won_records) / len(records)
    mean_rounds = float(np.mean([r["rounds_survived"] for r in records]))

    # Pet pick frequency and win tracking
    pet_picks_total = defaultdict(int)
    pet_win_games = defaultdict(int)
    pet_game_count = defaultdict(int)

    for r in records:
        picked_pets = {p["pet"] for p in r["pet_picks"]}
        for pet in picked_pets:
            pet_picks_total[pet] += sum(1 for p in r["pet_picks"] if p["pet"] == pet)
            pet_game_count[pet] += 1
            if r["won"]:
                pet_win_games[pet] += 1

    pet_pick_frequency = dict(pet_picks_total)
    pet_win_rate = {
        pet: pet_win_games[pet] / pet_game_count[pet]
        for pet in pet_game_count
    }

    # Synergy pairs: top 20 by co-occurrence
    pair_count = defaultdict(int)
    pair_wins = defaultdict(int)

    for r in records:
        pets_in_game = sorted({p["pet"] for p in r["pet_picks"]})
        for a, b in combinations(pets_in_game, 2):
            key = (a, b)
            pair_count[key] += 1
            if r["won"]:
                pair_wins[key] += 1

    top_pairs = sorted(pair_count.items(), key=lambda x: x[1], reverse=True)[:20]
    synergy_pairs = [
        {
            "pets": list(pair),
            "co_occurrence": count,
            "win_rate": pair_wins[pair] / count,
        }
        for pair, count in top_pairs
    ]

    # Win rate by round: among games that reached round N, what fraction were won?
    max_round = max(r["rounds_survived"] for r in records)
    win_rate_by_round = {}
    for rnd in range(1, max_round + 1):
        games_at_round = [r for r in records if r["rounds_survived"] >= rnd]
        if games_at_round:
            win_rate_by_round[rnd] = sum(1 for r in games_at_round if r["won"]) / len(games_at_round)

    return {
        "win_rate": win_rate,
        "mean_rounds_survived": mean_rounds,
        "pet_pick_frequency": pet_pick_frequency,
        "pet_win_rate": pet_win_rate,
        "synergy_pairs": synergy_pairs,
        "win_rate_by_round": {str(k): v for k, v in win_rate_by_round.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/sap_ppo_final.zip")
    parser.add_argument("--n-games", type=int, default=10_000)
    args = parser.parse_args()

    print(f"Running {args.n_games} evaluation games...")
    records = run_evaluation(args.model, args.n_games)

    with open("data/eval_results.json", "w") as f:
        json.dump(records, f)
    print("Saved data/eval_results.json")

    stats = compute_stats(records)
    with open("data/stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved data/stats.json  win_rate={stats['win_rate']:.3f}")


if __name__ == "__main__":
    main()
