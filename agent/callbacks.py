import copy

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from src.game.enemy_pool import EnemyPool


class WinRateCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq: int = 100_000, n_eval_episodes: int = 200, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self._next_eval = eval_freq

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_eval:
            wins, lives_list, rounds_list = self._run_eval()
            self._next_eval += self.eval_freq
            win_rate = float(np.mean(wins))
            mean_lives = float(np.mean(lives_list))
            mean_round = float(np.mean(rounds_list))
            self.logger.record("eval/win_rate", win_rate)
            self.logger.record("eval/mean_lives", mean_lives)
            self.logger.record("eval/mean_round", mean_round)
            if self.verbose:
                print(
                    f"[WinRateCallback] step={self.num_timesteps} "
                    f"win_rate={win_rate:.3f} mean_lives={mean_lives:.2f} mean_round={mean_round:.2f}"
                )
        return True

    def _run_eval(self):
        env = self.eval_env
        n_envs = env.num_envs
        episodes_done = 0
        wins, lives_list, rounds_list = [], [], []
        obs = env.reset()
        episode_lives = [10] * n_envs
        episode_rounds = [1] * n_envs

        while episodes_done < self.n_eval_episodes:
            masks = np.array(env.env_method("action_masks"))
            actions, _ = self.model.predict(obs, action_masks=masks, deterministic=True)
            obs, _, dones, infos = env.step(actions)
            for i, (done, info) in enumerate(zip(dones, infos)):
                episode_lives[i] = info.get("lives", episode_lives[i])
                episode_rounds[i] = info.get("round", episode_rounds[i])
                if done:
                    wins.append(info.get("wins", 0) >= 10)
                    lives_list.append(episode_lives[i])
                    rounds_list.append(episode_rounds[i])
                    episode_lives[i] = 10
                    episode_rounds[i] = 1
                    episodes_done += 1
                    if episodes_done >= self.n_eval_episodes:
                        break

        return wins, lives_list, rounds_list


class SelfPlayCallback(BaseCallback):
    """
    Every regen_freq steps, collects team snapshots from the current model
    and replaces the enemy pool so opponents scale with agent skill.
    """

    def __init__(
        self,
        data,
        win_rate_cb: WinRateCallback,
        regen_freq: int = 500_000,
        n_games: int = 500,
        pool_path: str = "data/enemy_pool.pkl",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.data = data
        self.win_rate_cb = win_rate_cb
        self.regen_freq = regen_freq
        self.n_games = n_games
        self.pool_path = pool_path
        self._next_regen = regen_freq

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_regen:
            self._regenerate_pool()
        return True

    def _regenerate_pool(self):
        self._next_regen += self.regen_freq
        print(f"\n[SelfPlay] Regenerating enemy pool at step {self.num_timesteps}...")
        current_pool = EnemyPool.load(self.pool_path)
        pool_dict = self._collect_teams(current_pool)

        if not pool_dict:
            print("[SelfPlay] No teams collected — skipping.")
            return

        total = sum(len(v) for v in pool_dict.values())
        print(f"[SelfPlay] Collected {total} snapshots across rounds {sorted(pool_dict.keys())}")

        new_pool = EnemyPool(pool_dict)
        new_pool.save(self.pool_path)

        self.model.env.set_attr("enemy_pool", new_pool)
        self.win_rate_cb.eval_env.set_attr("enemy_pool", new_pool)

        n_train = self.model.env.num_envs
        self.model._last_obs = self.model.env.reset()
        self.model._last_episode_starts = np.ones((n_train,), dtype=bool)
        print(f"[SelfPlay] Done.\n")

    def _collect_teams(self, current_pool) -> dict:
        from env.sap_env import SAPEnv
        pool_dict: dict[int, list] = {}
        for i in range(self.n_games):
            if i % 100 == 0:
                print(f"[SelfPlay]   game {i}/{self.n_games}")
            env = SAPEnv(self.data, current_pool)
            obs, _ = env.reset()
            done = False
            while not done:
                masks = env.action_masks()
                action, _ = self.model.predict(obs, action_masks=masks, deterministic=False)
                action = int(action)
                if action == 41:
                    snapshot = copy.deepcopy(env.gs.player_team)
                    pool_dict.setdefault(env.gs.round, []).append(snapshot)
                obs, _, term, trunc, _ = env.step(action)
                done = term or trunc
        return pool_dict
