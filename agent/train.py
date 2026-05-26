import glob
import os

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

from src.game.data_loader import DataLoader
from src.game.enemy_pool import EnemyPool
from env.sap_env import SAPEnv
from agent.callbacks import WinRateCallback, SelfPlayCallback

TOTAL_STEPS  = 3_000_000
N_TRAIN_ENVS = 8
N_EVAL_ENVS  = 4
EVAL_FREQ    = 100_000
CKPT_FREQ    = 100_000
REGEN_FREQ   = 500_000
REGEN_GAMES  = 500


def make_env(data, enemy_pool):
    def _init():
        return SAPEnv(data, enemy_pool)
    return _init


def train():
    data = DataLoader().load()
    enemy_pool = EnemyPool.load("data/enemy_pool.pkl")

    env      = VecMonitor(SubprocVecEnv([make_env(data, enemy_pool) for _ in range(N_TRAIN_ENVS)]))
    eval_env = VecMonitor(SubprocVecEnv([make_env(data, enemy_pool) for _ in range(N_EVAL_ENVS)]))

    checkpoints = sorted(glob.glob("checkpoints/sap_ppo_*_steps.zip"))
    if checkpoints:
        latest = checkpoints[-1]
        steps_done = int(os.path.basename(latest).split("_")[2])
        print(f"Resuming from {latest} ({steps_done:,} steps done)")
        model = MaskablePPO.load(latest, env=env, tensorboard_log="runs/")
    else:
        steps_done = 0
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="runs/",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
        )

    remaining = TOTAL_STEPS - steps_done
    if remaining <= 0:
        print("Already at 3M steps. Done.")
        return

    win_rate_cb = WinRateCallback(eval_env=eval_env, eval_freq=EVAL_FREQ, n_eval_episodes=200)
    self_play_cb = SelfPlayCallback(
        data=data,
        win_rate_cb=win_rate_cb,
        regen_freq=REGEN_FREQ,
        n_games=REGEN_GAMES,
        pool_path="data/enemy_pool.pkl",
    )

    callbacks = [
        CheckpointCallback(save_freq=CKPT_FREQ, save_path="checkpoints/", name_prefix="sap_ppo"),
        win_rate_cb,
        self_play_cb,
    ]

    model.learn(total_timesteps=remaining, callback=callbacks, reset_num_timesteps=False)
    model.save("checkpoints/sap_ppo_final")


if __name__ == "__main__":
    train()
