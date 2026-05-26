import gymnasium
import numpy as np
from gymnasium import spaces

from src.game.game_state import GameState

PET_INDEX = {
    "ant": 0, "bat": 1, "beaver": 2, "bison": 3, "blowfish": 4,
    "camel": 5, "cat": 6, "cow": 7, "cricket": 8, "crocodile": 9,
    "dodo": 10, "dog": 11, "dolphin": 12, "dragon": 13, "duck": 14,
    "elephant": 15, "fish": 16, "flamingo": 17, "giraffe": 18, "hedgehog": 19,
    "hippo": 20, "kangaroo": 21, "leopard": 22, "monkey": 23, "mosquito": 24,
    "otter": 25, "ox": 26, "peacock": 27, "penguin": 28, "rat": 29,
}
FOOD_INDEX = {"apple": 0, "salad": 1, "sushi": 2, "pizza": 3}


def _pet_features(pet) -> np.ndarray:
    feats = np.zeros(25, dtype=np.float32)
    if pet is None:
        return feats
    feats[0] = pet.attack / 50.0
    feats[1] = pet.health / 50.0
    feats[2] = pet.level / 3.0
    feats[3] = 1.0 if pet.is_alive else 0.0
    feats[4] = 1.0
    pet_idx = min(PET_INDEX.get(pet.name.lower(), 19), 19)
    feats[5 + pet_idx] = 1.0
    return feats


def _food_features(slot) -> np.ndarray:
    feats = np.zeros(5, dtype=np.float32)
    if slot is None or slot.item is None:
        return feats
    feats[0] = 1.0
    food_name = slot.item.name.lower() if hasattr(slot.item, "name") else ""
    food_idx = FOOD_INDEX.get(food_name)
    if food_idx is not None:
        feats[1 + food_idx] = 1.0
    return feats


class SAPEnv(gymnasium.Env):
    metadata = {"render_modes": []}

    def __init__(self, data, enemy_pool, max_shop_actions=30):
        super().__init__()
        self.data = data
        self.enemy_pool = enemy_pool
        self.max_shop_actions = max_shop_actions
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(267,), dtype=np.float32)
        self.action_space = spaces.Discrete(42)
        self.gs = None
        self._shop_actions_this_turn = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.gs = GameState(self.data, enemy_pool=self.enemy_pool)
        self.gs.start_turn()
        self._shop_actions_this_turn = 0
        return self._get_obs(), {}

    def step(self, action: int):
        reward = 0.0
        terminated = False
        truncated = False

        if action == 41:
            result = self.gs.end_turn()
            reward = self._round_reward(result)
            terminated = result["game_over"] or result["victory"]
            if not terminated:
                self.gs.start_turn()
                self._shop_actions_this_turn = 0
        else:
            self._execute_shop_action(action)
            self._shop_actions_this_turn += 1
            if self._shop_actions_this_turn >= self.max_shop_actions:
                result = self.gs.end_turn()
                reward = self._round_reward(result)
                terminated = result["game_over"] or result["victory"]
                if not terminated:
                    self.gs.start_turn()
                    self._shop_actions_this_turn = 0

        obs = self._get_obs()
        info = {"lives": self.gs.lives, "wins": self.gs.wins, "round": self.gs.round}
        return obs, reward, terminated, truncated, info

    def _round_reward(self, result: dict) -> float:
        label = result["result"].training_label
        r = 1.0 if label == "win" else -1.0
        if result["victory"]:
            r += 10.0
        if result["game_over"]:
            r -= 10.0
        r += result["lives"] * 0.05
        return r

    def _execute_shop_action(self, action: int):
        if 0 <= action <= 24:
            s, t = divmod(action, 5)
            self.gs.buy_pet(s, t)
        elif 25 <= action <= 29:
            self.gs.sell_pet(action - 25)
        elif 30 <= action <= 34:
            self.gs.buy_food(0, action - 30)
        elif 35 <= action <= 39:
            self.gs.buy_food(1, action - 35)
        elif action == 40:
            self.gs.reroll()

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(42, dtype=bool)
        gs = self.gs
        shop = gs.shop
        gold = gs.gold

        for s in range(min(len(shop.pet_slots), 5)):
            slot = shop.pet_slots[s]
            if not slot.is_empty() and gold >= 3:
                shop_pet = slot.item
                for t in range(5):
                    team_pet = gs.player_team.slots[t]
                    if team_pet is None or team_pet.name == shop_pet.name:
                        mask[s * 5 + t] = True

        for t in range(5):
            if gs.player_team.slots[t] is not None:
                mask[25 + t] = True

        if len(shop.food_slots) > 0 and not shop.food_slots[0].is_empty() and gold >= 3:
            for t in range(5):
                if gs.player_team.slots[t] is not None:
                    mask[30 + t] = True

        if len(shop.food_slots) > 1 and not shop.food_slots[1].is_empty() and gold >= 3:
            for t in range(5):
                if gs.player_team.slots[t] is not None:
                    mask[35 + t] = True

        if gold >= 1:
            mask[40] = True

        mask[41] = True
        return mask

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(267, dtype=np.float32)
        gs = self.gs
        idx = 0

        for t in range(5):
            obs[idx:idx + 25] = _pet_features(gs.player_team.slots[t])
            idx += 25

        for s in range(5):
            if s < len(gs.shop.pet_slots):
                slot = gs.shop.pet_slots[s]
                pet = slot.item if not slot.is_empty() else None
            else:
                pet = None
            obs[idx:idx + 25] = _pet_features(pet)
            idx += 25

        for f in range(2):
            if f < len(gs.shop.food_slots):
                obs[idx:idx + 5] = _food_features(gs.shop.food_slots[f])
            idx += 5

        obs[idx + 0] = min(gs.gold / 10.0, 1.0)
        obs[idx + 1] = min(gs.round / 15.0, 1.0)
        obs[idx + 2] = gs.lives / 10.0
        obs[idx + 3] = min(gs.wins / 10.0, 1.0)
        obs[idx + 4] = min(gs.win_streak / 5.0, 1.0)
        obs[idx + 5] = min(gs.loss_streak / 5.0, 1.0)
        obs[idx + 6] = 1.0 if gs.gold >= 1 else 0.0
        idx += 7

        assert idx == 267
        return obs

    def render(self):
        pass
