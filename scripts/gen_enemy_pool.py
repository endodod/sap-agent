from src.game.data_loader import DataLoader
from src.game.ai_simulator import AISimulator
from src.game.enemy_pool import EnemyPool

data = DataLoader().load()
sim = AISimulator(data)
pool_dict = sim.run_games(n=500)
ep = EnemyPool(pool_dict)
ep.save("data/enemy_pool.pkl")
print("Saved data/enemy_pool.pkl")
