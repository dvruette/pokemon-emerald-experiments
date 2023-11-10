import hnswlib
import numpy as np
from pygba import PokemonEmerald, PyGBA

class CustomEmeraldWrapper(PokemonEmerald):
    def __init__(
        self,
        badge_reward: float = 10.0,
        champion_reward: float = 100.0,
        visit_city_reward: float = 5.0,
        money_reward: float = 0.0,
        seen_pokemon_reward: float = 0.2,
        caught_pokemon_reward: float = 1.0,
        exploration_reward: float = 0.1,
        exploration_dist_thresh: float = 6.0,  # GBA screen is 7x5 tiles
        max_hnsw_count: int = 100000,
    ):
        self.badge_reward = badge_reward
        self.champion_reward = champion_reward
        self.visit_city_reward = visit_city_reward
        self.money_reward = money_reward
        self.seen_pokemon_reward = seen_pokemon_reward
        self.caught_pokemon_reward = caught_pokemon_reward
        self.exploration_reward = exploration_reward
        self.exploration_dist_thresh = exploration_dist_thresh
        self.max_hnsw_count = max_hnsw_count

        self._prev_reward = 0.0
        self._game_state = {}
        self._reward_info = {}
        self._location_store = {}

    def get_exploration_reward(self, state):
        location = state.get("location", None)
        if location is not None:
            map_id = (location["mapGroup"], location["mapNum"])
            if map_id not in self._location_store:
                self._location_store[map_id] = hnswlib.Index(space='l2', dim=2)
                self._location_store[map_id].init_index(max_elements=self.max_hnsw_count, ef_construction=100, M=16)
            index = self._location_store[map_id]
            pos = np.array([[location["x"], location["y"]]])
            if index.get_current_count() == 0:
                index.add_items(pos, np.array([index.get_current_count()]))
            else:
                labels, distances = index.knn_query(pos, k = 1)
                if distances[0][0] >= self.exploration_dist_thresh:
                    # print(f"distances[0][0] : {distances[0][0]} similar_frame_dist : {self.similar_frame_dist}")
                    index.add_items(
                        pos, np.array([index.get_current_count()])
                    )
        return sum(index.get_current_count() for index in self._location_store.values()) * self.exploration_reward

    def reward(self, gba, observation):
        self._game_state.update(self.game_state(gba))
        state = self._game_state

        self._reward_info = {
            "badge_reward": state.get("num_badges", 0) * self.badge_reward,
            "champion_reward": state.get("is_champion", 0) * self.champion_reward,
            "visit_city_reward": sum(state.get("visited_cities", {}).values()) * self.visit_city_reward,
            "money_reward": state.get("money", 0) * self.money_reward,
            "seen_pokemon_reward": state.get("num_seen_pokemon", 0) * self.seen_pokemon_reward,
            "caught_pokemon_reward": state.get("num_caught_pokemon", 0) * self.caught_pokemon_reward,
            "exploration_reward": self.get_exploration_reward(state),
        }
        reward = sum(self._reward_info.values())
        self._reward_info["total_reward"] = reward

        prev_reward = self._prev_reward
        self._prev_reward = reward
        self._prev_game_state = state.copy()
        return reward - prev_reward
    
    def reset(self, gba):
        self._game_state = {}
        self._location_store = {}
        self._prev_reward = self.reward(gba, None)
    
    def info(self, gba, observation):
        if self._game_state is None:
            self._game_state = self.game_state(gba)

        return {
            "game_state": self._game_state,
            "prev_reward": self._prev_reward,
            "rewards": self._reward_info,
        }
