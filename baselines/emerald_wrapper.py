import hnswlib
import numpy as np
from pygba import PokemonEmerald, PyGBA

class CustomEmeraldWrapper(PokemonEmerald):
    def __init__(
        self,
        badge_reward: float = 10.0,
        pokedex_reward: float = 10,
        pokenav_reward: float = 10,
        champion_reward: float = 100.0,
        visit_city_reward: float = 5.0,
        money_gained_reward: float = 0.001,
        # setting this below half of money_gained_reward leaves an exploit
        # where the agent can just buy and sell stuff at the pokemart..
        # but setting it too high will make it scared of battling
        money_lost_reward: float = 0.0004,
        seen_pokemon_reward: float = 0.2,
        caught_pokemon_reward: float = 1.0,
        exploration_reward: float = 0.01,
        exploration_dist_thresh: float = 6.0,  # GBA screen is 7x5 tiles
        max_hnsw_count: int = 100000,
    ):
        self.badge_reward = badge_reward
        self.pokedex_reward = pokedex_reward
        self.pokenav_reward = pokenav_reward
        self.champion_reward = champion_reward
        self.visit_city_reward = visit_city_reward
        self.money_gained_reward = money_gained_reward
        self.money_lost_reward = money_lost_reward
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
        pos = state.get("pos", None)
        if location is not None and pos is not None:
            map_id = (location["mapGroup"], location["mapNum"])
            if map_id != (0, 0):  # map_id is set to 0 while loading a new area
                if map_id not in self._location_store:
                    self._location_store[map_id] = hnswlib.Index(space='l2', dim=2)
                    self._location_store[map_id].init_index(max_elements=self.max_hnsw_count, ef_construction=100, M=16)
                index = self._location_store[map_id]
                pos = np.array([pos["x"], pos["y"]])
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

        # Game state can get funky during loading screens, so we just wait until
        # we get a valid observation.
        if observation is not None and observation.sum() < 1:
            return 0.0

        # don't give any reward for visiting the first town, as the player spawns there
        visited_cities = max(0, sum(state.get("visited_cities", {}).values()) - 1)
        # player starts with $3000 cash
        earned_money = (state.get("money", 3000) - 3000)
        # in case the the game state glitches and gives the player a lot of money, we ignore values over 100k
        if earned_money > 0 and earned_money < 100_000:
            money_rew = earned_money * self.money_gained_reward
        elif earned_money < 0:
            money_rew = earned_money * self.money_lost_reward
        else:
            money_rew = 0.0

        self._reward_info = {
            "visit_city_rew": visited_cities * self.visit_city_reward,
            "seen_pokemon_rew": state.get("num_seen_pokemon", 0) * self.seen_pokemon_reward,
            "caught_pokemon_rew": state.get("num_caught_pokemon", 0) * self.caught_pokemon_reward,
            "exploration_rew": self.get_exploration_reward(state),
            "money_rew": money_rew,
            "pokedex_rew": (1.0 if state.get("has_pokedex", False) else 0.0) * self.pokedex_reward,
            "pokenav_rew": (1.0 if state.get("has_pokenav", False) else 0.0) * self.pokenav_reward,
            "badge_rew": state.get("num_badges", 0) * self.badge_reward,
            # "champion_rew": state.get("is_champion", 0) * self.champion_reward,
        }
        reward = sum(self._reward_info.values())
        self._reward_info["total_reward"] = reward

        prev_reward = self._prev_reward
        self._prev_reward = reward
        return reward - prev_reward
    
    def reset(self, gba):
        self._game_state = {}
        self._location_store = {}
        self._prev_reward = 0.0
        self._prev_reward = self.reward(gba, None)
    
    def info(self, gba, observation):
        if self._game_state is None:
            self._game_state = self.game_state(gba)

        return {
            "game_state": self._game_state,
            "prev_reward": self._prev_reward,
            "rewards": self._reward_info,
        }
