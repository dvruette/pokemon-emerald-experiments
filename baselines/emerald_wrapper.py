from typing import Literal
import hnswlib
import numpy as np
from pygba import GameWrapper, PyGBA
from pygba.game_wrappers.pokemon_emerald import (
    get_game_state,
    read_species_info,
    read_experience_tables,
    count_flags,
    count_changed_flags,
    get_gained_exp,
)


class ExplorationTracker:
    def __init__(
        self,
        distance_threshold: float = 6.0,  # GBA screen is 7x5 tiles
        revisit_cooldown: int = 8192,
        max_hnsw_count: int = 100_000,
    ):
        self.distance_threshold = distance_threshold
        self.revisit_cooldown = revisit_cooldown
        self.max_hnsw_count = max_hnsw_count
        self.location_store = {}
        self.cooldown_store = {}

        self.curr_step = 0
        self.total_visits = 0

    def reset(self):
        self.location_store = {}
        self.cooldown_store = {}
        self.curr_step = 0

    def add_location(self, map_id, x, y):
        if map_id not in self.location_store:
            self.location_store[map_id] = hnswlib.Index(space='l2', dim=2)
            self.location_store[map_id].init_index(max_elements=self.max_hnsw_count, ef_construction=100, M=16)

        index = self.location_store[map_id]
        pos = np.array([x, y])
        label = index.get_current_count()
        pos_hash = hash((map_id, label))
        if index.get_current_count() == 0:
            index.add_items(pos, np.array([pos_hash]))
            self.cooldown_store[pos_hash] = self.curr_step + self.cooldown_steps
            self.total_visits += 1
        else:
            labels, distances = index.knn_query(pos, k=1)
            label, dist = labels[0][0], distances[0][0]
            if dist < self.distance_threshold:
                if self.cooldown_store[label] > self.curr_step:
                    self.cooldown_store[label] = self.curr_step + self.cooldown_steps
                    self.total_visits += 1
            else:  # dist >= self.distance_threshold
                index.add_items(pos, np.array([pos_hash]))
                self.cooldown_store[pos_hash] = self.curr_step + self.cooldown_steps
                self.total_visits += 1
    
    def update(self, location=None, pos=None):
        if location is not None and pos is not None:
            map_id = (location["mapGroup"], location["mapNum"])
            if map_id != (0, 0):  # map_id is set to 0 while loading a new area
                self.add_location(map_id, pos["x"], pos["y"])
        
        self.curr_step += 1

    def total_size(self):
        return sum(index.get_current_count() for index in self.location_store.values())
    
    def total_visited(self):
        return self.total_visits


class HealingTracker:
    def __init__(
        self,
        consistency_threshold: int = 4,
    ):
        self.consistency_threshold = consistency_threshold

        self.reset()

    def reset(self):
        self.curr_party = {}
        self.prev_party = {}
        self.candidate_party = {}
        self.curr_consistency = 0
        self.total_healed_amount = 0

    def update(self, party):
        is_consistent = True
        seen_ids = set()
        for mon in party:
            mon_id = mon["box"]["personality"]
            seen_ids.add(mon_id)

            mon_hp = mon["hp"]
            max_hp = mon["maxHp"]
            if mon_id not in self.candidate_party:
                self.candidate_party[mon_id] = (mon_hp, max_hp)
                is_consistent = False
            elif self.candidate_party[mon_id] != (mon_hp, max_hp):
                self.candidate_party[mon_id] = (mon_hp, max_hp)
                is_consistent = False

        for mon_id in self.candidate_party.keys():
            if mon_id not in seen_ids:
                del self.candidate_party[mon_id]
                is_consistent = False

        if is_consistent:
            self.curr_consistency += 1
        if self.curr_consistency >= self.consistency_threshold:
            self.curr_party = self.candidate_party.copy()
            self.curr_consistency = 0
            self.candidate_party = {}

        for mon_id in self.curr_party.keys():
            if mon_id not in seen_ids:
                del self.curr_party[mon_id]

        for mon_id in self.curr_party.keys():
            if mon_id in self.prev_party():
                prev_hp, prev_max_hp = self.prev_party[mon_id]
                curr_hp, curr_max_hp = self.curr_party[mon_id]
                if prev_hp > 0:  # don't count healing from fainted pokemon
                    prev_missing_hp = prev_max_hp - prev_hp
                    curr_missing_hp = curr_max_hp - curr_hp
                    healed_amount = max(0, prev_missing_hp - curr_missing_hp)
                    self.total_healed_amount += healed_amount / curr_max_hp

        self.prev_party = self.curr_party.copy()

    def total_healed(self):
        return self.total_healed_amount


class CustomEmeraldWrapper(GameWrapper):
    def __init__(
        self,
        badge_reward: float = 10.0,
        pokedex_reward: float = 10,
        pokenav_reward: float = 10,
        champion_reward: float = 100.0,
        visit_city_reward: float = 5.0,
        money_gained_reward: float = 2e-4,
        # setting this below half of money_gained_reward leaves an exploit
        # where the agent can just buy and sell stuff at the pokemart..
        # but setting it too high will make it scared of battling
        money_lost_reward: float = 1e-4,
        seen_pokemon_reward: float = 0.2,
        caught_pokemon_reward: float = 1.0,
        trainer_beat_reward: float = 2.0,
        event_reward: float = 0.075,
        exp_reward_transform: Literal["linear", "sqrt", "log", "tanh"] = "tanh",
        exp_reward_shape: float = 0.003,
        exp_reward_scale: float = 5,
        heal_reward: float = 0.05,
        exploration_reward: float = 0.02,
        revisit_reward: float = 0.01,
        revisit_cooldown: int = 8192,
        exploration_dist_thresh: float = 6.0,  # GBA screen is 7x5 tiles
        max_hnsw_count: int = 100_000,
        reward_scale: float = 1.0,
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
        self.trainer_beat_reward = trainer_beat_reward
        self.event_reward = event_reward
        self.exp_reward_transform = exp_reward_transform
        self.exp_reward_shape = exp_reward_shape
        self.exp_reward_scale = exp_reward_scale
        self.heal_reward = heal_reward
        self.exploration_reward = exploration_reward
        self.revisit_reward = revisit_reward
        self.reward_scale = reward_scale

        self.exploration_tracker = ExplorationTracker(
            distance_threshold=exploration_dist_thresh,
            revisit_cooldown=revisit_cooldown,
            max_hnsw_count=max_hnsw_count,
        )
        self._total_script_flags = 0
        self._prev_reward = 0.0
        self._game_state = {}
        self._prev_game_state = {}
        self._reward_info = {}
        self._location_store = {}

    def reward(self, gba: PyGBA, observation):
        self._game_state.update(get_game_state(gba))
        state = self._game_state

        # Game state can get funky during loading screens, so we just wait until
        # we get a valid observation.
        if observation is not None and observation.sum() < 1:
            self.exploration_tracker.update()
            return 0.0
        
        self.exploration_tracker.update(state.get("location", None), state.get("pos", None))

        trainer_flags = state.get("trainer_flags", None)
        num_trainer_flags = count_flags(trainer_flags)

        new_script_flags = state.get("script_flags", None)
        prev_script_flags = self._prev_game_state.get("script_flags", None)
        changed_script_flags = count_changed_flags(prev_script_flags, new_script_flags)
        self._total_script_flags += changed_script_flags

        species_info = read_species_info(gba)
        experience_tables = read_experience_tables(gba)
        all_mons = list(map(lambda x: x["box"], state.get("party", []))) + state.get("boxes", [])
        total_gained_exp = sum(get_gained_exp(mon, species_info, experience_tables) for mon in all_mons)
        if total_gained_exp >= 0:
            if self.exp_reward_transform == "linear":
                exp_reward = self.reward_scale * total_gained_exp
            elif self.exp_reward_transform == "sqrt":
                exp_reward = self.exp_reward_scale * total_gained_exp ** self.exp_reward_shape
            elif self.exp_reward_transform == "log":
                exp_reward = self.exp_reward_scale * np.log(self.exp_reward_shape * total_gained_exp + 1)
            elif self.exp_reward_transform == "tanh":
                exp_reward = self.exp_reward_scale * np.tanh(self.exp_reward_shape * total_gained_exp)
            # round exp reward to 0.01
            exp_reward = round(exp_reward * 100) / 100
        else:
            print()
            print(f"WARNING: total_gained_exp >= 0: {total_gained_exp}")
            print()
            exp_reward = self._reward_info.get("exp_rew", 0.0)

        # don't give any reward for visiting the first town, as the player spawns there
        visited_cities = max(0, sum(state.get("visited_cities", {}).values()) - 1)
        # player starts with $3000 cash
        earned_money = (state.get("money", 3000) - 3000)
        # in case the the game state glitches and gives the player a lot of money, we ignore values over 100k
        if earned_money != 0 and (self.money_gained_reward != 0.0 or self.money_lost_reward != 0.0):
            if earned_money > 0 and earned_money < 100_000:
                money_rew = earned_money * self.money_gained_reward
            elif earned_money < 0:
                money_rew = earned_money * self.money_lost_reward
        else:
            money_rew = 0.0

        # exploration reward
        total_visited = self.exploration_tracker.total_visited()
        total_size = self.exploration_tracker.total_size()
        num_revisited = total_visited - total_size

        # heal reward
        healed_amount = self.healing_tracker.total_healed()

        self._reward_info = {
            "visit_city_rew": visited_cities * self.visit_city_reward,
            "seen_poke_rew": state.get("num_seen_pokemon", 0) * self.seen_pokemon_reward,
            "caught_poke_rew": state.get("num_caught_pokemon", 0) * self.caught_pokemon_reward,
            "explore_rew": total_size * self.exploration_reward,
            "revisit_rew": num_revisited * self.revisit_reward,
            "heal_rew": healed_amount * self.heal_reward,
            "money_rew": money_rew,
            "pokedex_rew": (1.0 if state.get("has_pokedex", False) else 0.0) * self.pokedex_reward,
            "pokenav_rew": (1.0 if state.get("has_pokenav", False) else 0.0) * self.pokenav_reward,
            "badge_rew": state.get("num_badges", 0) * self.badge_reward,
            "champ_rew": state.get("is_champion", 0) * self.champion_reward,
            "trainer_rew": num_trainer_flags * self.trainer_beat_reward,
            "event_rew": self._total_script_flags * self.event_reward,
            "exp_rew": exp_reward,
        }
        reward = sum(self._reward_info.values())
        self._reward_info["total_reward"] = reward

        prev_reward = self._prev_reward
        self._prev_reward = reward
        self._prev_game_state = state.copy()
        return self.reward_scale * (reward - prev_reward)
    
    def reset(self, gba: PyGBA):
        self.exploration_tracker.reset()
        self._game_state = {}
        self._location_store = {}
        self._total_script_flags = 0
        self._prev_reward = 0.0
        self._prev_reward = self.reward(gba, None)
        self._prev_game_state = {}
    
    def info(self, gba: PyGBA, observation):
        if self._game_state is None:
            self._game_state = get_game_state(gba)

        return {
            "game_state": self._game_state,
            "prev_reward": self._prev_reward,
            "rewards": self._reward_info,
        }
