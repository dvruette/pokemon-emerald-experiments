from pygba import PokemonEmerald, PyGBA

class CustomEmeraldWrapper(PokemonEmerald):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reward(self, gba: PyGBA, observation):
        reward = super().reward(gba, observation)
        state = self._game_state

        # TODO: implement artificial curiosity to encourage exploration

        return reward
