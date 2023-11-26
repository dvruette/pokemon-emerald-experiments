from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        
        is_done = self.training_env.env_method("check_if_done")
        for i, done in enumerate(is_done):
            if done:
                info = self.training_env.env_method("get_last_agent_stats", indices=[i])[0]
                for key,val in info.items():
                    self.logger.record_mean(f"env_stats/{key}", val)

        return True
    
    def _on_rollout_end(self) -> None:
        all_infos = self.training_env.env_method("get_last_agent_stats")
        for info in all_infos:
            if info is None:
                continue
            for key,val in info.items():
                self.logger.record_mean(f"env_stats/mean_{key}", val)
