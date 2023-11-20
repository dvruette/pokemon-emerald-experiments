from stable_baselines3.common.callbacks import BaseCallback

def merge_dicts_by_mean(dicts):
    sum_dict = {}
    count_dict = {}

    for d in dicts:
        for k, v in d.items():
            if isinstance(v, (int, float)): 
                sum_dict[k] = sum_dict.get(k, 0) + v
                count_dict[k] = count_dict.get(k, 0) + 1

    mean_dict = {}
    for k in sum_dict:
        mean_dict[k] = sum_dict[k] / count_dict[k]

    return mean_dict

class TensorboardCallback(BaseCallback):

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        
        is_done = self.training_env.env_method("check_if_done")
        for i, done in enumerate(is_done):
            if done:
                infos = self.training_env.get_attr("agent_stats", indices=[i])[0]
                for key,val in infos[-1].items():
                    self.logger.record_mean(f"env_stats/{key}", val)

        return True
    
    def _on_rollout_end(self) -> None:
        all_infos = self.training_env.get_attr("agent_stats")
        for infos in all_infos:
            for key,val in infos[-1].items():
                self.logger.record_mean(f"env_stats/mean_{key}", val)
