from sbtc_ray import base_cfg  # Adjust if base_cfg is elsewhere
import util
from ray import tune


class HumanoidRslRlJobCfg(base_cfg.RslRlJobCfg):
    def __init__(self, cfg: dict = {}):
        # Set up the default IsaacLab config structure
        cfg = util.populate_isaac_ray_cfg_args(cfg)

        # Task setup
        cfg["runner_args"]["--task"] = tune.choice(["Isaac-Humanoid-v0"])  # adjust if variant differs
        cfg["runner_args"]["headless_singleton"] = "--headless"
        cfg["runner_args"]["--livestream"] = 0
        cfg["runner_args"]["--num_envs"] = 1024 
        cfg["runner_args"]["--max_iterations"] = 500 
        cfg["runner_args"]["--seed"] = 42
        # --- TUNE REWARD TERMS ---
        # These are typical reward terms in Cartpole
        reward_terms = [
            "progress", 
            "alive", 
            "upright", 
            "move_to_target", 
            "action_l2",
            "energy",
            "joint_pos_limits",
        ]
        for term in reward_terms:
            cfg["hydra_args"][f"env.rewards.{term}.weight"] = tune.uniform(-1.0, 1.0)

        # --- TUNE PPO HYPERPARAMETERS ---
        super().__init__(
            cfg,
            # below will tune PPO as well
            # sample_clip_param=True,
            # sample_num_learning_epochs=True,
            # sample_num_mini_batches=True,
            # sample_desired_kl=True,
            # choose_entropy_coef=True,
        )
