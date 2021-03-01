import hprl
import envs
from runner.nets import IActor, ICritic


class Runner(object):

    def run(self):

        env_config = envs.get_default_config_for_single()
        env = self._make_env(env_config)

        model_config = {
            "input_space": env.get_local_state_space(),
            "output_space": env.get_local_action_space()
        }
        agents_id = env.get_agents_id()
        models = {}
        for id in agents_id:
            models[id] = self._make_model(model_config)

        trainer_config = self._get_config(env)
        trainer = hprl.create_trainer(
            config=trainer_config,
            env=env,
            models=models,
        )
        config = {
            "episode": 500,
            "train_batch": 1,
            "eval_episode": 10,
            "log_dir": "records",
        }
        episode = config["episode"]
        train_batch = config["train_batch"]
        eval_episode = config["eval_episode"]
        log_dir = config["log_dir"]
        for _ in range(train_batch):
            train_records = trainer.train(int(episode / train_batch))
            eval_records = trainer.eval(eval_episode)
            trainer.log_result(log_dir)

    def _get_config(self, env: hprl.MultiAgentEnv):
        capacity = 200000
        learning_rate = 1e-4
        batch_size = 256
        discount_factor = 0.99
        eps_init = 1.0
        eps_min = 0.01
        eps_frame = 300000
        update_period = 1000
        action_space = env.get_local_action_space()
        state_space = env.get_central_state_space()

        policy_config = {
            "learning_rate": learning_rate,
            "discount_factor": discount_factor,
            "update_period": update_period,
            "action_space": action_space,
            "state_space": state_space,
            "eps_frame": eps_frame,
            "eps_init": eps_init,
            "eps_min": eps_min,
        }
        buffer_config = {
            "type": hprl.ReplayBufferTypes.Common,
            "capacity": capacity,
        }
        exec_config = {
            "batch_size": batch_size,
            "base_dir": "records",
            "check_frequency": 500,
        }
        trainner_config = {
            "type": hprl.TrainnerTypes.IQL,
            "executing": exec_config,
            "policy": policy_config,
            "buffer": buffer_config,
        }
        return trainner_config

    def _make_env(self, config):
        env = envs.make(config)
        return env

    def _make_model(self, config):
        acting_net = ICritic(
            input_space=config["input_space"],
            output_space=config["output_space"],
        )
        target_net = ICritic(
            input_space=config["input_space"],
            output_space=config["output_space"],
        )
        model = {
            "acting": acting_net,
            "target": target_net,
        }
        return model
