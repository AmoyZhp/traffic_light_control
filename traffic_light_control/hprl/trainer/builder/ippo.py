def build_ppo_trainer(
    config,
    env: MultiAgentEnv,
    models: Dict[str, nn.Module],
):
    policy_config = config["policy"]
    executing_config = config["executing"]

    critic_lr = policy_config["critic_lr"]
    actor_lr = policy_config["actor_lr"]
    discount_factor = policy_config["discount_factor"]
    update_period = policy_config["update_period"]
    inner_epoch = policy_config["inner_epoch"]
    clip_param = policy_config["clip_param"]
    advg_type = policy_config["advg_type"]

    logger.info("create IAC trainer")
    logger.info("\t critic lr : %f", critic_lr)
    logger.info("\t discount factor : %f", discount_factor)
    logger.info("\t update period : %d", update_period)
    logger.info("\t inner epoch : %d", inner_epoch)
    logger.info("\t clip param : %f : ", clip_param)

    agents_id = env.agents_id
    loss_fn = nn.MSELoss()
    policies = {}
    for id in agents_id:
        model = models[id]
        critic_net = model["critic_net"]
        critic_target_net = model["critic_target_net"]
        actor_net = model["actor_net"]
        action_space = policy_config["action_space"][id]
        state_space = policy_config["state_space"][id]
        policies[id] = PPO(
            critic_net=critic_net,
            critic_target_net=critic_target_net,
            inner_epoch=inner_epoch,
            clip_param=clip_param,
            actor_net=actor_net,
            critic_lr=critic_lr,
            actor_lr=actor_lr,
            discount_factor=discount_factor,
            update_period=update_period,
            action_space=action_space,
            state_space=state_space,
            critic_loss_fn=loss_fn,
            advantage_type=advg_type,
        )

    recorder = Printer()
    if executing_config["recording"]:
        recorder = TorchRecorder(executing_config["record_base_dir"])
        logger.info("\t training will be recorded")
    trainer = IOnPolicyTrainer(
        type=PolicyTypes.PPO,
        policies=policies,
        env=env,
        recorder=recorder,
        config=executing_config,
    )
    return trainer


def get_ppo_test_setting():
    config, model = get_ac_test_setting()
    config["type"] = PolicyTypes.PPO
    config["policy"]["inner_epoch"] = 16
    config["policy"]["clip_param"] = 0.2
    return config, model
