from runner.model.iac import ICritic


def make_iql_model(config, agents_id):
    models = {}
    for id in agents_id:
        acting_net = ICritic(
            input_space=config["local_state"][id],
            output_space=config["local_action"][id],
        )
        target_net = ICritic(
            input_space=config["local_state"][id],
            output_space=config["local_action"][id],
        )
        models[id] = {
            "acting_net": acting_net,
            "target_net": target_net,
        }
    return models
