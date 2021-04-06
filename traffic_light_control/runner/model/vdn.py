from runner.model.iac import ICritic


def make_vdn_model(config, agents_id):
    acting_nets = {}
    target_nets = {}
    for id in agents_id:
        acting_nets[id] = ICritic(
            input_space=config["local_state"][id],
            output_space=config["local_action"][id],
        )
        target_nets[id] = ICritic(
            input_space=config["local_state"][id],
            output_space=config["local_action"][id],
        )
    model = {
        "acting_nets": acting_nets,
        "target_nets": target_nets,
    }
    return model
