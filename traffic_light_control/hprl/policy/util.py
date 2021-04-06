from typing import List
from hprl.typing import Action, Reward, State, Terminal, Trajectory, TrajectoryTuple
import torch


def state_mapping(s: State):
    central = None
    if s.central is not None:
        central = torch.tensor(
            s.central,
            dtype=torch.float,
        )
    else:
        central = torch.empty(1)
    local = {}
    for id in s.local.keys():
        local[id] = torch.tensor(
            s.local[id],
            dtype=torch.float,
        )
    return State(central=central, local=local)


def action_mapping(a: Action):
    central = None
    if a.central is not None:
        central = torch.tensor(
            [a.central],
            dtype=torch.long,
        )
    local = {}
    for id in a.local.keys():
        local[id] = torch.tensor(
            [a.local[id]],
            dtype=torch.long,
        )
    return Action(central=central, local=local)


def reward_mapping(r: Reward):
    central = None
    if r.central is not None:
        central = torch.tensor(
            [r.central],
            dtype=torch.float,
        )
    local = {}
    for id in r.local.keys():
        local[id] = torch.tensor(
            [r.local[id]],
            dtype=torch.float,
        )
    return Reward(central=central, local=local)


def terminal_mapping(t: Terminal):

    central = None
    if t.central is not None:
        central = torch.tensor(
            t.central,
            dtype=torch.long,
        )
    local = {}
    for id in t.local.keys():
        local[id] = torch.tensor(
            t.local[id],
            dtype=torch.long,
        )
    return Terminal(central=central, local=local)


def parase_trajectory_to_tensor(traj: Trajectory):
    states = map(state_mapping, traj.states)
    actions = map(action_mapping, traj.actions)
    rewards = map(action_mapping, traj.rewards)
    terminal = terminal_mapping(traj.terminal)
    return Trajectory(states=list(states),
                      actions=list(actions),
                      rewards=list(rewards),
                      terminal=terminal)


def parase_traj_list(batch_data: List[Trajectory]):
    if batch_data is None or not batch_data:
        return

    agents_id = batch_data[0].states[0].local.keys()

    batch_data = list(map(parase_trajectory_to_tensor, batch_data))
    batch_seq_central_s = []
    batch_seq_local_s = {id: [] for id in agents_id}
    batch_seq_central_a = []
    batch_seq_local_a = {id: [] for id in agents_id}
    batch_seq_central_r = []
    batch_seq_local_r = {id: [] for id in agents_id}
    batch_seq_central_t = []
    batch_seq_local_t = {id: [] for id in agents_id}
    for traj in batch_data:
        seq_central_s = []
        seq_central_r = []
        seq_central_t = []
        seq_central_a = []
        seq_local_s = {id: [] for id in agents_id}
        seq_local_a = {id: [] for id in agents_id}
        seq_local_r = {id: [] for id in agents_id}
        seq_local_t = {id: [] for id in agents_id}

        for s in traj.states:
            if s.central is not None:
                seq_central_s.append(s.central.unsqueeze(0))
            for id in agents_id:
                seq_local_s[id].append(s.local[id].unsqueeze(0))
        for a in traj.actions:
            if a.central is not None:
                seq_central_a.append(a.central.unsqueeze(0))
            for id in agents_id:
                seq_local_a[id].append(a.local[id].unsqueeze(0))
        for r in traj.rewards:
            if r.central is not None:
                seq_central_r.append(r.central.unsqueeze(0))
            for id in agents_id:
                seq_local_r[id].append(r.local[id].unsqueeze(0))

        seq_central_t.append(traj.terminal.central.unsqueeze(0))
        for id in agents_id:
            seq_local_t[id].append(traj.terminal.local[id].unsqueeze(0))

        # the data below is seq_len * data_space
        if seq_central_s:
            seq_central_s = torch.cat(seq_central_s, 0)
            batch_seq_central_s.append(seq_central_s.unsqueeze(0))
        if seq_central_a:
            seq_central_a = torch.cat(seq_central_a, 0)
            batch_seq_central_a.append(seq_central_a.unsqueeze(0))
        if seq_central_r:
            seq_central_r = torch.cat(seq_central_r, 0)
            batch_seq_central_r.append(seq_central_r.unsqueeze(0))
        if seq_central_t:
            seq_central_t = torch.cat(seq_central_t, 0)
            batch_seq_central_t.append(seq_central_t.unsqueeze(0))
        for id in agents_id:
            seq_local_s[id] = torch.cat(seq_local_s[id], 0)
            seq_local_a[id] = torch.cat(seq_local_a[id], 0)
            seq_local_r[id] = torch.cat(seq_local_r[id], 0)
            seq_local_t[id] = torch.cat(seq_local_t[id], 0)
            batch_seq_local_s[id].append(seq_local_s[id].unsqueeze(0))
            batch_seq_local_a[id].append(seq_local_a[id].unsqueeze(0))
            batch_seq_local_r[id].append(seq_local_r[id].unsqueeze(0))
            batch_seq_local_t[id].append(seq_local_t[id].unsqueeze(0))
    if batch_seq_central_s:
        batch_seq_central_s = torch.cat(batch_seq_central_s, 0)
    if batch_seq_central_a:
        batch_seq_central_a = torch.cat(batch_seq_central_a, 0)
    if batch_seq_central_r:
        batch_seq_central_r = torch.cat(batch_seq_central_r, 0)
    if batch_seq_central_t:
        batch_seq_central_t = torch.cat(batch_seq_central_t, 0)
    for id in agents_id:
        batch_seq_local_s[id] = torch.cat(batch_seq_local_s[id], 0)
        batch_seq_local_a[id] = torch.cat(batch_seq_local_a[id], 0)
        batch_seq_local_r[id] = torch.cat(batch_seq_local_r[id], 0)
        batch_seq_local_t[id] = torch.cat(batch_seq_local_t[id], 0)
    traj_tuple = TrajectoryTuple(
        states={
            "central": batch_seq_central_s,
            "local": batch_seq_local_s
        },
        actions={
            "central": batch_seq_central_a,
            "local": batch_seq_local_a
        },
        rewards={
            "central": batch_seq_central_r,
            "local": batch_seq_local_r,
        },
        terminals={
            "central": batch_seq_central_t,
            "local": batch_seq_local_t
        },
    )
    return traj_tuple


def to_tensor_for_trajectory(batch_data: List[TrajectoryTuple]):
    def np_to_tensor(data: TrajectoryTuple):
        states = map(
            lambda s: torch.tensor(s, dtype=torch.float).unsqueeze(0),
            data.states,
        )
        actions = map(lambda a: torch.tensor(a, dtype=torch.long).view(-1, 1),
                      data.actions)
        rewards = map(lambda r: torch.tensor(r, dtype=torch.float).view(-1, 1),
                      data.rewards)
        return Trajectory(
            states=list(states),
            actions=list(actions),
            rewards=list(rewards),
            terminal=data.terminal,
        )

    batch_data = list(map(np_to_tensor, batch_data))
    states = []
    actions = []
    rewards = []
    for data in batch_data:
        # cated shape is 1 * seq_len * data_space
        seq_s = torch.cat(data.states, 0).unsqueeze(0)
        seq_a = torch.cat(data.actions, 0).unsqueeze(0)
        seq_r = torch.cat(data.rewards, 0).unsqueeze(0)
        states.append(seq_s)
        actions.append(seq_a)
        rewards.append(seq_r)

    # why not cat here because they may have not equal length
    return states, actions, rewards


def compute_reward_to_go(rewards: torch.tensor, device=None):
    # reward shape is : batch * seq_len * 1
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight = torch.triu(
        torch.ones(
            (rewards.shape[0], rewards.shape[1], rewards.shape[1]))).to(device)
    rtg = weight.matmul(rewards)

    return rtg