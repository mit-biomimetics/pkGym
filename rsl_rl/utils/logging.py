import torch

def update_infos_with_episode_sums(infos,
                                   episode_sums,
                                   dones,
                                   max_episode_length):

    for key, val in episode_sums.items():
        infos['episode'][key] = torch.mean(episode_sums[key][dones]) \
                                        / max_episode_length
        episode_sums[key][dones] = 0.
