if __name__ == '__main__':
    from glow.benchmark.paper import *
    from glow.benchmark.run import run_segment

    # piggy-back off of existing configs (ensure consistent params)
    config_list = [Config(label='segment_hcp',
                          source='hcp',
                          **(common_dict | hcp_dict)),
                   Config(label='segment_wgn',
                          source='wgn',
                          **(common_dict | wgn_dict))]
    for config in config_list:
        config.run_all(run_fnc=run_segment, verbose=True)
