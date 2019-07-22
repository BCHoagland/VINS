import pickle
import charles

class Config:
    env = 'Hopper-v2'
    actors = 4
    lr = 3e-4
    max_timesteps = 1e5
    trajectory_length = 1
    vis_iter = 500
    storage_size = 1000000
    batch_size = 128
    epochs = 1
    explore_steps = 10000

    testing_steps = 10000

def expert_demos():
    try:
        with open('.tmp/demos', 'rb') as f:
            demos = pickle.load(f)
            print('Loaded expert demos')
    except:
        print('Training expert')
        agent = charles.Agent(charles.TD3, Config)
        agent.train()

        print('Creating expert demos')
        demos = agent.demo()
        with open('.tmp/demos', 'wb') as f:
            pickle.dump(demos, f)
            print('Saved and loaded expert demos')

    return demos
