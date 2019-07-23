import pickle
import charles

from vins.visualize import alert

def expert_demos(expert_config, filename=None):
    if filename is None: filename = expert_config.env
    path = f'.tmp/{filename}-demos'
    try:
        with open(path, 'rb') as f:
            demos = pickle.load(f)
            alert('Loading expert demos', done=True)
    except:
        alert('Getting demos')
        agent = charles.Agent(charles.TD3, expert_config)
        agent.train()

        demos = expert.demo()
        with open(path, 'wb') as f:
            pickle.dump(demos, f)
            alert('Training expert and getting demos', done=True)

    return demos
