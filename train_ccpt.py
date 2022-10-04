from agents.PPO import PPO
from architectures.playtesting_arch import *
from runner.runner import Runner
from runner.parallel_runner import Runner as ParallelRunner
from motivation.random_network_distillation import RND
import os
import tensorflow as tf
import argparse
from envs.unity_env import PlayTestEnvironment
from reward_model.reward_model import GAIL

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Parse arguments for training
parser = argparse.ArgumentParser()
parser.add_argument('-mn', '--model-name', help="The name of the model", default='playtest')
parser.add_argument('-gn', '--game-name', help="The name of the game", default="games/playtest_env")
parser.add_argument('-wk', '--work-id', help="Work id for parallel training", default=0)
parser.add_argument('-sf', '--save-frequency', help="How mane episodes after save the model", default=3000)
parser.add_argument('-lg', '--logging', help="How many episodes after logging statistics", default=100)
parser.add_argument('-mt', '--max-timesteps', help="Max timestep per episode", default=200)
parser.add_argument('-pl', '--parallel', dest='parallel', action='store_true')

# Parse arguments for GAIL
parser.add_argument('-irl', '--inverse-reinforcement-learning', dest='use_reward_model', action='store_true')
parser.add_argument('-rf', '--reward-frequency', help="How many episode before update the reward model", default=1)
parser.add_argument('-rm', '--reward-model', help="The name of the reward model", default='vaffanculo_6000')
parser.add_argument('-dn', '--dems-name', help="The name of the demonstrations file", default='dem_playtest_3.pkl')

# Parse arguments for Intrinsic Motivation
parser.add_argument('-m', '--motivation', dest='use_motivation', action='store_true')

parser.set_defaults(use_reward_model=False)
parser.set_defaults(recurrent=False)
parser.set_defaults(parallel=False)
parser.set_defaults(use_motivation=False)

args = parser.parse_args()

eps = 1e-12

def callback(agent, env, runner):

    global last_key
    save_frequency = 100

    if runner.ep % save_frequency == 0:
        if isinstance(env, list):

            trajectories_for_episode = dict()
            actions_for_episode = dict()

            for e in env:
                for traj, acts in zip(e.trajectories_for_episode.values(), e.actions_for_episode.values()):
                    trajectories_for_episode[last_key] = traj
                    actions_for_episode[last_key] = acts
                    last_key += 1
                e.clear_buffers()
            positions = 0
        else:
            positions = len(env.pos_buffer.keys())
            trajectories_for_episode = env.trajectories_for_episode
            actions_for_episode = env.actions_for_episode

        print('Coverage of points: {}'.format(positions))

        # Save the trajectories
        json_str = json.dumps(trajectories_for_episode, cls=NumpyEncoder)
        f = open("arrays/{}/{}_trajectories_{}.json".format(model_name, model_name, runner.ep), "w")
        f.write(json_str)
        f.close()

        # Save the actions
        json_str = json.dumps(actions_for_episode, cls=NumpyEncoder)
        f = open("arrays/{}/{}_actions_{}.json".format(model_name, model_name, runner.ep), "w")
        f.write(json_str)
        f.close()

        del trajectories_for_episode
        del actions_for_episode


if __name__ == "__main__":

    # Algorithm arguments
    model_name = args.model_name
    game_name = args.game_name
    work_id = int(args.work_id)
    save_frequency = int(args.save_frequency)
    logging = int(args.logging)
    max_episode_timestep = int(args.max_timesteps)
    parallel = args.parallel
    # IRL
    use_reward_model = args.use_reward_model
    reward_model_name = args.reward_model
    dems_name = args.dems_name
    reward_frequency = int(args.reward_frequency)

    # Central buffer for parallel execution
    if parallel:
        last_key = 0
        if os.path.exists('arrays/{}'.format(model_name)):
            os.rmdir('arrays/{}'.format(model_name))
        os.makedirs('arrays/{}'.format(model_name))


    # Curriculum structure; here you can specify also the agent statistics (ATK, DES, DEF and HP)
    curriculum = {
        'current_step': 0,
        "thresholds": [3000, 3000],
        "parameters": {
            "agent_spawn_x": [0, 0, 0],
            "agent_spawn_z": [0, 0, 0],
            "agent_spawn_y": [1.7, 1.7, 1.7],
            "win_weight": [[0.5], [0.5], [0.5]],
            #"reward_weights": [[0, 0, 0.3, 0.5, 0.8, 1, 1], [0, 0, 0.3, 0.5, 0.8, 1, 1],
            #                   [0, 0, 0.3, 0.5, 0.8, 1, 1]],
            "reward_weights": [[0], [0],
                               [0]],
            "goal_area": [1, 1, 1]
        }
    }

    # Total episode of training
    total_episode = 1e10
    # Units of training (episodes or timesteps)
    frequency_mode = 'episodes'
    # Frequency of training (in episode)
    frequency = 10
    # Memory of the agent (in episode)
    memory = 10

    # Create agent
    graph = tf.compat.v1.Graph()
    with graph.as_default():
        tf.compat.v1.disable_eager_execution()
        sess = tf.compat.v1.Session(graph=graph)
        agent = PPO(sess, input_spec=input_spec, network_spec=network_spec, obs_to_state=obs_to_state, batch_fraction=0.2,
                    action_type='discrete', action_size=10, model_name=model_name, p_lr=7e-5, v_batch_fraction=0.2,
                    v_num_itr=10, memory=memory, c2=0.1,
                    v_lr=7e-5, recurrent=args.recurrent, frequency_mode=frequency_mode, distribution='gaussian',
                    p_num_itr=10, with_circular=True)
        # Initialize variables of models
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

    # If we use intrinsic motivation, create the model
    motivation = None
    if args.use_motivation:
        with graph.as_default():
            tf.compat.v1.disable_eager_execution()
            motivation_sess = tf.compat.v1.Session(graph=graph)
            motivation = RND(motivation_sess, input_spec=input_spec_rnd, network_spec_predictor=network_spec_rnd_predictor,
                             network_spec_target=network_spec_rnd_target, lr=7e-5,
                             obs_to_state=obs_to_state_rnd, num_itr=30, motivation_weight=1)
            init = tf.compat.v1.global_variables_initializer()
            motivation_sess.run(init)

    # If we use IRL, create the reward model
    reward_model = None
    if args.use_reward_model:
        with graph.as_default():
            tf.compat.v1.disable_eager_execution()
            reward_sess = tf.compat.v1.Session(graph=graph)
            reward_model = GAIL(input_architecture=input_spec_irl, network_architecture=network_spec_irl,
                                obs_to_state=obs_to_state_irl, actions_size=9, policy=agent, sess=reward_sess, lr=7e-5,
                                name=model_name, fixed_reward_model=False, with_action=True, reward_model_weight=1)
            init = tf.compat.v1.global_variables_initializer()
            reward_sess.run(init)


    # Open the environment with all the desired flags
    if not parallel:
        # Open the environment with all the desired flags
        env = PlayTestEnvironment(game_name=game_name, no_graphics=True, worker_id=work_id,
                             max_episode_timesteps=max_episode_timestep)
    else:
        # If parallel, create more environments
        envs = []
        for i in range(10):
            envs.append(PlayTestEnvironment(game_name=game_name, no_graphics=True, worker_id=work_id + i,
                                       max_episode_timesteps=max_episode_timestep))

    # Create runner
    if not parallel:
        runner = Runner(agent=agent, frequency=frequency, env=env, save_frequency=save_frequency,
                        logging=logging, total_episode=total_episode, curriculum=curriculum,
                        frequency_mode=frequency_mode, curriculum_mode='episodes', callback_function=callback,
                        reward_model=reward_model, reward_frequency=reward_frequency, dems_name=dems_name,
                        motivation=motivation)
    else:
        runner = ParallelRunner(agent=agent, frequency=frequency, envs=envs, save_frequency=save_frequency,
                                logging=logging, total_episode=total_episode, curriculum=curriculum,
                                frequency_mode=frequency_mode, curriculum_mode='episodes', callback_function=callback,
                                reward_model=reward_model, reward_frequency=reward_frequency, dems_name=dems_name,
                                motivation=motivation)

    try:
        runner.run()
    finally:
        if not parallel:
            env.close()
        else:
            for env in envs:
                env.close()
