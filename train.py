#branch with variable device name
import argparse
import gym
import numpy as np
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tempfile
import time

import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.common.misc_util import (
    boolean_flag,
    pickle_load,
    pretty_eta,
    relatively_safe_pickle_dump,
    set_global_seeds,
    RunningAvg,
    SimpleMonitor
)
from baselines.common.schedules import LinearSchedule, PiecewiseSchedule
# when updating this to non-deperecated ones, it is important to
# copy over LazyFrames
from baselines.common.atari_wrappers_deprecated import wrap_dqn
from baselines.common.azure_utils import Container
from model import model, dueling_model, bootstrap_model


def parse_args():
    parser = argparse.ArgumentParser("DQN experiments for Atari games")
    # Environment
    parser.add_argument("--env", type=str, default="Seaquest", help="name of the game")
    parser.add_argument("--seed", type=int, default=42, help="which seed to use")
    parser.add_argument("--device_name", type=str, default='/device:CPU:0', help="CPU or GPU name")
    #parser.add_argument("--device_name", type=str, default='kwaa', help="CPU or GPU name")


    # Core DQN parameters
    parser.add_argument("--replay-buffer-size", type=int, default=int(1e6), help="replay buffer size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for Adam optimizer")
    parser.add_argument("--num-steps", type=int, default=int(4e7), help="total number of steps to run the environment for")
    parser.add_argument("--batch-size", type=int, default=32, help="number of transitions to optimize at the same time")
    parser.add_argument("--learning-freq", type=int, default=4, help="number of iterations between every optimization step")
    parser.add_argument("--target-update-freq", type=int, default=10000, help="number of iterations between every target network update")
    # Bells and whistles
    boolean_flag(parser, "double-q", default=True, help="whether or not to use double q learning")
    boolean_flag(parser, "dueling", default=False, help="whether or not to use dueling model")
    boolean_flag(parser, "bootstrap", default=True, help="whether or not to use bootstrap model")
    boolean_flag(parser, "prioritized", default=False, help="whether or not to use prioritized replay buffer")
    parser.add_argument("--prioritized-alpha", type=float, default=0.6, help="alpha parameter for prioritized replay buffer")
    parser.add_argument("--prioritized-beta0", type=float, default=0.4, help="initial value of beta parameters for prioritized replay")
    parser.add_argument("--prioritized-eps", type=float, default=1e-6, help="eps parameter for prioritized replay buffer")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./logs", help="directory in which training state and model should be saved.")
    parser.add_argument("--save-azure-container", type=str, default=None,
                        help="It present data will saved/loaded from Azure. Should be in format ACCOUNT_NAME:ACCOUNT_KEY:CONTAINER")
    parser.add_argument("--save-freq", type=int, default=1e6, help="save model once every time this many iterations are completed")
    boolean_flag(parser, "load-on-start", default=True, help="if true and model was previously saved then training will be resumed")
    return parser.parse_args()


def make_env(game_name):
    env = gym.make(game_name + "NoFrameskip-v4")  # Already performs a frame-skip of 4 @ baselines.common.atari_wrappers_deprecated
    monitored_env = SimpleMonitor(env)  # puts rewards and number of steps in info, before environment is wrapped
    env = wrap_dqn(monitored_env)  # applies a bunch of modification to simplify the observation space (downsample, make b/w)
    return env, monitored_env


def maybe_save_model(savedir, container, state, rewards, steps):
    """This function checkpoints the model and state of the training algorithm."""
    if savedir is None:
        return
    start_time = time.time()
    model_dir = "model-{}".format(state["num_iters"])
    U.save_state(os.path.join(savedir, model_dir, "saved"))
    if container is not None:
        container.put(os.path.join(savedir, model_dir), model_dir)
    relatively_safe_pickle_dump(state, os.path.join(savedir, 'training_state.pkl.zip'), compression=True)
    if container is not None:
        container.put(os.path.join(savedir, 'training_state.pkl.zip'), 'training_state.pkl.zip')
    relatively_safe_pickle_dump(state["monitor_state"], os.path.join(savedir, 'monitor_state.pkl'))
    if container is not None:
        container.put(os.path.join(savedir, 'monitor_state.pkl'), 'monitor_state.pkl')
    relatively_safe_pickle_dump(rewards, os.path.join(savedir, 'rewards.pkl'))
    if container is not None:
        container.put(os.path.join(savedir, 'rewards.pkl'), 'rewards.pkl')
    relatively_safe_pickle_dump(steps, os.path.join(savedir, 'steps.pkl'))
    if container is not None:
        container.put(os.path.join(savedir, 'steps.pkl'), 'steps.pkl')
    logger.log("Saved model in {} seconds\n".format(time.time() - start_time))


def maybe_load_model(savedir, container):
    """Load model if present at the specified path."""
    if savedir is None:
        return

    state_path = os.path.join(os.path.join(savedir, 'training_state.pkl.zip'))
    if container is not None:
        logger.log("Attempting to download model from Azure")
        found_model = container.get(savedir, 'training_state.pkl.zip')
    else:
        found_model = os.path.exists(state_path)
    if found_model:
        state = pickle_load(state_path, compression=True)
        model_dir = "model-{}".format(state["num_iters"])
        if container is not None:
            container.get(savedir, model_dir)
        U.load_state(os.path.join(savedir, model_dir, "saved"))
        logger.log("Loaded models checkpoint at {} iterations".format(state["num_iters"]))
        return state


if __name__ == '__main__':
    args = parse_args()
    print('device_name: ' +args.device_name)
    # Parse savedir and azure container.
    savedir = args.save_dir + "_" + args.env
    if args.save_azure_container is not None:
        account_name, account_key, container_name = args.save_azure_container.split(":")
        container = Container(account_name=account_name,
                              account_key=account_key,
                              container_name=container_name,
                              maybe_create=True)
        if savedir is None:
            # Careful! This will not get cleaned up. Docker spoils the developers.
            savedir = tempfile.TemporaryDirectory().name
    else:
        container = None
    # Create and seed the env.
    env, monitored_env = make_env(args.env)
    if args.seed > 0:
        set_global_seeds(args.seed)
        env.unwrapped.seed(args.seed)

    with U.make_session(120) as sess:
        # Create training graph and replay buffer
        if args.bootstrap :
            act, train, update_target, debug = deepq.build_train(
                make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape, name=name),
                q_func=bootstrap_model,
                bootstrap=args.bootstrap,
                num_actions=env.action_space.n,
                optimizer=tf.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-4),
                gamma=0.99,
                grad_norm_clipping=10,
                double_q=args.double_q,
                device_name=args.device_name
            )
        else:
            act, train, update_target, debug = deepq.build_train(
                make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape, name=name),
                q_func=dueling_model if args.dueling else model,
                num_actions=env.action_space.n,
                optimizer=tf.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-4),
                gamma=0.99,
                grad_norm_clipping=10,
                double_q=args.double_q,
                device_name=args.device_name

            )

        approximate_num_iters = args.num_steps / 4
        exploration = PiecewiseSchedule([
            (0, 1.0),
            (1e6 / 4, 0.1), # (approximate_num_iters / 50, 0.1),
            # (5e6 / 4, 0.01) # (approximate_num_iters / 5, 0.01)
        ], outside_value=0.01)
        learning_rate = PiecewiseSchedule([
            (0, 1e-4),
            (1e6 / 4, 1e-4), # (approximate_num_iters / 50, 0.1),
            (5e6 / 4, 5e-5) # (approximate_num_iters / 5, 0.01)
        ], outside_value=5e-5)

        if args.prioritized:
            replay_buffer = PrioritizedReplayBuffer(args.replay_buffer_size, args.prioritized_alpha)
            beta_schedule = LinearSchedule(approximate_num_iters, initial_p=args.prioritized_beta0, final_p=1.0)
        else:
            replay_buffer = ReplayBuffer(args.replay_buffer_size)

        U.initialize()
        update_target()
        num_iters = 0

        # Load the model
        state = maybe_load_model(savedir, container)
        if state is not None:
            num_iters, replay_buffer = state["num_iters"], state["replay_buffer"],
            monitored_env.set_state(state["monitor_state"])

        start_time, start_steps = None, None
        steps_per_iter = RunningAvg(0.999)
        iteration_time_est = RunningAvg(0.999)
        obs = env.reset()

        # Main training loop
        head = np.random.randint(10)        #Initial head initialisation
        while True:
            num_iters += 1
            # Take action and store transition in the replay buffer.
            if args.bootstrap:
                action = act(np.array(obs)[None], head=head, update_eps=exploration.value(num_iters))[0]
            else:
                action = act(np.array(obs)[None], update_eps=exploration.value(num_iters))[0]
            new_obs, rew, done, info = env.step(action)
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs
            if done:
                obs = env.reset()
                head = np.random.randint(10)

            if (num_iters > max(0.5 * args.batch_size, args.replay_buffer_size // 20) and
                    num_iters % args.learning_freq == 0):
                # Sample a bunch of transitions from replay buffer
                if args.prioritized:
                    experience = replay_buffer.sample(args.batch_size, beta=beta_schedule.value(num_iters))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(args.batch_size)
                    weights = np.ones_like(rewards)
                # Minimize the error in Bellman's equation and compute TD-error
                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights, learning_rate.value(num_iters))
                # Update the priorities in the replay buffer
                if args.prioritized:
                    new_priorities = np.abs(td_errors) + args.prioritized_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)
            # Update target network.
            if num_iters % args.target_update_freq == 0:
                update_target()

            if start_time is not None:
                steps_per_iter.update(info['steps'] - start_steps)
                iteration_time_est.update(time.time() - start_time)
            start_time, start_steps = time.time(), info["steps"]

            # Save the model and training state.
            if num_iters > 0 and (num_iters % args.save_freq == 0 or info["steps"] > args.num_steps):
                maybe_save_model(savedir, container, {
                    'replay_buffer': replay_buffer,
                    'num_iters': num_iters,
                    'monitor_state': monitored_env.get_state()
                }, info["rewards"], info["steps"])

            if info["steps"] > args.num_steps:
                break

            if done:
                steps_left = args.num_steps - info["steps"]
                completion = np.round(info["steps"] / args.num_steps, 1)

                logger.record_tabular("% completion", completion)
                logger.record_tabular("steps", info["steps"])
                logger.record_tabular("iters", num_iters)
                logger.record_tabular("episodes", len(info["rewards"]))
                logger.record_tabular("reward (100 epi mean)", np.mean(info["rewards"][-100:]))
                logger.record_tabular("head for episode", (head+1))
                logger.record_tabular("exploration", exploration.value(num_iters))
                if args.prioritized:
                    logger.record_tabular("max priority", replay_buffer._max_priority)
                fps_estimate = (float(steps_per_iter) / (float(iteration_time_est) + 1e-6)
                                if steps_per_iter._value is not None else "calculating...")
                logger.dump_tabular()
                logger.log()
                logger.log("ETA: " + pretty_eta(int(steps_left / fps_estimate)))
                logger.log()
