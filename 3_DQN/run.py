from maze_env import Maze
from RL_brain import DeepQNetwork


def run_maze():
    step = 0
    for episode in range(300):
        # init
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation (state)
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            # '_' means next state
            observation_, reward, done = env.step(action)

            # DQN save memory
            RL.store_transition(observation, action, reward, observation_)

            # start learning after 200 step (prevent learning too early)
            # learn each 5 step
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            if done:
                break
            step += 1   # total step

    print('over')
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    RL = DeepQNetwork(
        env.n_actions,
        env.n_features,
        learning_rate=0.01,
        reward_decay=0.9,
        e_greedy=0.9,
        replace_target_iter=200,    # change target_net every 200 step
        memory_size=2000,           # max memory
        # output_graph=True         # out tensorboard doc
    )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()  # check nn cost curve
