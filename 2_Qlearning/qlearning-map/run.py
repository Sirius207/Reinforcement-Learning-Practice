
from maze_env import Maze
from RL_brain import QLearningTable


def update():
    for episode in range(100):
        # initial observation (1,1)
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation (state)
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            # '_' means next state
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_

            if done:
                break

    print('over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
