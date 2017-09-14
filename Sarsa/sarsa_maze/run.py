
from maze_env import Maze
from RL_brain import SarsaTable


def update():
    for episode in range(100):
        # initial observation (1,1)
        observation = env.reset()

        # !!! RL choose action based on observation (state); Sarsa choose first
        action = RL.choose_action(str(observation))

        while True:
            # fresh env
            env.render()

            # RL take action and get next observation and reward
            # '_' means next state
            observation_, reward, done = env.step(action)

            # !!! RL choose action based on NEXT observation (state)
            action_ = RL.choose_action(str(observation_))

            # RL learn from this transition (s, a, r, s, a) ==> Sarsa
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # swap observation
            observation = observation_

            # SWAP ACTION
            action = action_

            if done:
                break

    print('over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
