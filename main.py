from pomcp import POUCT, POMCP
from generator import Histogram, random, update_histogram_belief
from particles import Particles


def test_pomcp(problem, nsteps=10) -> None:

    planner = POMCP(
        agent=problem.agent,
        max_depth=3,
        discount_factor=0.95,
        planning_time=0.5,
        c_UCB=110,
        value_init=-100,
        rollout_policy=problem.agent.policy_model,
    )

    for i in range(nsteps):  # Step 4

        print("==== Step %d ====" % (i + 1))
        print(">> True state:", problem.env.cur_state)
        # print(">> Belief:", problem.agent.cur_belief)

        action = planner.plan()
        print(">> Action:", action)

        reward = problem.env.reward_model.sample(
            problem.env.cur_state, action, problem.agent.history, None
        )
        observation = problem.agent.observation_model.sample(
            problem.env.cur_state, action
        )
        print(">> Reward:", reward)
        print(">> Observation:", observation)

        # Step 5
        # Update the belief. If the planner is POMCP, planner.update
        # also automatically updates agent belief.
        problem.agent.update_history(action, observation)
        planner.update(problem.agent, action, observation)

        if reward != -1:
            break

        problem.env.cur_state.render(history=problem.agent.history)
        print("\n")
        print(input("Press enter to continue..."))


if __name__ == "__main__":

    ######### Tiger #########

    # from envs.tiger.problem import (
    #     State_Tiger,
    #     Problem_Tiger,
    # )

    # N = 3

    # init_true_state = random.choice([State_Tiger(f"tiger-{s}", N) for s in range(N)])
    # init_belief = Histogram({State_Tiger(f"tiger-{s}", N): 1 / N for s in range(N)})
    # init_belief_particles = Particles.from_histogram(
    #     histogram=Histogram({State_Tiger(f"tiger-{s}", N): 1 / N for s in range(N)})
    # )

    # tiger_problem = Problem_Tiger(N, 0.1, init_true_state, init_belief_particles)
    # test_pomcp(tiger_problem)

    ######### Battleship #########
    from envs.battleship.types import (
        State_Battleship,
        get_random_state,
        Action_Battleship,
    )
    from envs.battleship.problem import Problem_Battleship

    init_true_state = get_random_state()

    n_particles = 1000
    init_belief = Particles([get_random_state() for _ in range(n_particles)])
    battleship_problem = Problem_Battleship(init_true_state, init_belief)

    planner = POMCP(
        agent=battleship_problem.agent,
        max_depth=3,
        discount_factor=0.95,
        num_sims=1000,
        c_UCB=110,
        value_init=-100,
        rollout_policy=battleship_problem.agent.policy_model,
    )
    action = planner.plan()
    action = Action_Battleship(battleship_problem.env.cur_state.ships[0].pos)
    reward = battleship_problem.env.reward_model.sample(
        battleship_problem.env.cur_state, action, battleship_problem.agent.history, None
    )
    observation = battleship_problem.agent.observation_model.sample(
        battleship_problem.env.cur_state, action
    )
    print(action, reward, observation)

    battleship_problem.agent.update_history(action, observation)
    planner.update(battleship_problem.agent, action, observation)
