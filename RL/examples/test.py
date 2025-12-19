from envrionment import BallEnv
from dm_control import viewer
from pid import PIDPolicy

if __name__ == "__main__":
    env = BallEnv()
    policy = PIDPolicy(env)

    viewer.launch(environment_loader=env, policy=policy, title="UR5e Ball Balancing", width=1024, height=768)
