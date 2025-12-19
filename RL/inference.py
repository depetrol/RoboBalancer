import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo
from ur5e_env import UR5eBounceEnv
import time
import os
import shutil

def main():
    video_dir = "outputs/videos"
    collect_episode = 5

    os.makedirs(video_dir, exist_ok=True)

    env = UR5eBounceEnv(render_mode="rgb_array")
    
    env = RecordVideo(
        env,
        video_folder=video_dir,
        name_prefix="ur5e_rollout",
        episode_trigger=lambda x: True 
    )

    models_dir = "outputs/ppo"
    model_path = f"{models_dir}/ppo_ur5e_pingpong_final"
    
    if not os.path.exists(f"{model_path}.zip"):
        print(f"Error: Model file '{model_path}.zip' not found.")
        print("Please run 'train.py' first to train and save the agent.")
        return

    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    print(f"Model loaded. Recording video to {video_dir}...")

    obs, _ = env.reset()
    
    try:
        episodes = 0
        while True:
            action, _states = model.predict(obs, deterministic=True)

            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                print("Ball dropped! Episode finished. Video saved.")
                obs, _ = env.reset()
                episodes += 1
                if episodes >= collect_episode:
                    break

    except KeyboardInterrupt:
        print("\nInference stopped by user.")
    finally:
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    main()
