import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from gymnasium.wrappers import RecordVideo
import os
import multiprocessing

from ur5e_env import UR5eBounceEnv

class VideoRecorderCallback(BaseCallback):
    def __init__(self, save_freq, video_dir):
        super().__init__()
        self.save_freq = save_freq
        self.video_dir = video_dir
        self.last_save = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_save >= self.save_freq:
            self.last_save = self.num_timesteps
            
            record_env = UR5eBounceEnv(render_mode="rgb_array")

            record_env = RecordVideo(
                record_env,
                video_folder=self.video_dir,
                name_prefix="rollout",
                episode_trigger=lambda x: x == 0,
                disable_logger=True
            )

            obs, _ = record_env.reset()
            done = False
            truncated = False
            
            while not (done or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, truncated, _ = record_env.step(action)
            
            record_env.close()
            print(f"Recorded video at {self.num_timesteps} steps.")
            
        return True

def main():
    models_dir = "outputs/ppo"
    log_dir = "outputs/logs"
    video_dir = "outputs/training_videos"
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    num_cpu = multiprocessing.cpu_count()
    print(f"Detected {num_cpu} CPUs. Launching parallel environments...")

    env = make_vec_env(
        UR5eBounceEnv, 
        n_envs=num_cpu, 
        seed=0, 
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"render_mode": None}
    )

    final_model_path = f"{models_dir}/ppo_ur5e_pingpong_final.zip"
    model_name = "ppo_ur5e_pingpong_final"

    if os.path.exists(final_model_path):
        print(f"Loading existing model from {final_model_path}...")
        model = PPO.load(final_model_path, env=env, tensorboard_log=log_dir)
    else:
        print("No existing model found. Creating new model...")
        model = PPO("MlpPolicy", env, verbose=1, 
                    learning_rate=3e-4, 
                    n_steps=2048,
                    batch_size=128 * num_cpu, 
                    ent_coef=0.01,
                    tensorboard_log=log_dir,
                    device="auto")          

    checkpoint_callback = CheckpointCallback(
        save_freq=1_000_000, 
        save_path=models_dir,
        name_prefix="ur5e_bounce"
    )

    video_callback = VideoRecorderCallback(
        save_freq=1_000_000,
        video_dir=video_dir
    )

    print("Starting parallel training...")
    try:
        model.learn(total_timesteps=120_000_000, 
                    callback=[checkpoint_callback, video_callback], 
                    progress_bar=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current progress...")
    finally:
        model.save(f"{models_dir}/{model_name}")
        print(f"Model saved to {models_dir}/{model_name}")
        env.close()

if __name__ == "__main__":
    main()
