import numpy as np
import dm_env
from dm_env import specs
from dm_control import mujoco, viewer
import time

class BallEnv(dm_env.Environment):
    def __init__(self):
        self.physics = mujoco.Physics.from_xml_path("./ur5e/scene.xml")
        nu = self.physics.model.nu
        self._action_spec = specs.BoundedArray(
            shape=(nu,), dtype=np.float32,
            minimum=-np.ones(nu, dtype=np.float32),
            maximum=np.ones(nu, dtype=np.float32),
            name="ctrl"
        )
        self._obs_spec = {
            "ball_pos": specs.Array(shape=(3,), dtype=np.float32, name="ball_pos")
        }

    def _get_obs(self):
        ball_pos = self.physics.data.xpos[self.physics.model.name2id("ball", "body")].copy()
        return {"ball_pos": ball_pos.astype(np.float32)}
    
    def render_image(self, camera_name="tracking_cam"):
        print(f"Rendering from camera: {camera_name}")
        try:
            camera_id = self.physics.model.name2id(camera_name, "camera")
        except:
            print(f"Camera '{camera_name}' not found, using default.")
            camera_id = -1 

        pixels = self.physics.render(height=240, width=320, camera_id=camera_id)
        return pixels

    def reset(self):
        self.physics.reset()
        obs = self._get_obs()
        return dm_env.restart(obs)

    def step(self, action):
        self.physics.data.ctrl[:] = np.asarray(action, dtype=np.float32)
        self.physics.step()
        obs = self._get_obs()
        reward = 0.0
        return dm_env.transition(reward=reward, observation=obs)

    def action_spec(self): 
        return self._action_spec

    def observation_spec(self): 
        return self._obs_spec
