import gymnasium as gym
import numpy as np
import mujoco
import os

class UR5eBounceEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        
        xml_path = "./ur5e/scene.xml" 
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        self.action_scale = 0.1 
        
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)
        
        self.ball_goal = np.array([-0.4, -0.3, 1])
        self.home_q = np.array([0, -2, 2.5, -2, -0.1, 1.3])

        self.dt = 0.02 
        self.episode_length = 0
        self.model.opt.timestep = 0.002 
        
        self.render_mode = render_mode
        self.viewer = None
        self.renderer = None
        self.camera_obj = None 

        if self.render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)

            self.camera_obj = mujoco.MjvCamera()
            self.camera_obj.azimuth = 90      
            self.camera_obj.elevation = -20  
            self.camera_obj.distance = 1 
            self.camera_obj.lookat = self.ball_goal + [0,0,-0.5]
        
        self.ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'ping_pong_ball')
        self.ball_joint_addr = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'ping_pong_ball')]
        self.ball_vel_addr = self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'ping_pong_ball')]
        
        self.actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) 
                             for n in ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3']]
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) 
                          for n in ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']]
        robot_body_names = [
            "base_link",
            "shoulder_link",
            "upper_arm_link",
            "forearm_link",
            "wrist_1_link",
            "wrist_2_link",
            "wrist_3_link",
            "tool_paddle",
        ]
        self.robot_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in robot_body_names
        ]
    def _has_self_collision(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2

            b1 = self.model.geom_bodyid[g1]
            b2 = self.model.geom_bodyid[g2]

            if (b1 in self.robot_body_ids and
                b2 in self.robot_body_ids and
                b1 != b2):
                return True

        return False
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.data.qpos[self.joint_ids] = self.home_q + np.random.uniform(-0.1, 0.1, 6)
        self.data.qvel[self.joint_ids] = 0
        self.data.ctrl[self.actuator_ids] = self.data.qpos[self.joint_ids]
        
        self.data.qpos[self.ball_joint_addr:self.ball_joint_addr+3] = [self.ball_goal[0] + np.random.uniform(-0.05, 0.05), self.ball_goal[1] + np.random.uniform(-0.05, 0.05), self.ball_goal[2] + np.random.uniform(-0.05, 0.05)]
        self.data.qpos[self.ball_joint_addr+3:self.ball_joint_addr+7] = [1, 0, 0, 0]
        self.data.qvel[self.ball_vel_addr:self.ball_vel_addr+6] = 0
        self.episode_length = 0
        mujoco.mj_forward(self.model, self.data)

        if self.render_mode == "rgb_array" and self.renderer and self.camera_obj:
            self.renderer.update_scene(self.data, camera=self.camera_obj)

        return self._get_obs(), {}

    def step(self, action):
        self.episode_length += 1
        scaled_action = action * self.action_scale
        
        current_q = self.data.qpos[self.joint_ids]
        target_q = current_q + scaled_action
        
        target_q = np.clip(target_q, -3.14, 3.14)
        self.data.ctrl[self.actuator_ids] = target_q
        
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        
        ball_pos = self.data.xpos[self.ball_body_id]
        paddle_pos = self.data.body("tool_paddle").xpos

        ball_vx, ball_vy, ball_vz = obs[3], obs[4], obs[5]

        terminated = False
        reward = 0.0

        goal_dist = np.linalg.norm(ball_pos[:2] - self.ball_goal[:2])
        reward -= goal_dist * 10

        dist = np.linalg.norm(ball_pos[:2] - paddle_pos[:2])

        if dist > 0.05:
            reward -= dist

        if ball_pos[2] < paddle_pos[2] - 0.05:
            reward -= 100.0
        if ball_pos[2] < 0.1:
            terminated = True
            
        if 0.2 < ball_pos[2] < 0.4:
            reward += 10
            reward -= abs(ball_vz) * 5
        
        if 0.1 < paddle_pos[2] < 0.3:
            reward += 10

        horizontal_speed = np.sqrt(ball_vx**2 + ball_vy**2)
        reward -= horizontal_speed
        if self._has_self_collision():
            terminated = True
            reward -= 200.0 
        truncated = False

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "rgb_array" and self.renderer:
            self.renderer.update_scene(self.data, camera=self.camera_obj)
            return self.renderer.render()
        elif self.render_mode == "human":
            return self._render_frame()

    def _get_obs(self):
        b_pos = self.data.xpos[self.ball_body_id]
        b_vel_6d = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.ball_body_id, b_vel_6d, 0)
        b_vel = b_vel_6d[3:]
        
        q_pos = self.data.qpos[self.joint_ids]
        q_vel = self.data.qvel[self.joint_ids]
        
        return np.concatenate([b_pos, b_vel, q_pos, q_vel]).astype(np.float32)

    def _render_frame(self):
        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()
        
    def close(self):
        if self.viewer:
            self.viewer.close()
        if self.renderer:
            self.renderer.close()
