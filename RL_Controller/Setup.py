import pybullet as p
import time
import pybullet_data
import numpy as np
import torch
from Model import (
    actor,
    critic,
    target_actor,
    target_critic,
    replay_buffer,
    train_actor_critic,
    Reward,
)
import torch.optim as optim


def simulate_lidar(robot_id, num_rays=360, lidar_range=10.0):
    """Simulate LiDAR data using ray tests in PyBullet."""
    # Get robot position and orientation
    position, orientation = p.getBasePositionAndOrientation(robot_id)

    # Generate evenly spaced rays around the robot
    angles = np.linspace(0, 2 * np.pi, num_rays)
    rays_start = []
    rays_end = []

    for angle in angles:
        start = np.array(position) + np.array([0, 0, 0.1])  # Slightly above ground
        end = start + lidar_range * np.array([np.cos(angle), np.sin(angle), 0])
        rays_start.append(start)
        rays_end.append(end)

    # Perform ray tests
    ray_results = p.rayTestBatch(rays_start, rays_end)

    # Extract distances
    distances = []
    for result in ray_results:
        hit_fraction = result[2]  # Fraction of the ray length where hit occurred
        distances.append(hit_fraction * lidar_range)

    return distances


def get_camera_image(robot_id, width=320, height=240):
    """Capture a camera image from the robot's perspective."""
    # Get robot position and orientation
    position, orientation = p.getBasePositionAndOrientation(robot_id)
    orientation_matrix = p.getMatrixFromQuaternion(orientation)

    # Set camera parameters
    camera_target = [
        position[0] + orientation_matrix[0],
        position[1] + orientation_matrix[3],
        position[2] + 0.5,
    ]  # Point in front of the robot
    camera_up_vector = [0, 0, 1]

    view_matrix = p.computeViewMatrix(position, camera_target, camera_up_vector)
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=width / height, nearVal=0.01, farVal=100
    )

    # Get image
    width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
        width, height, viewMatrix=view_matrix, projectionMatrix=projection_matrix
    )
    return rgb_img, depth_img, seg_img


# Connect to PyBullet
physicsClient = p.connect(p.GUI)

# Set up environment and load URDFs
p.setAdditionalSearchPath(
    pybullet_data.getDataPath()
)  # Optionally set a search path for plane
planeId = p.loadURDF("plane.urdf")  # Load a plane for the robot to drive on

GoalX = 48
GoalY = 28
GoalLocation = [GoalX, GoalY]
# Path to your TurtleBot URDF file
# turtlebot_path = "turtlebot.urdf"
# turtlebot = p.loadURDF(turtlebot_path, [2.5, 2.5, 0.5])
block1 = p.loadURDF("block.urdf", [27, 5, 2.5])
block2 = p.loadURDF("block.urdf", [14, 12, 2.5])
block3 = p.loadURDF("block.urdf", [40, 20, 2.5])
block4 = p.loadURDF("block.urdf", [15, 25, 2.5])
goalBlock = p.loadURDF("goalBlock.urdf", [GoalX, GoalY, 1])
wall_1 = p.loadURDF(
    "longWall.urdf", [25, 30.5, 1.25], p.getQuaternionFromEuler([0, 0, 0])
)  # Back wall
wall_2 = p.loadURDF(
    "longWall.urdf", [25, -0.5, 1.25], p.getQuaternionFromEuler([0, 0, 0])
)  # Back wall
wall_3 = p.loadURDF(
    "shortWall.urdf", [0.5, 15, 1.25], p.getQuaternionFromEuler([0, 0, 1.57])
)  # Right wall\
wall_4 = p.loadURDF(
    "shortWall.urdf", [50.5, 15, 1.25], p.getQuaternionFromEuler([0, 0, 1.57])
)  # Right wall\

object_names = {
    block1: "Block 1",
    block2: "Block 2",
    block3: "Block 3",
    block4: "Block 4",
    goalBlock: "Goal Block",
    wall_1: "Wall 1",
    wall_2: "Wall 2",
    wall_3: "Wall 3",
    wall_4: "Wall 4",
}

object_pos = {
    block1: [[27, 5, 2.5], [0, 0, 0, 1]],
    block2: [[27, 5, 2.5], [0, 0, 0, 1]],
    block3: [[40, 20, 2.5], [0, 0, 0, 1]],
    block4: [[15, 25, 2.5], [0, 0, 0, 1]],
    goalBlock: [GoalX, GoalY, 1],
    wall_1: [[25, 30.5, 1.25], p.getQuaternionFromEuler([0, 0, 0])],
    wall_2: [[25, -0.5, 1.25], p.getQuaternionFromEuler([0, 0, 0])],
    wall_3: [[0.5, 15, 1.25], p.getQuaternionFromEuler([0, 0, 1.57])],
    wall_4: [[50.5, 15, 1.25], p.getQuaternionFromEuler([0, 0, 1.57])],
}
camera_distance = 20  # Zoom level (increase or decrease as needed)
camera_yaw = 0  # Horizontal rotation (0 to look straight down)
camera_pitch = -89.9  # Vertical angle, -90 degrees for top-down
camera_target = [25, 15, 0]  # Center the camera on the environment

p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target)

p.stepSimulation()
# Set up simulation properties
p.setGravity(0, 0, -9.8)
p.setRealTimeSimulation(1)

# Define wheel joint indices (update these based on your URDF joint order)
left_wheel_joint = 1
right_wheel_joint = 2


# Simple control loop to move the robot
# while True:
# Get robot's position and orientation
def States(turtlebot):
    position, orientation = p.getBasePositionAndOrientation(turtlebot)
    euler_orientation = p.getEulerFromQuaternion(
        orientation
    )  # Convert to Euler angles for better readability
    goal_distance = np.linalg.norm(np.array(position[:2]) - np.array(GoalLocation[:2]))
    if -0.5 < goal_distance < 0.5:
        goal_distance = 0
    closest_distance = float("inf")
    for obstacle in [block1, block2, block3, block4, wall_1, wall_2, wall_3, wall_4]:
        closest_points = p.getClosestPoints(
            turtlebot, obstacle, distance=100.0
        )  # Large search distance
        if closest_points:
            closest_distance = min(
                closest_distance, closest_points[0][8]
            )  # Closest point distance is at index 8

    return np.array(
        [position[0], position[1], orientation[2], closest_distance, goal_distance]
    )

    # TODO Pass the model (the x,y,yaw,distance to object,distance to goal)
    # TODO get the wheel velocitys from model and set the wheels to it.

    contact_points = p.getContactPoints(bodyA=turtlebot)
    for contact in contact_points:
        body_b = contact[2]  # ID of the other object in contact
        if body_b in object_names:
            print("Collision detected with:", object_names[body_b])
            p.disconnect()
    time.sleep(1.0 / 2000.0)  # Adjust simulation step rate


def run_simulation_and_train(
    actor,
    critic,
    target_actor,
    target_critic,
    replay_buffer,
    num_episodes=1000,
    max_time_steps=50,
    batch_size=64,
):
    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

    for episode in range(num_episodes):
        # Reset environment (replace with actual simulation setup)
        # turtlebot = p.loadURDF(turtlebot_path, [2.5, 2.5, 0.5])
        State = States(turtlebot)
        total_reward = 0
        print(episode)
        break_flag = 0

        for t in range(max_time_steps):
            state = torch.tensor(State, dtype=torch.float32).unsqueeze(0)
            action = actor(state).detach().numpy()[0]  # Predict action
            # action = np.clip(action, 0, 10)  # Clip actions to valid range
            p.setJointMotorControl2(
                turtlebot,
                left_wheel_joint,
                p.VELOCITY_CONTROL,
                targetVelocity=action[0],
            )
            p.setJointMotorControl2(
                turtlebot,
                right_wheel_joint,
                p.VELOCITY_CONTROL,
                targetVelocity=action[1],
            )

            contact_points = p.getContactPoints(bodyA=turtlebot)
            for contact in contact_points:
                body_b = contact[2]  # ID of the other object in contact
                if body_b in object_names:
                    next_state = States(turtlebot)
                    print(next_state)
                    print("Collision detected with:", object_names[body_b])
                    p.setJointMotorControl2(
                        turtlebot,
                        left_wheel_joint,
                        p.VELOCITY_CONTROL,
                        targetVelocity=0,
                    )
                    p.setJointMotorControl2(
                        turtlebot,
                        right_wheel_joint,
                        p.VELOCITY_CONTROL,
                        targetVelocity=0,
                    )
                    p.resetBasePositionAndOrientation(
                        turtlebot, [2.5, 2.5, 0.5], [0, 0, 0, 1]
                    )
                    p.resetBasePositionAndOrientation(
                        body_b, object_pos[body_b][0], object_pos[body_b][1]
                    )
                    d_obs_St, d_obs_St1 = State[3], next_state[3]
                    d_goal_St, d_goal_St1 = State[4], next_state[4]
                    reward, done = Reward(d_obs_St, 0, d_goal_St, d_goal_St1, t)
                    total_reward += reward

                    # Store transition
                    replay_buffer.add((state, action, reward, next_state, done))
                    State = next_state
                    break_flag = 1

            if break_flag:
                break

            time.sleep(1)
            # Simulate environment step (replace with actual environment step)
            next_state = States(turtlebot)  # Dummy next state
            d_obs_St, d_obs_St1 = State[3], next_state[3]
            d_goal_St, d_goal_St1 = State[4], next_state[4]
            reward, done = Reward(d_obs_St, d_obs_St1, d_goal_St, d_goal_St1, t)
            total_reward += reward

            # Store transition
            replay_buffer.add((state, action, reward, next_state, done))
            State = next_state

            if done:
                # p.removeBody(turtlebot)
                break

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

        # Train after each episode
        train_actor_critic(
            actor,
            critic,
            target_actor,
            target_critic,
            replay_buffer,
            actor_optimizer,
            critic_optimizer,
            batch_size,
        )


# Disconnect from the simulation
run_simulation_and_train(actor, critic, target_actor, target_critic, replay_buffer)
# p.disconnect()
