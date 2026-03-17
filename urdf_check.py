
# urdf_check.py
import os
import numpy as np
from isaacgym import gymapi, gymutil
import torch
from isaacgym import gymtorch
from collections import deque
import math
from pathlib import Path

URDF_ROOT = str(Path(__file__).parent)
URDF_FILE = "myobody.urdf"

SPAWN_HEIGHT = 1.4  # The bottom of the cylinder starts at ground level
DISABLE_GRAVITY = False
FIX_BASE_LINK = True  # Allow free-floating dynamics
USE_TORQUE_MODE_DEFAULT = True  # Start in torque mode for dynamic simulation
POS_STEP = 0.1
EFFORT_STEP = 0.1

gym = gymapi.acquire_gym()

sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0) if DISABLE_GRAVITY else gymapi.Vec3(0.0, 0.0, -9.81)

# Physics parameters for stable dynamic simulation
sim_params.dt = 1.0 / 1000.0  # 1 kHz physics
sim_params.substeps = 4  # Substeps for stability
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0
sim_params.physx.bounce_threshold_velocity = 0.2
sim_params.physx.max_depenetration_velocity = 1.0
sim_params.physx.default_buffer_size_multiplier = 5.0

args = gymutil.parse_arguments(description="URDF probe")
compute_device_id = 0
graphics_device_id = 0
sim = gym.create_sim(compute_device_id, graphics_device_id, gymapi.SIM_PHYSX, sim_params)
assert sim is not None, "Failed to create sim"

# Add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)  # Z-up
plane_params.static_friction = 1.0
plane_params.dynamic_friction = 1.0
plane_params.restitution = 0.0
gym.add_ground(sim, plane_params)

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
assert viewer is not None, "Failed to create viewer"

asset_opts = gymapi.AssetOptions()
asset_opts.fix_base_link = FIX_BASE_LINK
asset_opts.disable_gravity = DISABLE_GRAVITY
# Cast enums to int to avoid DeprecationWarning
asset_opts.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
asset_opts.flip_visual_attachments = False
asset_opts.collapse_fixed_joints = False  # Keep all rigid bodies for per-link coloring

asset = gym.load_asset(sim, URDF_ROOT, URDF_FILE, asset_opts)
assert asset is not None, f"Failed to load asset {URDF_FILE} from {URDF_ROOT}"

dof_count = gym.get_asset_dof_count(asset)
dof_names = gym.get_asset_dof_names(asset)
dof_props_asset = gym.get_asset_dof_properties(asset)  # per-DOF limits, stiffness, etc.
 

print(f"Loaded URDF with {dof_count} DOFs:")
for i, name in enumerate(dof_names):
    lower = dof_props_asset['lower'][i]
    upper = dof_props_asset['upper'][i]
    has_lim = bool(dof_props_asset['hasLimits'][i])
    stiff = dof_props_asset['stiffness'][i]
    damp  = dof_props_asset['damping'][i]
    effort = dof_props_asset['effort'][i]
    vel_limit = dof_props_asset['velocity'][i]
    friction = dof_props_asset['friction'][i]
    print(f"  [{i:02d}] {name:24s} limits=({lower:.3f}, {upper:.3f}) hasLimits={has_lim}")
    print(f"       effort={effort:.1f} Nm, vel_limit={vel_limit:.1f} rad/s")
    print(f"       URDF damping={damp:.4f}, URDF friction={friction:.4f}, PD Kp={stiff:.1f}")

env = gym.create_env(sim, gymapi.Vec3(-1, -1, -1), gymapi.Vec3(1, 1, 1), 1)
pose = gymapi.Transform()
# Spawn robot exactly at world origin (0,0,0)
pose.p = gymapi.Vec3(0.0, 0.0, SPAWN_HEIGHT)
actor = gym.create_actor(env, asset, pose, "robot", 0, 1)

# ── Apply realistic colors to robot and human links ──────────────────
rb_names = gym.get_asset_rigid_body_names(asset)
rb_name_to_idx = {n: i for i, n in enumerate(rb_names)}

DARK_CHARCOAL = gymapi.Vec3(0.18, 0.18, 0.19)
DARK_METALLIC = gymapi.Vec3(0.15, 0.15, 0.16)
FOOT_GRAY     = gymapi.Vec3(0.22, 0.22, 0.24)
BONE_COLOR    = gymapi.Vec3(0.92, 0.90, 0.82)

ROBOT_COLORS = {
    'LINK_BASE':    DARK_CHARCOAL,
    'LINK_R_BASE':  DARK_CHARCOAL,  'LINK_L_BASE':  DARK_CHARCOAL,
    'LINK_R_HIP':   DARK_CHARCOAL,  'LINK_L_HIP':   DARK_CHARCOAL,
    'LINK_R_THIGH': DARK_CHARCOAL,  'LINK_L_THIGH': DARK_CHARCOAL,
    'LINK_R_SHANK': DARK_CHARCOAL,  'LINK_L_SHANK': DARK_CHARCOAL,
    'LINK_R_ANK':   DARK_METALLIC,  'LINK_L_ANK':   DARK_METALLIC,
    'LINK_R_FOOT':  FOOT_GRAY,      'LINK_L_FOOT':  FOOT_GRAY,
}

HUMAN_BONE_LINKS = [
    'sacrum', 'pelvis', 'femur_r', 'femur_l', 'tibia_r', 'tibia_l',
    'patella_r', 'patella_l', 'talus_r', 'talus_l', 'calcn_r', 'calcn_l',
    'toes_r', 'toes_l',
    'l5', 'l4', 'l3', 'l2', 'l1', 't',
    'c7', 'c6', 'c5', 'c4', 'c3', 'c2', 'c1', 'head',
    'humerus_r', 'humerus_l', 'ulna_r', 'ulna_l', 'radius_r', 'radius_l',
    'hand_r', 'hand_l',
]

robot_colored = 0
for name, color in ROBOT_COLORS.items():
    idx = rb_name_to_idx.get(name)
    if idx is not None:
        gym.set_rigid_body_color(env, actor, idx, gymapi.MESH_VISUAL, color)
        robot_colored += 1
    else:
        print(f"  [WARN] Robot link '{name}' not found in rigid bodies (collapsed?)")

bone_colored = 0
for name in HUMAN_BONE_LINKS:
    idx = rb_name_to_idx.get(name)
    if idx is not None:
        gym.set_rigid_body_color(env, actor, idx, gymapi.MESH_VISUAL, BONE_COLOR)
        bone_colored += 1
    else:
        print(f"  [WARN] Human bone '{name}' not found in rigid bodies (collapsed?)")

print(f"\nColor summary: {robot_colored}/{len(ROBOT_COLORS)} robot links, {bone_colored}/{len(HUMAN_BONE_LINKS)} human bones")
print(f"Total rigid bodies in asset: {len(rb_names)}")

# Position the viewer camera closer to the robot.
# Use a slight offset in +x so the camera looks toward the origin where the robot stands.
close_cam_pos = gymapi.Vec3(1.2, 0.6, 0.6)  # distance ~1.4m from origin
close_cam_target = gymapi.Vec3(0.0, 0.0, 0.3)
gym.viewer_camera_look_at(viewer, env, close_cam_pos, close_cam_target)

# Map this actor's DOFs to the global sim DOF tensor indices
sim_dof_indices = [gym.get_actor_dof_index(env, actor, i, gymapi.DOMAIN_SIM) for i in range(dof_count)]

# Acquire the global DOF force tensor (size = total DOFs in the sim)
dof_force_tensor = gym.acquire_dof_force_tensor(sim)
dof_force = gymtorch.wrap_tensor(dof_force_tensor)   # torch tensor view
dof_force.zero_()

# Acquire the global DOF state tensor for reading positions/velocities
dof_state_tensor = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(dof_state_tensor)  # shape: [num_dofs, 2] where [:, 0]=pos, [:, 1]=vel

# Start with explicit DOF props on the actor
actor_dof_props = gym.get_actor_dof_properties(env, actor)
for i in range(dof_count):
    # Override effort limits — human joints have effort=0 in the URDF,
    # which prevents the PD controller from applying any force.
    actor_dof_props['effort'][i] = 1000.0

    if USE_TORQUE_MODE_DEFAULT:
        actor_dof_props['driveMode'][i] = int(gymapi.DOF_MODE_EFFORT)
        actor_dof_props['stiffness'][i] = 0.0  # Zero stiffness for pure torque control
        actor_dof_props['damping'][i] = 0.0    # Zero damping for free motion
    else:
        actor_dof_props['driveMode'][i] = int(gymapi.DOF_MODE_POS)
        actor_dof_props['stiffness'][i] = 20000.0
        actor_dof_props['damping'][i] = 200.0

gym.set_actor_dof_properties(env, actor, actor_dof_props)

# Also reset all DOFs to zero position/velocity so the model starts still
if dof_count > 0:
    init_state = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
    for i in range(dof_count):
        init_state[i]['pos'] = 0.0
        init_state[i]['vel'] = 0.0
    gym.set_actor_dof_states(env, actor, init_state, gymapi.STATE_ALL)

# Init DOF state/targets
if dof_count > 0:
    dof_state = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
    q  = np.array([s['pos'] for s in dof_state], dtype=np.float32)
    qd = np.zeros_like(q)
    targets = q.copy()
    efforts = np.zeros_like(q)
else:
    q = qd = targets = efforts = np.zeros(0, dtype=np.float32)

# Key bindings
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_LEFT,  "select_prev")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_RIGHT, "select_next")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_UP,    "inc")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_DOWN,  "dec")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_M,     "toggle_mode")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R,     "reset")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Q,     "quit")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_C,     "toggle_com")

selected = 0
use_torque_mode = USE_TORQUE_MODE_DEFAULT
show_com = True
sim_time = 0.0  # Track simulation time

# Rigid body COM visualization setup
rb_count = gym.get_actor_rigid_body_count(env, actor)
rb_indices = [gym.get_actor_rigid_body_index(env, actor, i, gymapi.DOMAIN_SIM) for i in range(rb_count)]
rb_props = gym.get_actor_rigid_body_properties(env, actor)
rb_masses = np.array([p.mass for p in rb_props], dtype=np.float32) if rb_props is not None else np.ones(rb_count, dtype=np.float32)
total_mass = float(np.sum(rb_masses)) if rb_count > 0 else 1.0

# Keep a short history of COM positions per link so we only draw the last N traces
TRACE_HISTORY = 10
rb_traces = [deque(maxlen=TRACE_HISTORY) for _ in range(rb_count)]

# Helper to draw a small crosshair at a position using lines (compatible across Gym versions)
def draw_crosshair(center: gymapi.Vec3, size: float, color: gymapi.Vec3):
    h = size * 0.5
    # X axis
    p1 = gymapi.Vec3(center.x - h, center.y, center.z)
    p2 = gymapi.Vec3(center.x + h, center.y, center.z)
    gymutil.draw_line(p1, p2, color, gym, viewer, env)
    # Y axis
    p1 = gymapi.Vec3(center.x, center.y - h, center.z)
    p2 = gymapi.Vec3(center.x, center.y + h, center.z)
    gymutil.draw_line(p1, p2, color, gym, viewer, env)
    # Z axis
    p1 = gymapi.Vec3(center.x, center.y, center.z - h)
    p2 = gymapi.Vec3(center.x, center.y, center.z + h)
    gymutil.draw_line(p1, p2, color, gym, viewer, env)

# Global rigid body state tensor to read rigid body poses (body origin, NOT COM)
rb_state_tensor = gym.acquire_rigid_body_state_tensor(sim)
rb_state = gymtorch.wrap_tensor(rb_state_tensor)  # shape: [num_bodies, 13]

# Batch quaternion-vector rotation (q in [qx,qy,qz,qw] order)
def rotate_vec_batch(qxyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    # q: (N,4) -> u=(x,y,z), s=w; v: (N,3)
    u = qxyzw[:, 0:3]
    s = qxyzw[:, 3:4]
    cross1 = np.cross(u, v)
    cross2 = np.cross(u, cross1 + s * v)
    return v + 2.0 * cross2

def apply_controls():
    if dof_count == 0:
        return
    if use_torque_mode:
        # Apply effort to each DOF using sim-domain indices
        for i in range(dof_count):
            dof_force[sim_dof_indices[i]] = float(efforts[i])
        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(dof_force))
    else:
        gym.set_actor_dof_position_targets(env, actor, targets)

print("\nControls:\n  LEFT/RIGHT : select DOF\n  UP/DOWN    : nudge (+/-) selected DOF (pos target or effort)\n  M          : toggle Position-mode <-> Torque-mode\n  R          : reset all DOFs to 0\n  Q          : quit\n")
print("  C          : toggle COM markers (per-link and whole-body)\n")
print("\n⚠️  DYNAMIC SIMULATION MODE:")
print("  - Base is FREE-FLOATING (not fixed)")
print("  - Robot will FALL under gravity in torque mode")
print("  - Apply torques with UP/DOWN to control motion")
print("  - Switch to Position mode (M) to hold robot in place\n")

def wrapToPi(a):
    return math.atan2(math.sin(a), math.cos(a))

while not gym.query_viewer_has_closed(viewer):
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "select_prev" and evt.value > 0:
            selected = (selected - 1) % max(1, dof_count)
            print(f"\nSelected DOF {selected}: {dof_names[selected] if dof_count>0 else 'N/A'}")
        elif evt.action == "select_next" and evt.value > 0:
            selected = (selected + 1) % max(1, dof_count)
            print(f"\nSelected DOF {selected}: {dof_names[selected] if dof_count>0 else 'N/A'}")
        elif evt.action == "toggle_mode" and evt.value > 0:
            use_torque_mode = not use_torque_mode
            for i in range(dof_count):
                if use_torque_mode:
                    actor_dof_props["driveMode"][i] = int(gymapi.DOF_MODE_EFFORT)
                    actor_dof_props["stiffness"][i] = 0.0  # No stiffness - free motion
                    actor_dof_props["damping"][i] = 0.0    # No damping - pure dynamics
                else:
                    actor_dof_props["driveMode"][i] = int(gymapi.DOF_MODE_POS)
                    actor_dof_props["stiffness"][i] = 20000.0
                    actor_dof_props["damping"][i] = 200.0
            gym.set_actor_dof_properties(env, actor, actor_dof_props)
            if use_torque_mode:
                # Reset efforts to zero when switching to torque mode
                efforts[:] = 0.0
            else:
                # Snap all joints to default (zero) configuration
                targets[:] = 0.0
                dof_state = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
                for i in range(dof_count):
                    dof_state[i]['pos'] = 0.0
                    dof_state[i]['vel'] = 0.0
                gym.set_actor_dof_states(env, actor, dof_state, gymapi.STATE_ALL)
                gym.set_actor_dof_position_targets(env, actor, targets)
            print("Mode:", "TORQUE (effort) - Robot will fall under gravity!" if use_torque_mode else "POSITION - Robot held by PD controller")
        elif evt.action == "inc" and evt.value > 0 and dof_count > 0:
            if use_torque_mode:
                efforts[selected] += EFFORT_STEP
                print(f"\nEffort[{selected}] = {efforts[selected]:.3f}")
            else:
                targets[selected] += POS_STEP
                print(f"\nTarget[{selected}] = {targets[selected]:.3f}")
        elif evt.action == "dec" and evt.value > 0 and dof_count > 0:
            if use_torque_mode:
                efforts[selected] -= EFFORT_STEP
                print(f"\nEffort[{selected}] = {efforts[selected]:.3f}")
            else:
                targets[selected] -= POS_STEP
                print(f"\nTarget[{selected}] = {targets[selected]:.3f}")
        elif evt.action == "reset" and evt.value > 0 and dof_count > 0:
            targets[:] = 0.0
            efforts[:] = 0.0
            dof_state = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
            for i in range(dof_count):
                dof_state[i]['pos'] = 0.0
                dof_state[i]['vel'] = 0.0
            gym.set_actor_dof_states(env, actor, dof_state, gymapi.STATE_ALL)
            print("Reset DOFs to zero.")
        elif evt.action == "quit" and evt.value > 0:
            gym.destroy_viewer(viewer)
            gym.destroy_sim(sim)
            raise SystemExit
        elif evt.action == "toggle_com" and evt.value > 0:
            show_com = not show_com
            print("COM markers:", "ON" if show_com else "OFF")

    apply_controls()
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    sim_time += sim_params.dt  # Increment simulation time
    
    # Refresh DOF state tensor to get updated positions and velocities
    gym.refresh_dof_state_tensor(sim)
    
    # Print current state of selected DOF continuously with simulation time
    if dof_count > 0:
        # Read from the tensor using sim-domain indices
        # dof_states shape is [num_dofs, 2] where [:, 0]=position, [:, 1]=velocity
        q_selected = dof_states[sim_dof_indices[selected], 0].item()
        qd_selected = dof_states[sim_dof_indices[selected], 1].item()
        
        # Show effort command in torque mode
        if use_torque_mode:
            print(f"\r[t={sim_time:6.2f}s] [{dof_names[selected]}] pos={q_selected:+8.4f} rad | vel={qd_selected:+8.4f} rad/s | effort_cmd={efforts[selected]:+8.2f} Nm", end='', flush=True)
        else:
            print(f"\r[t={sim_time:6.2f}s] [{dof_names[selected]}] pos={q_selected:+8.4f} rad | vel={qd_selected:+8.4f} rad/s | target={targets[selected]:+8.4f} rad", end='', flush=True)
    
    # Refresh rigid body states and draw COM markers
    gym.refresh_rigid_body_state_tensor(sim)
    if show_com and rb_count > 0:
        # rb_state is [N,13]: [px,py,pz, qx,qy,qz,qw, vx,vy,vz, wx,wy,wz]
        rb_state_reshaped = rb_state.view(-1, 13)
        # Extract positions for the specific actor's bodies via sim-domain indices
        rb_pos = rb_state_reshaped[rb_indices, 0:3].cpu().numpy()
        rb_quat_xyzw = rb_state_reshaped[rb_indices, 3:7].cpu().numpy()
        # Local COM offsets from body frames
        rb_com_local = np.array([[p.com.x, p.com.y, p.com.z] for p in rb_props], dtype=np.float32)
        # Rotate local COM offsets into world and add to body origins
        rb_com_world = rb_pos + rotate_vec_batch(rb_quat_xyzw, rb_com_local)

        # update per-link trace buffers with the newest COMs
        for i in range(rb_count):
            rb_traces[i].append(rb_com_world[i].copy())

        # Clear previously drawn lines for a clean trace redraw
        gym.clear_lines(viewer)

        # Draw only the last TRACE_HISTORY crosshairs per link (smaller size for traces)
        green = gymapi.Vec3(0.0, 1.0, 0.0)
        for i in range(rb_count):
            for pos in rb_traces[i]:
                center = gymapi.Vec3(float(pos[0]), float(pos[1]), float(pos[2]))
                draw_crosshair(center, size=0.04, color=green)
        # Compute whole-body COM (mass-weighted average)
        com_total = (rb_com_world * rb_masses[:, None]).sum(axis=0) / max(total_mass, 1e-8)
    red = gymapi.Vec3(1.0, 0.0, 0.0)
    center_total = gymapi.Vec3(float(com_total[0]), float(com_total[1]), float(com_total[2]))
    # print("COM total = ({:+.3f}, {:+.3f}, {:+.3f}) m".format(com_total[0], com_total[1], com_total[2]), end='', flush=True)
    draw_crosshair(center_total, size=0.10, color=red)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)