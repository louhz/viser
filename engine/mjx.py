import mujoco  as mj

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

import imageio
import cv2


def apply_action_and_step(mj_model: mj.MjModel, mj_data: mj.MjData, action,robot_action):
    """
    Places 'action' into mj_data.ctrl, then calls mj_step for one timestep.
    If you want continuous torque control or position servo, ensure your XML
    actuators are defined for these joints.
    """
    mj_data.ctrl[:7] = robot_action
    mj_data.ctrl[7:] = action
    mj.mj_step(mj_model, mj_data)





def _name2id_or_none(model, objtype, name: str):
    idx = mj.mj_name2id(model, objtype, name)
    return idx if idx != -1 else None

def find_entity_kind_and_id(model: mj.MjModel, name: str):
    """Return ('body'|'geom'|'site', idx) for a named entity."""
    bid = _name2id_or_none(model, mj.mjtObj.mjOBJ_BODY, name)
    if bid is not None:
        return 'body', bid
    gid = _name2id_or_none(model, mj.mjtObj.mjOBJ_GEOM, name)
    if gid is not None:
        return 'geom', gid
    sid = _name2id_or_none(model, mj.mjtObj.mjOBJ_SITE, name)
    if sid is not None:
        return 'site', sid
    raise ValueError(f"Could not find body/geom/site named '{name}' in the model.")

def mat9_to_quat_wxyz(mat9):
    """Convert a 3x3 rotation matrix (flattened row-major length-9) to quaternion [w,x,y,z]."""
    R = np.asarray(mat9, dtype=float).reshape(3, 3)
    t = np.trace(R)
    if t > 0.0:
        S = np.sqrt(t + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2,1] - R[1,2]) / S
        qy = (R[0,2] - R[2,0]) / S
        qz = (R[1,0] - R[0,1]) / S
    else:
        if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            qw = (R[2,1] - R[1,2]) / S
            qx = 0.25 * S
            qy = (R[0,1] + R[1,0]) / S
            qz = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            qw = (R[0,2] - R[2,0]) / S
            qx = (R[0,1] + R[1,0]) / S
            qy = 0.25 * S
            qz = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            qw = (R[1,0] - R[0,1]) / S
            qx = (R[0,2] + R[2,0]) / S
            qy = (R[1,2] + R[2,1]) / S
            qz = 0.25 * S
    # Normalize for safety
    q = np.array([qw, qx, qy, qz], dtype=float)
    return q / np.linalg.norm(q)

def quat_wxyz_to_euler_zyx(q):
    """Convert [w,x,y,z] to yaw(Z), pitch(Y), roll(X) in radians (ZYX TaitBryan)."""
    w, x, y, z = q
    # yaw (Z)
    yaw = np.arctan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z))
    # pitch (Y)
    sp = 2.0 * (w*y - z*x)
    sp = np.clip(sp, -1.0, 1.0)
    pitch = np.arcsin(sp)
    # roll (X)
    roll = np.arctan2(2.0 * (w*x + y*z), 1.0 - 2.0 * (x*x + y*y))
    return yaw, pitch, roll

def get_world_pose(model: mj.MjModel, data: mj.MjData, entity_kind: str, entity_id: int):
    """Return (pos[3], quat_wxyz[4]) of a body/geom/site in the world frame."""
    if entity_kind == 'body':
        pos = data.xpos[entity_id].copy()
        quat = data.xquat[entity_id].copy()  # MuJoCo stores [w,x,y,z]
    elif entity_kind == 'geom':
        pos = data.geom_xpos[entity_id].copy()
        quat = mat9_to_quat_wxyz(data.geom_xmat[entity_id])
    elif entity_kind == 'site':
        pos = data.site_xpos[entity_id].copy()
        quat = mat9_to_quat_wxyz(data.site_xmat[entity_id])
    else:
        raise ValueError(f"Unknown entity_kind: {entity_kind}")
    return pos, quat




def seed_state_from_actuator_targets(mj_model: mj.MjModel, mj_data: mj.MjData, ctrl_vec: np.ndarray):
    """
    Set qpos from actuator targets for actuators that directly drive a joint.
    Only sets hinge/slide joints. Then zero velocities and forward the model.
    """
    assert ctrl_vec.shape[0] == mj_model.nu, f"Expected {mj_model.nu} ctrl, got {ctrl_vec.shape[0]}"

    # Helper to get qpos index and size for a joint
    def _qpos_span(jid: int):
        adr = mj_model.jnt_qposadr[jid]
        jtype = mj_model.jnt_type[jid]
        if jtype == mj.mjtJoint.mjJNT_HINGE or jtype == mj.mjtJoint.mjJNT_SLIDE:
            return adr, 1
        # Ignore ball/free here; they’re uncommon on Franka/Allegro joints
        return None, 0

    for ai in range(mj_model.nu):
        trn_type = mj_model.actuator_trntype[ai]
        if trn_type == mj.mjtTrn.mjTRN_JOINT:
            jid = mj_model.actuator_trnid[ai, 0]
            adr, size = _qpos_span(jid)
            if size == 1 and adr is not None:
                # Optional: clamp to joint range if limited
                if mj_model.jnt_limited[jid]:
                    lo, hi = mj_model.jnt_range[jid]
                    mj_data.qpos[adr] = float(np.clip(ctrl_vec[ai], lo, hi))
                else:
                    mj_data.qpos[adr] = float(ctrl_vec[ai])

    # Start from rest
    mj_data.qvel[:] = 0.0
    mj_data.qacc[:] = 0.0
    mj.mj_forward(mj_model, mj_data)
def main():
    mj_file = '/home/louhz/viewer/viser/examples/assets/franka_leap_demo/scene_ketchup_render.xml'
    mj_model = mj.MjModel.from_xml_path(mj_file)
    mj_data = mj.MjData(mj_model)

    # Reset via keyframe (optional)
    home_pose_id = mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_KEY, "home_pose")
    mj.mj_resetDataKeyframe(mj_model, mj_data, home_pose_id)

    # --- your existing target setup (arm/hand) ---
    robot_action = np.array([-0.488, 0.14 , 0.31, -1.95, 1.36, 1.43, 0.25])
    raw_action  = np.zeros(16)  # start hand at zeros

    full_ctrl = np.zeros(mj_model.nu)
    full_ctrl[:7] = robot_action
    full_ctrl[7:7+len(raw_action)] = raw_action
    seed_state_from_actuator_targets(mj_model, mj_data, full_ctrl)  # calls mj_forward

    # --- sim timing ---
    dt = mj_model.opt.timestep
    settle_seconds = 0.5
    settle_steps   = int(settle_seconds / dt)
    duration = 4.0
    sim_steps = int(duration / dt)
    video_fps = 60

    # --- identify the object to log ---
    target_name = "ketchup"  # change if your body/geom/site is named differently
    entity_kind, entity_id = find_entity_kind_and_id(mj_model, target_name)

    # --- open the log file ---
    log_path = "ketchup_pose.txt"
    with open(log_path, "w") as logf:
        logf.write("# time  px  py  pz   qw  qx  qy  qz   yaw  pitch  roll  (ZYX, radians)\n")

        # ---------- pre-roll (no rendering) ----------
        for _ in range(settle_steps):
            apply_action_and_step(mj_model, mj_data, raw_action, robot_action)
            # grab world pose after the step
            pos, quat = get_world_pose(mj_model, mj_data, entity_kind, entity_id)
            yaw, pitch, roll = quat_wxyz_to_euler_zyx(quat)
            logf.write(f"{mj_data.time:.6f}  {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}  "
                       f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}  "
                       f"{yaw:.6f} {pitch:.6f} {roll:.6f}\n")

        # ---------- main rollout (with rendering) ----------
        frames = []
        height, width = 480, 640

        # hand action you wanted to command next
        hand_action_next = np.array([
            0 , 0 , 0 , 0, 0.8 , 0,
            0, 0.8, 0 , 0, 0, 0,
            0 , 0, 0 , 0
        ])

        with mj.Renderer(mj_model, height=height, width=width) as renderer:
            for _ in range(sim_steps):
                apply_action_and_step(mj_model, mj_data, hand_action_next, robot_action)

                # log world pose at this step
                pos, quat = get_world_pose(mj_model, mj_data, entity_kind, entity_id)
                yaw, pitch, roll = quat_wxyz_to_euler_zyx(quat)
                logf.write(f"{mj_data.time:.6f}  {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}  "
                           f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}  "
                           f"{yaw:.6f} {pitch:.6f} {roll:.6f}\n")

                # render
                renderer.update_scene(mj_data, camera="render")
                frames.append(renderer.render())

    print(f"Saved world poses to: {log_path}")
    print("Final sim time:", mj_data.time)
    print("Number of contacts at end:", mj_data.ncon)

    show_video(frames, fps=video_fps)
    save_video(frames, fps=video_fps, filename="ketchup.mp4")

    
def show_video(frames, fps):
    """Display frames using Matplotlib."""
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    img = ax.imshow(frames[0])

    def update(frame):
        img.set_data(frame)
        return (img,)

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / fps)
    plt.show()

def save_video(frames, fps, filename="output.mp4"):
    """
    Saves video from a list of frames using imageio.

    :param frames: List of frames (numpy arrays).
    :param fps: Frames per second.
    :param filename: Output file name with extension (e.g., "output.mp4").
    """
    with imageio.get_writer(filename, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)

if __name__ == "__main__":
    main()
