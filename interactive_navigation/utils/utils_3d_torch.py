import torch
import numpy as np
from PIL import Image, ImageDraw

def get_rotation_matrix(agent_rot):
    #######
    # Construct the rotation matrix. Ref: https://en.wikipedia.org/wiki/Rotation_matrix
    #######

    r_y = torch.Tensor([[torch.cos(torch.deg2rad(agent_rot[1])), 0, torch.sin(torch.deg2rad(agent_rot[1]))],
                        [0, 1, 0],
                        [-torch.sin(torch.deg2rad(agent_rot[1])), 0, torch.cos(torch.deg2rad(agent_rot[1]))]])
    r_x = torch.Tensor([[1, 0, 0],
                        [0, torch.cos(torch.deg2rad(agent_rot[0])), -torch.sin(torch.deg2rad(agent_rot[0]))],
                        [0, torch.sin(torch.deg2rad(agent_rot[0])), torch.cos(torch.deg2rad(agent_rot[0]))]])
    r = torch.mm(r_x, r_y)
    return r.to(agent_rot.device)

def get_rotation_matrix_batch(agent_rot):
    # agent_rot: Bx3
    # output: Bx3x3

    m = []
    for r in agent_rot:
        m.append(get_rotation_matrix(r).unsqueeze(0))
    return torch.cat(m, dim=0)

def get_object_rotation_matrix(rot):
    #######
    # Construct the Left-hand coordinate system rotation matrix. Ref: ???
    #######

    r_y = torch.Tensor([[torch.cos(torch.deg2rad(rot[1])), 0, torch.sin(torch.deg2rad(rot[1]))],
                        [0, 1, 0],
                        [-torch.sin(torch.deg2rad(rot[1])), 0, torch.cos(torch.deg2rad(rot[1]))]])
    r_x = torch.Tensor([[1, 0, 0],
                        [0, torch.cos(torch.deg2rad(rot[0])), -torch.sin(torch.deg2rad(rot[0]))],
                        [0, torch.sin(torch.deg2rad(rot[0])), torch.cos(torch.deg2rad(rot[0]))]])
    r_z = torch.Tensor([[torch.cos(torch.deg2rad(rot[2])), -torch.sin(torch.deg2rad(rot[2])), 0],
                        [torch.sin(torch.deg2rad(rot[2])), torch.cos(torch.deg2rad(rot[2])), 0],
                        [0, 0, 1]])
    r = torch.mm(torch.mm(r_y, r_x), r_z)
    return r.to(rot.device)

def get_object_rotation_matrix_batch(rot):
    #######
    # Construct the Left-hand coordinate system rotation matrix. Ref: ???
    #######

    z = torch.zeros(len(rot)).to(rot.device).unsqueeze(1)
    o = torch.ones(len(rot)).to(rot.device).unsqueeze(1)

    # r_y
    c = torch.cos(torch.deg2rad(rot[:, 1])).unsqueeze(1)
    s = torch.sin(torch.deg2rad(rot[:, 1])).unsqueeze(1)
    r_y_0 = torch.cat([c, z, s], dim=1).unsqueeze(1)
    r_y_1 = torch.cat([z, o, z], dim=1).unsqueeze(1)
    r_y_2 = torch.cat([-s, z, c], dim=1).unsqueeze(1)
    r_y = torch.cat([r_y_0, r_y_1, r_y_2], dim=1)

    # r_x
    c = torch.cos(torch.deg2rad(rot[:, 0])).unsqueeze(1)
    s = torch.sin(torch.deg2rad(rot[:, 0])).unsqueeze(1)
    r_x_0 = torch.cat([o, z, z], dim=1).unsqueeze(1)
    r_x_1 = torch.cat([z, c, -s], dim=1).unsqueeze(1)
    r_x_2 = torch.cat([z, s, c], dim=1).unsqueeze(1)
    r_x = torch.cat([r_x_0, r_x_1, r_x_2], dim=1)

    # r_z
    c = torch.cos(torch.deg2rad(rot[:, 2])).unsqueeze(1)
    s = torch.sin(torch.deg2rad(rot[:, 2])).unsqueeze(1)
    r_z_0 = torch.cat([c, -s, z], dim=1).unsqueeze(1)
    r_z_1 = torch.cat([s, c, z], dim=1).unsqueeze(1)
    r_z_2 = torch.cat([z, z, o], dim=1).unsqueeze(1)
    r_z = torch.cat([r_z_0, r_z_1, r_z_2], dim=1)

    r = torch.bmm(torch.bmm(r_y, r_x), r_z)
    return r.to(rot.device)

def get_object_rotation_matrix_batch_loop(rot):
    # agent_rot: Bx3
    # output: Bx3x3

    m = []
    for r in rot:
        m.append(get_object_rotation_matrix(r).unsqueeze(0))
    return torch.cat(m, dim=0)

def get_object_affine_matrix(obj_pose):
    # obj_pose: Bx6
    # output: Bx3x4

    m = get_object_rotation_matrix_batch(obj_pose[:, 3:])
    t = obj_pose[:, :3].unsqueeze(2)
    m = torch.cat((m, t), dim=2)
    return m

def project_to_3d(pos, depth, half_fov, w, h):
    # pos: BxNx2
    # depth: BxNx1
    # half_fov: Bx1
    # w: Bx1
    # h: Bx1
    # output: BxNx3

    n = pos.shape[1]
    x = (pos[:, :, 0] * 2 / w.repeat(1, n)) - 1.0
    y = ((1.0 - (pos[:, :, 1] / h.repeat(1, n))) * 2) - 1.0
    pos_3d = [(x * (depth.squeeze(2) * torch.tan(torch.deg2rad(half_fov)).repeat(1, n))).unsqueeze(2),
              (y * (depth.squeeze(2) * torch.tan(torch.deg2rad(half_fov)).repeat(1, n))).unsqueeze(2),
              depth]
    return torch.cat(pos_3d, dim=2)

def project_to_2d(pos, half_fov, w, h):
    # pos: BxNx3
    # half_fov: Bx1
    # w: Bx1
    # h: Bx1
    # output: BxNx2

    n = pos.shape[1]
    pos_2d = [(pos[:, :, 0] / (pos[:, :, 2] * torch.tan(torch.deg2rad(half_fov)).repeat(1, n))).unsqueeze(2),
              (pos[:, :, 1] / (pos[:, :, 2] * torch.tan(torch.deg2rad(half_fov)).repeat(1, n))).unsqueeze(2)]
    pos_2d = torch.cat(pos_2d, dim=2)

    x = (w.repeat(1, n) * ((pos_2d[:, :, 0] + 1.0) / 2.0)).long().unsqueeze(2)
    y = (h.repeat(1, n) * (1 - ((pos_2d[:, :, 1] + 1.0) / 2.0))).long().unsqueeze(2)
    return torch.cat([x, y], dim=2)

def project_to_agent_coordinate(pos, agent_pos, r):
    # pos: BxNx3
    # agent_pos: Bx3
    # r: Bx3x3
    # output: BxNx3

    n = pos.shape[1]
    pos_diff = torch.transpose((pos - agent_pos.unsqueeze(1).repeat(1, n, 1)), 1, 2)
    # since AI2THOR is left-handed coordinate system, we need to turn it to the right-handed to use the rotation matrix
    pos_diff[:, 2, :] *= -1
    new_pos = torch.bmm(r, pos_diff)
    # turn back to the left-handed coordinate system
    new_pos[:, 2, :] *= -1
    return torch.transpose(new_pos, 1, 2)

def project_to_global_coordinate(pos, agent_pos, r):
    # pos: BxNx3
    # agent_pos: Bx3
    # r: Bx3x3
    # output: BxNx3

    n = pos.shape[1]
    new_pos = torch.transpose(pos, 1, 2)
    new_pos[:, 2, :] *= -1
    new_pos = torch.bmm(r, new_pos)
    new_pos[:, 2, :] *= -1
    new_pos = torch.transpose(new_pos, 1, 2)
    new_pos += agent_pos.unsqueeze(1).repeat(1, n, 1)
    return new_pos

def project_2d_to_3d(points, depth, agent_pose, w, h, half_fov):
    # point: BxNx2
    # depth: BxNx1
    # agent_pose: Bx6
    # w: Bx1
    # h: Bx1
    # half_fov: Bx1

    points_agent_coordinate_3d = project_to_3d(points, depth, half_fov, w, h)
    rotation_matrix = get_rotation_matrix_batch(agent_pose[:, 3:])
    rotation_matrix = torch.inverse(rotation_matrix)
    points_global_coordinate_3d = project_to_global_coordinate(points_agent_coordinate_3d,
                                                               agent_pose[:, :3],
                                                               rotation_matrix)
    return points_global_coordinate_3d

def local_project_2d_to_3d(points, depth, w, h, half_fov):
    # point: BxNx2
    # depth: BxNx1
    # agent_pose: Bx6
    # w: Bx1
    # h: Bx1
    # half_fov: Bx1

    points_agent_coordinate_3d = project_to_3d(points, depth, half_fov, w, h)
    return points_agent_coordinate_3d

def project_3d_to_2d(points, agent_pose, w, h, half_fov):
    # point: BxNx2
    # agent_pose: Bx6
    # w: Bx1
    # h: Bx1
    # half_fov: Bx1

    rotation_matrix = get_rotation_matrix_batch(agent_pose[:, 3:])
    points_agent_coordinate = project_to_agent_coordinate(points, agent_pose[:, :3], rotation_matrix)
    points_agent_coordinate_2d = project_to_2d(points_agent_coordinate, half_fov, w, h)
    return points_agent_coordinate_2d

def plot_next_keypoints(points, frame, agent_pose=None):
    frame = frame[0,0].detach().cpu().numpy()
    points = points.detach().cpu().unsqueeze(0)
    w = torch.Tensor([224])
    h = torch.Tensor([224])
    half_fov = torch.Tensor([45])
    if not isinstance(agent_pose, type(None)):
        points_2d = project_3d_to_2d(points, agent_pose, w, h, half_fov)
    else:
        points_2d = project_to_2d(points, half_fov, w, h)
    points_2d = [[ele[1], ele[0]] for ele in points_2d[0].numpy()]
    img = Image.fromarray(frame)
    img = draw_point(img, points_2d)
    return img

def save_keypoints_results(points):
    w = torch.Tensor([224])
    h = torch.Tensor([224])
    half_fov = torch.Tensor([45])
    points = points.detach().cpu()[5:9]
    points_2d = []
    for p in points:
        points_2d.append(project_to_2d(p.unsqueeze(0), half_fov, w, h))
    points_2d = torch.cat(points_2d).numpy()
    np.save("storage/keypoints_2d.npy", points_2d)

def dict_to_list(pos):
    return [pos["x"], pos["y"], pos["z"]]

def project_2d_points_to_3d(metadatas, points, depths):
    # events: B
    # points: BxNx2
    # depths: BxNx1

    agent_pose = []
    w, h, half_fov = [], [], []
    for metadata in metadatas:
        pose = dict_to_list(metadata["cameraPosition"])
        pose.append(metadata["agent"]["cameraHorizon"])
        pose.append(metadata["agent"]["rotation"]["y"])
        pose.append(0)
        agent_pose.append(pose)
        w.append(metadata["screenWidth"])
        h.append(metadata["screenHeight"])
        half_fov.append(metadata["fov"] // 2)
    agent_pose = torch.Tensor(agent_pose).to(points.device)
    w = torch.Tensor(w).unsqueeze(1).to(points.device)
    h = torch.Tensor(h).unsqueeze(1).to(points.device)
    half_fov = torch.Tensor(half_fov).unsqueeze(1).to(points.device)

    new_points = project_2d_to_3d(points, depths, agent_pose, w, h, half_fov)
    return new_points

def local_project_2d_points_to_3d(metadatas, points, depths):
    # events: B
    # points: BxNx2
    # depths: BxNx1

    w, h, half_fov = [], [], []
    for metadata in metadatas:
        w.append(metadata["screenWidth"])
        h.append(metadata["screenHeight"])
        half_fov.append(metadata["fov"] // 2)
    w = torch.Tensor(w).unsqueeze(1).to(points.device)
    h = torch.Tensor(h).unsqueeze(1).to(points.device)
    half_fov = torch.Tensor(half_fov).unsqueeze(1).to(points.device)

    new_points = local_project_2d_to_3d(points, depths, w, h, half_fov)
    return new_points

def project_3d_points_to_2d(metadatas, points):
    # events: B
    # points: BxNx2

    agent_pose = []
    w, h, half_fov = [], [], []
    for metadata in metadatas:
        pose = dict_to_list(metadata["cameraPosition"])
        pose.append(metadata["agent"]["cameraHorizon"])
        pose.append(metadata["agent"]["rotation"]["y"])
        pose.append(0)
        agent_pose.append(pose)
        w.append(metadata["screenWidth"])
        h.append(metadata["screenHeight"])
        half_fov.append(metadata["fov"] // 2)
    agent_pose = torch.Tensor(agent_pose).to(points.device)
    w = torch.Tensor(w).unsqueeze(1).to(points.device)
    h = torch.Tensor(h).unsqueeze(1).to(points.device)
    half_fov = torch.Tensor(half_fov).unsqueeze(1).to(points.device)

    new_points = project_3d_to_2d(points, agent_pose, w, h, half_fov)
    return new_points

def get_affine_matrix(pose):
    # pose: Bx6
    # output: Bx4x4

    b = pose.shape[0]
    r = get_object_rotation_matrix_batch(pose[:, 3:])
    t = pose[:, :3].unsqueeze(2)
    m = torch.cat((r, t), dim=2)
    o = torch.Tensor([0, 0, 0, 1]).to(pose.device)
    m = torch.cat((m, o.view(1, 1, 4).repeat(b, 1, 1)), dim=1)
    return m

def get_gt_affine_matrix(metadatas_a, metadatas_b, ids):
    # metadatas_a: B
    # metadatas_b: B
    # ids: B
    # output: Bx4x4

    obj_a_pose = []
    for i, metadata in enumerate(metadatas_a):
        obj_a_pos = dict_to_list(metadata["objects"][ids[i]]["position"])
        obj_a_rot = dict_to_list(metadata["objects"][ids[i]]["rotation"])
        obj_a_pose.append(obj_a_pos + obj_a_rot)
    obj_a_pose = torch.Tensor(obj_a_pose)
    m_a = get_affine_matrix(obj_a_pose)
    m_a = torch.inverse(m_a)

    obj_b_pose = []
    for i, metadata in enumerate(metadatas_b):
        obj_b_pos = dict_to_list(metadata["objects"][ids[i]]["position"])
        obj_b_rot = dict_to_list(metadata["objects"][ids[i]]["rotation"])
        obj_b_pose.append(obj_b_pos + obj_b_rot)
    obj_b_pose = torch.Tensor(obj_b_pose)
    m_b = get_affine_matrix(obj_b_pose)

    m = torch.bmm(m_b, m_a)
    m[:, 0, 2] *= -1
    m[:, 1, 2] *= -1
    m[:, 2, 0] *= -1
    m[:, 2, 1] *= -1
    m[:, 2, 3] *= -1
    return m

def get_gt_affine_matrix_by_pose(obj_a_pose, obj_b_pose):
    # obj_a_pose: Bx6
    # obj_b_pose: Bx6
    # output: Bx4x4

    m_a = get_affine_matrix(obj_a_pose)
    m_a = torch.inverse(m_a)

    m_b = get_affine_matrix(obj_b_pose)

    m = torch.bmm(m_b, m_a)
    m[:, 0, 2] *= -1
    m[:, 1, 2] *= -1
    m[:, 2, 0] *= -1
    m[:, 2, 1] *= -1
    m[:, 2, 3] *= -1
    return m

def draw_convex_hull_w_gt_affine_matrix_torch(event_a, event_b, id):
    from scipy.spatial import ConvexHull

    def get_convex_hull(event, objectId):
        mask = event.instance_masks[objectId]
        index = np.transpose(np.array(np.where(mask)), (1, 0))
        hull = ConvexHull(index)
        points = index[hull.vertices, :]
        return points

    objectId = event_a.metadata["objects"][id]["objectId"]
    points = get_convex_hull(event_a, objectId)
    img = Image.fromarray(event_a.frame, "RGB")
    img = draw_point(img, points)
    img.save("convex_hull_a.png")

    b = 2
    n = len(points)

    metadatas_a = [event_a.metadata] * b
    metadatas_b = [event_b.metadata] * b
    ids = [id] * b
    M = get_gt_affine_matrix(metadatas_a, metadatas_b, ids)
    depths = [[event_a.depth_frame[x, y] for (x, y) in points]] * b
    depths = torch.Tensor(depths).unsqueeze(2)
    points = [[[y, x] for (x, y) in points]] * b
    points = torch.Tensor(points)

    points_3d = project_2d_points_to_3d(metadatas_a, points, depths)
    points_3d = torch.cat((points_3d, torch.ones((b, n, 1))), dim=2)
    points_3d = torch.transpose(points_3d, 1, 2)
    points_3d[:, 2, :] *= -1
    next_points_3d = torch.bmm(M, points_3d)
    next_points_3d[:, 2, :] *= -1
    next_points_3d = torch.transpose(next_points_3d, 1, 2)[:, :, :3]

    next_points_2d = project_3d_points_to_2d(metadatas_b, next_points_3d)

    for i in range(b):
        img = Image.fromarray(event_b.frame, "RGB")
        ps = next_points_2d[i].detach().numpy()
        ps = np.array([[y, x] for (x, y) in ps])
        img = draw_point(img, ps)
        img.save("convex_hull_b_{}.png".format(i))

def draw_point(img, points):
    draw = ImageDraw.Draw(img)
    for p in points:
        shape = [(p[1] - 1, p[0] - 1),
                 (p[1] + 1, p[0] + 1)]
        draw.ellipse(shape, fill=(255, 255, 0))
    return img

def get_corners(mask, depth_mask):
    x, y = np.where(mask)
    if len(x) == 0 or len(y) == 0:
        corners = [[0, 0]] * 8
        depths = [[0]] * 8
        return np.array(corners), np.array(depths)
    else:
        corners = []
        depths = []

    x1 = np.max(x)
    y1 = y[np.argmax(x)]
    corners.append([y1, x1])
    depths.append(depth_mask[x1, y1])

    y2 = np.max(y)
    x2 = x[np.argmax(y)]
    corners.append([y2, x2])
    depths.append(depth_mask[x2, y2])

    x3 = np.min(x)
    y3 = y[np.argmin(x)]
    corners.append([y3, x3])
    depths.append(depth_mask[x3, y3])

    y4 = np.min(y)
    x4 = x[np.argmin(y)]
    corners.append([y4, x4])
    depths.append(depth_mask[x4, y4])

    s = x + y
    idx = np.argmax(s)
    x5 = x[idx]
    y5 = y[idx]
    corners.append([y5, x5])
    depths.append(depth_mask[x5, y5])

    idx = np.argmin(s)
    x6 = x[idx]
    y6 = y[idx]
    corners.append([y6, x6])
    depths.append(depth_mask[x6, y6])

    s = x - y
    idx = np.argmax(s)
    x7 = x[idx]
    y7 = y[idx]
    corners.append([y7, x7])
    depths.append(depth_mask[x7, y7])

    idx = np.argmin(s)
    x8 = x[idx]
    y8 = y[idx]
    corners.append([y8, x8])
    depths.append(depth_mask[x8, y8])
    return np.array(corners), np.array(depths)

if __name__ == "__main__":
    rot = torch.Tensor([0, 0, 0])
    r = get_rotation_matrix(rot)
    xx = 0
