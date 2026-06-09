import torch
import numpy as np
import math

device = 'cpu'

def get_problem(name, *args, **kwargs):
    name = name.lower()

    PROBLEM = {
        'polygons': PolygonProblem,
        'omnitest': OmniTest,
        'sympart': SYMPART,
        'idmp1': IDMP1,
        'idmp2': IDMP2,
        'idmp3': IDMP3,
        'idmp4': IDMP4,
        'idmp5': IDMP5,
        'idmp6': IDMP6,
        'idmp7': IDMP7,
        'idmp8': IDMP8,
        'idmp9': IDMP9,
        'idmp10': IDMP10,
        'idmp11': IDMP11,
        'idmp12': IDMP12,
        'mmf1': MMF1,
        'mmf2': MMF2,
        'mmf3': MMF3,
        'mmf4': MMF4,
        'mmf5': MMF5,
        'mmf6': MMF6,
        'mmf7': MMF7,
        'mmf8': MMF8,
        'mmf9': MMF9,
        'mmf1_e': MMF1_e,
        'mmf1_z': MMF1_z,
        'mmf14': MMF14,
        'mmf14_a': MMF14_a,
    }

    if name not in PROBLEM:
        raise Exception("Problem not found.")

    return PROBLEM[name](*args, **kwargs)

def create_polygons(row, col, distance, M):
    polygons = []
    for i in range(row):
        for j in range(col):
            center = np.array([i * distance, j * distance])
            theta = np.linspace(0, 2 * np.pi, M, endpoint=False)
            radius = 1.0
            polygon = np.stack([
                center[0] + radius * np.cos(theta),
                center[1] + radius * np.sin(theta)
            ], axis=1)
            polygons.append(polygon)
    return polygons

class PolygonProblem():
    def __init__(self, n_dim=2):
        self.n_obj = 3
        self.n_dim = n_dim
        self.row = 2
        self.col = 3
        self.distance = 5
        self.polygons = create_polygons(self.row, self.col, self.distance, self.n_obj)

        self.lbound = torch.full((self.n_dim,), -50.0, dtype=torch.float32, device=device)
        self.ubound = torch.full((self.n_dim,), 50.0, dtype=torch.float32, device=device)

        self.ideal_point = np.array([0, 0, 0])
        self.nadir_point = np.array([1.75, 1.75, 1.75])

    def evaluate(self, X):
        X = X * (self.ubound - self.lbound) + self.lbound

        n_samples = X.shape[0]
        N = self.row * self.col
        pop_obj = torch.full((n_samples, self.n_obj), float('inf'), dtype=torch.float32, device=device)

        for i in range(N):
            polygon = self.polygons[i]
            tiled_vertices = np.tile(polygon, (1, self.n_dim // 2))
            vertices_tensor = torch.tensor(tiled_vertices, dtype=torch.float32, device=device)

            dists = torch.norm(X[:, None, :] - vertices_tensor[None, :, :], dim=2)
            pop_obj = torch.minimum(pop_obj, dists)

        return pop_obj

class OmniTest():
    def __init__(self, n_dim=2):
        self.n_obj = 2
        self.n_dim = n_dim
        self.lbound = torch.full((self.n_dim,), 0, dtype=torch.float32, device=device)
        self.ubound = torch.full((self.n_dim,), 6, dtype=torch.float32, device=device)
        self.ideal_point = np.array([-2, -2])
        self.nadir_point = np.array([0, 0])
    def evaluate(self, X):
        X = X * (self.ubound - self.lbound) + self.lbound
        F1 = torch.sum(torch.sin(torch.pi * X), dim=1)  # shape: (batch_size,)
        F2 = torch.sum(torch.cos(torch.pi * X), dim=1)  # shape: (batch_size,)
        objs = torch.stack([F1, F2]).T
        return objs

class SYMPART():
    def __init__(self, n_dim=2):
        self.n_obj = 2
        self.n_dim = n_dim
        self.a = 1
        self.b = 10
        self.c = 10
        # self.w = np.pi / 4
        self.w = 0

        # Calculate the inverted rotation matrix, store for fitness evaluation
        self.IRM = np.array([
            [np.cos(self.w), np.sin(self.w)],
            [-np.sin(self.w), np.cos(self.w)]])

        r = max(self.b, self.c)

        self.lbound = torch.full((self.n_dim,), -10*r, dtype=torch.float32, device=device)
        self.ubound = torch.full((self.n_dim,), 10*r, dtype=torch.float32, device=device)
        self.ideal_point = np.array([0, 0])
        self.nadir_point = np.array([4, 4])
    def evaluate(self, X):
        X = X * (self.ubound - self.lbound) + self.lbound
        if self.w == 0:
            X1 = X[:, 0]
            X2 = X[:, 1]
        else:
            # If rotated, we rotate it back by applying the inverted rotation matrix to X
            Y = X @ torch.tensor(self.IRM, dtype=X.dtype, device=X.device).T
            X1 = Y[:, 0]
            X2 = Y[:, 1]

        a, b, c = self.a, self.b, self.c
        t1_hat = torch.sign(X1) * torch.ceil((torch.abs(X1) - a - c / 2) / (2 * a + c))
        t2_hat = torch.sign(X2) * torch.ceil((torch.abs(X2) - b / 2) / b)
        one = torch.ones_like(X1)
        t1 = torch.sign(t1_hat) * torch.min(torch.stack([torch.abs(t1_hat), one], dim=0), dim=0)[0]
        t2 = torch.sign(t2_hat) * torch.min(torch.stack([torch.abs(t2_hat), one], dim=0), dim=0)[0]
        p1 = X1 - t1 * c
        p2 = X2 - t2 * b
        f1 = (p1 + a) ** 2 + p2 ** 2
        f2 = (p1 - a) ** 2 + p2 ** 2
        objs = torch.stack([f1, f2]).T
        return objs

class IDMP1():
    def __init__(self, a=3):
        self.a = a
        self.n_obj = 2
        self.n_dim = 2
        psize = torch.tensor([0.1, 0.1], device=device)
        center = torch.tensor([-0.5, 0.5], device=device)
        self.points = torch.stack((center - psize, center + psize), dim=0)
        self.lbound = torch.tensor([-1.0, -1.0], dtype=torch.float32, device=device)
        self.ubound = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
        self.ideal_point = np.array([0, 0])
        self.nadir_point = np.array([0.2, 0.2])
    def evaluate(self, X):
        X = X * (self.ubound - self.lbound) + self.lbound
        N = X.shape[0]
        a = self.a
        F = torch.zeros((N, 2), device=device)
        for i in range(2):
            point_i = self.points[i].unsqueeze(0).expand(N, -1)
            t = torch.abs(X[:, 0:1] - point_i)
            t[:, 0] += torch.abs(X[:, 1] + 0.5)
            t[:, 1] += a * torch.abs(X[:, 1] - 0.5)
            F[:, i] = torch.min(t, dim=1).values
        return F

class IDMP2():
    def __init__(self, a=0.4):
        self.a = a
        psize = torch.tensor([0.1, 0.1], device=device)
        center = torch.tensor([-0.5, 0.5], device=device)
        self.points = torch.vstack([center - psize, center + psize])
        self.n_obj = 2
        self.n_dim = 2
        self.lbound = torch.tensor([-1.0, -1.0], device=device)
        self.ubound = torch.tensor([1.0, 1.0], device=device)
        self.ideal_point = np.array([0, 0])
        self.nadir_point = np.array([0.2, 0.2])

    def evaluate(self, X):
        X = X * (self.ubound - self.lbound) + self.lbound
        N = X.shape[0]
        F = torch.zeros((N, 2), device=device)
        a = self.a
        for i in range(2):
            point_i = self.points[i].unsqueeze(0).expand(N, -1)
            t = torch.abs(X[:, 0:1] - point_i)
            t[:, 0] += 100 * torch.abs(X[:, 1] + 0.5) ** 2
            t[:, 1] += 100 * torch.abs(X[:, 1] - 0.5) ** (2 - a)
            F[:, i] = torch.min(t, dim=1).values

        return F


class IDMP3():
    def __init__(self, a=0.4):
        self.a = a
        psize = torch.tensor([0.1, 0.1], device=device)
        center = torch.tensor([-0.5, 0.5], device=device)
        self.points = torch.vstack([center - psize, center + psize])
        self.n_obj = 2
        self.n_dim = 2
        self.lbound = torch.tensor([-1.0, -1.0], device=device)
        self.ubound = torch.tensor([1.0, 1.0], device=device)
        self.ideal_point = np.array([0, 0])
        self.nadir_point = np.array([0.2, 0.2])

    def evaluate(self, X):
        X = X * (self.ubound - self.lbound) + self.lbound
        N = X.shape[0]
        F = torch.zeros((N, 2), device=device)
        a = self.a
        for i in range(2):
            point_i = self.points[i].unsqueeze(0).expand(N, -1)
            t = torch.abs(X[:, 0:1] - point_i)
            t[:, 0] += 100 * torch.abs(X[:, 1] + 0.5) ** 2
            t[:, 1] += 100 * (X[:, 1] - 0.5 + a * (X[:, 0] - 0.5)) ** 2
            F[:, i] = torch.min(t, dim=1).values

        return F


class IDMP4():
    def __init__(self, a=4):
        self.a = a
        psize = torch.tensor([0.1, 0.1], device=device)
        center = torch.tensor([-0.5, 0.5], device=device)
        self.points = torch.vstack([center - psize, center + psize])
        self.n_obj = 2
        self.n_dim = 2
        self.lbound = torch.tensor([-1.0, -1.0], device=device)
        self.ubound = torch.tensor([1.0, 1.0], device=device)
        self.ideal_point = np.array([0, 0])
        self.nadir_point = np.array([0.2, 0.2])

    def evaluate(self, X):
        X = X * (self.ubound - self.lbound) + self.lbound
        N = X.shape[0]
        F = torch.zeros((N, 2), device=device)
        a = self.a
        x2 = X[:, 1]
        for i in range(2):
            point_i = self.points[i].unsqueeze(0).expand(N, -1)
            t = torch.abs(X[:, 0:1] - point_i)
            t[:, 0] += 100 * (torch.abs(x2 + 0.5) ** 2 + 1 - torch.cos(2 * np.pi * (x2 + 0.5)))
            t[:, 1] += 100 * (torch.abs(x2 - 0.5) ** 2 + 1 - torch.cos(a * 2 * np.pi * (x2 - 0.5)))
            F[:, i] = torch.min(t, dim=1).values

        return F

class IDMP5():
    def __init__(self):
        self.n_obj = 3
        self.n_dim = 3
        self.lbound = torch.tensor([-1.0, -1.0, -1.0], device=device)
        self.ubound = torch.tensor([1.0, 1.0, 1.0], device=device)
        # 1. 直接使用 MATLAB 导出的 pgon 顶点数值
        pgon = torch.tensor([
            [-0.8660, -0.5000],
            [0, 1.000],
            [0.8660, -0.5000]
        ], dtype=torch.float32, device=device)
        psize = torch.tensor([0.1, 0.1, 0.1, 0.1], device=device)
        centers = torch.tensor([[-0.5,-0.5],[0.5,-0.5],[0.5,0.5],[-0.5,0.5]], device=device)
        # 完全对齐 MATLAB: Points(M, 2, 4)
        self.points = torch.empty((self.n_obj, 2, 4), device=device)

        for i in range(4):
            self.points[:, :, i] = pgon * psize[i] + centers[i]

        self.ideal_point = np.array([0.00287324, 0.00372583, 0.00287324])
        self.nadir_point = np.array([0.17081657, 0.17091574, 0.17081657])

    def evaluate(self, X):
        # X: normalized -> real
        X = X * (self.ubound - self.lbound) + self.lbound
        N = X.shape[0]
        F = torch.zeros((N, self.n_obj), device=device)
        x12 = X[:, :2]  # (N,2)
        x3 = X[:, 2]  # (N,)
        for i in range(self.n_obj):
            # 等价 MATLAB: reshape(obj.Points(i,:,:),[2,4])'
            V = self.points[i].T  # (4,2)
            # pdist2
            diff = x12[:, None, :] - V[None, :, :]  # (N,4,2)
            temp = torch.norm(diff, dim=2)  # (N,4)
            # 不平衡惩罚（逐项一致）
            temp[:, 0] += 1.0 * torch.abs(x3 + 0.6)
            temp[:, 1] += 2.0 * torch.abs(x3 + 0.2)
            temp[:, 2] += 3.0 * torch.abs(x3 - 0.2)
            temp[:, 3] += 4.0 * torch.abs(x3 - 0.6)
            # min(temp,[],2)
            F[:, i] = torch.min(temp, dim=1).values

        return F


class IDMP6():
    def __init__(self):
        self.n_obj = 3
        self.n_dim = 3
        self.lbound = torch.tensor([-1.0, -1.0, -1.0], device=device)
        self.ubound = torch.tensor([1.0, 1.0, 1.0], device=device)
        # 1. 直接使用 MATLAB 导出的 pgon 顶点数值
        pgon = torch.tensor([
            [-0.8660, -0.5000],
            [0, 1.000],
            [0.8660, -0.5000]
        ], dtype=torch.float32, device=device)
        psize = torch.tensor([0.1,0.1,0.1,0.1], device=device)
        centers = torch.tensor([[-0.5,-0.5],[0.5,-0.5],[0.5,0.5],[-0.5,0.5]], device=device)
        self.points = torch.zeros((self.n_obj, 4, 2), device=device)
        for i in range(4):
            self.points[:, i, :] = pgon * psize[i] + centers[i]

        self.pw = torch.tensor([2.0,1.8,1.6,1.4], device=device)
        self.bias_center = torch.tensor([0.6,0.2,-0.2,-0.6], device=device)

        self.ideal_point = np.array([0.00287324, 0.00372583, 0.00287324])
        self.nadir_point = np.array([0.17081657, 0.17091574, 0.17081657])

    def evaluate(self, X):
        X = X * (self.ubound - self.lbound) + self.lbound
        N = X.shape[0]
        F = torch.zeros((N, self.n_obj), device=device)
        x12 = X[:, :2]
        x3 = X[:, 2]

        for i in range(self.n_obj):
            diff = x12[:, None, :] - self.points[i][None, :, :]  # (N,4,2)
            temp = torch.norm(diff, dim=2)
            temp += 100.0 * torch.abs(x3[:, None] - (-self.bias_center)) ** self.pw  # 广播计算
            F[:, i] = torch.min(temp, dim=1).values

        return F


class IDMP7():
    def __init__(self):
        self.n_obj = 3
        self.n_dim = 3
        self.lbound = torch.tensor([-1.0,-1.0,-1.0], device=device)
        self.ubound = torch.tensor([1.0,1.0,1.0], device=device)
        # 1. 直接使用 MATLAB 导出的 pgon 顶点数值
        pgon = torch.tensor([
            [-0.8660, -0.5000],
            [0, 1.000],
            [0.8660, -0.5000]
        ], dtype=torch.float32, device=device)
        psize = torch.tensor([0.1,0.1,0.1,0.1], device=device)
        centers = torch.tensor([[-0.5,-0.5],[0.5,-0.5],[0.5,0.5],[-0.5,0.5]], device=device)
        self.points = torch.zeros((self.n_obj,4,2), device=device)
        for i in range(4):
            self.points[:,i,:] = pgon * psize[i] + centers[i]

        self.ideal_point = np.array([0.00287324, 0.00372583, 0.00287324])
        self.nadir_point = np.array([0.17081657, 0.17091574, 0.17081657])

    def evaluate(self, X):
        X = X * (self.ubound - self.lbound) + self.lbound
        N = X.shape[0]
        F = torch.zeros((N,self.n_obj), device=device)
        x1 = X[:,0]; x2 = X[:,1]; x3 = X[:,2]
        x12 = X[:, :2]

        for i in range(self.n_obj):
            diff = x12[:, None, :] - self.points[i][None, :, :]
            temp = torch.norm(diff, dim=2)
            t2 = (x1-0.5)+(x2+0.5)
            t3 = (x1-0.5)+(x2-0.5)
            t4 = (x1+0.5)+(x2-0.5)
            temp[:,0] += 100.0*(x3+0.6)**2
            temp[:,1] += 100.0*(x3+0.2+0.1*t2)**2
            temp[:,2] += 100.0*(x3-0.2+0.2*t3)**2
            temp[:,3] += 100.0*(x3-0.6+0.3*t4)**2
            F[:,i] = torch.min(temp,dim=1).values

        return F

class IDMP8():
    def __init__(self):
        self.n_obj = 3
        self.n_dim = 3
        self.lbound = torch.tensor([-1.0,-1.0,-1.0], device=device)
        self.ubound = torch.tensor([1.0,1.0,1.0], device=device)
        theta = np.pi / 2 - np.linspace(0, 2 * np.pi, self.n_obj + 1)[:-1]
        pgon = torch.tensor([
            [-0.8660, -0.5000],
            [0, 1.000],
            [0.8660, -0.5000]
        ], dtype=torch.float32, device=device)
        psize = torch.tensor([0.1,0.1,0.1,0.1], device=device)
        centers = torch.tensor([[-0.5,-0.5],[0.5,-0.5],[0.5,0.5],[-0.5,0.5]], device=device)
        self.points = torch.zeros((self.n_obj,4,2), device=device)
        for j in range(4):
            self.points[:,j,:] = pgon * psize[j] + centers[j]

        self.ideal_point = np.array([0.00287324, 0.00372583, 0.00287324])
        self.nadir_point = np.array([0.17081657, 0.17091574, 0.17081657])

    def evaluate(self, X):
        X = X * (self.ubound - self.lbound) + self.lbound
        N = X.shape[0]
        F = torch.zeros((N,self.n_obj), device=device)
        x1 = X[:,0]; x2 = X[:,1]; x3 = X[:,2]
        x12 = X[:,:2]

        for i in range(self.n_obj):
            diff = x12[:,None,:] - self.points[i][None,:,:]
            temp = torch.norm(diff, dim=2)
            temp[:,0] += 100*((x3+0.6)**2 + 1 - torch.cos(1*2*np.pi*(x3+0.6)))
            temp[:,1] += 100*((x3+0.2)**2 + 1 - torch.cos(2*2*np.pi*(x3+0.2)))
            temp[:,2] += 100*((x3-0.2)**2 + 1 - torch.cos(3*2*np.pi*(x3-0.2)))
            temp[:,3] += 100*((x3-0.6)**2 + 1 - torch.cos(4*2*np.pi*(x3-0.6)))
            F[:,i] = torch.min(temp, dim=1).values

        return F


class IDMP9():
    def __init__(self):
        self.n_obj = 4
        self.n_dim = 4
        self.lbound = torch.tensor([-1.0,-1.0,-1.0,-1.0], device=device)
        self.ubound = torch.tensor([1.0,1.0,1.0,1.0], device=device)
        theta = np.pi / 2 - np.linspace(0, 2 * np.pi, self.n_obj + 1)[:-1]
        pgon = torch.tensor([
            [-0.7071, -0.7071],
            [-0.7071, 0.7071],
            [0.7071, 0.7071],
            [0.7071, -0.7071],
        ], dtype=torch.float32, device=device)
        psize = torch.tensor([0.1,0.1,0.1,0.1], device=device)
        centers = torch.tensor([[-0.5,-0.5],[0.5,-0.5],[0.5,0.5],[-0.5,0.5]], device=device)
        self.points = torch.zeros((self.n_obj,4,2), device=device)
        for j in range(4):
            self.points[:,j,:] = pgon * psize[j] + centers[j]

        self.ideal_point = np.array([0.0, 0.0, 0.0, 0.0])
        self.nadir_point = np.array([0.2, 0.2, 0.2, 0.2])

    def evaluate(self, X):
        X = X * (self.ubound - self.lbound) + self.lbound
        N = X.shape[0]
        F = torch.zeros((N,self.n_obj), device=device)
        x12 = X[:,:2]
        x3 = X[:,2]
        x4 = X[:,3]

        for i in range(self.n_obj):
            diff = x12[:,None,:] - self.points[i][None,:,:]
            temp = torch.norm(diff, dim=2)
            temp[:,0] += 1.0*(torch.abs(x3+0.6)+torch.abs(x4+0.6))
            temp[:,1] += 2.0*(torch.abs(x3+0.2)+torch.abs(x4+0.2))
            temp[:,2] += 3.0*(torch.abs(x3-0.2)+torch.abs(x4-0.2))
            temp[:,3] += 4.0*(torch.abs(x3-0.6)+torch.abs(x4-0.6))
            F[:,i] = torch.min(temp, dim=1).values

        return F


class IDMP10():
    def __init__(self):
        self.n_obj = 4
        self.n_dim = 4
        self.lbound = torch.tensor([-1.0,-1.0,-1.0,-1.0], device=device)
        self.ubound = torch.tensor([1.0,1.0,1.0,1.0], device=device)
        theta = np.pi / 2 - np.linspace(0, 2 * np.pi, self.n_obj + 1)[:-1]
        pgon = torch.tensor([
            [-0.7071, -0.7071],
            [-0.7071, 0.7071],
            [0.7071, 0.7071],
            [0.7071, -0.7071],
        ], dtype=torch.float32, device=device)
        psize = torch.tensor([0.1,0.1,0.1,0.1], device=device)
        centers = torch.tensor([[-0.5,-0.5],[0.5,-0.5],[0.5,0.5],[-0.5,0.5]], device=device)
        self.points = torch.zeros((self.n_obj,4,2), device=device)
        for j in range(4):
            self.points[:,j,:] = pgon * psize[j] + centers[j]

        self.ideal_point = np.array([0.0, 0.0, 0.0, 0.0])
        self.nadir_point = np.array([0.2, 0.2, 0.2, 0.2])

    def evaluate(self, X):
        X = X * (self.ubound - self.lbound) + self.lbound
        N = X.shape[0]
        F = torch.zeros((N,self.n_obj), device=device)
        x12 = X[:,:2]
        x3 = X[:,2]
        x4 = X[:,3]

        for i in range(self.n_obj):
            diff = x12[:,None,:] - self.points[i][None,:,:]
            temp = torch.norm(diff, dim=2)
            temp[:,0] += 100*(torch.abs(x3+0.6)**2 + torch.abs(x4+0.6)**2)
            temp[:,1] += 100*(torch.abs(x3+0.2)**1.8 + torch.abs(x4+0.2)**1.8)
            temp[:,2] += 100*(torch.abs(x3-0.2)**1.6 + torch.abs(x4-0.2)**1.6)
            temp[:,3] += 100*(torch.abs(x3-0.6)**1.4 + torch.abs(x4-0.6)**1.4)
            F[:,i] = torch.min(temp, dim=1).values

        return F


class IDMP11():
    def __init__(self):
        self.n_obj = 4
        self.n_dim = 4
        self.lbound = torch.tensor([-1.0, -1.0, -1.0, -1.0], device=device)
        self.ubound = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device)
        theta = np.pi / 2 - np.linspace(0, 2 * np.pi, self.n_obj + 1)[:-1]
        pgon = torch.tensor([
            [-0.7071, -0.7071],
            [-0.7071, 0.7071],
            [0.7071, 0.7071],
            [0.7071, -0.7071],
        ], dtype=torch.float32, device=device)
        psize = torch.tensor([0.1, 0.1, 0.1, 0.1], device=device)
        centers = torch.tensor([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], device=device)

        self.points = torch.zeros((self.n_obj, 4, 2), device=device)
        for j in range(4):
            self.points[:, j, :] = pgon * psize[j] + centers[j]

        self.ideal_point = np.array([0.0, 0.0, 0.0, 0.0])
        self.nadir_point = np.array([0.2, 0.2, 0.2, 0.2])

    def evaluate(self, X):
        X = X * (self.ubound - self.lbound) + self.lbound
        N = X.shape[0]
        F = torch.zeros((N, self.n_obj), device=device)
        x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        x12 = X[:, :2]

        t2 = (x1 - 0.5) + (x2 + 0.5)
        t3 = (x1 - 0.5) + (x2 - 0.5)
        t4 = (x1 + 0.5) + (x2 - 0.5)

        for i in range(self.n_obj):
            diff = x12[:, None, :] - self.points[i][None, :, :]  # (N,4,2)
            temp = torch.norm(diff, dim=2)
            temp[:, 0] += 100 * ((x3 + 0.6) ** 2 + (x4 + 0.6) ** 2)
            temp[:, 1] += 100 * ((x3 + 0.2 + 0.05 * t2) ** 2 + (x4 + 0.2 + 0.05 * t2) ** 2)
            temp[:, 2] += 100 * ((x3 - 0.2 + 0.1 * t3) ** 2 + (x4 - 0.2 + 0.1 * t3) ** 2)
            temp[:, 3] += 100 * ((x3 - 0.6 + 0.15 * t4) ** 2 + (x4 - 0.6 + 0.15 * t4) ** 2)
            F[:, i] = torch.min(temp, dim=1).values

        return F


class IDMP12():
    def __init__(self):
        self.n_obj = 4
        self.n_dim = 4
        self.lbound = torch.tensor([-1.0, -1.0, -1.0, -1.0], device=device)
        self.ubound = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device)
        theta = np.pi / 2 - np.linspace(0, 2 * np.pi, self.n_obj + 1)[:-1]
        pgon = torch.tensor([
            [-0.7071, -0.7071],
            [-0.7071, 0.7071],
            [0.7071, 0.7071],
            [0.7071, -0.7071],
        ], dtype=torch.float32, device=device)
        psize = torch.tensor([0.1, 0.1, 0.1, 0.1], device=device)
        centers = torch.tensor([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], device=device)

        self.points = torch.zeros((self.n_obj, 4, 2), device=device)
        for j in range(4):
            self.points[:, j, :] = pgon * psize[j] + centers[j]

        self.ideal_point = np.array([0.0, 0.0, 0.0, 0.0])
        self.nadir_point = np.array([0.2, 0.2, 0.2, 0.2])

    def evaluate(self, X):
        X = X * (self.ubound - self.lbound) + self.lbound
        N = X.shape[0]
        F = torch.zeros((N, self.n_obj), device=device)
        x12 = X[:, :2]
        x3, x4 = X[:, 2], X[:, 3]

        for i in range(self.n_obj):
            diff = x12[:, None, :] - self.points[i][None, :, :]  # (N,4,2)
            temp = torch.norm(diff, dim=2)
            temp[:, 0] += 100 * ((x3 + 0.6) ** 2 + 1 - torch.cos(1 * 2 * np.pi * (x3 + 0.6))) + 100 * (x4 + 0.6) ** 2
            temp[:, 1] += 100 * ((x3 + 0.2) ** 2 + 1 - torch.cos(2 * 2 * np.pi * (x3 + 0.2))) + 100 * (x4 + 0.2) ** 2
            temp[:, 2] += 100 * ((x3 - 0.2) ** 2 + 1 - torch.cos(3 * 2 * np.pi * (x3 - 0.2))) + 100 * (x4 - 0.2) ** 2
            temp[:, 3] += 100 * ((x3 - 0.6) ** 2 + 1 - torch.cos(4 * 2 * np.pi * (x3 - 0.6))) + 100 * (x4 - 0.6) ** 2
            F[:, i] = torch.min(temp, dim=1).values

        return F


class MMF1():
    def __init__(self, n_dim=2):
        self.n_obj = 2
        self.n_dim = n_dim
        self.lbound = torch.tensor([1.0, -1.0], dtype=torch.float32, device=device)
        self.ubound = torch.tensor([3.0, 1.0], dtype=torch.float32, device=device)
        self.ideal_point = np.array([0, 0])
        self.nadir_point = np.array([1, 1])

    def evaluate(self, X):
        X = X * (self.ubound - self.lbound) + self.lbound
        x1 = torch.abs(X[:, 0] - 2.0)
        x2 = X[:, 1]
        f1 = x1
        f2 = 1.0 - torch.sqrt(x1) + 2.0 * (x2 - torch.sin(6 * np.pi * x1 + np.pi)) ** 2
        objs = torch.stack([f1, f2]).T
        return objs


class MMF2():
    def __init__(self, n_dim=2):
        self.n_obj = 2
        self.n_dim = n_dim
        self.lbound = torch.tensor([0.0, 0.0], dtype=torch.float32, device=device)
        self.ubound = torch.tensor([1.0, 2.0], dtype=torch.float32, device=device)
        self.ideal_point = np.array([0, 0])
        self.nadir_point = np.array([1, 1])

    def evaluate(self, X):
        X = X * (self.ubound - self.lbound) + self.lbound
        x1 = X[:, 0]
        x2 = X[:, 1]
        x2 = x2 - (x2 > 1).float()
        f1 = x1
        y2 = x2 - torch.sqrt(x1)
        f2 = 1 - torch.sqrt(x1) + 2 * ((4 * y2 ** 2) - 2 * torch.cos(
            20 * y2 * torch.pi / torch.sqrt(torch.tensor(2.0, device=device))) + 2)
        objs = torch.stack([f1, f2], dim=1)
        return objs


class MMF3():
    def __init__(self, n_dim=2):
        self.n_obj = 2
        self.n_dim = n_dim
        self.lbound = torch.tensor([0.0, 0.0], dtype=torch.float32, device=device)
        self.ubound = torch.tensor([1.0, 1.5], dtype=torch.float32, device=device)
        self.ideal_point = np.array([0, 0])
        self.nadir_point = np.array([1, 1])

    def evaluate(self, X):
        X = X * (self.ubound - self.lbound) + self.lbound
        x1 = X[:, 0]
        x2 = X[:, 1]
        f1 = x1
        tmp = torch.zeros_like(x1, device=device)
        mask1 = (0 <= x2) & (x2 <= 0.5)
        mask2 = (0.5 < x2) & (x2 < 1) & (x1 <= 0.25)
        mask3 = (0.5 < x2) & (x2 < 1) & (x1 > 0.25)
        mask4 = (1 <= x2) & (x2 <= 1.5)
        mask_else = ~(mask1 | mask2 | mask3 | mask4)
        tmp[mask1] = x2[mask1] - torch.sqrt(x1[mask1])
        tmp[mask2] = x2[mask2] - 0.5 - torch.sqrt(x1[mask2])
        tmp[mask3] = x2[mask3] - torch.sqrt(x1[mask3])
        tmp[mask4] = x2[mask4] - 0.5 - torch.sqrt(x1[mask4])
        tmp[mask_else] = x2[mask_else] - torch.sqrt(x1[mask_else])
        f2 = 1 - torch.sqrt(x1) + 2 * ((4 * tmp ** 2) - 2 * torch.cos(
            20 * tmp * torch.pi / torch.sqrt(torch.tensor(2.0, device=device))) + 2)
        objs = torch.stack([f1, f2], dim=1)
        return objs

class MMF4():
    def __init__(self, n_dim=2):
        self.n_obj = 2
        self.n_dim = n_dim
        self.lbound = torch.tensor([-1.0, 0.0], dtype=torch.float32, device=device)
        self.ubound = torch.tensor([1.0, 2.0], dtype=torch.float32, device=device)
        self.ideal_point = np.array([0, 0])
        self.nadir_point = np.array([1, 1])
    def evaluate(self, X):
        X = X * (self.ubound - self.lbound) + self.lbound
        X1 = X[:, 0]
        X2 = X[:, 1]
        X2_mod = torch.where(X2 > 1.0, X2 - 1.0, X2)
        f1 = torch.abs(X1)
        f2 = 1.0 - X1 ** 2 + 2.0 * (X2_mod - torch.sin(torch.pi * torch.abs(X1))) ** 2
        objs = torch.stack([f1, f2]).T
        return objs

class MMF5():
    def __init__(self, n_dim=2):
        self.n_obj = 2
        self.n_dim = n_dim
        self.lbound = torch.tensor([1.0, -1.0], dtype=torch.float32, device=device)
        self.ubound = torch.tensor([3.0, 3.0], dtype=torch.float32, device=device)
        self.ideal_point = np.array([0, 0])
        self.nadir_point = np.array([1, 1])

    def evaluate(self, X):
        X = X * (self.ubound - self.lbound) + self.lbound
        x1 = X[:, 0]
        x2 = X[:, 1]
        x2_mod = x2 - 2.0 * (x2 > 1.0).float()
        f1 = torch.abs(x1 - 2.0)
        f2 = 1.0 - torch.sqrt(f1) + 2.0 * (x2_mod - torch.sin(6 * torch.pi * f1 + torch.pi)) ** 2
        return torch.stack([f1, f2], dim=1)

class MMF6():
    def __init__(self, n_dim=2):
        self.n_obj = 2
        self.n_dim = n_dim
        self.lbound = torch.tensor([1.0, -1.0], dtype=torch.float32, device=device)
        self.ubound = torch.tensor([3.0, 2.0], dtype=torch.float32, device=device)
        self.ideal_point = np.array([0, 0])
        self.nadir_point = np.array([1, 1])

    def evaluate(self, X):
        X = X * (self.ubound - self.lbound) + self.lbound
        x1 = X[:, 0]
        x2 = X[:, 1]

        # 定义区间掩码
        a1, a2 = 7/6, 8/6
        b1, b2 = 9/6, 10/6
        c1, c2 = 11/6, 2
        d1, d2 = 2, 13/6
        e1, e2 = 14/6, 15/6
        f1, f2 = 16/6, 17/6

        S1 = (x1 > a1) & (x1 <= a2)
        S2 = (x1 > b1) & (x1 <= b2)
        S3 = (x1 > c1) & (x1 <= c2)
        S4 = (x1 > d1) & (x1 <= d2)
        S5 = (x1 > e1) & (x1 <= e2)
        S6 = (x1 > f1) & (x1 <= f2)

        G1 = (x1 > 1) & (x1 <= a1)
        G2 = (x1 > 4/3) & (x1 <= 3/2)
        G3 = (x1 > 5/3) & (x1 <= c1)

        H1 = (x1 > d1 + 1/6*1) & (x1 <= e1)
        H2 = (x1 > e1) & (x1 <= f1)
        H3 = (x1 > f1) & (x1 <= 3)

        new_v2 = x2.clone()
        # -1 < x2 <= 0
        mask1 = (-1 < x2) & (x2 <= 0) & (S1 | S2 | S3)
        mask2 = (-1 < x2) & (x2 <= 0) & (S4 | S5 | S6)
        # 1 < x2 <= 2
        mask3 = (1 < x2) & (x2 <= 2) & (G1 | G2 | G3)
        mask4 = (1 < x2) & (x2 <= 2) & (H1 | H2 | H3)
        # 0 < x2 <= 1
        mask5 = (0 < x2) & (x2 <= 1) & (G1 | G2 | G3 | H1 | H2 | H3)
        mask6 = (0 < x2) & (x2 <= 1) & (S1 | S2 | S3 | S4 | S5 | S6)

        new_v2[mask3 | mask4 | mask6] -= 1.0
        # mask1, mask2, mask5 不变

        f1 = torch.abs(x1 - 2.0)
        f2 = 1 - torch.sqrt(f1) + 2 * (new_v2 - torch.sin(6 * torch.pi * f1 + torch.pi))**2
        return torch.stack([f1, f2], dim=1)

class MMF7():
    def __init__(self, n_dim=2):
        self.n_obj = 2
        self.n_dim = n_dim
        self.lbound = torch.tensor([1.0, -1.0], dtype=torch.float32, device=device)
        self.ubound = torch.tensor([3.0, 1.0], dtype=torch.float32, device=device)
        self.ideal_point = np.array([0, 0])
        self.nadir_point = np.array([1, 1])
    def evaluate(self, X):
        X = X * (self.ubound - self.lbound) + self.lbound
        x1 = X[:, 0]
        x2 = X[:, 1]
        x2_mod = x2 - 2.0 * (x2 > 1.0).float()
        f1 = torch.abs(x1 - 2.0)
        term = 0.3 * (f1 ** 2) * torch.cos(24 * torch.pi * f1 + 4 * torch.pi) + 0.6 * f1
        inside = x2_mod - term * torch.sin(6 * torch.pi * f1 + torch.pi)
        f2 = 1.0 - torch.sqrt(f1) + inside ** 2
        return torch.stack([f1, f2], dim=1)

class MMF8():
    def __init__(self, n_dim=2):
        self.n_obj = 2
        self.n_dim = n_dim
        self.lbound = torch.tensor([-torch.pi, 0.0], dtype=torch.float32, device=device)
        self.ubound = torch.tensor([torch.pi, 9.0], dtype=torch.float32, device=device)
        self.ideal_point = np.array([0, 0])
        self.nadir_point = np.array([1, 1])

    def evaluate(self, X):
        X = X * (self.ubound - self.lbound) + self.lbound
        x1 = X[:, 0]
        x2 = X[:, 1]
        x2_mod = torch.where(x2 > 4.0, x2 - 4.0, x2)
        f1 = torch.sin(torch.abs(x1))
        f2 = torch.sqrt(1.0 - f1**2) + 2.0 * (x2_mod - (f1 + torch.abs(x1))) ** 2
        return torch.stack([f1, f2], dim=1)


class MMF9():
    def __init__(self, n_dim=2, device='cpu'):
        self.n_obj = 2
        self.n_dim = n_dim
        self.lbound = torch.tensor([0.1, 0.1], dtype=torch.float32, device=device)
        self.ubound = torch.tensor([1.1, 1.1], dtype=torch.float32, device=device)
        self.ideal_point = np.array([0.0, 0.0])
        self.nadir_point = np.array([1.2, 10.0])

    def evaluate(self, X):
        X_real = X * (self.ubound - self.lbound) + self.lbound
        x1 = X_real[:, 0]
        x2 = X_real[:, 1]
        f1 = x1
        g = 2 - torch.pow(torch.sin(2 * torch.pi * x2), 6)
        f2 = g / x1
        return torch.stack([f1, f2], dim=1)


class MMF1_e():
    def __init__(self, n_dim=2, device='cpu'):
        self.n_obj = 2
        self.n_dim = n_dim
        self.device = device

        # 根据 MATLAB Setting 定义边界
        # lower = [1, -20], upper = [3, 20]
        self.lbound = torch.tensor([1.0, -20.0], dtype=torch.float32, device=device)
        self.ubound = torch.tensor([3.0, 20.0], dtype=torch.float32, device=device)

        # 理想点与纳迪尔点（参考值）
        self.ideal_point = np.array([0.0, 0.0])
        self.nadir_point = np.array([1.0, 1.0])

    def evaluate(self, X):
        """
        X: 输入张量，假设已经归一化到 [0, 1] 之间
        """
        # 映射到实际决策空间边界
        X_real = X * (self.ubound - self.lbound) + self.lbound

        x1 = X_real[:, 0]
        x2 = X_real[:, 1]

        # 计算第一个目标函数 f1 = abs(x1 - 2)
        f1 = torch.abs(x1 - 2.0)

        # 预计算通用项
        sqrt_f1 = torch.sqrt(f1)
        sin_term = torch.sin(6.0 * torch.pi * f1 + torch.pi)

        # 分段逻辑处理
        # index1: x1 < 2
        # f2 = 1 - sqrt(f1) + 2 * (x2 - sin(6*pi*f1 + pi))^2
        f2_cond1 = 1.0 - sqrt_f1 + 2.0 * torch.pow(x2 - sin_term, 2)

        # index2: x1 >= 2
        # f2 = 1 - sqrt(f1) + 2 * (x2 - exp(x1) * sin(6*pi*f1 + pi))^2
        f2_cond2 = 1.0 - sqrt_f1 + 2.0 * torch.pow(x2 - torch.exp(x1) * sin_term, 2)

        # 使用 torch.where 根据条件合并结果
        f2 = torch.where(x1 < 2.0, f2_cond1, f2_cond2)

        return torch.stack([f1, f2], dim=1)


import torch
import numpy as np


class MMF1_z():
    def __init__(self, n_dim=2, device='cpu'):
        self.n_obj = 2
        self.n_dim = n_dim
        self.device = device

        # 根据 MATLAB Setting 定义边界
        # lower = [1, -1], upper = [3, 1]
        self.lbound = torch.tensor([1.0, -1.0], dtype=torch.float32, device=device)
        self.ubound = torch.tensor([3.0, 1.0], dtype=torch.float32, device=device)

        # 理想点与纳迪尔点参考
        self.ideal_point = np.array([0.0, 0.0])
        self.nadir_point = np.array([1.0, 1.0])

    def evaluate(self, X):
        """
        X: 输入张量，形状为 (pop_size, n_dim)，取值范围 [0, 1]
        """
        # 1. 映射到实际决策空间 [1, 3] x [-1, 1]
        X_real = X * (self.ubound - self.lbound) + self.lbound

        x1 = X_real[:, 0]
        x2 = X_real[:, 1]

        # 2. 计算第一个目标 f1 = abs(x1 - 2)
        f1 = torch.abs(x1 - 2.0)

        # 3. 计算分段函数 f2
        # 公共部分: 1 - sqrt(f1)
        common_part = 1.0 - torch.sqrt(f1)

        # 情况 1: x1 < 2，使用 sin(6*pi*f1 + pi)
        sin1 = torch.sin(6.0 * torch.pi * f1 + torch.pi)
        f2_cond1 = common_part + 2.0 * torch.pow(x2 - sin1, 2)

        # 情况 2: x1 >= 2，使用 sin(2*pi*f1 + pi)
        sin2 = torch.sin(2.0 * torch.pi * f1 + torch.pi)
        f2_cond2 = common_part + 2.0 * torch.pow(x2 - sin2, 2)

        # 4. 合并结果
        f2 = torch.where(x1 < 2.0, f2_cond1, f2_cond2)

        return torch.stack([f1, f2], dim=1)


class MMF14():
    def __init__(self, n_obj=3, n_dim=3, device='cpu'):
        self.n_obj = n_obj
        self.n_dim = n_dim
        self.device = device

        # 根据 MATLAB Setting 定义边界: [0, 1] 空间
        self.lbound = torch.zeros(n_dim, dtype=torch.float32, device=device)
        self.ubound = torch.ones(n_dim, dtype=torch.float32, device=device)

        # 理想点与纳迪尔点（参考 DTLZ2 结构）
        self.ideal_point = np.zeros(n_obj)
        self.nadir_point = np.ones(n_obj) * 2.0  # 因为 1+g 的最大值约为 2.0

    def evaluate(self, X):
        """
        X: 输入张量，形状为 (batch_size, n_dim)，取值范围 [0, 1]
        """
        # 映射到实际边界（虽然 MMF14 原生就是 [0, 1]）
        X_real = X * (self.ubound - self.lbound) + self.lbound

        N = X_real.shape[0]
        M = self.n_obj
        num_of_peak = 2

        # 1. 计算 g 函数: g = 2 - (sin(num_of_peak * pi * X(:,end)))^2
        # 注意使用最后一个维度 X[:, -1]
        g = 2.0 - torch.pow(torch.sin(num_of_peak * torch.pi * X_real[:, -1]), 2)
        g = g.view(-1, 1)  # 形状变为 (N, 1) 以便后续广播计算

        # 2. 准备球面变换所需的三角函数项 (类似 DTLZ2)
        # 取前 M-1 个变量进行角度变换
        theta = X_real[:, :M - 1] * (torch.pi / 2.0)

        # 计算 cos 和 sin
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        # 3. 核心转换逻辑：利用累乘计算球面坐标
        # MATLAB: fliplr(cumprod([ones(N,1), cos(X(:,1:M-1)*pi/2)], 2))
        ones = torch.ones((N, 1), device=self.device)
        cos_combined = torch.cat([ones, cos_theta], dim=1)
        # 计算累乘并翻转
        cum_cos = torch.cumprod(cos_combined, dim=1)
        flipped_cum_cos = torch.flip(cum_cos, dims=[1])

        # MATLAB: [ones(N,1), sin(X(:,M-1:-1:1)*pi/2)]
        # 注意 sin 部分需要把变量索引倒过来 (M-1:-1:1)
        flipped_sin_theta = torch.flip(sin_theta, dims=[1])
        sin_combined = torch.cat([ones, flipped_sin_theta], dim=1)

        # 4. 最终目标值计算: PopObj = (1+g) * flipped_cum_cos * sin_combined
        PopObj = (1.0 + g) * flipped_cum_cos * sin_combined

        return PopObj


import torch
import numpy as np


class MMF14_a():
    def __init__(self, n_obj=3, n_dim=3, device='cpu'):
        self.n_obj = n_obj
        self.n_dim = n_dim
        self.device = device

        # 边界设定：[0, 1] 空间
        self.lbound = torch.zeros(n_dim, dtype=torch.float32, device=device)
        self.ubound = torch.ones(n_dim, dtype=torch.float32, device=device)

        self.ideal_point = np.zeros(n_obj)
        self.nadir_point = np.ones(n_obj) * 2.0

    def evaluate(self, X):
        """
        X: 输入张量 (batch_size, n_dim)，取值 [0, 1]
        """
        # 映射到实际边界
        X_real = X * (self.ubound - self.lbound) + self.lbound

        N = X_real.shape[0]
        M = self.n_obj
        num_of_peak = 2

        # 1. 计算非线性关联变量 x_g
        # MATLAB: X(:,end) - 0.5*sin(pi*X(:,end-1))
        # 注意：Python 中 X[:, -1] 是最后一个，X[:, -2] 是倒数第二个
        x_last = X_real[:, -1]
        x_penultimate = X_real[:, -2]
        x_g = x_last - 0.5 * torch.sin(torch.pi * x_penultimate)

        # 2. 计算 g 函数
        # g = 2 - (sin(num_of_peak * pi * (x_g + 1/(2*num_of_peak))))^2
        shift = 1.0 / (2.0 * num_of_peak)
        g = 2.0 - torch.pow(torch.sin(num_of_peak * torch.pi * (x_g + shift)), 2)
        g = g.view(-1, 1)  # 形状 (N, 1)

        # 3. 球面坐标变换 (与 MMF14 逻辑一致)
        # 角度变量 theta 取前 M-1 个维度
        theta = X_real[:, :M - 1] * (torch.pi / 2.0)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        # 计算 cos 链乘 (fliplr + cumprod)
        ones = torch.ones((N, 1), device=self.device)
        cos_combined = torch.cat([ones, cos_theta], dim=1)
        flipped_cum_cos = torch.flip(torch.cumprod(cos_combined, dim=1), dims=[1])

        # 计算 sin 项 (索引倒序 M-1:-1:1)
        flipped_sin_theta = torch.flip(sin_theta, dims=[1])
        sin_combined = torch.cat([ones, flipped_sin_theta], dim=1)

        # 最终映射
        PopObj = (1.0 + g) * flipped_cum_cos * sin_combined

        return PopObj