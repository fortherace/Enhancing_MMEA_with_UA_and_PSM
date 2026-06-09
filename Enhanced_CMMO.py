import matlab.engine
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from problem import get_problem
from pymoo.indicators.hv import HV
from cluster_multimodal import cluster_multimodal
from torch.utils.data import DataLoader, Dataset
from model import ParetoSetModel
import schedulefree
from typing import List, Tuple
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import timeit

from scipy.spatial.distance import cdist

def select_nondominated_from_archive(all_X: List[np.ndarray], all_F: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stack all generations, run non-dominated sorting on the union, and return rank-1 set.
    """
    if len(all_X) == 0:
        return np.empty((0,)), np.empty((0,))
    X_all = np.vstack(all_X)
    F_all = np.vstack(all_F)
    nds = NonDominatedSorting()
    I = nds.do(F_all, only_non_dominated_front=True)
    return X_all[I], F_all[I]

# ------------------------- Archive Only -------------------------
class ArchiveCallback:
    """
    A callback for pymoo that:
    Archives population (X, F) at each generation.
    """

    def __init__(self):
        self.archive_X: List[np.ndarray] = []
        self.archive_F: List[np.ndarray] = []

    def __call__(self, algorithm):
        # collect current population
        pop = algorithm.pop
        if pop is None or len(pop) == 0:
            return

        X = pop.get("X")
        F = pop.get("F")

        # store into archive
        if X is not None and F is not None:
            self.archive_X.append(np.array(X))
            self.archive_F.append(np.array(F))

def das_dennis_recursion(ref_dirs, ref_dir, n_partitions, beta, depth):
    if depth == len(ref_dir) - 1:
        ref_dir[depth] = beta / (1.0 * n_partitions)
        ref_dirs.append(ref_dir[None, :])
    else:
        for i in range(beta + 1):
            ref_dir[depth] = 1.0 * i / (1.0 * n_partitions)
            das_dennis_recursion(ref_dirs, np.copy(ref_dir), n_partitions, beta - i, depth + 1)


def das_dennis(n_partitions, n_dim):
    if n_partitions == 0:
        return np.full((1, n_dim), 1 / n_dim)
    else:
        ref_dirs = []
        ref_dir = np.full(n_dim, np.nan)
        das_dennis_recursion(ref_dirs, ref_dir, n_partitions, n_partitions, 0)
        return np.concatenate(ref_dirs, axis=0)


def select_psl_ea_solutions(pref, psl_pf, ea_pf, psl_ps, ea_ps, z_min, z_max, mode='hybrid'):
    n_obj = psl_pf.shape[1]

    # ---------- 归一化 ----------
    psl_norm = (psl_pf - z_min) / (z_max - z_min)
    ea_norm = (ea_pf - z_min) / (z_max - z_min)

    final_pf = []
    final_ps = []

    psl_num = 0
    ea_num = 0

    for pre in pref:

        # ---------- PSL 最匹配解 ----------
        num = np.dot(pre, psl_norm.T)
        denom = np.linalg.norm(pre) * np.linalg.norm(psl_norm, axis=1)
        cos_psl = num / denom
        cos_psl[np.isnan(cos_psl)] = 0

        idx_psl = np.argmax(cos_psl)

        sug_solution = psl_pf[idx_psl]
        sug_ps = psl_ps[idx_psl]

        sug_max_cos = np.max(cos_psl)

        # ---------- EA 最匹配解 ----------
        num = np.dot(pre, ea_norm.T)
        denom = np.linalg.norm(pre) * np.linalg.norm(ea_norm, axis=1)
        cos_ea = num / denom
        cos_ea[np.isnan(cos_ea)] = 0

        idx_ea = np.argmax(cos_ea)

        ea_solution = ea_pf[idx_ea]
        ea_ps_sol = ea_ps[idx_ea]

        ea_max_cos = np.max(cos_ea)

        if mode == 'psl':
            final_pf.append(sug_solution)
            final_ps.append(sug_ps)
            psl_num += 1
        elif mode == 'ea':
            final_pf.append(ea_solution)
            final_ps.append(ea_ps_sol)
            ea_num += 1
        else:
            # PSL 被 EA 支配
            if np.all(sug_solution > ea_solution):
                final_pf.append(ea_solution)
                final_ps.append(ea_ps_sol)
                ea_num += 1

            # PSL 支配 EA
            elif np.all(sug_solution <= ea_solution):
                final_pf.append(sug_solution)
                final_ps.append(sug_ps)
                psl_num += 1

            # EA 更符合偏好
            elif ea_max_cos > sug_max_cos:
                final_pf.append(ea_solution)
                final_ps.append(ea_ps_sol)
                ea_num += 1

            # PSL 更符合偏好
            else:
                final_pf.append(sug_solution)
                final_ps.append(sug_ps)
                psl_num += 1

    final_pf = np.array(final_pf)
    final_ps = np.array(final_ps)

    return final_pf, final_ps, psl_num, ea_num


def IGDX(true_PS, approx_PS):
    # 计算距离矩阵
    dist_matrix = cdist(true_PS, approx_PS, metric='euclidean')
    min_dist = np.min(dist_matrix, axis=1)
    return np.mean(min_dist)


def IGDF(true_PF, approx_PF):
    dist_matrix = cdist(true_PF, approx_PF, metric='euclidean')
    min_dist = np.min(dist_matrix, axis=1)
    return np.mean(min_dist)


def CR(true_PS, approx_PS, eps=1e-12):
    """
    Cover Rate (CR) in product form as in MMOEA/DC papers
    approx_PS: (N,D) numpy array, obtained PS
    true_PS: (M,D) numpy array, true PS
    """
    xmin = np.min(true_PS, axis=0)
    xmax = np.max(true_PS, axis=0)

    # 对每个解逐维截断并归一化
    clipped = np.minimum(np.maximum(approx_PS, xmin), xmax)
    sigma = (clipped - xmin) / (xmax - xmin + eps)  # shape: (N,D)

    # 每个解的 CR：乘积形式再开 2^D 次方根
    D = approx_PS.shape[1]
    CR_per_sol = np.prod(sigma ** (1 / 2 ** D), axis=1)

    # 所有解取平均
    CR_total = np.mean(CR_per_sol)
    return CR_total

def PSP(true_PS, approx_PS):
    """
    PSP = CR / IGDX
    """
    cr = CR(true_PS, approx_PS)
    igdx = IGDX(true_PS, approx_PS)
    return cr / (igdx + 1e-12)

ins_list = ['mmf1', 'mmf2', 'mmf3', 'mmf4', 'mmf5', 'mmf6', 'mmf7', 'mmf8', 'mmf9', 'mmf1_e', 'mmf1_z', 'mmf14', 'mmf14_a', 'omnitest', 'sympart']

# number of independent runs
n_run = 31
# number of sampled solutions for gradient estimation
n_sample = 5
# number of Pretraining epochs
pretrain_epochs = 100
# sampling method for evolutionary gradient approximation
sampling_method = "Bernoulli"

# device
device = 'cpu'
# ----------------------------------------------------- Preparation --------------------------------------------------------
eng = matlab.engine.start_matlab()
matlab_dir = r'D:\Project\PyCharm_Code\Enhancing_MMEA_with_UA_and_PSM\MMEAs\CMMO'
eng.cd(matlab_dir, nargout=0)
eng.addpath(matlab_dir, nargout=0)
for test_ins in ins_list:
    print(test_ins)
    small_igdx_psl = []
    small_igdf_psl = []
    small_hv_psl = []
    small_psp_psl = []
    small_igdx_ea = []
    small_igdf_ea = []
    small_hv_ea = []
    small_psp_ea = []
    small_igdx = []
    small_igdf = []
    small_psp = []
    small_hv = []


    large_igdx_psl = []
    large_igdf_psl = []
    large_hv_psl = []
    large_psp_psl = []
    large_igdx_ea = []
    large_igdf_ea = []
    large_hv_ea = []
    large_psp_ea = []
    large_igdx = []
    large_igdf = []
    large_psp = []
    large_hv = []

    problem = get_problem(test_ins)
    n_dim = problem.n_dim
    n_obj = problem.n_obj
    lbound = problem.lbound.cpu().numpy()
    ubound = problem.ubound.cpu().numpy()
    ideal_point = problem.ideal_point
    nadir_point = problem.nadir_point

    if test_ins in ['omnitest', 'sympart', 'mmf1', 'mmf2', 'mmf3', 'mmf4', 'mmf5', 'mmf6', 'mmf7', 'mmf8', 'mmf1_e', 'mmf1_z', 'mmf9', 'mmf14', 'mmf14_a']:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        True_PF = np.loadtxt(os.path.join(base_dir, f'data/PF/{test_ins}_pf.dat'))
        True_PS = np.loadtxt(os.path.join(base_dir, f'data/PS/{test_ins}_ps.dat'))

    if n_obj == 2:
        n_pref_update = 5
        MaxFE = 30000     # 100%
        pop_size = 100
    elif n_obj == 3:
        n_pref_update = 8
        MaxFE = 60000     # 100%
        pop_size = 105
    else:
        n_pref_update = 8
        MaxFE = 60000     # 100%
        pop_size = 120

    # repeatedly run the algorithm n_run times
    for run_iter in range(n_run):
        print(run_iter)
        start = timeit.default_timer()
        # ------------------------------------------ Evolutionary Search ---------------------------------------------------
        func_name = f"run_cmmo_{test_ins}"
        cmd = f"[arch_X_nd, arch_F_nd] = {func_name}({pop_size}, {MaxFE});"
        eng.evalc(cmd)
        arch_X_nd = np.array(eng.workspace['arch_X_nd'])
        arch_F_nd = np.array(eng.workspace['arch_F_nd'])

        print("number of non-dominated solutions:", arch_F_nd.shape)
        stop1 = timeit.default_timer()
        print("Evolutionary Search Time:", stop1 - start)
        # ------------------------------------------ Clustering ---------------------------------------------------
        lb = arch_X_nd.min(axis=0)
        ub = arch_X_nd.max(axis=0)
        pareto_set = (arch_X_nd - lb) / (ub - lb)
        pareto_front = (arch_F_nd - ideal_point) / (nadir_point - ideal_point)
        prefs = pareto_front / pareto_front.sum(axis=1, keepdims=True)
        if n_obj == 2:
            pref_dirs = np.stack([np.linspace(0, 1, 10), 1 - np.linspace(0, 1, 10)]).T
        if n_obj == 3:
            pref_dirs = das_dennis(3, 3)
        if n_obj == 4:
            pref_dirs = das_dennis(2, 4)

        if test_ins == 'mmf8':
            eps = 0.2
        else:
            eps = 0.3

        n_clusters, groups_X, groups_F= cluster_multimodal(
            pareto_set,
            pareto_front,
            prefs,
            pref_dirs,
            eps=eps,
            min_samples=5
        )
        print("The number of final groups:", n_clusters)
        stop2 = timeit.default_timer()
        print("Clustering Time:", stop2 - stop1)
        # ------------------------------------------ Pre-training ---------------------------------------------------
        all_prefs = []
        all_X = []
        all_ps_ids = []
        for i, Xc in enumerate(groups_X):
            approx_pareto_set = Xc * (ub - lb) + lb
            approx_pareto_set = (approx_pareto_set - lbound) / (ubound - lbound)
            all_X.append(approx_pareto_set)
        for i, Fc in enumerate(groups_F):
            approx_pareto_front = Fc
            # approx_pareto_front += 0.1
            prefs = approx_pareto_front / approx_pareto_front.sum(axis=1, keepdims=True)
            all_prefs.append(prefs)
            all_ps_ids.append(np.full(prefs.shape[0], i, dtype=int))

        all_prefs_tensor = torch.tensor(np.vstack(all_prefs), dtype=torch.float32, device=device)
        all_X_tensor = torch.tensor(np.vstack(all_X), dtype=torch.float32, device=device)
        all_ps_ids_tensor = torch.tensor(np.concatenate(all_ps_ids), dtype=torch.long, device=device)

        class PreTrainDataset(Dataset):
            def __init__(self, prefs, x, ids):
                self.prefs = prefs
                self.x = x
                self.ids = ids

            def __len__(self):
                return len(self.prefs)

            def __getitem__(self, idx):
                return self.prefs[idx], self.x[idx], self.ids[idx]

        pretrain_dataset = PreTrainDataset(all_prefs_tensor, all_X_tensor, all_ps_ids_tensor)
        pretrain_loader = DataLoader(pretrain_dataset, batch_size=128, shuffle=True)

        psmodel = ParetoSetModel(n_dim=n_dim, n_obj=n_obj, n_heads=n_clusters)
        optimizer_p = schedulefree.AdamWScheduleFree(psmodel.parameters(), lr=0.0025,
                                                     warmup_steps=10)

        loss_history = []
        for pretrain_epoch in range(pretrain_epochs):
            psmodel.train()
            optimizer_p.train()
            for batch_prefs, batch_X, batch_id in pretrain_loader:
                pred_X = psmodel(batch_prefs, batch_id)
                loss = torch.mean((pred_X - batch_X) ** 2)
                optimizer_p.zero_grad()
                loss.backward()
                loss_history.append(loss.item())
                optimizer_p.step()

        stop3 = timeit.default_timer()
        print("Pre-training Time: ", stop3 - stop2)
        final_ps = np.empty((0, n_dim))
        final_pf = np.empty((0, n_obj))
        num_PSL = 0
        num_EA = 0
        Only_PSL_ps = np.empty((0, n_dim))
        Only_PSL_pf = np.empty((0, n_obj))
        Only_EA_ps = np.empty((0, n_obj))
        Only_EA_pf = np.empty((0, n_obj))

        if n_obj == 2:
            pref_size = 100
            pref = np.stack([np.linspace(0, 1, 100), 1 - np.linspace(0, 1, 100)]).T
        if n_obj == 3:
            pref_size = 105
            pref = das_dennis(13, 3)
        if n_obj == 4:
            pref_size = 120
            pref = das_dennis(7, 4)
        pref = torch.tensor(pref).to(device).float()

        for i, (Xc, Fc) in enumerate(zip(groups_X, groups_F)):
            pref_i = pref[i::n_clusters]
            ps_ids = torch.full((pref_i.shape[0],), i, dtype=torch.long, device=device)
            sol = psmodel(pref_i, ps_ids)
            pf = problem.evaluate(sol)
            ps = sol.detach().cpu().numpy()
            PS_PSL = ps * (ubound - lbound) + lbound
            PF_PSL = pf.detach().cpu().numpy()
            PS_EA = Xc * (ub - lb) + lb
            PF_EA = Fc * (nadir_point - ideal_point) + ideal_point
            select_pf, select_ps, psl_num, ea_num = select_psl_ea_solutions(
                pref=pref_i.cpu().numpy(),
                psl_pf=PF_PSL,
                ea_pf=PF_EA,
                psl_ps=PS_PSL,
                ea_ps=PS_EA,
                z_min=ideal_point,
                z_max=nadir_point,
                mode="hybrid",
            )
            final_ps = np.vstack((final_ps, select_ps))
            final_pf = np.vstack((final_pf, select_pf))
            num_EA = num_EA + ea_num
            num_PSL = num_PSL + psl_num
            select_pf, select_ps, psl_num, ea_num = select_psl_ea_solutions(
                pref=pref_i.cpu().numpy(),
                psl_pf=PF_PSL,
                ea_pf=PF_EA,
                psl_ps=PS_PSL,
                ea_ps=PS_EA,
                z_min=ideal_point,
                z_max=nadir_point,
                mode="psl",
            )
            Only_PSL_ps = np.vstack((Only_PSL_ps, select_ps))
            Only_PSL_pf = np.vstack((Only_PSL_pf, select_pf))

            select_pf, select_ps, psl_num, ea_num = select_psl_ea_solutions(
                pref=pref_i.cpu().numpy(),
                psl_pf=PF_PSL,
                ea_pf=PF_EA,
                psl_ps=PS_PSL,
                ea_ps=PS_EA,
                z_min=ideal_point,
                z_max=nadir_point,
                mode="ea",
            )
            Only_EA_ps = np.vstack((Only_EA_ps, select_ps))
            Only_EA_pf = np.vstack((Only_EA_pf, select_pf))

        # save_dir = os.path.join("Graph", "CMMO(our)", test_ins.upper())
        # os.makedirs(save_dir, exist_ok=True)
        #
        # # =========================
        # # 2. 保存 PF / PS
        # # =========================
        # pf_path = os.path.join(save_dir, "PF.npy")
        # ps_path = os.path.join(save_dir, "PS.npy")
        #
        # np.save(pf_path, final_pf)
        # np.save(ps_path, final_ps)

        T = (True_PF - ideal_point) / (nadir_point - ideal_point)
        results_F_norm = (final_pf - ideal_point) / (nadir_point - ideal_point)
        hv = HV(ref_point=np.array([1.1] * n_obj))
        True_HV = hv(T)
        hv_value = hv(results_F_norm)
        a = True_HV - hv_value
        small_hv.append(a)
        results_F_psl = (Only_PSL_pf - ideal_point) / (nadir_point - ideal_point)
        hv = HV(ref_point=np.array([1.1] * n_obj))
        hv_value_psl = hv(results_F_psl)
        b = True_HV - hv_value_psl
        small_hv_psl.append(b)
        results_F_ea = (Only_EA_pf - ideal_point) / (nadir_point - ideal_point)
        hv = HV(ref_point=np.array([1.1] * n_obj))
        hv_value_ea = hv(results_F_ea)
        c = True_HV - hv_value_ea
        small_hv_ea.append(c)
        igdx_psl = IGDX(True_PS, Only_PSL_ps)
        igdf_psl = IGDF(True_PF, Only_PSL_pf)
        psp_psl = PSP(True_PS, Only_PSL_ps)
        small_igdx_psl.append(float(igdx_psl))
        small_igdf_psl.append(float(igdf_psl))
        small_psp_psl.append(float(psp_psl))

        igdx_ea = IGDX(True_PS, Only_EA_ps)
        igdf_ea = IGDF(True_PF, Only_EA_pf)
        psp_ea = PSP(True_PS, Only_EA_ps)
        small_igdx_ea.append(float(igdx_ea))
        small_igdf_ea.append(float(igdf_ea))
        small_psp_ea.append(float(psp_ea))
        print("来自PSL:", num_PSL)
        print("来自EA:", num_EA)

        igdx = IGDX(True_PS, final_ps)
        igdf = IGDF(True_PF, final_pf)
        psp = PSP(True_PS, final_ps)
        small_igdx.append(float(igdx))
        small_igdf.append(float(igdf))
        small_psp.append(float(psp))

        # ------------------------------------------ Large Solution Set ---------------------------------------------------
        final_ps = np.empty((0, n_dim))
        final_pf = np.empty((0, n_obj))
        num_PSL = 0
        num_EA = 0
        Only_PSL_ps = np.empty((0, n_dim))
        Only_PSL_pf = np.empty((0, n_obj))
        Only_EA_ps = np.empty((0, n_obj))
        Only_EA_pf = np.empty((0, n_obj))

        if n_obj == 2:
            pref = np.stack([np.linspace(0, 1, 1000), 1 - np.linspace(0, 1, 1000)]).T
        if n_obj == 3:
            pref_size = 990
            pref = das_dennis(43, 3)
        if n_obj == 4:
            pref_size = 969
            pref = das_dennis(16, 4)
        pref = torch.tensor(pref).to(device).float()
        for i, (Xc, Fc) in enumerate(zip(groups_X, groups_F)):
            pref_i = pref[i::n_clusters]
            ps_ids = torch.full((pref_i.shape[0],), i, dtype=torch.long, device=device)
            sol = psmodel(pref_i, ps_ids)
            pf = problem.evaluate(sol)
            ps = sol.detach().cpu().numpy()
            PS_PSL = ps * (ubound - lbound) + lbound
            PF_PSL = pf.detach().cpu().numpy()
            PS_EA = Xc * (ub - lb) + lb
            PF_EA = Fc * (nadir_point - ideal_point) + ideal_point
            select_pf, select_ps, psl_num, ea_num = select_psl_ea_solutions(
                pref=pref_i.cpu().numpy(),
                psl_pf=PF_PSL,
                ea_pf=PF_EA,
                psl_ps=PS_PSL,
                ea_ps=PS_EA,
                z_min=ideal_point,
                z_max=nadir_point,
                mode="hybrid",
            )
            final_ps = np.vstack((final_ps, select_ps))
            final_pf = np.vstack((final_pf, select_pf))
            num_EA = num_EA + ea_num
            num_PSL = num_PSL + psl_num
            select_pf, select_ps, psl_num, ea_num = select_psl_ea_solutions(
                pref=pref_i.cpu().numpy(),
                psl_pf=PF_PSL,
                ea_pf=PF_EA,
                psl_ps=PS_PSL,
                ea_ps=PS_EA,
                z_min=ideal_point,
                z_max=nadir_point,
                mode="psl",
            )
            Only_PSL_ps = np.vstack((Only_PSL_ps, select_ps))
            Only_PSL_pf = np.vstack((Only_PSL_pf, select_pf))

            select_pf, select_ps, psl_num, ea_num = select_psl_ea_solutions(
                pref=pref_i.cpu().numpy(),
                psl_pf=PF_PSL,
                ea_pf=PF_EA,
                psl_ps=PS_PSL,
                ea_ps=PS_EA,
                z_min=ideal_point,
                z_max=nadir_point,
                mode="ea",
            )
            Only_EA_ps = np.vstack((Only_EA_ps, select_ps))
            Only_EA_pf = np.vstack((Only_EA_pf, select_pf))

        stop4 = timeit.default_timer()
        print("Solution Generation and selection: ", stop4 - stop3)

        T = (True_PF - ideal_point) / (nadir_point - ideal_point)
        results_F_norm = (final_pf - ideal_point) / (nadir_point - ideal_point)
        hv = HV(ref_point=np.array([1.1] * n_obj))
        True_HV = hv(T)
        hv_value = hv(results_F_norm)
        d = True_HV - hv_value
        large_hv.append(d)
        results_F_psl = (Only_PSL_pf - ideal_point) / (nadir_point - ideal_point)
        hv = HV(ref_point=np.array([1.1] * n_obj))
        hv_value_psl = hv(results_F_psl)
        e = True_HV - hv_value_psl
        large_hv_psl.append(e)
        results_F_ea = (Only_EA_pf - ideal_point) / (nadir_point - ideal_point)
        hv = HV(ref_point=np.array([1.1] * n_obj))
        hv_value_ea = hv(results_F_ea)
        f = True_HV - hv_value_ea
        large_hv_ea.append(f)

        igdx_psl = IGDX(True_PS, Only_PSL_ps)
        igdf_psl = IGDF(True_PF, Only_PSL_pf)
        psp_psl = PSP(True_PS, Only_PSL_ps)
        large_igdx_psl.append(float(igdx_psl))
        large_igdf_psl.append(float(igdf_psl))
        large_psp_psl.append(float(psp_psl))

        igdx_ea = IGDX(True_PS, Only_EA_ps)
        igdf_ea = IGDF(True_PF, Only_EA_pf)
        psp_ea = PSP(True_PS, Only_EA_ps)
        large_igdx_ea.append(float(igdx_ea))
        large_igdf_ea.append(float(igdf_ea))
        large_psp_ea.append(float(psp_ea))
        print("来自PSL:", num_PSL)
        print("来自EA:", num_EA)

        igdx = IGDX(True_PS, final_ps)
        igdf = IGDF(True_PF, final_pf)
        psp = PSP(True_PS, final_ps)
        large_igdx.append(float(igdx))
        large_igdf.append(float(igdf))
        large_psp.append(float(psp))

    print(f"{test_ins} merged small IGDX：", small_igdx)
    print(f"{test_ins} model small IGDX：", small_igdx_psl)
    print(f"{test_ins} EA small IGDX：", small_igdx_ea)
    print("\n")
    print(f"{test_ins} merged small IGDF：", small_igdf)
    print(f"{test_ins} model small IGDF：", small_igdf_psl)
    print(f"{test_ins} EA small IGDF：", small_igdf_ea)
    print("\n")
    print(f"{test_ins} merged small HV：", small_hv)
    print(f"{test_ins} model small HV：", small_hv_psl)
    print(f"{test_ins} EA small HV：", small_hv_ea)
    print("\n")
    print(f"{test_ins} merged small PSP: ", small_psp)
    print(f"{test_ins} model small PSP: ", small_psp_psl)
    print(f"{test_ins} EA small PSP: ", small_psp_ea)
    print("\n")
    print(f"{test_ins} merged large IGDX：", large_igdx)
    print(f"{test_ins} model large IGDX：", large_igdx_psl)
    print(f"{test_ins} EA large IGDX：", large_igdx_ea)
    print("\n")
    print(f"{test_ins} merged large IGDF：", large_igdf)
    print(f"{test_ins} model large IGDF：", large_igdf_psl)
    print(f"{test_ins} EA large IGDF：", large_igdf_ea)
    print("\n")
    print(f"{test_ins} merged large HV：", large_hv)
    print(f"{test_ins} model large HV：", large_hv_psl)
    print(f"{test_ins} EA large HV：", large_hv_ea)
    print("\n")
    print(f"{test_ins} merged large PSP: ", large_psp)
    print(f"{test_ins} model large PSP: ", large_psp_psl)
    print(f"{test_ins} EA large PSP: ", large_psp_ea)







