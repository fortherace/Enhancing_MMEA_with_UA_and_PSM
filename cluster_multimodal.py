import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist


def assign_to_reference_directions(prefs, pref_dirs):
    dist_matrix = cdist(prefs, pref_dirs, metric='euclidean')
    labels = np.argmin(dist_matrix, axis=1)
    return labels


def cluster_multimodal(approx_pareto_set, approx_pareto_front, prefs, pref_dirs, eps=0.3, min_samples=5):
    """
    对多目标近似帕累托解进行多模态聚类。

    参数:
        approx_pareto_set : ndarray, 决策空间解集 (N x D)
        approx_pareto_front : ndarray, 目标空间解集 (N x M)
        prefs : ndarray, 每个解对应的偏好向量
        pref_dirs : ndarray, 参考方向集
        eps : float, DBSCAN的epsilon
        min_samples : int, DBSCAN的最小样本数
        T_PS : ndarray, 可选，真实帕累托前沿用于可视化

    返回:
        n_clusters : int, 模态数
        groups_X : list of ndarray, 每个模态的决策空间集合
        groups_F : list of ndarray, 每个模态的目标空间集合
    """

    # ---------- Step 1: 将点分配到参考方向 ----------
    labels = assign_to_reference_directions(prefs, pref_dirs)

    # ---------- Step 2: 对每个参考向量下的解做 DBSCAN，找子模态数最多的参考向量 ----------
    best_label = None
    best_subclusters = None
    best_sub_labels = None
    max_clusters = -1

    for lb in np.unique(labels):
        idx = np.where(labels == lb)[0]
        X_sub = approx_pareto_set[idx]

        if len(X_sub) < min_samples:
            continue

        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_sub)
        sub_labels = db.labels_

        n_clusters = len(set(sub_labels)) - (1 if -1 in sub_labels else 0)
        # print(f"参考向量 {lb} 下子模态数: {n_clusters}")

        if n_clusters > max_clusters:
            max_clusters = n_clusters
            best_label = lb
            best_subclusters = []
            best_sub_labels = sub_labels

            for cid in range(n_clusters):
                mask = sub_labels == cid
                best_subclusters.append(X_sub[mask])

    if best_subclusters is None:
        raise ValueError("没有找到满足条件的初始模态，请检查 eps 和 min_samples 设置。")


    # ---------- Step 3: 其余解按“最近原型”并入 ----------
    final_clusters_X = [list(cluster) for cluster in best_subclusters]
    final_clusters_F = [list() for _ in best_subclusters]

    # 把初始参考向量下的解对应的 F 也放进去
    idx_best = np.where(labels == best_label)[0]
    X_best = approx_pareto_set[idx_best]
    F_best = approx_pareto_front[idx_best]

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_best)
    sub_labels = db.labels_

    for cid in range(len(best_subclusters)):
        mask = sub_labels == cid
        final_clusters_F[cid].extend(F_best[mask])

    # 剩余解
    idx_rest = np.where(labels != best_label)[0]
    X_rest = approx_pareto_set[idx_rest]
    F_rest = approx_pareto_front[idx_rest]

    # 初始类目标空间原型
    F_prototypes = [np.mean(np.vstack(Fc), axis=0) for Fc in final_clusters_F]

    # 按目标空间距离排序剩余解
    obj_dist_list = []
    for i, f in enumerate(F_rest):
        dists = [np.linalg.norm(f - fp) for fp in F_prototypes]
        obj_dist_list.append((i, np.min(dists)))
    obj_dist_list.sort(key=lambda x: x[1])

    # 按决策空间最近类并入
    for t, (idx, _) in enumerate(obj_dist_list):
        x = X_rest[idx]
        f = F_rest[idx]

        dec_dists = [
            np.min(np.linalg.norm(np.array(cluster) - x, axis=1))
            for cluster in final_clusters_X
        ]
        cid = np.argmin(dec_dists)

        final_clusters_X[cid].append(x)
        final_clusters_F[cid].append(f)

    # 转为 ndarray
    final_clusters_X = [np.array(c) for c in final_clusters_X]
    final_clusters_F = [np.array(c) for c in final_clusters_F]

    n_clusters = len(final_clusters_X)
    return n_clusters, final_clusters_X, final_clusters_F
