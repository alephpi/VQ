import numpy as np
from itertools import product

def angle_stats(pts):
    G = pts @ pts.T
    np.fill_diagonal(G, -np.inf)
    nn_cos    = np.max(G, axis=1)          # 每个点的最近邻余弦
    nn_angles = np.degrees(np.arccos(np.clip(nn_cos, -1, 1)))
    return nn_angles.min(), nn_angles.max(), nn_angles.mean(), nn_angles.std()

def find_sphere_code_gradient_descent(D, N, steps=5000, lr=0.001, print_every=500):
    # 随机初始化并投影到球面
    points = np.random.randn(N, D)
    points /= np.linalg.norm(points, axis=1, keepdims=True)

    for step in range(steps + 1):
        
        # ---- 打印最小角度 ----
        if step % print_every == 0:
            min_a, max_a = get_min_angle_deg(points)
            print(f"Step {step:5d} | min NN angle = {min_a:.3f}°  "
                  f"max NN angles = {max_a:.3f}°")

        # ---- 排斥力梯度（Riesz 能量 E = Σ 1/||xi-xj||）----
        G = points @ points.T
        np.fill_diagonal(G, 1.0)

        # ||xi - xj||² = 2 - 2·G[i,j]（利用单位向量性质）
        dist2 = 2 - 2 * G                        # (N, N)
        np.fill_diagonal(dist2, np.inf)

        diff = points[:, None, :] - points[None, :, :]  # (N, N, D)

        # d/d(xi) Σ_{j≠i} 1/dist_ij = -Σ_{j≠i} (xi-xj) / dist_ij³
        grad = -np.sum(diff / dist2[:, :, None] ** 1.5, axis=1)  # (N, D)

        # ---- 更新 + 重新投影到球面 ----
        points -= lr * grad
        points /= np.linalg.norm(points, axis=1, keepdims=True)

    return points

def hypercube_sphere_code(D, L):
    """
    超立方体面投影球面码

    思路：
      - D 维超立方体有 2D 个面，每个面固定某维度为 ±1
      - 其余 D-1 个维度在 linspace(-1, 1, L) 的网格上取值
      - 用 set 去重（角点/棱上的点被多个面共享）
      - 全部投影到单位球面
    """
    grid = np.linspace(-1, 1, L)          # 每条边 L 个等分码位

    if D == 1:
        pts = np.array([[-1.0], [1.0]])
        return pts

    # 预先构造 (L^(D-1), D-1) 的网格组合
    mesh = np.meshgrid(*([grid] * (D - 1)), indexing="ij")
    base = np.stack(mesh, axis=-1).reshape(-1, D - 1)

    pts_list = []
    for dim in range(D):
        for sign in (+1.0, -1.0):
            pts = np.empty((base.shape[0], D))
            pts[:, dim] = sign
            other_dims = [d for d in range(D) if d != dim]
            pts[:, other_dims] = base
            pts_list.append(pts)

    pts = np.vstack(pts_list)
    # 去重（角点/棱上重复）
    pts = np.unique(np.round(pts, 12), axis=0)
    pts_sphere = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return pts_sphere

print(f"{'L':>4} | {'N':>6} | {'最小角':>9} | {'最大角':>9} | {'均值':>9} | {'std':>8}")
print("─" * 54)
for L in [2, 3, 4, 5, 6, 8, 10, 15, 20]:
    pts = hypercube_sphere_code(D=3, L=L)
    mn, mx, mu, sd = angle_stats(pts)
    print(f"{L:>4} | {len(pts):>6} | {mn:>8.3f}° | {mx:>8.3f}° | {mu:>8.3f}° | {sd:>8.3f}°")

print(f"{'D':>4} | {'N':>6} | {'最小角':>9} | {'最大角':>9} | {'均值':>9} | {'std':>8}")
print("─" * 54)
for D in range(2,11):
    pts = hypercube_sphere_code(D=D, L=5)
    # if len(pts) > 65536:
    #     break
    mn, mx, mu, sd = angle_stats(pts)
    print(f"{D:>4} | {len(pts):>6} | {mn:>8.3f}° | {mx:>8.3f}° | {mu:>8.3f}° | {sd:>8.3f}°")

# 示例：3维球面上放20个点
# points = find_sphere_code_gradient_descent(D=3, N=130, steps=1000000)