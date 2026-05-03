import numpy as np
import math
from itertools import product

def angle_stats(pts, pairwise=False):
    if pairwise:
        G = pts @ pts.T
    else:
        G = pts.dot(pts[0])
    # norm_squre = np.diag(G)[0]
    # replace all 1 by -inf to ignore self-pairs in max calculation
    m = G.max()
    
    # np.fill_diagonal(G, -np.inf)  # 直接修改 G 的对角线元素，避免创建新数组
    G[G>(m-1e-10)] = -np.inf
    if pairwise:
        identical_pairs = (np.count_nonzero(G == -np.inf) - len(pts))//2
    else:
        identical_pairs = (np.count_nonzero(G == -np.inf) - 1)//2
    print(f"Identical pairs: {identical_pairs}")
    # G = G / norm_squre  # 余弦值
    if len(G.shape) < 2:
        G = G.reshape(1, -1)
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

def test_hypercube_sphere_code():
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

def generate_E8_shell(m: int) -> np.ndarray:
    """
    生成 E8 格第 m 层的所有格点（范数² = 2m 的所有向量）。

    E8 格由两部分构成：
      整数部分：所有坐标为整数且坐标之和为偶数的向量（D8 子格）
      半整数部分：所有坐标为半整数（n + 1/2，n ∈ Z）且坐标之和为偶数的向量

    参数
    ----
    m : 层序号（m = 1 给出 240 个最短向量，m = 0 只有零向量）

    返回
    ----
    list of tuple，每个 tuple 是长度为 8 的坐标组（整数层为 int，半整数层为 float）
    """
    if m < 0:
        raise ValueError("m 必须 >= 0")

    target = 2 * m
    points = []

    # ------------------------------------------------------------------ #
    # 第一部分：整数坐标                                                    #
    # 条件：sum(xi^2) = 2m，sum(xi) 为偶数                                 #
    # ------------------------------------------------------------------ #
    def gen_int(dim: int, rem: int, coord_sum: int, coords: list):
        if dim == 8:
            if rem == 0 and coord_sum % 2 == 0:
                points.append(tuple(coords))
            return
        max_v = math.isqrt(rem)
        for v in range(-max_v, max_v + 1):
            sq = v * v
            if sq <= rem:
                coords.append(v)
                gen_int(dim + 1, rem - sq, coord_sum + v, coords)
                coords.pop()

    gen_int(0, target, 0, [])

    # ------------------------------------------------------------------ #
    # 第二部分：半整数坐标                                                  #
    # xi = ni + 1/2，ni ∈ Z                                               #
    # sum((ni+1/2)^2) = 2m  =>  sum(ni^2 + ni) = 2m - 2                  #
    # sum(xi) 为偶数  =>  sum(ni) 为偶数（因 sum(xi) = sum(ni) + 4）       #
    # 注意：ni(ni+1) >= 0 对所有整数恒成立                                  #
    # ------------------------------------------------------------------ #
    if m >= 1:
        target_h = 2 * m - 2

        def max_n_for(rem: int) -> int:
            # ni(ni+1) <= rem 的最大正整数解
            if rem == 0:
                return 0
            return int((-1 + math.sqrt(1 + 4 * rem)) / 2) + 1

        def gen_half(dim: int, rem: int, n_sum: int, ns: list):
            if dim == 8:
                if rem == 0 and n_sum % 2 == 0:
                    points.append(tuple(n + 0.5 for n in ns))
                return
            bound = max_n_for(rem)
            # n < 0 时：n(n+1) = (-k)(1-k) = k(k-1)，bound 同样适用
            for n in range(-bound - 1, bound + 2):
                contrib = n * (n + 1)   # 恒 >= 0
                if contrib > rem:
                    continue
                ns.append(n)
                gen_half(dim + 1, rem - contrib, n_sum + n, ns)
                ns.pop()

        gen_half(0, target_h, 0, [])

    points = np.array(points)
    return points / np.linalg.norm(points, axis=1, keepdims=True)

def generate_E8_shells(ms:list):
    all_points = []
    for m in ms:
        pts = generate_E8_shell(m)
        all_points.append(pts)
    return np.vstack(all_points)


# ---------------------------------------------------------------------- #
# 验证工具                                                                 #
# ---------------------------------------------------------------------- #
def verify_shell(points: list, m: int):
    """验证返回的格点是否全部满足 E8 的格点条件以及范数² = 2m。"""
    target = 2 * m
    errors = []
    for p in points:
        norm_sq = sum(x * x for x in p)
        # 判断是整数型还是半整数型
        is_half = any(x != int(x) for x in p)
        if is_half:
            # 半整数：所有坐标应为 n + 0.5，坐标之和应为偶数
            ns = [x - 0.5 for x in p]
            if any(n != int(n) for n in ns):
                errors.append(f"非法半整数点：{p}")
                continue
            if round(sum(p)) % 2 != 0:
                errors.append(f"半整数点坐标和不为偶数：{p}")
        else:
            # 整数：坐标之和应为偶数
            if sum(int(x) for x in p) % 2 != 0:
                errors.append(f"整数点坐标和不为偶数：{p}")
        if abs(norm_sq - target) > 1e-9:
            errors.append(f"范数² 错误（期望 {target}，实际 {norm_sq:.4f}）：{p}")
    return errors

def test_E8_shell(m, pairwise=False):
    pts = generate_E8_shell(m)
    print(f"E8 shell m={m} has {len(pts)} points.")
    min_a, max_a, mean_a, std_a = angle_stats(pts, pairwise=pairwise)
    print(f"  Min NN angle: {min_a:.3f}°")
    print(f"  Max NN angle: {max_a:.3f}°")
    print(f"  Mean NN angle: {mean_a:.3f}°")
    print(f"  Std NN angle: {std_a:.3f}°")
    return min_a

def test_E8_shells(ms, pairwise=False):
    pts = generate_E8_shells(ms)
    print(f"E8 shell m={ms} has {len(pts)} points.")
    min_a, max_a, mean_a, std_a = angle_stats(pts, pairwise=pairwise)
    print(f"  Min NN angle: {min_a:.3f}°")
    print(f"  Max NN angle: {max_a:.3f}°")
    print(f"  Mean NN angle: {mean_a:.3f}°")
    print(f"  Std NN angle: {std_a:.3f}°")
    return min_a
if __name__ == "__main__":
    angles = []
    angles_pairwise = []
    for pairwise in [False, True]:
        if pairwise:
            l = angles_pairwise
        else:
            l = angles
        # l.append(test_E8_shell(1, pairwise=pairwise))
        # l.append(test_E8_shell(2, pairwise=pairwise))
        # l.append(test_E8_shell(3, pairwise=pairwise))
        # l.append(test_E8_shell(4, pairwise=pairwise))
        # l.append(test_E8_shell(5, pairwise=pairwise))
        # l.append(test_E8_shells([2,3,5], pairwise=pairwise))
        # l.append(test_E8_shells([1,2], pairwise=pairwise))
        # l.append(test_E8_shells([1,3], pairwise=pairwise))
        # l.append(test_E8_shells([1,5], pairwise=pairwise))
        # l.append(test_E8_shells([2,3], pairwise=pairwise))
        # l.append(test_E8_shells([2,5], pairwise=pairwise))
        # l.append(test_E8_shells([3,5], pairwise=pairwise))
    # test_E8_shell(6, pairwise=False)
    test_E8_shells([1,4], pairwise=True)
    print(f"{angles}")
    print(f"{angles_pairwise}")