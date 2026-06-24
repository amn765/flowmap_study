# 阶段 5 · MeanFlow 训练与实现（约 4-5 天）

> 目标：把阶段 4 的恒等式落成可运行的代码。核心是用 **JVP（Jacobian-vector product）** 一次前向算出 `v·∂_z u + ∂_t u`，再配上时间采样、损失加权、CFG，最后在一个 2D 玩具分布上真正训出一步生成。
> （梯度提示：数学已在阶段 4 备齐，这里全是工程化 —— 把每个公式对应到几行 PyTorch。）

主线资料：[MeanFlow 论文](https://arxiv.org/abs/2505.13447) 第 4 节 + 官方代码（[JAX 官方实现](https://github.com/Gsunshine/meanflow) / [PyTorch 实现](https://github.com/zhuyu-cs/MeanFlow)）+ [PyTorch torch.func.jvp 官方文档](https://pytorch.org/docs/stable/generated/torch.func.jvp.html)。

## 任务清单

### ☐ 5-1 训练损失与 stop-gradient 目标
- 把恒等式做成 loss。目标项 `u_tgt` 用 stop-gradient（`detach()`），只让左边的 `u_θ(z_t,r,t)` 回传梯度：
  ```python
  # v = eps - x   (线性路径的瞬时速度)
  # dudt = vjp/jvp 算出的全导数 v·∂z u + ∂t u
  u_tgt = (v - (t - r) * dudt).detach()      # stop-gradient
  loss = ((u_pred - u_tgt) ** 2).mean()
  ```
- 为什么 stop-grad：目标里含网络自身输出，若也回传会鼓励网络去「迁就目标」而塌缩 —— 和阶段 2 一致性损失同理。
- 思考题：这和一致性模型的 EMA 目标网络相比，MeanFlow 用普通 `detach()` 够不够？（论文：够，因为目标由解析恒等式给出，不靠移动目标稳住。）
- **完成标志**：能写出 loss 三行，并解释 stop-gradient 的必要性。

### ☐ 5-2 ⭐ 用 JVP 计算全导数
- 关键洞察（阶段 4-3）：`du/dt = v·∂_z u + ∂_t u` 正是 `u_θ` 在输入 `(z, r, t)` 处、沿切向量 `(v, 0, 1)` 的**方向导数**。用**前向模式自动微分**一次拿到：
  ```python
  import torch
  from torch.func import jvp

  def u_fn(z, r, t):
      return model(z, r, t)

  # 切向量：z 方向给 v，r 方向给 0，t 方向给 1
  tangents = (v, torch.zeros_like(r), torch.ones_like(t))
  u_pred, dudt = jvp(u_fn, (z_t, r, t), tangents)
  # u_pred = u_θ(z_t,r,t);  dudt = v·∂z u + ∂t u  一次前向同时得到
  ```
- 关键点：JVP 是**前向模式**，和反向传播相容，且只需一次额外前向，比有限差分稳、比反向求 Jacobian 省。
- 思考题：为什么不用有限差分 `(u(t+Δ)-u(t))/Δ` 近似 `du/dt`？（数值误差 + 需多次前向 + 对 Δ 敏感，呼应阶段 2-5 的「离散→连续」品味。）
- **完成标志**：能独立写出这段 `jvp` 调用，并讲清切向量 `(v,0,1)` 每一维的来历。

### ☐ 5-3 时间 (r, t) 采样策略
- 每个样本要采一对 `(r, t)`，满足 `r ≤ t`。常见做法：
  - 各自从分布（如 `U(0,1)` 或 logit-normal）采样，再令 `r, t = min, max`。
  - **按比例混入 `r = t`**：一部分样本强制 `r=t`（退化成 flow matching，稳住地基），其余 `r<t`（学真正的平均速度跨区间）。论文里这个比例是重要超参（例如 25%~75% 取 r=t）。
- 思考题：`r=t` 的比例太高 / 太低分别会怎样？（太高 ≈ 退回 FM 不会一步；太低 ≈ 地基不稳难收敛。）
- **完成标志**：能写出一段采样 `(r,t)` 并按比例置 `r=t` 的代码，解释比例的作用。

### ☐ 5-4 自适应损失加权
- 不同 `(r,t)` 的目标量纲差异大，直接 MSE 会被大残差样本主导。MeanFlow 用**自适应权重**（adaptive weighting），例如按目标范数归一：
  ```python
  # 论文式：w = 1 / (||Δ||² + c)^p  之类，stop-grad 权重
  w = (1.0 / (error.detach() ** 2 + 1e-3)).pow(p)
  loss = (w * error ** 2).mean()
  ```
- 思想同 EDM 的 loss weighting：让各时间区间对梯度的贡献均衡，训练更稳。
- 思考题：权重为什么要 `detach`？如果权重也带梯度会发生什么？
- **完成标志**：能解释自适应加权解决什么问题，并写出一种带 stop-grad 权重的 loss。

### ☐ 5-5 集成 Classifier-Free Guidance (CFG)
- MeanFlow 能把 CFG **吸进平均速度场本身**，采样时不必再跑两次（有条件/无条件）前向，保持一步生成的低成本。
- 做法概念：训练时按 CFG 思路构造「被引导的速度」当作 `v`，让 `u_θ` 直接学到含引导的平均速度；采样仍是单次前向。
- 思考题：标准扩散的 CFG 采样每步要 2 次前向。MeanFlow 把引导「烘焙」进网络后，一步采样的 NFE 是多少？
- **完成标志**：能说清 MeanFlow 如何把 CFG 融入训练，从而采样仍只需 1 次前向。

### ☐ 5-6 ⭐ 玩具实现：2D 分布一步生成
- 动手把全部拼起来，在一个 2D toy 分布（如 two-moons / 8-gaussians）上训练并一步采样：
  ```python
  # 训练一步（核心循环骨架，亲手补全）
  x = sample_data(B)                       # 真实数据 z_0
  eps = torch.randn_like(x)                # 噪声 z_1
  r, t = sample_times(B)                   # r<=t，含一定比例 r=t
  z_t = (1 - t) * x + t * eps              # 插值点
  v   = eps - x                            # 瞬时速度（解析）
  tangents = (v, torch.zeros_like(r), torch.ones_like(t))
  u_pred, dudt = torch.func.jvp(lambda z,r,t: model(z,r,t), (z_t,r,t), tangents)
  u_tgt = (v - (t - r) * dudt).detach()
  loss = ((u_pred - u_tgt) ** 2).mean()
  loss.backward(); opt.step(); opt.zero_grad()

  # 一步采样
  with torch.no_grad():
      e = torch.randn(N, 2)
      x_gen = e - model(e, torch.zeros(N,1), torch.ones(N,1))   # z_0 = z_1 - u(z_1,0,1)
  ```
- 把生成点和真实分布画在一起对比；再试 2 步、4 步采样看质量提升。
- **完成标志**：toy 模型能一步采样出与目标分布形状吻合的样本，且训练 loss 正常下降。

### ☐ 5-7 自测：口述题
合上资料，能流畅回答：
1. 为什么用 JVP 而不是有限差分算 `du/dt`？切向量是什么？
2. `(r,t)` 采样为什么要混入 `r=t`？比例高低各有什么后果。
3. 一步采样那行代码是怎么从恒等式来的？

## 完成标志
- 有一份能跑通的 2D MeanFlow 训练 + 一步采样脚本，loss 下降、采样分布对得上。
- 能把恒等式里每一项对应到代码里的具体一行（v / jvp / stop-grad / 采样）。
