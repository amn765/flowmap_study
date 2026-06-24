# Flow Map & MeanFlow 学习计划

> 目标：从「已掌握 flow matching（CFM / rectified flow / OT-CFM）」到「吃透流映射方法族与 MeanFlow，能读论文、读源码、复现一步生成、提出改进」。
> 总时长：按内容走（不卡死时间），全程约 24-32 个学习日；宁慢勿断。

## 设计思路

你已经会用速度场 + ODE 多步采样，所以这套计划**从 flow matching 之后接起**，主线只有一条：**如何跳过数值积分，做到少步乃至一步生成**。

梯度怎么做平缓：先把视角从「学瞬时速度」切到「学流映射」（阶段 0-1，只对齐动机与数学）→ 再看这条路上最成功的前身一致性模型（阶段 2）→ 收拢成统一框架，看清所有方法的坐标（阶段 3）→ 然后才进入核心 MeanFlow 的恒等式推导（阶段 4）→ 把公式落成 JVP 代码并跑通玩具实验（阶段 5）→ 最后精读论文、读源码、上真实数据集复现并对比前沿（阶段 6）。每个阶段只比上一阶段「多迈一小步」，且每个任务都是可勾选、有完成标志的小目标。

## 学习路线总览

| 阶段 | 主题 | 时长 | 里程碑 |
|------|------|------|--------|
| 0 | [从速度场到流映射](plan/stage0-从速度场到流映射.md) | 约 2-3 天 | 说清「为什么要 flow map」，建立方法族全景 |
| 1 | [流映射的数学基础](plan/stage1-流映射的数学基础.md) | 约 3-4 天 | 会写双时间流映射、半群性质与 ∂t Φ=v |
| 2 | [一致性模型谱系](plan/stage2-一致性模型谱系.md) | 约 3-4 天 | 理解 CM/CT/CTM「自洽当损失」的范式 |
| 3 | [流映射统一框架](plan/stage3-流映射统一框架.md) | 约 3-4 天 | 用一张表定位 FMM/Shortcut/MeanFlow |
| 4 | [MeanFlow 核心](plan/stage4-MeanFlow核心.md) | 约 4-5 天 | 闭卷推出 MeanFlow 恒等式与一步采样 |
| 5 | [MeanFlow 训练与实现](plan/stage5-MeanFlow训练与实现.md) | 约 4-5 天 | 用 JVP 跑通 2D 一步生成 |
| 6 | [论文源码与前沿](plan/stage6-论文源码与前沿.md) | 约 5-7 天 | 复现图像一步生成 + 提出改进 idea |

## 在线访问（手机也能学）

已部署到 GitHub Pages。手机浏览器打开后选「添加到主屏幕」即可当 App 用。两个页面共享同一份进度（同一浏览器内），换设备用面板里的「导出 / 导入进度」同步。

| 入口 | 网址 |
|------|------|
| 📖 学习站（阅读文档 + 勾选任务） | https://amn765.github.io/flowmap_study/ |
| 📊 打卡面板（进度 / 热力图 / 连续天数） | https://amn765.github.io/flowmap_study/tracker/ |
| 💾 GitHub 仓库 | https://github.com/amn765/flowmap_study |

## 怎么使用这套计划

1. **打开进度面板**：双击 `tracker/index.html` 即可用（不依赖联网，本地直接跑）。每完成一个任务就勾选，进度条、热力图、连续天数会实时更新。进度存在浏览器 localStorage 里，换浏览器/电脑前记得用「导出进度」备份。
2. **读文档**：推荐直接用线上学习站 https://amn765.github.io/flowmap_study/ ，文档正文、勾选都正常。⚠️ 注意：若**本地双击** `index.html`，浏览器禁止用 `fetch` 读本地 `plan/*.md`（CORS），文档正文会读不出来 —— 本地想读正文就直接打开 `plan/` 文件夹里的 markdown。打卡面板 `tracker/index.html` 不受此限制，本地随便用。
3. **每天的节奏**：打开当前阶段的 markdown → 挑 1~2 个任务 → 动手验证 → 回到面板打勾。即使某天没时间，也点一下「今日打卡」保住连续天数。
4. **黄金法则**：
   - 所有公式**亲手推一遍**、所有代码**亲手敲一遍**，不要复制粘贴。
   - 每个任务做完后，合上资料**口述一遍学到了什么**——讲不出来就是没懂（每个阶段都配了口述自测题）。
   - 卡住超过 1 小时就先跳过、做标记，往往学到后面会自然解开。

## 推荐资料（全程通用）

- MeanFlow：*Mean Flows for One-step Generative Modeling*（Geng, Deng, Bai, Kolter, He, 2025）—— 核心论文，反复读。
- *Flow Map Matching*（Boffi, Albergo, Vanden-Eijnden, 2024）—— 统一框架。
- *Consistency Models*（Song et al., 2023）与 *Consistency Trajectory Models*（Kim et al., 2024）—— 前身谱系。
- *One Step Diffusion via Shortcut Models*（Frans et al., 2024）—— 步长条件自一致。
- PyTorch `torch.func.jvp` 官方文档 —— 阶段 5 实现关键。
