# XHAND1 MuJoCo 复现工程（左/右手，5×120 触觉）

这是我整理出来的一套 XHAND1 MuJoCo 复现结果，目标是把双手模型、触觉和验证流程都放在一个地方，方便直接运行和重复生成。

模型结构和几何以正式 URDF v1.3 为准。

## 环境准备

本分支使用完全位于仓库内的 uv 环境，不依赖 Conda，不使用 `sudo`，也不让 uv
自动下载 Python。主机需要已有 `/usr/bin/python3.10`、`curl`、`tar`、
`sha256sum`、OSMesa、`ffmpeg` 和 `ffprobe`。

首次初始化：

```bash
./scripts/bootstrap_uv.sh
```

脚本只下载固定的 Linux x86-64 `uv 0.12.5` 归档，并在解压前核对 SHA-256
`68a509da24b06b4223a1c0175fb5eb5bc79342b76cbeff0cfe51ac3f5b17b6b2`。
uv 位于未跟踪的 `.tools/`，虚拟环境位于 `.venv/`，缓存位于
`/tmp/xhand1-uv-cache`。`uv.lock` 固定 MuJoCo 3.10.0 及全部 Python 依赖和
发行文件哈希。

后续所有 Python 命令都通过包装器运行：

```bash
./scripts/uv.sh sync --locked
./scripts/uv.sh run --frozen python --version
```

包装器禁止 Python 自动下载，清除 ROS 常见的 `PYTHONPATH`/`PYTHONHOME`
注入，并默认设置 `MUJOCO_GL=osmesa`。确实需要交互式 viewer 时可显式使用
`MUJOCO_GL=glfw ./scripts/uv.sh ...`。

## 快速运行

先运行全量无界面验收：

```bash
./scripts/uv.sh run --frozen python verify_xhand.py --side left
./scripts/uv.sh run --frozen pytest -q
```

打开右手交互场景：

```bash
MUJOCO_GL=glfw ./scripts/uv.sh run --frozen python demo_control.py --side right
```

左手：

```bash
MUJOCO_GL=glfw ./scripts/uv.sh run --frozen python demo_control.py --side left
```

只加载手本体、不创建地面、支撑、球或方块：

```bash
MUJOCO_GL=glfw ./scripts/uv.sh run --frozen python simulate_hand_only.py \
  --side left --pose open
```

该入口直接编译 `xhand_left.xml` 或 `xhand_right.xml`，手根固定，模型中只有
XHAND 的 30 个 body、12 个 hinge joint、12 个 actuator 和触觉 site/sensor；
不含 freejoint 或 mocap body。右侧 `Control` 面板的 12 个 slider 可直接拖动。
无窗口检查可增加 `--headless --seconds 1`。

交互键：

- `O`：张开
- `P`：捏取橙色球（推荐首次测试）
- `C`：握拳
- `R`：重置手和球
- `T`：显示/隐藏 600 个 taxel site
- `F`：显示/隐藏接触点和接触力

MuJoCo 右侧 `Control` 面板可直接拖动 12 个 actuator。首次拖动后，Demo 会停止
覆盖 `data.ctrl` 并切换到手动保持模式；后续拖动立即生效。按 `O`、`P` 或 `C`
可从当前 slider 数值重新进入对应预设轨迹。如果只想用原生 viewer：

```bash
MUJOCO_GL=glfw ./scripts/uv.sh run --frozen python -m mujoco.viewer --mjcf scene_right.xml
```

无界面运行捏取和触觉输出：

```bash
./scripts/uv.sh run --frozen python demo_control.py --side right --pose pinch --headless --seconds 2.5
```

## 左手三指抬升方块

`grasp_cube.py` 是兼容入口和单一 CLI；实现按职责位于 `xhand_grasp/`：配置与
实验注册、`MjSpec` 场景、接触面分类、轨迹、仿真、硬判据、搜索、渲染和产物
序列化彼此分离。场景从 `xhand_left.xml` 在内存构建，不修改原 XML：手根是固定
body，方块有 freejoint，只有拇指、食指、中指的八个 actuator 接收 smoothstep
时变目标；无名指和小指目标始终严格为零。

仓库同时保留 schema v1 的历史基线和 schema v2 的掌心向下、指定对向接触面
实验。为避免默认配置仍指向 v1 所造成的混淆，v2 命令应始终显式传入：

```text
grasp_configs/left_opposed_face_palm_down.json
```

### v2 单次运行

先完成下节的调参，再运行其固化的 30 mm、20 g、滑动摩擦 0.8 最佳配置；JSON、
NPZ 和 MP4 均来自同一次仿真：

```bash
./scripts/uv.sh run --frozen python grasp_cube.py run \
  --config artifacts/left_opposed_face_palm_down/tune/best_config.json \
  --output-dir artifacts/left_opposed_face_palm_down/nominal \
  --video --video-filename nominal.mp4
```

`run` 仅在全部硬判据通过时返回 `0`，未通过时返回 `2`，但两种情况都会完整发布
可审计产物。输出目录必须尚不存在，防止旧视频与新指标混用。

### v2 调参与鲁棒性

使用固定种子搜索静态手 pose 和八个三指目标；完整预算由版本化配置定义：

```bash
./scripts/uv.sh run --frozen python grasp_cube.py tune \
  --config grasp_configs/left_opposed_face_palm_down.json \
  --output-dir artifacts/left_opposed_face_palm_down/tune \
  --workers 16 --seed 20260821
```

随后应对 `tune` 写出的完整 `best_config.json` 执行 100 组物性网格和 50 次固定
种子局部扰动：

```bash
./scripts/uv.sh run --frozen python grasp_cube.py robustness \
  --config artifacts/left_opposed_face_palm_down/tune/best_config.json \
  --output artifacts/left_opposed_face_palm_down/robustness.json \
  --workers 16 --seed 20260821
```

鲁棒性 JSON 不省略失败组合，并内嵌 `hardest_passing_config`：它不仅包含尺寸、
质量和摩擦，还保留同一次网格试验使用的完整手 pose、控制目标、时序和判据。
无需人工抄写参数即可重新运行该样本并生成视频：

```bash
./scripts/uv.sh run --frozen python grasp_cube.py run \
  --hardest-from artifacts/left_opposed_face_palm_down/robustness.json \
  --output-dir artifacts/left_opposed_face_palm_down/hardest-passing \
  --video --video-filename hardest_passing.mp4
```

如果网格没有任何通过项，`hardest_passing_config` 为 `null`，上述命令会明确拒绝
运行，而不会静默退回名义配置。

### 产物一致性契约

`run` 先在目标目录同级的临时 staging 目录中完成所有文件，最后一次性发布目录；
仿真、编码或元数据采集失败时不会留下可被误认为完成结果的输出目录。

- `result.json` 保存实际运行后派生的 `experiment_status`、完整 summary、最终解析
  配置、环境与源码/模型/锁文件哈希，以及各产物 SHA-256。schema v2 另保存全部
  `normalized_acceptance_margins`、其中的最小值和对应 `limiting_metric`；失败运行
  同样保留负余量，不会只报告通过项。
- `resolved_config.json` 与 `result.json.config` 完全一致；输入配置中的
  `initial_near_miss` 等说明不会覆盖本次运行的实际通过/失败状态。
- `trace.npz` 保存完整 1 ms 时序。schema v2 还保存 `face_order`、
  `finger_order`、`actuator_order`，因此接触面、三指和 actuator 数值轴无需依赖
  隐含下标；`target_face_topology` 可由逐帧接触力、触觉和纯度重新计算。
- `video_frame_steps` 给出每帧对应的仿真 step；MP4 来自同一个仿真循环，写出后
  会完整解码并由 `ffprobe` 核对编码、尺寸、帧率和帧数。
- 使用 `--hardest-from` 时，metadata 同时记录 robustness JSON 的路径、文件哈希
  和 `hardest_passing_config` 选择器。

### 当前 v2 结果状态

结论：**未在声明范围内验证成功**。固定种子 `20260821` 的完整运行包含 250,000
个名义运动学样本、512 个名义动力学候选、2,048 个局部精调候选、2,048 个尺寸/
物性 fallback 候选，以及 256 次 finalist 扰动探针；4,608 个候选中没有一个通过
全部硬约束。五种 fallback 尺寸各自另做 25,000 个静态几何样本。

保留的最佳近失仍是名义 `30 mm / 20 g / μ=0.8`：

- 方块中位/最低抬升 `10.022/10.005 mm`，保持高度跨度 `0.056 mm`，姿态漂移
  `1.17°`，末速度 `0.00473 m/s`，最大穿透 `0.760 mm`。
- 掌面向下角 `11.20°`；手根固定；无掌部、无名指、小指、错误面或活动指非末端
  接触；支撑在保持阶段前已脱离。
- 拇指与中指在固化的 `-Y/+Y` 目标面职责均为 `100%`，但食指目标面职责和触觉
  均为 `0%`，三指同时拓扑也为 `0%`。这五项硬检查失败，最小归一化余量为
  `-4.0`（限制指标 `index_contact_duty`），所以不能把该轨迹称为三指成功。
- 完整物性网格 `0/100`，五种尺寸各 `0/20`；固定种子局部扰动 `0/50`。
  因此 `hardest_passing_config` 为 `null`，没有生成虚假的“最难通过”视频。

审计产物位于 `artifacts/left_opposed_face_palm_down/`：`tune/` 保存分阶段计数、
前 20 个候选、finalist 探针明细与最佳配置，`nominal/` 和 `best-near-miss/` 保存同源 JSON/NPZ/MP4，
`robustness.json` 完整保存 100+50 项。仓库内的 v2 配置仍明确标记为
`initial_near_miss`，避免把失败配置静默固化成已验证成功。

schema v1 的既有实验或历史产物不能作为 schema v2 的成功证据。

### 36–50 mm 大尺寸 campaign

独立实验 `left_opposed_face_palm_down_large_cube_lift` 将尺寸扩展到
36/38/40/42/44/46/48/50 mm，并保持旧 v2 实验及产物不变。其版本化配置还固化
了固定 20 g 几何发现、等密度复验、独立绝对 pregrasp/final 目标采样、单次搜索
边界扩展及 75+25 鲁棒性 case family：

```bash
./scripts/uv.sh run --frozen python grasp_cube.py tune \
  --config grasp_configs/left_opposed_face_palm_down_large_cube.json \
  --output-dir artifacts/left_opposed_face_palm_down_large_cube/tune \
  --workers 16 --seed 20260821

./scripts/uv.sh run --frozen python grasp_cube.py run \
  --config artifacts/left_opposed_face_palm_down_large_cube/tune/best_config.json \
  --output-dir artifacts/left_opposed_face_palm_down_large_cube/nominal \
  --video --video-filename nominal.mp4

./scripts/uv.sh run --frozen python grasp_cube.py robustness \
  --config artifacts/left_opposed_face_palm_down_large_cube/tune/best_config.json \
  --output artifacts/left_opposed_face_palm_down_large_cube/robustness.json \
  --workers 16 --seed 20260821
```

当前固定种子完整实验的结论仍是 **未在 36–50 mm 声明范围内验证成功**。搜索
实际完成 1,190,000 个静态样本；36 mm 和 37 mm 精筛均没有候选满足严格静态
晋级 gate，因此只保留明确标注的诊断近失并执行一次边界扩展和 128 次局部精调。
最佳 36 mm/20 g 近失已达到中位/最低抬升 11.762/11.586 mm，掌面角 5.246°，
但中指有效接触与三指同时拓扑均为 0，且存在活动指非末端辅助接触。固定 20 g
硬通过数为 0，故没有进入等密度名义精调；鲁棒性为等密度 `0/75`、固定质量对照
`0/25`、局部扰动 `0/50`。

完整逐尺寸计数、失败硬检查、边界诊断、哈希和视频验证记录在
`artifacts/left_opposed_face_palm_down_large_cube/EXPERIMENT_REPORT.md`。名义与
显式 best-near-miss JSON/NPZ/MP4 均已保存；因没有任何通过网格项，
`hardest_passing_config` 为 `null`，没有生成误导性的 hardest-passing 视频。

### 52–64 mm 先抓稳再抬升 campaign

schema v3 实验
`left_opposed_face_palm_down_larger_cube_grasp_then_lift` 不再按固定时间无条件
进入抬升。控制器执行 `SETTLE → CLOSE → VERIFY → MANIPULATE → HOLD`：三指在
固化对向面上连续 0.25 s 满足力、纯度、触觉、支撑、稳定性和安全 gate 后，才从
下一帧发送相对抓取姿态的 `manipulation_delta_rad`。验证超时或出现禁触、越限、
超穿透等硬违规时进入 `ABORT`，保持抓取姿态且操作进度严格为零。总仿真时长固定
为 4.75 s，提前完成验证所节省的时间并入最终 HOLD。

正式配置与命令：

```bash
./scripts/uv.sh run --frozen python grasp_cube.py tune \
  --config grasp_configs/left_opposed_face_palm_down_larger_cube_grasp_then_lift.json \
  --output-dir artifacts/left_opposed_face_palm_down_larger_cube_grasp_then_lift/tune \
  --workers 16 --seed 20260821

./scripts/uv.sh run --frozen python grasp_cube.py run \
  --config artifacts/left_opposed_face_palm_down_larger_cube_grasp_then_lift/tune/best_config.json \
  --output-dir artifacts/left_opposed_face_palm_down_larger_cube_grasp_then_lift/best-near-miss \
  --video --video-filename best_near_miss.mp4

# 仅当 tune 生成 best_constant_density_config.json 时运行：
./scripts/uv.sh run --frozen python grasp_cube.py robustness \
  --config artifacts/left_opposed_face_palm_down_larger_cube_grasp_then_lift/tune/best_constant_density_config.json \
  --output artifacts/left_opposed_face_palm_down_larger_cube_grasp_then_lift/robustness.json \
  --workers 16 --seed 20260821
```

搜索先覆盖 52/54/56/58/60/62/64 mm，再对三个最佳粗尺寸的相邻 1 mm 尺寸
补筛，并只让动态稳定抓取通过项进入操作增量搜索。固定 20 g 通过仍只是几何对照；
最终验证要求按 30 mm/20 g 锚点计算的等密度质量通过。schema-v3 NPZ 在 v2
接触证据之外保存控制状态、逐项 gate、连续通过步数、抓取锁存、操作进度和四个
事件帧；JSON 中分别报告 `grasp_success`、`manipulation_success` 和
`full_success`。

当前固定种子正式实验的结论为 **未在 52–64 mm 声明范围内验证成功**。搜索
完成 1,295,000 个实际静态样本（相邻尺寸去重前最大声明预算 1,365,000），选择
58 mm 与 62 mm 各执行 256 个初始动力学候选和 512 个抓取局部精调；两者稳定
抓取均为 `0/768`。因此操作目标从未获授权，fixed full、等密度复验和 density full
均为 0。最佳 58 mm/20 g 近失的掌面角、穿透、支撑和稳定性合格，但拇指与食指
目标面力为 0，中指还存在 1.4995 N 非末端接触；VERIFY 连续 gate 最大计数为 0，
最终抬升仅约 0.062 mm。

因没有合法等密度名义配置，100+50 鲁棒性套件按协议未运行，并在
`artifacts/left_opposed_face_palm_down_larger_cube_grasp_then_lift/robustness_not_run.json`
中保存明确原因。完整预算、逐尺寸成功率、失败约束、环境/源码哈希和同源
JSON/NPZ/MP4 验证见同目录的 `EXPERIMENT_REPORT.md`。

### 62 mm 相对 pose rescue

在不改动上述宽域 campaign 及其失败产物的前提下，独立实验
`left_opposed_face_palm_down_larger_cube_relative_pose_rescue` 围绕 62 mm 静态候选
重新联合搜索方块在手根坐标系中的 XYZ、手根 roll/pitch/yaw、方块 yaw、抓取目标
和受限的相对操作增量。它仍使用同一 schema-v3 状态机、固定手根、三指隔离、对向
面接触与全部硬阈值；一次性边界扩展后 index/mid distal 绝对目标上限为 1.8 rad，
没有再次扩展或放宽验收条件。

定向搜索找到并重复验证了一个 **62 mm 等密度名义成功项**：方块质量
`176.539259 g`、摩擦 `0.8`，中位/最低抬升 `11.2196/11.1692 mm`，食指与中指
接触 `+X` 面、拇指接触 `-X` 面。三指操作阶段目标面占空比分别为
`100%/99.967%/100%`，同时拓扑 `99.967%`；掌面角 `0.483°`，保持高度波动
`0.209 mm`，姿态漂移 `1.675°`，末速度 `0.000236 m/s`，最大穿透
`0.199 mm`。54/54 项硬检查全部通过，手根、inactive 控制、错误面、非末端和禁触
违规均为零。

```bash
# 复现已固化的等密度名义成功项
./scripts/uv.sh run --frozen python grasp_cube.py run \
  --config grasp_configs/left_opposed_face_palm_down_larger_cube_relative_pose_rescue_validated.json \
  --output-dir artifacts/left_opposed_face_palm_down_larger_cube_relative_pose_rescue/nominal-reproduction \
  --video --video-filename nominal.mp4

# 运行登记的 100 组物性网格和 50 次固定种子扰动
./scripts/uv.sh run --frozen python grasp_cube.py robustness \
  --config grasp_configs/left_opposed_face_palm_down_larger_cube_relative_pose_rescue_validated.json \
  --output artifacts/left_opposed_face_palm_down_larger_cube_relative_pose_rescue/robustness-reproduction.json \
  --workers 16 --seed 20260821
```

该配置的名义任务已验证成功，但尚不能标记为鲁棒通过：完整网格通过 `38/100`
（等密度 `28/75`、固定 20 g 对照 `10/25`），局部扰动通过 `34/50`，低于
`45/50` 门槛；摩擦 `0.4/0.6` 的 40 个网格项均失败。等密度、`μ=0.8` 的
0.5 mm 尺寸边界探测在 `59.0–64.0 mm` 全部通过，向下首个失败点为
`58.5 mm`，向上在 64 mm 达到声明范围边界。名义与 hardest-passing 的
JSON/NPZ/MP4、完整失败组合、边界探测和哈希见
`artifacts/left_opposed_face_palm_down_larger_cube_relative_pose_rescue/SEARCH_REPORT.md`。

### 导出多段已通过的物体轨迹

`catalog` 子命令只接受完整鲁棒性报告中已经记录为 `passed` 且
`stage_status.full_success` 的零基网格索引；它用注册表重新生成精确物性并逐项
复跑。全部复跑成功后才原子发布目录，任何一项失败都不会留下半成品。可选标签
仅用于报告，不改变稳定的 `grid_NNN/` 目录名。

```bash
./scripts/uv.sh run --frozen python grasp_cube.py catalog \
  --config grasp_configs/left_opposed_face_palm_down_larger_cube_relative_pose_rescue_validated.json \
  --robustness artifacts/left_opposed_face_palm_down_larger_cube_relative_pose_rescue/robustness.json \
  --grid-index 2 --label 2=small_light \
  --grid-index 18 --label 18=small_high_friction \
  --grid-index 37 --label 37=nominal \
  --grid-index 38 --label 38=friction_pair \
  --grid-index 57 --label 57=large_heavy \
  --grid-index 74 --label 74=boundary_heaviest \
  --grid-index 88 --label 88=mass_control \
  --output-dir artifacts/left_opposed_face_palm_down_larger_cube_relative_pose_rescue/trajectory_catalog-reproduction \
  --video
```

每段轨迹包含 `resolved_config.json`、`result.json`、`trace.npz`、逐仿真步的
`object_trajectory.csv` 和可选 MP4。顶层 `catalog.json` 汇总物体尺寸、质量、
密度、显式惯量、三向摩擦、接触参数、手—物初始 pose，以及 initial、
grasp-acquired、manipulation-start、manipulation-end、final 五个关键帧。
当前目录含 7 条全硬通过轨迹，覆盖 60–64 mm、20–213.599 g 和滑动摩擦
0.8–1.2；完整参数与轨迹摘要见 `trajectory_catalog/TRAJECTORY_CATALOG.md`。

### 在 MuJoCo Viewer 中运行或回放轨迹

项目的 uv 包装脚本默认使用无窗口的 OSMesa。打开交互式 Viewer 时必须在命令前
显式设置 `MUJOCO_GL=glfw`。Viewer 默认从目录解析 resolved config，创建新的独立
物理会话，重新执行真实控制和每个 `mj_step`；目录 NPZ 只用于无参数覆盖时的精确
结果对照：

```bash
MUJOCO_GL=glfw ./scripts/uv.sh run --frozen python grasp_cube.py view \
  --catalog artifacts/left_opposed_face_palm_down_larger_cube_relative_pose_rescue/trajectory_catalog/catalog.json \
  --trajectory nominal \
  --loop
```

可用标签为 `small_light`、`small_high_friction`、`nominal`、
`friction_pair`、`large_heavy`、`boundary_heaviest` 和 `mass_control`。
也可绕过目录，直接指定一对产物。此时仍重新运行物理，并把 NPZ 当作参考：

```bash
MUJOCO_GL=glfw ./scripts/uv.sh run --frozen python grasp_cube.py view \
  --config artifacts/left_opposed_face_palm_down_larger_cube_relative_pose_rescue/trajectory_catalog/grid_037/resolved_config.json \
  --trace artifacts/left_opposed_face_palm_down_larger_cube_relative_pose_rescue/trajectory_catalog/grid_037/trace.npz \
  --speed 1.0 --loop
```

只有显式增加 `--state-replay` 才会逐帧恢复记录状态：

```bash
MUJOCO_GL=glfw ./scripts/uv.sh run --frozen python grasp_cube.py view \
  --state-replay \
  --catalog artifacts/left_opposed_face_palm_down_larger_cube_relative_pose_rescue/trajectory_catalog/catalog.json \
  --trajectory nominal --loop
```

窗口中按空格暂停/继续，按 `R` 从头运行，按 `L` 切换循环；还可用
`--start-paused` 从首帧暂停打开。Linux 上需要有效的 `DISPLAY` 或
`WAYLAND_DISPLAY`。

## Schema-v4 倾斜下压与三指等高接触实验

配置
`grasp_configs/left_opposed_face_palm_tilted_down_aligned_contacts_grasp_then_lift.json`
注册了独立实验
`left_opposed_face_palm_tilted_down_aligned_contacts_grasp_then_lift`。它搜索
59–64 mm 等密度方块、10/12.5/15/17.5/20° 五个手指下倾带和 2–8 mm 下压；
抓取验证要求连续 0.25 s 三指目标面有效且重力方向高度跨度不超过 5 mm，操作与保持
阶段的等高占空比至少 70%。完整预算为 600,000 个静态样本、最多 640 个动态候选、
2,560 次抓取精调、1,280 次操作精调和 80 次 1 ms 复验。

正式搜索命令如下；成功配置与近失均由同一个真实仿真内核重新执行后写入 catalog：

```bash
./scripts/uv.sh run --frozen python grasp_cube.py tune \
  --config grasp_configs/left_opposed_face_palm_tilted_down_aligned_contacts_grasp_then_lift.json \
  --workers 16 --seed 20260821
```

本次固定种子正式运行已完整覆盖 600,000 个样本，但静态硬门晋级数为 0；因此没有
获得成功抓取或操作轨迹，五个倾角带均仅发布诚实近失。最佳近失的独立 50 次
Latin-hypercube 扰动为 `0/50`，未达到 `45/50` 鲁棒门槛。未放宽 5 mm 等高、
2 mm 穿透、对向面或 10/8 mm 抬升判据。逐带统计、物理参数和失败诊断见
`artifacts/left_opposed_face_palm_tilted_down_aligned_contacts_grasp_then_lift/EXPERIMENT_REPORT.md`。

查看 12.5° 带近失时，默认会重新执行控制和物理：

```bash
MUJOCO_GL=glfw ./scripts/uv.sh run --frozen python grasp_cube.py view \
  --catalog artifacts/left_opposed_face_palm_tilted_down_aligned_contacts_grasp_then_lift/tune/trajectory_catalog/catalog.json \
  --trajectory near_miss_tilt_12p5
```

也可以直接运行某条 resolved config，并保存实际看到的本次结果：

```bash
MUJOCO_GL=glfw ./scripts/uv.sh run --frozen python grasp_cube.py view \
  --config artifacts/left_opposed_face_palm_tilted_down_aligned_contacts_grasp_then_lift/tune/trajectory_catalog/near_miss_tilt_12p5/resolved_config.json \
  --output-dir /tmp/xhand-v4-viewer-run
```

Viewer 支持 `--edge-mm`、互斥的 `--mass-g/--density-scale`、`--friction`、
`--finger-down-deg`、hand roll/yaw、`--press-mm`、`--cube-in-root-mm X Y Z` 和
`--cube-rpy-deg R P Y`。覆盖参数后会重新校验并重新仿真，状态标为
`parameter_override_run`，不会继承 catalog 的结论。例如：

```bash
MUJOCO_GL=glfw ./scripts/uv.sh run --frozen python grasp_cube.py view \
  --catalog artifacts/left_opposed_face_palm_tilted_down_aligned_contacts_grasp_then_lift/tune/trajectory_catalog/catalog.json \
  --trajectory near_miss_tilt_12p5 \
  --edge-mm 62 --density-scale 1.0 --friction 0.8 \
  --finger-down-deg 15 --hand-roll-deg 0 --hand-yaw-deg -2 \
  --press-mm 5 --cube-rpy-deg 0 0 30 \
  --output-dir /tmp/xhand-v4-override-run
```

窗口中新增 `C` 键切换三色接触质心、平均水平面和 ±2.5 mm 等高带；Space、`R`、
`L` 分别用于暂停、重启和循环。要查看 catalog 中记录的状态而不重新仿真，增加
`--state-replay`。

## Schema-v5 远距离、高拇指弯曲与指腹优先实验

配置
`grasp_configs/left_opposed_face_palm_tilted_down_far_hand_fingertip_grasp_then_lift.json`
将初始手根—方块中心距离限制为 138–160 mm，并把
`left_hand_thumb_bend_joint_actuator` 的抓取搜索范围提高到 0.85–1.35 rad。
Viewer 仍会从 resolved config 重新执行控制和真实物理；catalog alias 选中的
`resolved_config.json` 与 `trace.npz` 会先经过 SHA-256 校验。v5 不再接受
`--press-mm`，手物距离覆盖应使用 `--root-cube-distance-mm`，它沿 resolved pose
中的局部手物射线重算固定手根位置。

只有最佳候选经重新仿真满足全部硬约束时，catalog 才包含 `best_nominal`。成功轨迹的
Viewer 命令模板为：

```bash
MUJOCO_GL=glfw ./scripts/uv.sh run --frozen python grasp_cube.py view \
  --catalog artifacts/left_opposed_face_palm_tilted_down_far_hand_fingertip_grasp_then_lift/tune/trajectory_catalog/catalog.json \
  --trajectory best_nominal \
  --joint-monitor left_hand_thumb_bend_joint_actuator
```

固定种子 `20260821` 的完整预算结果已生成 `best_nominal -> tilt_17p5`：60 mm、
160 g、摩擦 0.8，初始手物距离 146.81 mm，thumb bend 抓取目标 0.956 rad；
最终窗口抬升中位/最低值为 10.683/10.677 mm，操作阶段三指等高占空比为
72.99%，三指 force-weighted 指腹力比例均为 100%。该配置是名义硬通过，
但独立 50 次固定种子扰动仅 3 次通过，未达到 45/50 鲁棒门槛；尺寸复验仅
60 mm 和 61 mm 通过。完整结果分别见 `tune/tune_results.json` 和仓库产物
目录下的 `robustness.json`。

若声明预算内没有完整硬通过项，catalog 不会创建 `best_nominal`，而只把真实复验后的
最佳近失标为 `best_attempt`：

```bash
MUJOCO_GL=glfw ./scripts/uv.sh run --frozen python grasp_cube.py view \
  --catalog artifacts/left_opposed_face_palm_tilted_down_far_hand_fingertip_grasp_then_lift/tune/trajectory_catalog/catalog.json \
  --trajectory best_attempt \
  --joint-monitor left_hand_thumb_bend_joint_actuator
```

黄色箭头显示被监视关节的世界坐标轴；终端每 0.1 s 报告目标角、实际角、误差、接触力、
指腹力比例和有效 taxel 数。未显式指定 `--joint-monitor` 时，schema-v5 默认监视上述
thumb bend actuator。实时与状态回放 Viewer 默认使用以方块为观察中心的自由相机；
鼠标左键拖动旋转视角、右键拖动平移视角，滚轮缩放。固定的
`three_finger_camera` 仅继续用于离线 MP4 渲染。

### 抓取优先的高拇指、多尺寸轨迹

`grasp_configs/left_far_hand_high_thumb_grasp_acquisition_manifest.json` 将操作增量固定为
零，只搜索并复验“先抓住”。它发布 6 条通过轨迹：60 mm 方块的 thumb bend 目标
1.15/1.16/1.17 rad，以及 61/62/64 mm 等密度方块。最高通过目标为 1.17 rad；
`best_grasp` 对应 60 mm、160 g、摩擦 0.8、手物距离 157.453 mm。该项连续
250 ms 全部抓取 gate 通过，三指均为 100% 指腹力，成功窗口等高跨度最大
4.948 mm。`1.20 rad` 的 266 个候选没有通过，未放宽 5 mm 判据。

重新生成独立 catalog：

```bash
MUJOCO_GL=osmesa ./scripts/uv.sh run --frozen python grasp_cube.py grasp-catalog \
  --manifest grasp_configs/left_far_hand_high_thumb_grasp_acquisition_manifest.json \
  --output-dir /tmp/xhand-high-thumb-grasp-catalog
```

在可拖动自由视角中重新执行最佳抓取的真实控制和物理：

```bash
MUJOCO_GL=glfw ./scripts/uv.sh run --frozen python grasp_cube.py view \
  --catalog artifacts/left_opposed_face_palm_tilted_down_far_hand_fingertip_grasp_then_lift/grasp_acquisition_high_thumb/trajectory_catalog/catalog.json \
  --trajectory best_grasp \
  --joint-monitor left_hand_thumb_bend_joint_actuator \
  --start-paused --loop
```

catalog 的 `trace.npz` 是完整 live-physics 对照；每条额外的 `grasp_trace.npz` 严格在
抓取锁存帧结束，不含 MANIPULATE/HOLD。这里的成功范围仅为 `grasp_acquisition`：
尚未验证抬升或完整操作。详细参数和限制见
`artifacts/left_opposed_face_palm_tilted_down_far_hand_fingertip_grasp_then_lift/grasp_acquisition_high_thumb/EXPERIMENT_REPORT.md`。

### Schema-v6 抓取前保持物体初始位姿

`grasp_configs/left_opposed_face_palm_down_pose_preserving_grasp.json` 基于抓稳后的
手物关系反求无碰撞预抓取姿态，并把活动手指直接初始化到该预抓取关节状态。方块仍是
带 freejoint 的自由体：没有 weld、mocap 或运行时方块 qpos 重写。控制流程在 0.5 s
`SETTLE` 中保持预抓取姿态，再用分关节 smoothstep 闭合；从仿真首个物理步前的方块
pose 到抓取锁存帧（含该帧）逐帧检查：最大平移 0.5 mm、最大姿态变化 1°、支撑持续
存在，且 `SETTLE` 期间任何手部 collision geom 都不得接触方块。

当前最佳 `pose_preserving_id94` 使用 60 mm、160 g、摩擦 0.8 的方块，thumb bend
目标 1.17 rad。抓取在 1.750 s 锁存，锁存前最大平移 0.400 mm、最大姿态变化
0.896°；验证窗口内三指目标面、等高和指腹接触占空比均为 100%，高度跨度最大
4.365 mm。该 catalog 的验证范围明确是 `grasp_acquisition_pose_preserved`，操作增量
为零，因此它不是抬升成功结果。

在可拖动自由视角中重新执行这条真实物理抓取：

```bash
MUJOCO_GL=glfw ./scripts/uv.sh run --frozen python grasp_cube.py view \
  --catalog artifacts/left_opposed_face_palm_down_pose_preserving_grasp/grasp_acquisition_pose_preserved/catalog.json \
  --trajectory best_pose_preserving_grasp \
  --joint-monitor left_hand_thumb_bend_joint_actuator \
  --start-paused --loop
```

Viewer 只在启动时把自由相机对准方块，之后不会重写相机；左键拖动旋转、右键拖动
平移、滚轮缩放。Space 控制暂停，R 从同一 resolved config 重新开始，L 切换循环。

### 六条固定物体 pose 的抓取复验

六条高拇指 source 都已独立重调 hand root pose、pregrasp 关节姿态和各指
close timing。方块的初始世界 pose、尺寸、质量和摩擦保持各自 source 的值不变。
方块仍是具有 freejoint 的自由体，没有 weld，也没有在仿真中 reset 或重写方块
qpos。从第一个物理步到抓取锁存帧（包含锁存帧）的 sticky 验收上限为
0.5 mm 平移和 1° 姿态变化，因此不能通过先推动方块、再返回初始 pose 来通过。

| Source alias | 方块边长 | 质量 | Thumb bend 目标 | 锁存前最大平移 | 锁存前最大姿态变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `grasp_117060` | 60 mm | 160.000 g | 1.17 rad | 0.399839 mm | 0.895815° |
| `grasp_116060` | 60 mm | 160.000 g | 1.17 rad | 0.399674 mm | 0.896010° |
| `grasp_115060` | 60 mm | 160.000 g | 1.17 rad | 0.400177 mm | 0.896290° |
| `grasp_117061` | 61 mm | 168.134 g | 1.17 rad | 0.470705 mm | 0.619016° |
| `grasp_116062` | 62 mm | 176.539 g | 1.16 rad | 0.412312 mm | 0.582889° |
| `grasp_110064` | 64 mm | 194.181 g | 1.12 rad | 0.327032 mm | 0.704008° |

产物 catalog 位于
`artifacts/left_opposed_face_palm_down_pose_preserving_grasp/six_seed_dynamic_tune/trajectory_catalog/catalog.json`。
使用每个 source 一个候选的最小预算，可对六条证据种子重新执行动力学复验：

```bash
MUJOCO_GL=osmesa ./scripts/uv.sh run --frozen python \
  scripts/tune_pose_preserving_seed_campaign.py \
  --count-per-source 1 --workers 1 \
  --output-dir /tmp/xhand-pose-preserving-six-seed-count1
```

将完整六源调参目录发布为 Viewer 可读 catalog（目标目录必须尚未存在）：

```bash
./scripts/uv.sh run --frozen python -m xhand_grasp.pose_preserving_seed_catalog \
  artifacts/left_opposed_face_palm_down_pose_preserving_grasp/six_seed_dynamic_tune \
  artifacts/left_opposed_face_palm_down_pose_preserving_grasp/six_seed_dynamic_tune/trajectory_catalog \
  --expected-count 6 --best-source-id 117060
```

在可拖动自由视角中重新执行最佳轨迹的真实控制和物理：

```bash
MUJOCO_GL=glfw ./scripts/uv.sh run --frozen python grasp_cube.py view \
  --catalog artifacts/left_opposed_face_palm_down_pose_preserving_grasp/six_seed_dynamic_tune/trajectory_catalog/catalog.json \
  --trajectory best_nominal \
  --joint-monitor left_hand_thumb_bend_joint_actuator \
  --start-paused --loop
```

分别查看六条 source 时，将 `--trajectory` 设为以下任一 alias：

```text
grasp_117060
grasp_116060
grasp_115060
grasp_117061
grasp_116062
grasp_110064
```

例如，查看 62 mm source：

```bash
MUJOCO_GL=glfw ./scripts/uv.sh run --frozen python grasp_cube.py view \
  --catalog artifacts/left_opposed_face_palm_down_pose_preserving_grasp/six_seed_dynamic_tune/trajectory_catalog/catalog.json \
  --trajectory grasp_116062 \
  --joint-monitor left_hand_thumb_bend_joint_actuator \
  --start-paused --loop
```

上述六条结果只验证了“在支撑上保持物体初始 pose 并抓稳”，操作增量仍为零；
尚未验证操作轨迹或抬升成功。

### Schema v7：高拇指目标、可变尺寸与固定初始物体 Pose

v7 是独立实验
`left_opposed_face_palm_down_high_thumb_variable_size_pose_preserving_grasp_then_lift`，
不会修改上面的六条 v6 证据。它在 52–70 mm 方块、1.25–1.45 rad thumb bend
命令目标上搜索；质量固定为 160 g，摩擦固定为 0.8。每个尺寸的方块中心 XY 和
yaw 在候选生成时一次确定，底面放在支撑顶面。之后方块始终保持 freejoint，搜索只
能调整固定手根、三指 pregrasp/terminal 目标和分指闭合时序，不能 weld、mocap、
重采样或逐帧重写方块 pose。

查看完整声明预算而不创建产物：

```bash
./scripts/uv.sh run --frozen python \
  scripts/tune_high_thumb_variable_size.py --dry-run
```

执行可恢复的正式搜索：

```bash
MUJOCO_GL=osmesa ./scripts/uv.sh run --frozen python \
  scripts/tune_high_thumb_variable_size.py --workers 3
```

中断后使用完全相同的参数并增加 `--resume`；resume 会校验模板、源码、六条源轨迹、
候选配置及已有 JSON/NPZ 的 SHA-256，输入发生变化时拒绝混用旧结果。也可用
`--edges-mm`、`--thumb-targets-rad` 和各阶段预算参数建立独立诊断目录。

完整 campaign 发布为三个 Viewer 可读 catalog：主 catalog、仅抓稳 catalog 和仅
完整抬升 catalog。预算内不足 12 条时仍会写出真实结果、配额缺失项和最佳近失，
但不会创建整体成功 alias。

```bash
./scripts/uv.sh run --frozen python -m xhand_grasp.high_thumb_size_catalog \
  artifacts/left_opposed_face_palm_down_high_thumb_variable_size_pose_preserving \
  artifacts/left_opposed_face_palm_down_high_thumb_variable_size_pose_preserving/trajectory_catalog
```

只有 catalog 中存在 `highest_thumb_target` 时，下面的命令才代表已验证抓稳轨迹；
Viewer 会重新执行真实控制和 `mj_step`，而不是播放 NPZ 状态动画：

```bash
MUJOCO_GL=glfw ./scripts/uv.sh run --frozen python grasp_cube.py view \
  --catalog artifacts/left_opposed_face_palm_down_high_thumb_variable_size_pose_preserving/trajectory_catalog/grasp_catalog.json \
  --trajectory highest_thumb_target \
  --joint-monitor left_hand_thumb_bend_joint_actuator \
  --start-paused --loop
```

固定种子 `20260821` 的完整声明预算已经执行：1,000,000 个真实 collision-geom
静态闭合样本产生 11,713 个静态通过项并保留 400 条；随后完成 400 条粗动力学、
2,560 条局部动力学、36 条 `1 mm / 0.01 rad` 细化和 48 条锁定 `1 ms` 的
exact 复验。exact 硬抓稳通过数为 0，因此没有发布抓稳或抬升成功 alias，也没有把
近失冒充成功。停止原因为 `insufficient_hard_passing_diverse_grasps`。

最佳 exact 近失是 67 mm、160 g、摩擦 0.8、thumb bend 命令 1.25 rad；实际 thumb
qpos 最大值为 1.20763 rad。VERIFY 中三指目标面、等高和指腹接触均为 100%，等高
p95 为 2.923 mm，但连续全部 gate 只有 68 ms，且抓稳前方块最大平移 0.958 mm、
最大旋转 1.618°，超过 0.5 mm / 1° 硬门槛。它仅作为失败诊断，可在自由视角 Viewer
中重新执行：

```bash
MUJOCO_GL=glfw ./scripts/uv.sh run --frozen python grasp_cube.py view \
  --config artifacts/left_opposed_face_palm_down_high_thumb_variable_size_pose_preserving/trajectory_catalog/diagnostics/best_near_miss/resolved_config.json \
  --joint-monitor left_hand_thumb_bend_joint_actuator \
  --start-paused --loop
```

绑定的失败诊断 JSON、NPZ 和 MP4 位于
`artifacts/left_opposed_face_palm_down_high_thumb_variable_size_pose_preserving/rendered_best_near_miss/`；
视频已经过完整解码及 `ffprobe` 校验。

## 复现流程

本次三指验证直接使用仓库中已提交的模型产物，复现顺序如下：

1. 执行 `./scripts/bootstrap_uv.sh`。
2. 用锁文件同步：`./scripts/uv.sh sync --locked`。
3. 跑上游验收，确认 mesh、关节、执行器、触觉和接触都正常：

```bash
./scripts/uv.sh run --frozen python verify_xhand.py --side left
```

4. 跑本分支测试：

```bash
./scripts/uv.sh run --frozen pytest -q
```

5. 运行已固化的 62 mm 等密度成功配置并查看实际硬判据；成功时退出码为 `0`：

```bash
./scripts/uv.sh run --frozen python grasp_cube.py run \
  --config grasp_configs/left_opposed_face_palm_down_larger_cube_relative_pose_rescue_validated.json \
  --output-dir artifacts/left_opposed_face_palm_down_larger_cube_relative_pose_rescue/reproduction \
  --video --video-filename nominal.mp4
```

`convert_xhand.py` 依赖仓库外的源资料并会重写模型产物，因此不属于本验证流程，
本次没有运行它。只有在明确需要重新转换、且源资料齐全时才应单独执行。

## 目录结构

```text
xhand1/
├── assets/left/             # 左手 30 个 STL
├── assets/right/            # 右手 30 个 STL
├── source_urdf/             # 从 zip 提取的只读源副本
├── tactile/                 # 标准化后的 T16/T30 transformed JSON
├── xhand_left.xml           # 左手可复用 MJCF 本体
├── xhand_right.xml          # 右手可复用 MJCF 本体
├── scene_left.xml           # 左手捏取场景
├── scene_right.xml          # 右手捏取场景
├── grasp_configs/           # 版本化三指任务配置
├── grasp_cube.py            # tune/run/robustness 单一 CLI
├── xhand_grasp/             # 模块化场景、仿真、判据、搜索和产物实现
├── pyproject.toml           # uv 项目与精确主依赖
├── uv.lock                  # 完整依赖和发行文件哈希
├── scripts/uv.sh            # 隔离环境包装器
├── scripts/bootstrap_uv.sh  # 固定 uv 下载与校验
├── tests/                   # 三指任务单元/动力学回归
├── convert_xhand.py         # 可重复执行的转换器
├── xhand_tactile.py         # 法向 touch + 三轴接触力
├── demo_control.py          # GUI/无界面控制 Demo
├── verify_xhand.py          # 自动验收
└── conversion_manifest.json # 源哈希、单位、关节顺序和数量
```

MuJoCo 的标准做法是在 `<asset><mesh .../></asset>` 中引用 STL/OBJ，不是把三角形顶点嵌入 XML。因此 XML 和 `assets/<side>/` 必须一起保留。本交付的 STL 本身已是米制，没有错误地再乘 `0.001`。

## 转换逻辑

`convert_xhand.py` 只使用 Python 标准库。它不属于三指验证；若另行确认要重建模型，
也必须通过相同 uv 包装器执行：

```bash
./scripts/uv.sh run --frozen python convert_xhand.py
```

它会：

1. 从 `Xhand-urdf.zip` 提取左右手 URDF 与每侧 30 个 STL。
2. 保留全部 `xyz/rpy`、轴、质量、惯量、限位和颜色。
3. 忽略 17 个无自由度装饰 link 中 SolidWorks 导出的退化 `1e-11` 惯量；它们是原 URDF 直接导入 MuJoCo 报 `inertia must have positive eigenvalues` 的根因。12 个动态 link 的正式惯量保留。
4. 为相邻刚体簇生成 self-contact exclude，消除关节外壳的初始穿透，但保留非相邻指节和指间接触。
5. 按 URDF effort 加入 12 个限力 position actuator：近端最大 `1.1 N·m`，远端/侧摆最大 `0.4 N·m`。控制程序只改 `ctrl`，不在每帧瞬移 `qpos`。
6. 将 transformed JSON 的 mm 坐标除以 1000，直接挂载到各指 `link2` 局部坐标系。

脚本是增量写入，不会删除源文件或清空目录。

## 触觉数据

每侧有 5 指 × 120 点：

- 拇指使用左/右手各自的 T30 transformed JSON。
- 食指、中指、无名指、小指共用 T16 transformed JSON。
- XML 中 600 个 `<touch>` 传感器的数据位于 `data.sensordata`，每点是法向力标量。

```python
from xhand_tactile import TactileReader

reader = TactileReader(model, data, "right")
normal = reader.normal_forces()          # (5, 120), N
force_link = reader.taxel_forces_link() # (5, 120, 3), link2 坐标系, N
force_sensor = reader.taxel_forces_sensor()  # (5, 120, 3), 交付传感器轴, N
```

MuJoCo 原生 `<touch>` 只输出法向标量，而实物 XHAND1 是三轴阵列。`xhand_tactile.py` 会把 `mj_contactForce` 从 contact frame 正确旋转到 world，根据 `geom1/geom2` 修正符号，再旋转到 `link2`/传感器坐标系并归属给最近 taxel。

site 的捕获半径为 `2.5 mm`，用于覆盖 STL 凸包接触点与硅胶测点之间约 `1.8–2.2 mm` 的偏差。相邻 site 可能重叠，所以不要把 600 路原生 touch 直接求和当作总力；总三轴力应使用 `taxel_forces_link().sum(axis=1)`。

## 在其他 MuJoCo 工程中使用

独立场景可直接包含本体：

```xml
<mujoco model="my_scene">
  <include file="xhand_right.xml"/>
  <!-- 地面、物体、灯光等 -->
</mujoco>
```

`xhand_right.xml`/`xhand_left.xml` 的手根默认固定到 world。若要装到机械臂，需把对应 XML 中的根 `<body name="*_hand_link">` 子树挂到机械臂末端 body，并保留它的 `<asset>`、`<contact>`、`<actuator>` 和 `<sensor>` 段。不要给手内 12 个关节增加 freejoint。

## 已验证的验收项

`verify_xhand.py` 对左右手都检查：

- 30 个 mesh、12 个关节、12 个 actuator、600 个 site/600 路 sensor。
- 模型 compile、`mj_forward`、1.5 s 握拳动力学和关节限位。
- 张开位零自穿透。
- 拇指+食指对球的真实接触与非零 touch。
- 五根手指分别用 mocap 探针激活，原生法向力与重建三轴力都必须非零。
- `qpos`/`qvel` 全部有限，无 NaN/Inf。

MuJoCo 3.10.0 下的当前验收结果为 `ALL CHECKS PASSED`。

## 工程注意

当前 collision 使用正式 URDF 的高面数 STL，MuJoCo 会对 mesh 做凸包接触。这适合模型验收、可视化与一般操作。如果后续要开数千并行环境做 RL，建议另外生成凸分解/基元 collision，但不要替换当前可视 STL。

## Schema v8：法向对齐抓取与平滑近竖直抬升

实验 `left_opposed_face_palm_down_high_thumb_normal_aligned_smooth_vertical_lift`
先在已有高质量 pose 上重调闭合控制，不足时再搜索新 pose，随后用真实物体响应
Jacobian 和五次 minimum-jerk 轨迹搜索三指抬升。模板位于
`grasp_configs/left_opposed_face_palm_down_high_thumb_normal_aligned_smooth_vertical_lift.json`；
默认覆盖 60–70 mm、160 g、摩擦系数 0.8 及拇指 bend 目标 1.25–1.45 rad。

主要硬判据包括：三指闭合方向均向内，静态夹角不超过 45°、CLOSE 阶段每指
p95 不超过 30°；抓稳前方块保持在 0.5 mm / 1° 内，三指 distal 指腹接触连续
通过 250 ms，且无掌部、非末端或 inactive finger 有效接触。操作结果的最后窗口
中位/最低抬升须达到 10/8 mm，水平位移不超过 2 mm、姿态漂移不超过 10°，
同时满足 0.2 mm 累计回退、0.020 m/s 峰值上升速度、0.12 m/s² 加速度、
2.5 m/s³ jerk 和 0.005 m/s HOLD 入口速度限制。

全量调参必须在冻结 uv 环境中运行。可以分两阶段执行（`--workers` 可按 CPU
资源调整）：

```bash
MUJOCO_GL=osmesa ./scripts/uv.sh run --frozen python \
  scripts/tune_normal_aligned_smooth_lift.py \
  --stage rescue --workers 12

MUJOCO_GL=osmesa ./scripts/uv.sh run --frozen python \
  scripts/tune_normal_aligned_smooth_lift.py \
  --stage lift --workers 12 \
  --rescue-report artifacts/left_opposed_face_palm_down_high_thumb_normal_aligned_smooth_vertical_lift/tune/rescue/search_report.json
```

也可以在尚未生成默认调参目录时一次完成两个阶段：

```bash
MUJOCO_GL=osmesa ./scripts/uv.sh run --frozen python \
  scripts/tune_normal_aligned_smooth_lift.py \
  --stage all --workers 12
```

中断后使用相同命令并追加 `--resume`；不要同时运行分阶段命令和 `all`。搜索结束后，
将硬通过轨迹及最佳近失用同一次 MuJoCo 重跑导出为 JSON、NPZ 和 MP4：

```bash
MUJOCO_GL=osmesa ./scripts/uv.sh run --frozen python \
  scripts/export_normal_aligned_smooth_lift_catalog.py \
  artifacts/left_opposed_face_palm_down_high_thumb_normal_aligned_smooth_vertical_lift/tune/lift \
  --output-dir artifacts/left_opposed_face_palm_down_high_thumb_normal_aligned_smooth_vertical_lift/trajectory_catalog
```

仅当 lift 硬通过时，以下固定配置才会存在并可用 `run` 复验：

```bash
MUJOCO_GL=osmesa ./scripts/uv.sh run --frozen python grasp_cube.py run \
  --config artifacts/left_opposed_face_palm_down_high_thumb_normal_aligned_smooth_vertical_lift/tune/lift/best_nominal_config.json \
  --output-dir artifacts/left_opposed_face_palm_down_high_thumb_normal_aligned_smooth_vertical_lift/nominal \
  --video --video-filename nominal.mp4
```

固定种子 `20260821` 的当前正式运行完成了 630/630 个候选（51 probe、192 trust、
384 refine、3 exact）。三个 exact 候选均先抓稳，但没有完整通过抬升硬判据，因此
当前 catalog 只有 `best_attempt`，没有 `best_nominal`。最佳 63 mm 候选的最后窗口
中位/最低抬升为 11.005/10.934 mm；主要失败项是 7.402 mm 水平位移、
18.574 m/s³ jerk，以及食指/中指操作目标面占空比 34.0%/28.4%。

通过 passive Viewer 查看该 exact 近失的同一次真实物理仿真，并自由拖动视角：

```bash
MUJOCO_GL=glfw ./scripts/uv.sh run --frozen python grasp_cube.py view \
  --catalog artifacts/left_opposed_face_palm_down_high_thumb_normal_aligned_smooth_vertical_lift/trajectory_catalog/catalog.json \
  --trajectory best_attempt \
  --joint-monitor left_hand_thumb_bend_joint_actuator \
  --start-paused --loop
```

窗口内左键拖动旋转、右键拖动平移、滚轮缩放；Space 暂停/继续，R 重启，L
切换循环。若后续搜索得到硬通过项，catalog 才会额外提供 `best_nominal`。没有硬
通过项时，调参不会创建成功别名，只会生成
`best_attempt_config.json`；复跑该近失仍应退出 `2`，不得表述为成功。导出器只有在
5 条成功轨迹同时覆盖至少 3 个尺寸和 2 个拇指目标分带时才退出 `0`，否则保留
实际结果并退出 `2`。

本实验的调参、发布产物分别位于
`artifacts/left_opposed_face_palm_down_high_thumb_normal_aligned_smooth_vertical_lift/tune/{rescue,lift}/`
和
`artifacts/left_opposed_face_palm_down_high_thumb_normal_aligned_smooth_vertical_lift/trajectory_catalog/`，
不会覆盖 v1–v7 的配置或产物。

对全部 exact 候选执行每条 16 次局部扰动，并对最佳候选额外执行 50 次注册的
pose/物性扰动：

```bash
MUJOCO_GL=osmesa ./scripts/uv.sh run --frozen python \
  scripts/run_normal_aligned_smooth_lift_robustness.py \
  artifacts/left_opposed_face_palm_down_high_thumb_normal_aligned_smooth_vertical_lift/tune/lift \
  --output artifacts/left_opposed_face_palm_down_high_thumb_normal_aligned_smooth_vertical_lift/robustness/perturbation_report.json \
  --workers 18 --seed 20260821
```

当前 3×16+50 共 98 次扰动均未完整通过；三条 exact 分别为 0/16，最佳 63 mm
候选为 0/50（要求 45/50）。由于名义 exact 本身未通过，报告固定标记为诊断结果，
不能创建成功别名或提升为鲁棒成功。

## Schema v9：实际接触 Grasp Pose 与平滑操纵

实验 `left_opposed_face_palm_down_actual_contact_grasp_pose_smooth_vertical_lift`
把“接触时的实际手形”与 actuator 命令彻底分开。配置中的
`grasp_pose.nominal_joint_qpos_rad` 表示待验证的八关节接触姿态；抓取是否成立只由
连续 250 ms 的真实 `qpos`、接触、物体 pose 和安全 gate 决定。
`contact_preload_targets_rad` 只用于维持抓力，即使把命令调到 1.5 rad，也不能替代
拇指实际 bend 全窗口位于 1.40–1.60 rad 的证据。最终候选还会把窗口中位 qpos
回写为实测 grasp pose，并从最初无接触状态重新抓取后才发布。

首次可恢复搜索（输出目录必须尚不存在）：

```bash
MUJOCO_GL=osmesa ./scripts/uv.sh run --frozen python grasp_cube.py tune \
  --config grasp_configs/left_opposed_face_palm_down_actual_contact_grasp_pose_smooth_vertical_lift.json \
  --output-dir artifacts/left_opposed_face_palm_down_actual_contact_grasp_pose_smooth_vertical_lift/tune/campaign \
  --target-success-count 1 --workers 12 \
  --evidence-grasp-anchor artifacts/left_opposed_face_palm_down_actual_contact_grasp_pose_smooth_vertical_lift/tune/dynamic_first_grasp/candidates/candidate_1616000576132672
```

找到首条完整轨迹后，可在同一套哈希绑定的候选流上扩展到五条；模型、锁文件、
配置、源码或 evidence anchor 有任何变化时，resume 会拒绝复用：

```bash
MUJOCO_GL=osmesa ./scripts/uv.sh run --frozen python grasp_cube.py tune \
  --config grasp_configs/left_opposed_face_palm_down_actual_contact_grasp_pose_smooth_vertical_lift.json \
  --output-dir artifacts/left_opposed_face_palm_down_actual_contact_grasp_pose_smooth_vertical_lift/tune/campaign \
  --resume --target-success-count 5 --workers 12 \
  --evidence-grasp-anchor artifacts/left_opposed_face_palm_down_actual_contact_grasp_pose_smooth_vertical_lift/tune/dynamic_first_grasp/candidates/candidate_1616000576132672
```

Viewer 会重新执行真实控制和每一个 `mj_step`。以下命令在 250 ms grasp window
锁存时自动暂停；此时终端同时显示窗口中位 qpos、锁存瞬时 qpos、nominal、preload
及误差。窗口内可用左键旋转、右键平移和滚轮缩放视角，Space 继续，R 重启，L 循环：

```bash
MUJOCO_GL=glfw ./scripts/uv.sh run --frozen python grasp_cube.py view \
  --catalog artifacts/left_opposed_face_palm_down_actual_contact_grasp_pose_smooth_vertical_lift/tune/campaign/catalogs/target_1/grasp_pose/catalog.json \
  --trajectory best_first \
  --pause-at-event grasp_lock \
  --joint-monitor left_hand_thumb_bend_joint_actuator \
  --loop
```

只有完整操纵硬通过轨迹才进入正式 robustness。每条最多执行 16 次局部扰动，
`best_first` 另执行 50 次 pose/物性扰动，至少 45/50 才标记为鲁棒：

```bash
MUJOCO_GL=osmesa ./scripts/uv.sh run --frozen python grasp_cube.py robustness \
  --config grasp_configs/left_opposed_face_palm_down_actual_contact_grasp_pose_smooth_vertical_lift.json \
  --search-root artifacts/left_opposed_face_palm_down_actual_contact_grasp_pose_smooth_vertical_lift/tune/campaign/catalogs/target_1/manipulation/catalog.json \
  --output artifacts/left_opposed_face_palm_down_actual_contact_grasp_pose_smooth_vertical_lift/robustness/perturbation_report.json \
  --workers 12 --seed 20260821
```

固定种子 `20260821` 的正式 campaign 已完成 550,000 个静态样本、782 个动态
抓取候选和 2,176 个完整操纵候选（18×64 trust + 8×128 局部精调）。经实测
250 ms qpos 窗口重新固化后共有 18 条 grasp pose 通过；抓取 catalog 的
`best_first` 是 67 mm、160 g、摩擦 0.8，拇指实际 bend 为
1.44975/1.45031/1.45056 rad（min/median/max），抓稳前方块平移 0.184 mm、
旋转 0.884°。

操纵阶段没有完整硬通过项，因此 manipulation catalog 只有 `best_attempt`。该 64 mm
候选的末窗最低/中位抬升为 8.964/9.028 mm，低于中位 10 mm 阈值；操作阶段
thumb/index/mid 目标面占空比分别为 35.1%/22.8%/20.2%，同时拓扑仅 20.2%，
滤波 jerk 为 21.374 m/s³，故不能提升为成功项，也不执行 robustness。可查看它的
真实物理重跑：

```bash
MUJOCO_GL=glfw ./scripts/uv.sh run --frozen python grasp_cube.py view \
  --catalog artifacts/left_opposed_face_palm_down_actual_contact_grasp_pose_smooth_vertical_lift/tune/campaign/catalogs/target_1/manipulation/catalog.json \
  --trajectory best_attempt \
  --pause-at-event grasp_lock \
  --joint-monitor left_hand_thumb_bend_joint_actuator \
  --loop
```

一次旧版 resume 暴露了失败 trace 的重复压缩缺陷；发布器、动态压缩和 committed-stage
恢复现均有回归保护。最终 18 条抓取、全部操纵 summary、24 条保留操纵 trace、
JSON/NPZ/MP4 和 catalog 哈希均已复验。另有 64 条低排名且未晋级的 dynamic 抓取
失败项，其已删除 NPZ 的历史 SHA-256 被旧 resume 覆盖且无法恢复，已在
`tune/campaign/recovery/dynamic_trace_digest_loss_audit.json` 中隔离登记；这些记录未被
用于 measured grasp、操纵搜索或最终 catalog。由于修复后源码哈希不同，这个历史
campaign 已封存；若要继续到 target 5，应使用当前代码新建输出目录，而非修改旧
manifest。

## Schema v10：选择不同的 measured Grasp 数据

72–90 mm 固定 160 g campaign 的正式 grasp catalog 只收录最终遴选项；
`dynamic/measured/` 则保留了所有通过实际接触窗口的 grasp 数据。可先列出
可选项，命令只读取并校验数据，不会打开 Viewer：

```bash
MUJOCO_GL=glfw ./scripts/uv.sh run --frozen python grasp_cube.py view \
  --measured-root artifacts/left_opposed_face_palm_down_larger_actual_contact_grasp_pose_smooth_vertical_lift/tune/campaign/dynamic/measured \
  --list-data
```

列表中的 `INDEX` 从 1 开始。按索引查看一条数据：

```bash
MUJOCO_GL=glfw ./scripts/uv.sh run --frozen python grasp_cube.py view \
  --measured-root artifacts/left_opposed_face_palm_down_larger_actual_contact_grasp_pose_smooth_vertical_lift/tune/campaign/dynamic/measured \
  --data-index 1 \
  --pause-at-event grasp_lock \
  --joint-monitor left_hand_thumb_bend_joint_actuator \
  --loop
```

也可以使用列表中的精确 candidate ID：

```bash
MUJOCO_GL=glfw ./scripts/uv.sh run --frozen python grasp_cube.py view \
  --measured-root artifacts/left_opposed_face_palm_down_larger_actual_contact_grasp_pose_smooth_vertical_lift/tune/campaign/dynamic/measured \
  --candidate-id 1616000032129682 \
  --pause-at-event grasp_lock \
  --joint-monitor left_hand_thumb_bend_joint_actuator \
  --loop
```

同一尺寸通常有多条数据；`--edge-rank` 同样从 1 开始，表示该尺寸
在列表稳定排序中的第几条：

```bash
MUJOCO_GL=glfw ./scripts/uv.sh run --frozen python grasp_cube.py view \
  --measured-root artifacts/left_opposed_face_palm_down_larger_actual_contact_grasp_pose_smooth_vertical_lift/tune/campaign/dynamic/measured \
  --select-edge-mm 82 --edge-rank 2 \
  --pause-at-event grasp_lock \
  --joint-monitor left_hand_thumb_bend_joint_actuator \
  --loop
```

`--data-index`、`--candidate-id` 和 `--select-edge-mm` 互斥。尺寸存在多个匹配时
必须显式给出 `--edge-rank`，避免静默选到另一条数据。旧的
`--catalog ... --trajectory ...` 和 `--config ...` 命令保持不变。

## Schema v13：89→60 mm 接触关系缩放实验

正式固定质量消融 campaign 位于：

```text
artifacts/left_opposed_face_palm_down_scaled_centered_spread_actual_grasp_then_lift/tune/campaign_yaw_bounded_v2
```

该 campaign 已完整扫描 88–60 mm，不因中间失败提前终止。方块固定为 160 g、
摩擦 0.8，并同时测试 `proportional_face_yz` 和 `absolute_face_yz` 两种接触点映射。
最终有 22 条 measured grasp pose 严格通过；Viewer catalog 按每个尺寸最多三条发布
18 条，覆盖 70、75、76、77、78、79、80、81 和 84 mm。最小抓稳尺寸为 70 mm。

查看总体排名最高的抓稳轨迹：

```bash
MUJOCO_GL=glfw ./scripts/uv.sh run --frozen python grasp_cube.py view \
  --catalog artifacts/left_opposed_face_palm_down_scaled_centered_spread_actual_grasp_then_lift/tune/campaign_yaw_bounded_v2/catalogs/target_5/grasp_pose/catalog.json \
  --trajectory best_nominal \
  --pause-at-event grasp_lock \
  --joint-monitor left_hand_thumb_bend_joint_actuator \
  --show-coordinate-frames \
  --loop
```

查看最小抓稳尺寸 70 mm：

```bash
MUJOCO_GL=glfw ./scripts/uv.sh run --frozen python grasp_cube.py view \
  --catalog artifacts/left_opposed_face_palm_down_scaled_centered_spread_actual_grasp_then_lift/tune/campaign_yaw_bounded_v2/catalogs/target_5/grasp_pose/catalog.json \
  --trajectory smallest_grasp_pass \
  --pause-at-event grasp_lock \
  --joint-monitor left_hand_thumb_bend_joint_actuator \
  --show-coordinate-frames \
  --loop
```

也可以把 `--trajectory` 改为 `edge_75_best`、`edge_76_best`、`edge_77_best`、
`edge_78_best`、`edge_79_best`、`edge_80_best`、`edge_81_best` 或 `edge_84_best`，
直接选择相应尺寸。`grasp_pose_1` 至 `grasp_pose_18` 可逐条查看全部发布数据。

2,080 条操纵候选中没有一条满足全部平滑抬升硬条件，因此没有
`smallest_lift_pass`。最佳近失是 81 mm：末窗中位/最低抬升为
9.281/9.205 mm，但食指在操作阶段目标面接触占空比只有 10.57%，同时还超过
2 mm 横移、10° 姿态漂移和 2.5 m/s³ jerk 阈值。它只标记为诊断项：

```bash
MUJOCO_GL=glfw ./scripts/uv.sh run --frozen python grasp_cube.py view \
  --catalog artifacts/left_opposed_face_palm_down_scaled_centered_spread_actual_grasp_then_lift/tune/campaign_yaw_bounded_v2/catalogs/target_5/manipulation/catalog.json \
  --trajectory best_attempt \
  --pause-at-event grasp_lock \
  --joint-monitor left_hand_thumb_bend_joint_actuator \
  --show-coordinate-frames \
  --loop
```

离线 MP4、对应 JSON 和逐元素一致的 NPZ 位于：

```text
artifacts/left_opposed_face_palm_down_scaled_centered_spread_actual_grasp_then_lift/tune/campaign_yaw_bounded_v2/diagnostics/best_manipulation_attempt/
```

18 条发布抓取各执行 16 次局部扰动，最优项另执行 50 次扰动；338 次均因
严格的 2 mm 接触点连续窗口丢失而失败。因此本轮结论仅为
`validated_fixed_160g_scaled_grasp_ablation` 的名义抓稳通过，不是鲁棒抓稳，
也不是完整抬升通过。

## Schema v14：三指接触保持规划抬升

v14 使用 20 节点、3 s 的接触约束前馈计划，并叠加逐指法向力 PI 反馈。
控制器严格按“上一帧观测 → 当前命令 → `mj_step` → 新观测”执行；失力时冻结
计划进度并只沿该指已验证的内收方向增加预载。可选的
`contact_feedback schema_version=2` 还会使用上一帧方块局部切向滑移进行冻结和
安全终止，schema 1 及所有旧实验保持原数值路径。物体、实际抓取、手物配对、
规划器和控制器的五个顶层 ID 均写入 NPZ 并在评估时重新核对。

最终源码上的正式自适应复验位于：

```text
artifacts/left_opposed_face_palm_down_contact_preserving_planned_lift/tune/formal_campaign_v14_1_adaptive_pose_followup_v2
```

该轮包含 64 个单轴 probe 和 64 个安全多维组合，共 128 个物理唯一候选。
最佳固定质量配置为 79 mm、160 g、摩擦 0.8；三指和三指同时目标面接触占空比
均为 100%，最长失联为 0 ms，中位/最低抬升为 10.875/10.818 mm，水平位移
0.524 mm，姿态漂移 3.657°。120 项验收中 119 项通过，唯一失败是滤波后峰值
jerk `2.9656 m/s³`，高于 `2.5 m/s³` 门槛。因此它只发布为 `best_attempt`，
没有 `best_first`、`best_nominal` 或 `best_robust`，也没有启动鲁棒性 sidecar。

使用真实 MuJoCo 物理重跑并在抓取锁存处暂停：

```bash
MUJOCO_GL=glfw ./scripts/uv.sh run --frozen python grasp_cube.py view \
  --catalog artifacts/left_opposed_face_palm_down_contact_preserving_planned_lift/tune/formal_campaign_v14_1_adaptive_pose_followup_v2/catalogs/target_1/manipulation/catalog.json \
  --trajectory best_attempt \
  --pause-at-event grasp_lock \
  --joint-monitor left_hand_thumb_bend_joint_actuator \
  --show-coordinate-frames \
  --loop
```

该结果只能标记为固定 160 g 的最佳近失消融配置，不能表述为完整通过或鲁棒通过。

### 食指/中指主屈曲关节基线对齐微调

在 79 mm 固定质量候选上，补充搜索将
`left_hand_index_joint1` 到 `left_hand_mid_joint1` 的连线进一步对齐方块局部
`+Y`。最终严格抓稳窗口的夹角 p95 从 `10.051°` 降至 `7.726°`，操作阶段
p95 为 `7.682°`；三指同时有效接触占空比为 `99.532%`，无活动指自碰撞，
中位/最低抬升为 `10.337/10.289 mm`。该结果仍因 jerk 和 HOLD 入口速度失败，
因此 catalog 别名是 `best_aligned_grasp`/`best_attempt`，不是完整操纵成功。

Viewer 中同时显示两个关节、连线和方块坐标系：

```bash
MUJOCO_GL=glfw ./scripts/uv.sh run --frozen python grasp_cube.py view \
  --catalog artifacts/left_opposed_face_palm_down_contact_preserving_planned_lift/tune/index_middle_joint1_y_alignment_refinement_v1/catalogs/target_1/manipulation/catalog.json \
  --trajectory best_aligned_grasp \
  --pause-at-event grasp_lock \
  --show-joint-pair left_hand_index_joint1 left_hand_mid_joint1 \
  --show-coordinate-frames \
  --loop
```

该 Viewer 会重新运行同一套 MuJoCo 控制和物理；保存的 163 个 trace 字段已与
独立重跑逐数组完全一致。按 `J` 可切换关节对标记，按 `F` 可切换坐标系。

### 食指/中指主屈曲关节近零夹角抓取

在不移动方块初始世界 pose、尺寸（79 mm）、质量（160 g）和摩擦（0.8）的
前提下，独立实验
`left_opposed_face_palm_down_joint_pair_aligned_contact_preserving_planned_lift`
联合微调了固定手根的 6D pose、八个活动关节的实际接触 pose、precontact 和
preload。最终候选 `15941124607131459` 在连续 250 ms 抓稳窗口中，
`left_hand_index_joint1` 到 `left_hand_mid_joint1` 的连线与方块局部 `+Y` 轴夹角
p50/p95/max 分别为 `0.1205/0.1428/0.1453°`，相对源候选的 p95
`7.7264°` 改善 `7.5837°`。

该候选严格抓稳通过：抓稳前方块仅平移 `0.0633 mm`、旋转 `0.4038°`，
SETTLE 无手物接触，三个目标接触点在 2 mm 区域内的占空比均为 100%，
6750 帧内没有活动手指自碰撞。三指 CLOSE 阶段法向内收方向 p95 均低于
20°。这是一条抓取成功轨迹；沿用旧 manipulation plan 后，操作阶段夹角 p95
增至 `4.5540°`，且食指连续失联 11 ms（门槛 10 ms）后安全 ABORT，因此不能
标记为完整抬升成功。

使用真实 MuJoCo 物理重跑并在抓取锁存帧暂停：

```bash
MUJOCO_GL=glfw ./scripts/uv.sh run --frozen python grasp_cube.py view \
  --catalog artifacts/left_opposed_face_palm_down_joint_pair_aligned_contact_preserving_planned_lift/tune/index_middle_joint1_near_zero_alignment_refinement_v2/catalogs/target_1/manipulation/catalog.json \
  --trajectory best_near_zero_grasp \
  --pause-at-event grasp_lock \
  --show-joint-pair left_hand_index_joint1 left_hand_mid_joint1 \
  --show-coordinate-frames \
  --loop
```

暂停后可拖动鼠标检查视角；按 `J` 切换关节/连线，按 `F` 切换坐标系，按
Space 继续仿真。

## 固定 Grasp Pose 的 MuJoCo 接触环境复测

针对 schema-v15 candidate `15204225421299876`，接触环境消融保持方块、手根
pose、八关节实际 grasp pose、控制器和 21 节点计划完全不变，只改变经过版本化
和哈希认证的 MuJoCo 接触/求解参数。正式 12-case 结果位于：

```text
artifacts/left_opposed_face_palm_down_joint_pair_near_zero_contact_preserving_planned_lift/tune/contact_environment_ablation_from_15204225421299876_v1/
```

原始环境已经是 Newton、elliptic cone、1 ms、`impratio=10`。基线重跑的 199
个旧 trace 字段与源证据逐元素完全一致；新 trace 只增加环境 ID 和接触数量审计
字段。拇指 1.841 mm 指标主要来自多点碰撞流形/力加权接触质心切换，而不是拇指
摩擦锥饱和。

`best_environment` 将 torsional friction 从 0.005 改为 0.0025，拇指最大质心
位移降至 0.713 mm，但这是材料消融，并在计划进度 19.07% 时因关节连线连续超限
终止，只抬升 0.660 mm。`best_same_object` 保持原接触材料并设置
`noslip_iterations=20`，拇指位移仅降至 1.831 mm，计划进度/中位抬升反而降至
38.87%/3.853 mm。因此没有环境 case 同时满足 1.5 mm 滑移门槛和完整操纵验收。

重新运行低滑移材料消融结果（catalog 会自动认证并加载环境 sidecar）：

```bash
MUJOCO_GL=glfw ./scripts/uv.sh run --frozen python grasp_cube.py view \
  --catalog artifacts/left_opposed_face_palm_down_joint_pair_near_zero_contact_preserving_planned_lift/tune/contact_environment_ablation_from_15204225421299876_v1/catalog.json \
  --trajectory best_environment \
  --pause-at-event grasp_lock \
  --show-joint-pair left_hand_index_joint1 left_hand_mid_joint1 \
  --show-coordinate-frames \
  --loop
```

将 `--trajectory` 改为 `best_same_object` 可查看 NoSlip=20 的同材料结果，改为
`baseline` 可查看逐元素复现的原始环境。也可用
`--contact-environment ENVIRONMENT_JSON` 对任意 resolved config 显式应用一个
严格版本化 sidecar；该运行会重新验收，不继承 catalog 成功状态。
