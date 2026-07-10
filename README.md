# XHAND1 MuJoCo 模型（左/右手，5×120 触觉）

这个目录是从 `Xhand1交付资料-带触觉` 非破坏性生成的 MuJoCo 工程。源压缩包、PDF、JSON 和 SDK 都没有被修改或删除。

最终模型以正式 URDF v1.3 为结构与几何源，而不是交付包中 2025-02 的右手触觉初版。

## 环境准备

这个仓库不是一个独立的 Python 项目，它默认依赖父目录 `mujoco` 里的运行环境。

推荐的环境是：

- Linux
- MuJoCo Python 包
- `numpy`
- 父目录 `mujoco` 中的 conda 环境 `mujoco-examples`

父目录已经提供了激活脚本，脚本会读取当前工作目录，所以要先进入 `mujoco` 根目录再激活：

```bash
cd /home/jingchen/projects/mujoco
source ./activate_mujoco.sh
cd xhand1
```

如果你是第一次在这台机器上装 GitHub CLI，可先执行：

```bash
sudo apt install gh
gh auth login
```

如果只想验证模型，不需要额外安装这个仓库自己的依赖；只要父环境里的 `mujoco` 和 `numpy` 可用即可。

## 快速运行

环境激活脚本依赖当前目录，所以要从 `mujoco` 根目录执行：

```bash
cd /home/jingchen/projects/mujoco
source ./activate_mujoco.sh
cd xhand1
```

先运行全量无界面验收：

```bash
python verify_xhand.py
```

打开右手交互场景：

```bash
python demo_control.py --side right
```

左手：

```bash
python demo_control.py --side left
```

交互键：

- `O`：张开
- `P`：捏取橙色球（推荐首次测试）
- `C`：握拳
- `R`：重置手和球
- `T`：显示/隐藏 600 个 taxel site
- `F`：显示/隐藏接触点和接触力

MuJoCo 右侧 `Control` 面板也可直接拖动 12 个 actuator。如果只想用原生 viewer：

```bash
python -m mujoco.viewer --mjcf scene_right.xml
```

无界面运行捏取和触觉输出：

```bash
python demo_control.py --side right --pose pinch --headless --seconds 2.5
```

## 复现流程

如果你想从原始交付资料重新生成本仓库的所有 MuJoCo 产物，按下面顺序执行：

1. 把原始交付包 `Xhand1交付资料-带触觉` 放在 `mujoco` 根目录同级，保持默认目录结构不变。
2. 激活父目录环境：

```bash
cd /home/jingchen/projects/mujoco
source ./activate_mujoco.sh
cd xhand1
```

3. 重新生成 XML、触觉 JSON 和 manifest：

```bash
python convert_xhand.py
```

4. 运行自动验收，确认 mesh、关节、执行器、触觉和接触都正常：

```bash
python verify_xhand.py
```

5. 打开交互场景做人工检查：

```bash
python demo_control.py --side right
python demo_control.py --side left
```

如果要无界面复现捏取和触觉输出，可以加：

```bash
python demo_control.py --side right --pose pinch --headless --seconds 2.5
```

当前仓库里已经包含了转换后的结果文件，所以一般情况下你只需要激活环境并运行 `verify_xhand.py` / `demo_control.py`，不必重新转换。

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
├── convert_xhand.py         # 可重复执行的转换器
├── xhand_tactile.py         # 法向 touch + 三轴接触力
├── demo_control.py          # GUI/无界面控制 Demo
├── verify_xhand.py          # 自动验收
└── conversion_manifest.json # 源哈希、单位、关节顺序和数量
```

MuJoCo 的标准做法是在 `<asset><mesh .../></asset>` 中引用 STL/OBJ，不是把三角形顶点嵌入 XML。因此 XML 和 `assets/<side>/` 必须一起保留。本交付的 STL 本身已是米制，没有错误地再乘 `0.001`。

## 转换逻辑

`convert_xhand.py` 只使用 Python 标准库，可重复执行：

```bash
python3 convert_xhand.py
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
