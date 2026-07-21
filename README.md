# 智能手机步态分析与运动轨迹追踪系统

> 基于智能手机传感器的端到端步态分析系统：采集 IMU（加速度计 + 陀螺仪）与 GNSS（GPS）数据，经 WebSocket 实时传输到后端，利用机器学习模型识别运动类型（走路 / 跑步 / 跳跃 / 高抬腿），并在电脑端网页实时展示运动状态、传感器波形与运动轨迹。

> ⚠️ 本文档为最新版。旧 `README.txt` 为早期版本，其中"实时逐帧判定""旧目录结构"等内容已过时，请以此文件为准。

---

## 一、项目介绍

### 1.1 项目目标
构建一个**低成本、便携式**的运动识别与轨迹追踪系统，覆盖"数据采集 → 预处理/特征工程 → 步态分类模型 → 数据可视化 → 实时 Web 应用"的完整闭环，为个人运动指导与科研分析提供支持。

### 1.2 支持的运动类型
| 英文标签（文件夹/模型内部） | 中文名称 |
| :--- | :--- |
| `walking` | 走路 |
| `running` | 跑步 |
| `jumping` | 跳跃 |
| `high_knees` | 高抬腿 |

### 1.3 判定模式（重要）
当前采用 **"停止后判定整段"** 模式，而非逐帧实时判定：
- 手机点 **「开始发送」** → 后端开始累积整段传感器数据（不判定）；
- 手机点 **「停止发送」** → 后端用这段时间的数据（就近窗口上限 30s，最短需 ≥10s）一次性提取特征并预测，将结果推送给电脑端；
- 电脑端在停止后弹出**最终判定卡片**（运动类型 / 置信度 / 时长 / 帧数）。

> 原因：单次数据包（几十毫秒）数据量太小，不足以稳定提取步态特征，硬判会大量误判。因此改为"采集一段 → 停止 → 整段判定"，结果更可靠。

### 1.4 技术栈
| 类别 | 技术 |
| :--- | :--- |
| 后端 | Python · FastAPI · Uvicorn · WebSocket |
| 机器学习 | NumPy · SciPy · scikit-learn（RandomForest）· Pandas |
| 可视化 | Matplotlib · Seaborn（离线混淆矩阵）；ECharts 5（前端波形）；高德地图 JS API 2.0（轨迹） |
| 前端 | HTML5 · CSS3（Tailwind via CDN）· JavaScript |

---

## 二、项目结构与各代码功能

### 2.1 目录结构（实际）
```
sports_recog/
├── src/
│   ├── backend/
│   │   ├── main.py              # FastAPI 服务 + WebSocket 通信
│   │   ├── classifier.py        # 步态分类引擎（停止后判定整段）
│   │   ├── start_ngrok.py       # 可选：通过 ngrok 将本地服务暴露到公网
│   │   └── models/
│   │       └── rf_model.pkl     # 随机森林模型（真实数据训练，后端自动加载）
│   ├── frontend/
│   │   ├── index.html           # 电脑端监控页面
│   │   └── mobile.html          # 手机端采集页面
│   └── training/
│       ├── config.py            # 配置文件（列名映射、标签映射）
│       ├── convert_real_data.py # 真实数据格式转换脚本
│       ├── features.py          # 特征提取（81 维）
│       ├── preprocess.py        # 数据加载 / 滤波 / 滑窗分割
│       └── train.py             # 模型训练脚本
├── data/
│   ├── raw/                     # 原始采集（陀螺仪 + 加速度计双文件，见采集命名约定）
│   ├── processed/               # 转换后的统一格式 CSV + confusion_matrix.png
│   └── simul/                   # 早期模拟演示数据（仅供对照，不参与训练）
├── requirements.txt             # Python 依赖
└── README.md                    # 本文档
```
> 注：项目根目录下另有一个旧路径 `backend/models/rf_model.pkl`，是早期 `train.py` 的遗留产物，**未被后端加载**（后端只认 `src/backend/models/rf_model.pkl`）。可安全删除，避免混淆（见第五节问题 9）。

每次打开运行前的第一件事就是： 激活虚拟环境     D:\JJproject\sports_recog\venv\Scripts\Activate.ps1

### 2.2 训练模块（`src/training/`）
- **`config.py`**：统一配置。`ACTIVE_CONFIG = DEMO_CONFIG`，列名为 `timestamp, acc_x/y/z, gyro_x/y/z`，`gps_cols=[]`（当前数据无 GPS）。`LABEL_MAP` 定义文件夹名（英文）→ 中文标签映射，新增运动类型需在此扩展。
- **`convert_real_data.py`**：把 Physics Toolbox 采集的**双文件**（`data/raw/<motion>.csv` 陀螺仪 `time,wx,wy,wz`；`data/raw/<motion>_a.csv` 加速度计 `time,ax,ay,az,atotal`）转换为统一格式。处理方式：取两文件**时间重叠区间**，线性插值重采样到 **50Hz**，合并为 `timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z`，丢弃冗余的 `atotal` 列；输出到 `data/processed/<label>/<label>_converted.csv`。`MOTION_MAP` 负责把原始文件名 `highknees` 映射到标签 `high_knees`。
- **`preprocess.py`**：
  - `load_csv_files()`：遍历 `data/processed` 下各运动文件夹，加载全部 CSV 并打标签；
  - `butter_lowpass` / `apply_filter()`：4 阶巴特沃斯**低通滤波**（截止 20Hz）去噪；
  - `preprocess()`：缺失值处理 + 逐列滤波；
  - `segment()`：滑动窗口分割（窗口 100 点 = 2s@50Hz，步长 50 = 50% 重叠）。
- **`features.py`**：`extract_features()` 从每个窗口提取 **81 维**特征——6 轴 × 11 时域特征（均值/标准差/方差/最值/极差/中位数/四分位/平均绝对值/均方）+ 3 个加速度轴间相关系数 + 3 轴 × 4 频域特征（FFT 能量和/均值/最大值/峰值位置）。
- **`train.py`**：训练主流程——加载 → 预处理 → 分窗 → 特征 → **RandomForest（200 树，max_depth=20）** → 80/20 分层划分 → 分类报告 + 5 折交叉验证 → 保存模型到 `src/backend/models/rf_model.pkl` → 输出混淆矩阵 `data/processed/confusion_matrix.png`。

### 2.3 后端模块（`src/backend/`）
- **`classifier.py`**：`Classifier` 类。`__init__` 加载 `rf_model.pkl`；`reset_session()`（手机「开始发送」清空缓冲）；`predict()`（仅把每帧累积进 `session_buffer`，返回原始数据供前端实时显示，**不再实时判定**）；`end_session()`（手机「停止发送」时调用：计算录制时长，<10s 判「数据不足」，否则取最近最多 1500 帧做特征提取 + 预测，返回 `activity/confidence/mode/sample_count/duration_sec`）。无模型时回退到基于加速度幅值的规则分类。
- **`main.py`**：FastAPI 主服务。`/ws/mobile` 接收手机数据并区分信令：`type:"start"`（清缓冲 + 广播 `session_start`）、`type:"stop"`（调用 `end_session` + 广播 `session_result`）、普通数据包（累积 + 广播实时 `data`）。`/ws/client` 仅保持连接。**静态文件挂载 `src/frontend`**，路由 `/` 返回电脑端、路由 `/mobile` 返回手机端。
- **`start_ngrok.py`**：可选公网隧道脚本，需自行配置 ngrok 及其 authtoken 后使用。
运行着代码  D:\JJproject\sports_recog\src\backend\main.py
同时，连接ngrok      ngrok http 8000


### 2.4 前端模块（`src/frontend/`）
- **`index.html`**（电脑端监控页）：通过 WebSocket 连接 `/ws/client`，`handleData()` 区分三种消息——`session_start`（显示「📡 采集中…」）、`data`（实时刷新传感器波形/参数，**不判定**）、`session_result`（弹出最终判定卡片）。含：运动状态卡、实时参数面板、高德地图轨迹（`AMap.Map`/`Polyline`/`Marker`，需填充 `YOUR_AMAP_KEY`）、ECharts 六通道波形图、系统日志。
http://localhost:8000/

- **`mobile.html`**（手机端采集页）：请求传感器权限（兼容 iOS 13+）、连接 `/ws/mobile`、开始/停止发送（分别发 `type:"start"` / `type:"stop"` 信令）、以 50Hz（20ms）发送 `acc/gyro/gps`。传感器读取：`acc = event.acceleration || event.accelerationIncludingGravity`（优先去重力线性加速度，部分安卓返回 null 时回退含重力），`gyro = event.rotationRate`。
https://surfboard-bagginess-raging.ngrok-free.dev/mobile

---

## 三、正常工作顺序（数据与调用链路）

系统要在端到端层面正常工作，必须让以下两条链路按序就绪：

### 3.1 离线训练链路（一次性，数据更新后重跑）
```
data/raw 双文件
   │  convert_real_data.py（重采样合并为统一 50Hz 六轴格式）
   ▼
data/processed/<label>/<label>_converted.csv
   │  train.py：load_csv_files → preprocess(滤波) → segment(2s窗) → extract_features(81维) → RandomForest
   ▼
src/backend/models/rf_model.pkl   +   data/processed/confusion_matrix.png
```
只有 `rf_model.pkl` 存在于 `src/backend/models/` 且由真实数据训练，在线判定才会走模型分支（否则走回退规则，准确率低）。

### 3.2 在线推理链路（每次运行）
```
手机端 mobile.html
   │  WebSocket /ws/mobile（start → 持续数据 → stop）
   ▼
main.py（接收并分类消息，转发给 classifier）
   │  classifier.py：累积整段 → 停止时 end_session() 判定
   ▼
broadcast（WebSocket /ws/client）
   ▼
电脑端 index.html（采集中 → 停止后弹出最终判定）
```
方向依赖：**训练链路必须先于在线链路完成**（模型就位）；在线链路中手机端「开始→停止」信令驱动后端判定，电脑端只消费结果。

---

## 四、执行流程（操作步骤）

> 数据采集部分（手机/手表导出 CSV 放入 `data/raw`）由你另行完成，本文档不展开。以下从"采集完成后"开始。采集产物命名需满足 `convert_real_data.py` 约定：每种运动提供 `<motion>.csv`（陀螺仪）与 `<motion>_a.csv`（加速度计），其中 `motion ∈ {walking, running, jumping, highknees}`（`highknees` 会自动映射到标签 `high_knees`）。

### 4.1 环境准备
- Python 3.8+（项目已自带虚拟环境 `venv/`）。
- 安装依赖（首次或新增依赖时）：
  ```bash
  cd D:\JJproject\sports_recog
  venv\Scripts\python.exe -m pip install -r requirements.txt
  ```
- 高德地图：打开 `src/frontend/index.html`，将 `YOUR_AMAP_KEY` 替换为你申请的高德地图 JS API（Web 端）Key；**不替换则地图区域显示提示，但不影响步态判定功能**。

### 4.2 训练模型（数据更新后必做）
```bash
# 1) 转换真实数据格式（输出到 data/processed）
cd D:\JJproject\sports_recog\src\training
..\..\venv\Scripts\python.exe convert_real_data.py

# 2) 训练并生成模型
..\..\venv\Scripts\python.exe train.py
```
执行后：`src/backend/models/rf_model.pkl` 被覆盖更新，`data/processed/confusion_matrix.png` 刷新。

### 4.3 启动系统
```bash
# 启动后端（保持此终端运行）
cd D:\JJproject\sports_recog\src\backend
..\..\venv\Scripts\python.exe main.py
```
看到 `Uvicorn running on http://0.0.0.0:8000` 即成功。

### 4.4 使用
1. **电脑端**：浏览器打开 `http://localhost:8000/`（请通过此地址访问，勿直接双击打开 HTML 文件，否则 WebSocket 无 host 会连接失败）。
2. **手机端**：手机与电脑连同一 WiFi，浏览器打开 `http://<电脑局域网IP>:8000/mobile`。
3. 手机端依次操作：**请求传感器权限**（iOS 必需）→ **连接** → **开始发送**（电脑端显示「📡 采集中…」、波形滚动）。
4. 运动一段（≥10 秒）后，手机端点 **停止发送**，电脑端弹出**最终判定卡片**（运动类型 + 置信度 + 时长 + 帧数）。

### 4.5 远程访问（可选）
若需脱离局域网，用 `start_ngrok.py` 建立 ngrok 隧道；手机端用 ngrok 提供的 HTTPS 地址访问 `/mobile`。注意电脑端若经 ngrok 访问空闲 WebSocket 可能被掐断，本地 `localhost` 访问最稳。

---

## 五、现存问题与优化方向

### 5.1 判定相关（影响准确率，优先）
1. **训练 / 推理特征尺度不一致（核心隐患）**
   训练用 2 秒窗（100 帧）提取特征，推理却把整段（就近窗口最长 30s）当一个大窗提 81 维特征。像"过零率"这类特征，2 秒窗是几次、整段窗是几百次，**量级差十几倍**，模型从未见过此量级，倾向判成训练里"中间幅值"的走路。
   → **优化**：推理时把整段切成多个 2 秒窗，逐窗预测后做**多数投票 / 置信度分布**，与训练完全对齐；顺带可输出"逐段判定"而非单次 argmax。

2. **训练数据量严重不足**
   当前每类仅 1 段（窗口数约 走路42 / 跑步24 / 跳跃22 / 高抬腿18），RandomForest 易过拟合到这几次采集的具体方式（手机位置、力度、个人习惯）。当前 90.9% 准确率是**同人同设备内**切分得到的，数字虚高、泛化能力差（5 折 CV 标准差 ±0.13）。
   → **优化**：每类补采到 5–10 段，覆盖不同人、不同握持、不同强度；报告中补充"跨受试者 / 跨设备"测试，如实说明 90.9% 的边界。

3. **手机端物理量偏移**
   训练数据用 Physics Toolbox 的**去重力**线性加速度；手机端 `event.acceleration` 在部分安卓会返回 `null` 而回退到 `accelerationIncludingGravity`（含 9.8 重力），分布直接错乱 → 误判。
   → **优化**：强制发送去重力线性加速度；回退含重力时在日志告警，必要时在后端统一加"去重力"步骤。

4. **缺少置信度阈值**
   低置信度时仍硬判为某一类（多为走路）。
   → **优化**：置信度低于阈值时显示"不确定 / 请延长采集"，而非强行给结果。

### 5.2 模型与架构升级
5. **换更强表格模型**：LightGBM / XGBoost 通常优于默认 RandomForest。
6. **上深度学习**：1D-CNN 或 LSTM 直接吃原始 IMU 序列、自动学表征，对"走/跑靠时序模式"类任务普遍更好，且天然规避手工特征尺度问题。
7. **任务拆解**：走/跑是周期性步态，跳/高抬腿是爆发性动作，本质不同。可拆成"周期性识别（走/跑）" + "爆发性检测（跳/高抬腿）"两个子模型，区分度更明显。

### 5.3 工程与体验
8. **"实时性"取舍**：当前为"分段判定"，解决了短时不可靠，但弱化了"实时"卖点。可在论文中描述为"准实时 / 分段判定"；或保留分段判定的同时加"滑动整段重判"让结果更稳。
9. **遗留废弃模型文件**：项目根 `backend/models/rf_model.pkl` 是旧路径产物、未被加载，易与 `src/backend/models/rf_model.pkl` 混淆，建议删除。
10. **地图室内不准（GNSS 漂移）**：室内 GPS 误差常达数十米，轨迹乱跳是物理限制而非代码 bug。建议：① 对 GPS 做滑动平均 / 卡尔曼滤波平滑（小改动，治标）；② **推荐做 IMU 航位推算（Dead Reckoning）**——加速度二次积分得位移 + 陀螺仪航向得方向，生成以起点为原点的相对轨迹，完全不依赖 GPS，且契合"低成本便携"卖点（需新增步数检测、航向估计、坐标推算模块）；③ 若电脑无外网，可换 Leaflet + OpenStreetMap 或纯相对轨迹 Canvas，避免依赖高德。

---

*文档生成时间：2026-07-21｜适用代码版本：包含「停止后判定整段」改动的当前分支。*
