# RoboChemGym

**面向长程化学操作的协议驱动生成式仿真框架**

[English](README.md) | [简体中文](README_CN.md)

![RoboChemGym 概览](docs/images/robochemgym-teaser.png)

RoboChemGym 将自然语言化学实验协议转换为可执行的机器人轨迹，通过面向
实验室场景的随机化扩展轨迹数据，并提供 Isaac Sim 中的策略部署与评测
入口。本项目构建于 LabUtopia 仿真环境之上。

当前仓库是首个源码版本，重点是提供完整的代码路径和可复现配置，不包含
训练完成的策略及其效果声明。

## 概览

RoboChemGym 包含三个相互衔接的模块：

1. **协议驱动的轨迹生成**：将文本转换为结构化 Agent Plan，校验动作和
   资产，构建场景，在 Isaac Sim 中执行，并保存报告和采样视频。
2. **随机化数据采集**：对当前支持的场景因素进行随机化，通过原子化
   manifest 流程保存成功 episode，并生成确定性数据划分。
3. **策略部署与评测**：检查 checkpoint 与环境的兼容性，在固定随机种子
   下执行 Isaac 评测，并生成任务级和动作级指标。

## 发布状态

| 方法模块 | 状态 |
|---|---|
| 协议驱动的轨迹生成 | 已发布 |
| 随机化数据采集 | 已发布 |
| 策略部署与评测 | 已发布 |

## 安装

环境要求：

- Ubuntu Linux
- Python 3.10
- NVIDIA GPU 及兼容驱动
- Isaac Sim `4.2.0.2`
- Git LFS

```bash
git clone https://github.com/Xeraphael/RoboChemGym.git
cd RoboChemGym
git lfs install
git lfs pull

conda create -n labutopia python=3.10 -y
conda activate labutopia
pip install "isaacsim[all,extscache]==4.2.0.2" \
  --extra-index-url https://pypi.nvidia.com
pip install -r requirements.txt
```

## API 配置

Action Agent 使用兼容 OpenAI chat-completions 的 API。所有配置必须通过环境
变量提供，不要提交 `.env` 文件。

```bash
cp .env.example .env
set -a
source .env
set +a
```

接口地址、模型、API Key、超时和重试次数对应的环境变量见
[`.env.example`](.env.example)。

## 快速开始

### 1. 运行用户 Protocol

将需要执行的实验流程写入任意文本文件，然后把文件路径传给 Action Agent：

```bash
python agent/main.py --protocol path/to/protocol.txt
```

例如，可以运行仓库内置的示例：

```bash
python agent/main.py --protocol protocols/example_protocol/protocol.txt
```

执行流程如下：

```text
文本 Protocol
  -> LLM 生成结构化 Agent Plan
  -> 确定性 Validator
  -> 编译 USD/YAML 并进行场景预检
  -> Isaac 执行和基于状态的成功判定
  -> 有界的参数/场景重试
  -> 报告、轨迹和采样尝试视频
```

每次运行会在 `outputs/action_agent/` 下创建独立目录。可以从已经校验的 Plan
继续执行，避免再次调用 LLM：

```bash
python agent/main.py --resume outputs/action_agent/<run-id>
```

缺少资产或核心原子动作时会阻止执行。对于当前无法观测的描述性语义，例如
“避免过热”，系统会记录非阻塞警告，并执行最接近的受支持物理动作。

旧代码生成后端会执行 LLM 生成的 Python，默认处于禁用状态。只有显式传入
`--allow-unsafe-codegen` 才会启用，并且只应在一次性隔离环境中使用。

仓库在 [`protocols/example_protocol/`](protocols/example_protocol/) 中提供了
该示例对应的可复用 Plan、校验产物、配置和场景。

### 2. 采集随机化 Episode

```bash
python collect.py --config config/collection/example_protocol.yaml \
  --/rtx/verifyDriverVersion/enabled=false
```

v0.1 配置会随机化物体位置、灯光强度与色温，以及已有的工作台材质。每个
episode 的实际随机化结果会写入 manifest。Episode 首先写入临时文件，只有
成功关闭文件后才会加入数据划分。

### 3. 部署并评测

```bash
python evaluate.py --config config/evaluation/example_protocol_act.yaml \
  --/rtx/verifyDriverVersion/enabled=false
```

评测结果写入
`outputs/evaluation/example_protocol_act/runs/<run-id>/`，并明确标记为
`running`、`completed` 或 `incomplete`。开始策略控制前，系统会检查数据集
身份、schema、相机顺序与尺寸、状态/动作约定以及夹爪约定是否与 checkpoint
一致。

## 评测输出

评测报告支持以下指标：

| 指标 | 说明 |
|---|---|
| 任务成功率 | 完成全部必要动作的 episode 比例 |
| 分步骤成功率 | Protocol 中每个有序步骤的成功次数 |
| 分动作成功率 | 按原子动作类型聚合的成功情况 |
| 失败分类 | 校验、资产、执行、超时和兼容性错误 |
| Episode 长度 | 每个 episode 执行的控制步数 |
| 终止距离 | 可观测时物体与目标之间的最终距离 |

v0.1 源码验收不包含训练完成的策略或策略效果声明。

## 代码结构

```text
agent/                    Protocol 解析、规划、校验与执行
controllers/              原子动作和机器人控制器
data_collectors/          相机、episode 记录和 HDF5 采集
pipeline/                 采集、训练和评测编排
policy/                   ACT 模型、数据集、运行器和工作空间
tasks/                    Isaac 任务定义
protocols/                可复现 Protocol bundle
config/                   采集、训练和评测配置
scripts/                  可复现启动命令
tests/                    功能测试
docs/                     项目图片和第三方声明
```

## 资产

当前公开源码只包含内置 `example_protocol` 示例场景中嵌入的资产：两个锥形瓶
变体、加热板和目标平台。Capability registry 中的其他条目仅用于兼容；在
提供可再分发资产并更新路径之前，Validator 会阻止使用这些资产。

旧的专有实验室场景和私有仪器资产库不属于本次发布。Isaac Sim 及其运行时
资产需要单独安装。

## Isaac Sim 驱动校验

RoboChemGym 固定使用 Isaac Sim `4.2.0.2`。项目入口会注入
`--/rtx/verifyDriverVersion/enabled=false`，用于绕过 Kit 106.1 的驱动版本
解析问题，不会关闭渲染或物理功能。也可以像上面的命令一样显式传入该参数。

## 引用

论文发表后将补充正式引用信息。软件元数据见 [`CITATION.cff`](CITATION.cff)。

## 许可证

RoboChemGym 源码使用 [MIT License](LICENSE)。第三方代码、Isaac Sim 和外部
资产保留各自许可证，详见[第三方声明](docs/THIRD_PARTY_NOTICES.md)。
