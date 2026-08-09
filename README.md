# Personal KB

一个本地优先的 Windows 桌面知识库。将 PDF、图片、音频、文本或整个文件夹放入你的资料库，Personal KB 会提取内容、建立索引，并基于原始资料回答问题。

> 当前为 Windows Preview。安装包不包含 Python、OpenViking、语音模型、API Key 或你的资料。

## 功能

- 导入 PDF、文本、Markdown、图片、音频和文件夹。
- PDF 文本提取、图片 OCR/图表说明，以及 WhisperX 本地音频转写。
- 混合检索：OpenViking 语义检索结合本地 BM25 词法检索。
- 基于检索原文的问答，并显示可展开的来源。
- 资源管理：按文件夹浏览、排序、筛选、批量处理、一键更新和回收站。
- 与 Obsidian 共用同一个知识库目录；按需创建关联笔记。
- 手动建立“相关知识”“同一课堂”“前后关系”等知识链接。

## 它如何保存资料

你在设置中选择的文件夹就是知识库本体，例如 `D:\资源库`。原始文件保留在该目录；应用数据会放在其中的 `.personal-kb/`：

```text
资料库/
  收件箱/                   新导入资料的默认位置
  数学/线性代数/             你自己组织的文件夹与资料
  notes/                    按需创建的 Obsidian Markdown 笔记
  .personal-kb/             由程序维护，请勿手动修改
    artifacts/              提取、OCR 与转写后的文本
    records/                每份资料的处理记录
    lexical-index.json      本地 BM25 索引
    relations.json          整个知识库的知识链接
    recycle/                回收站
```

移动或新放入资料后，在“资源管理”中使用“一键更新”扫描并处理变化。

## 快速开始

### 环境要求

| 项目 | 用途 | 是否必需 |
| --- | --- | --- |
| Windows 10/11 | 当前桌面端平台 | 是 |
| Python 3.11+ | 本地处理与检索脚本 | 是 |
| OpenViking 服务 | 语义索引与检索 | 是 |
| DeepSeek API Key | 将检索结果组织为回答 | 按需 |
| DashScope API Key | 图片 OCR、图表说明等视觉处理 | 按需 |
| NVIDIA GPU + CUDA | 加速 WhisperX 音频转写 | 建议 |
| Obsidian | 编辑笔记与查看图谱 | 可选 |

### 从源码运行

```powershell
git clone <你的仓库地址>
cd personal-kb

python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-live.txt

cd desktop
npm ci
npm run tauri dev
```

需要本地音频转写、GPU 或额外 OCR 能力时，再安装媒体依赖：

```powershell
python -m pip install -r requirements-media.txt
```

### 构建安装包

```powershell
cd desktop
npm run tauri -- build
```

安装包生成在：

```text
desktop/src-tauri/target/release/bundle/nsis/Personal KB_0.1.0_x64-setup.exe
```

### 首次使用

1. 打开应用，在“设置”中选择一个空文件夹作为默认知识库目录。
2. 按需保存 DeepSeek 与 DashScope API Key。密钥保存在 Windows 凭据管理器，不会写入资料库。
3. 启动本地 OpenViking 服务。
4. 在“资料库”拖入文件或文件夹，或直接将资料放入知识库目录。
5. 在“资源管理”中点击“一键更新”，等待状态变为“已处理”。
6. 在“搜索”中用完整问题检索资料。

详细说明见 [新手使用指南](docs/Personal-KB-新手使用指南.md)。

## 与 Obsidian 一起使用

在 Obsidian 中选择“打开文件夹作为仓库”，并选择你的知识库根目录，例如 `D:\资源库`，不要选择 `.personal-kb`。

Personal KB 不会自动为每份资料创建笔记。你可以在资源管理中选中资料并点击“创建关联笔记”，此时才会在 `notes/` 下生成 Markdown。知识链接不会生成笔记，只会记录到 `.personal-kb/relations.json`，为后续“知识网络”提供数据。

## 隐私与费用

- 原始资料、索引、处理产物和知识关系默认都在本地知识库文件夹内。
- WhisperX 默认本地运行；CPU 也可使用，但速度较慢。
- DashScope 仅在图片 OCR、图表说明等视觉任务时接收必要内容。
- DeepSeek 仅在回答时接收你的问题与检索到的上下文。
- API 调用可能产生费用，请在服务商控制台检查模型与计费设置。

## 已知限制

- 当前只支持 Windows。
- `.ppt` 和 `.pptx` 建议先导出为 PDF 再导入。
- 当前安装包不是完全独立版，首次使用仍需准备 Python 与 OpenViking。
- “知识网络”可视化编辑和自动关联建议仍在规划中。

## 开发与验证

```powershell
python -m pytest -q
python -m compileall -q scripts tests test_runs

cd desktop
npm run build
cd src-tauri
cargo check
```

仓库不会包含知识库内容、API Key、模型缓存、构建产物或 `node_modules`。提交前请再次检查 `.gitignore`，不要提交真实资料、转写文本、截图中的个人信息或 `.personal-kb`。

## 路线图

- [x] 导入、处理、管理与来源检索
- [x] Obsidian 协作与按需关联笔记
- [x] 集中式知识关系记录
- [ ] 知识网络：拖动资料节点并编辑关系连线
- [ ] 知识关系的查看、编辑与删除
- [ ] 自动建议可能相关的资料，由用户确认后写入
- [ ] 可选后端依赖的一键安装引导

## 上游与授权

Personal KB 基于 [TobyCh04/cornell-kb](https://github.com/TobyCh04/cornell-kb) 的资料处理、检索和 Obsidian 协作基础改造而来，并经原作者同意 Fork 与改造。

上游项目当前未声明统一许可证。请保留上游链接与授权记录；在获得原作者关于许可证的明确确认前，本项目不自行添加 MIT 或其他通用开源许可证。

## 贡献

欢迎提交问题和改进建议。提交前请阅读 [贡献指南](CONTRIBUTING.md)，并确保日志、截图和复现资料不包含个人知识库内容或密钥。
