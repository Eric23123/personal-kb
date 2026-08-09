# GitHub 发布材料

这份清单用于发布 Personal KB 的第一个公开预览版。发布前请以全新的 Git 克隆目录复核一次，不要使用你的真实知识库目录。

## 仓库信息

| GitHub 字段 | 建议内容 |
| --- | --- |
| Repository name | `personal-kb` |
| Description | 本地优先的 Windows 桌面知识库，支持资料导入、来源检索、Obsidian 协作与知识链接。 |
| Website | 暂不填写 |
| Topics | `knowledge-base`, `personal-knowledge-management`, `tauri`, `react`, `python`, `rag`, `obsidian`, `whisperx`, `openviking` |
| 可见性 | 先 Private 测试，确认无个人资料后再改为 Public |
| License | 暂不添加，等待上游作者确认许可证方案 |

## 发布前检查

- [ ] 已保留上游作者同意 Fork 与改造的聊天记录。
- [ ] README 中保留了上游仓库链接。
- [ ] 没有 API Key、`.env`、凭据导出、真实资料路径或个人信息。
- [ ] 没有真实 PDF、音频、图片、转写文本、`.personal-kb/` 或 Obsidian 私人笔记。
- [ ] `models/`、`desktop/node_modules/`、`desktop/dist/`、`desktop/src-tauri/target/`、`releases/` 未被 Git 跟踪。
- [ ] 运行 `git status --ignored` 检查忽略规则是否生效。
- [ ] 运行 `python -m pytest -q` 与 `cd desktop; npm run build`。
- [ ] 截图已脱敏：没有姓名、绝对路径、资料正文、API Key、余额或账户信息。
- [ ] 从 GitHub clone 到新目录后，README 的安装步骤可执行。

## 第一个 Release

**Tag**：`v0.1.0-preview`  
**Title**：`Personal KB v0.1.0 Preview`  
**Pre-release**：勾选  
**Latest release**：勾选

上传附件：

```text
Personal-KB_0.1.0_x64-setup.exe
```

不要上传模型、用户资料、`.personal-kb`、`node_modules` 或整个 `target` 目录。

## Release 正文

```markdown
## Personal KB v0.1.0 Preview

Personal KB 是一个本地优先的 Windows 桌面知识库。它可以导入 PDF、图片、音频和文本资料，建立索引并基于原始资料检索问答，同时可与 Obsidian 使用同一个知识库目录。

### 本次包含

- 文件与文件夹导入
- PDF 提取、图片 OCR/图表说明、WhisperX 本地音频转写
- OpenViking 语义检索与 BM25 本地检索
- 基于来源的问答与可展开来源
- 资源管理、一键更新与回收站
- Obsidian 协作、按需关联笔记与集中式知识链接

### 使用前请注意

- 当前仅支持 Windows。
- 这是预览版。安装包不包含 Python、OpenViking、WhisperX 模型或 API Key；请先阅读 README 完成本地配置。
- 你的资料、索引与 API Key 不随安装包提供，也不应提交到 GitHub。
- `.ppt` / `.pptx` 请先导出为 PDF 后再导入。

### 反馈问题

提交 Issue 时请提供复现步骤、系统版本、应用版本和脱敏后的错误信息。请勿上传真实课程资料、API Key 或包含隐私信息的截图。

### 上游致谢

本项目基于 https://github.com/TobyCh04/cornell-kb 改造，并经原作者同意 Fork 与修改。
```

## 首个 Issue 模板

```markdown
### 问题描述

### 复现步骤
1.
2.
3.

### 期望行为

### 实际行为

### 环境
- Personal KB 版本：
- Windows 版本：
- Python 版本：
- 是否使用 GPU：

### 脱敏后的日志或截图
```
