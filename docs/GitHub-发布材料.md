# GitHub 发布材料

Personal KB 当前采用**源码优先**的发布方式：GitHub 仓库是项目主页、使用入口和作品集展示页，不发布独立安装包。

## 仓库信息

| GitHub 字段 | 建议内容 |
| --- | --- |
| Repository name | `personal-kb` |
| Description | 本地优先的 Windows 个人知识库，支持资料导入、来源检索、Obsidian 协作与知识链接。 |
| Website | 暂不填写 |
| Topics | `knowledge-base`, `personal-knowledge-management`, `tauri`, `react`, `python`, `rag`, `obsidian`, `whisperx`, `openviking` |
| 可见性 | 先 Private 测试，确认无个人资料后再改为 Public |
| License | 暂不添加，等待上游作者确认许可证方案 |

## 发布前检查

- [ ] 已保留上游作者同意 Fork 与改造的聊天记录。
- [ ] README 中保留了上游仓库链接。
- [ ] 没有 API Key、`.env`、凭据导出、真实资料路径或个人信息。
- [ ] 没有真实 PDF、音频、图片、转写文本、`.personal-kb/` 或私人 Obsidian 笔记。
- [ ] `models/`、`desktop/node_modules/`、`desktop/dist/`、`desktop/src-tauri/target/`、`releases/` 未被 Git 跟踪。
- [ ] 运行 `git status --ignored` 检查忽略规则是否生效。
- [ ] 运行 `python -m pytest -q` 与 `cd desktop; npm run build`。
- [ ] 截图已脱敏：没有姓名、绝对路径、资料正文、API Key、余额或账户信息。
- [ ] 从 GitHub clone 到新目录后，README 的源码运行步骤可执行。

## 版本标记

当完成一个稳定阶段时，可以在 GitHub 创建 tag，例如：

```text
v0.1.0-preview
```

该 tag 用于固定可复现的源码版本，不需要上传 `.exe`、模型或任何用户资料。用户按照 README 的“从源码运行”完成本地配置即可。

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
- Personal KB 版本或 commit：
- Windows 版本：
- Python 版本：
- 是否使用 GPU：

### 脱敏后的日志或截图
```
