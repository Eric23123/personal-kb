# 贡献指南

感谢你愿意改进 Personal KB。

## 提交问题

请说明复现步骤、期望行为、实际行为、系统版本和应用版本。日志、截图和示例文件必须脱敏；不要上传 API Key、私人笔记、真实课程资料、转写文本或整个知识库目录。

## 提交改动

1. 从最新代码创建独立分支。
2. 保持改动聚焦，避免混入格式化或无关重构。
3. 为行为变化补充或更新测试。
4. 提交前运行：

```powershell
python -m pytest -q
cd desktop
npm run build
```

5. 在 Pull Request 中说明目的、验证方式和已知限制。

## 隐私边界

请勿提交 `.env`、API Key、Windows 凭据导出、模型、`node_modules`、构建产物、`.personal-kb`、真实资料或个人 Obsidian 笔记。提交前执行 `git status --ignored` 复核。

## 上游致谢

Personal KB 基于 `TobyCh04/cornell-kb` 改造。请保留上游来源说明；许可证事宜以原作者后续明确授权为准。
