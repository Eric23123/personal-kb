# Personal-KB 项目报告

**报告日期：** 2026-08-04  
**范围：** 仓库审阅、离线验证与改进评估。  
**生产环境变更：** 无。

## 执行摘要

Personal-KB 是一套边界清晰、以来源为依据的学习资料处理管线。其核心工程优势包括：失败即拒绝的源文件哈希清单、OpenViking 与 Hindsight 的职责分离、命名空间校验、来源追溯要求、隔离的摄入门禁，以及明确的生产回滚规则。当前的生产检索策略是 BM25 与稠密检索的 RRF 混合检索，默认不启用重排序；在取得具有代表性的课程数据并完成评估前，这一保守策略是合适的。

该仓库**目前不能声明离线门禁完全通过**。本次环境验证得到 4 个测试失败和 1 个源清单哈希不匹配问题。问题均可复现且范围局部，但在修复之前不能进行真实课程资料摄入。Python 编译检查和使用假客户端的端到端门禁均已通过。

## 项目功能

1. 接收本地文本、PDF、音频、图片和课件输入。
2. 通过 PyMuPDF、Whisper/MOSS、OCR 与图表处理流程生成绑定哈希的提取产物。
3. 将权威源材料写入 OpenViking 进行来源可追溯的检索，将个人和时间相关的学习状态保存在 Hindsight。
4. 使用经校验的查询规划、命名空间安全查询、带来源信息的内容读取，以及本地词法加稠密 RRF 融合。
5. 以 DeepSeek 生成结构化学习笔记，并将受管理的 Markdown 同步到按课程路由的 Obsidian 位置。
6. 以清单、资源盘点、幂等性、读回验证、测试资源清理和回滚证据保护真实摄入。

## 架构评估

| 区域 | 当前设计 | 评估 |
|---|---|---|
| 权威知识 | `viking://resources/personal-kb` 下的 OpenViking | 正确地以源材料为依据，URI 和来源信息保留为明确不变量。 |
| 个人学习记忆 | Hindsight | 与权威课程事实和检索正确分离。 |
| 检索 | BM25 + 稠密 RRF；重排序可选 | 保守且合理；没有留出的真实课程数据时不应提升模型或策略。 |
| 摄入安全 | SHA-256 清单、幂等性、盘点、清理 | 设计较强，但已提交的测试夹具目前不一致。 |
| 媒体管线 | PyMuPDF、Whisper/MOSS、GLM-OCR、Qwen3-VL | 分阶段的模型职责适当，GPU 可选依赖已隔离。 |
| 笔记输出 | DeepSeek 生成、修正流程、Obsidian 同步 | 工作流边界良好，但 Windows 下的命令行 Unicode 输出需要修复。 |
| CI | GitHub Actions、Python 3.11、pytest 与 compileall | 基线合理，但当前测试未全部通过。 |

## 已执行的验证

| 命令 | 结果 | 证据 |
|---|---|---|
| `python -m pytest -q` | 失败：327 通过，4 失败 | 2 项 CLI 帮助测试和 2 项可注入客户端的资源盘点测试失败。 |
| `python -m compileall -q scripts tests test_runs` | 通过 | 退出码为 0。 |
| `python test_runs/e2e_gate.py` | 通过：8/8 | 首次写入、幂等性、哈希不匹配、来源信息、源读回、Hindsight 隔离、回滚盘点及清理均通过。这是默认的假客户端门禁，不是线上服务验证。 |
| `python scripts/ingestion/source_manifest.py validate --manifest config/source_manifest.json --root .` | 失败 | `data/Test_whisper.txt` 的 SHA-256 与已提交夹具清单不匹配。 |

当前工作区没有 `.git` 目录。因此，无法执行 `git status`、最近提交检查和 `git diff --check`；无法基于这一工作区判断分支清洁状态或历史来源。

## 最高优先级问题

### P0：测试夹具清单无效

`data/Test_whisper.txt` 的 SHA-256 为 `D924F9105299C7266A161C1A4C2197CF4222EC15C0F060A019B475A07466B0EC`，但 `config/source_manifest.json` 记录的是 `f132644a77efd61b5c19d8c72b167e0c13951dd22eae0511bee9bf737643bd92`。

失败即拒绝的策略正确地阻止了此问题。不要直接替换清单哈希：应先确认文本夹具和清单中哪一方是权威版本，再更新非权威一侧，并为公开夹具清单添加回归断言。

### P0：完整离线测试套件未通过

1. `tests/test_note_generator_optimization.py::test_cli_exposes_bounded_worker_and_source_output_controls`
2. `tests/test_note_generator_routing.py::test_cli_only_offers_deepseek_backend_and_deprecated_ollama_url`

两项测试都会执行 `scripts/notes/note_generator.py --help`。Argparse 尝试输出一个警告符号（`U+26A0 U+FE0F`），在当前 Windows GBK 控制台中触发 `UnicodeEncodeError`。应使 CLI 帮助信息跨平台：为参数帮助使用 ASCII 文案，或在解析参数之前显式配置 UTF-8 标准输出。仅在输出契约确有需要时保留 Unicode，并在 Windows 上覆盖测试。

3. `tests/test_recovery_tooling.py::TestResourceInventory::test_snapshot_with_fake_client`
4. `tests/test_recovery_tooling.py::TestResourceInventory::test_snapshot_skips_dirs`

`scripts/ops/resource_inventory.py:snapshot()` 即使已传入假 `InventoryClient`，仍无条件导入 `openviking_sdk`。应只在 `client is None` 分支内导入并构造 `SyncHTTPClient`。这能维持声明的可注入离线契约，使 `requirements-dev.txt` 继续足以运行离线测试。

## 文档与就绪度风险

- 验证计数已陈旧或互相矛盾：Step 6 计划记录 276 个测试，`STATUS.md` 在不同检查点记录 313、331 和 334，而本次审阅观察到 327 个通过、4 个失败。下一次状态更新应使用带时间戳的命令输出作为唯一依据。
- `docs/production-readiness-runbook.md` 和 `e2e_gate.py` 的用法文本记录了 `python test_runs/e2e_gate.py --offline`，但解析器不接受 `--offline`。默认调用走假客户端离线路径；应记录这一实际命令，或添加该参数并提供覆盖测试。
- 真实课程摄入和端到端线上回答链路验证仍有意等待真实课程资料和隔离的 live-read 报告。这是适当的生产边界，而非实现遗漏。
- 此仓库快照缺少 Git 元数据。发布门禁前应恢复元数据，以便清洁工作区与差异检查具备实际意义。

## 已安装技能

以下 Codex 技能已安装到用户级技能目录，并会在下一次会话中可用：

| 技能 | 与 Personal-KB 的关联 |
|---|---|
| `pdf` | 用于媒体管线的 PDF 提取、检查、渲染和产物质量验证。 |
| `speech` | 用于与课堂音频和媒体准备相关的音频工作流。 |
| `transcribe` | 用于 Whisper/MOSS 路径的转录生成与验证。 |
| `security-best-practices` | 用于审查 API 密钥、本地文件摄入、服务端点和来源边界。 |

## 建议的下一里程碑

1. 在确认权威夹具内容后，修复公开夹具哈希不匹配。
2. 修复上述两类集中代码缺陷，并新增或保留对应的回归测试。
3. 运行 `python -m pytest -q`、`python -m compileall -q scripts tests test_runs`、清单验证和默认 E2E 门禁；将输出保存为带日期的证据产物。
4. 以该证据更新并统一 `STATUS.md`、Step 6 计划和生产运行手册，特别是实际的 E2E 命令。
5. 仅在离线门禁全部通过后，再收集经用户授权的隔离 live-read 报告。该验证期间不得修改生产 OpenViking 命名空间或 Hindsight bank。

## 已审阅来源

- `LUNA_DS_WORKING_FRAMEWORK.md`
- `README.md`
- `STATUS.md`
- `docs/plans/step6_openviking_plan.md`
- `docs/plans/step4_note_generator_plan.md`
- `docs/production-readiness-runbook.md`
- `.github/workflows/tests.yml`
- `scripts/notes/note_generator.py`
- `scripts/ops/resource_inventory.py`
