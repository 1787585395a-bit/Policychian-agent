# PolicyChain UI 验收说明

## 本次验收范围

首页仍以政策链接或大段政策正文为主输入，保留进度条、结构化运行日志、复制日志、长文本换行、错误提示和研究辅助声明。本次新增的是运行可观测性，不改变报告正文：

- 每个任务显示唯一 Run ID；
- 显示请求运行模式和实际运行模式；
- 分别显示政策分析、行业影响、公司匹配、报告生成状态；
- 明确显示是否启用了回退方案；
- 成功或失败后均可下载脱敏 JSON 运行日志；
- 报告区域只显示研究报告，不混入 recorder 调试事件。

## API 与交互验收

- `GET /api/research-status?job_id=...` 返回 `run_id`、`requested_run_mode`、`effective_run_mode`、`agent_status`、`fallback_used` 和 `log_download_available`。
- `agent_status` 使用 `policy`、`impact`、`company`、`report` 四个面向页面的键。
- 任务运行期间下载入口不可用；任务进入 `done` 或 `error` 后，下载入口指向 `GET /api/run-log?job_id=...`。
- 下载响应为 UTF-8 JSON，含 `Content-Disposition: attachment`；不存在的任务或日志返回清晰 JSON 错误。
- 兼容入口 `GET /api/run-logs/{run_id}` 与 `GET /api/run-logs?run_id=...` 保留。
- 下载内容由 `RunRecorder` 写入并经统一脱敏：凭据字段不可见，大段正文默认只保留字符数和哈希。

## 页面检查清单

- [ ] 首页在桌面和窄屏下均可正常加载，无横向溢出。
- [ ] 政策链接和长政策正文均可输入，长文本不会遮挡提交按钮。
- [ ] 提交后按钮进入“分析中”，进度、阶段和日志持续更新。
- [ ] 运行摘要显示 Run ID、请求/实际模式、四个 Agent 状态和回退情况。
- [ ] 成功时渲染完整研究报告，且下载运行日志按钮可用。
- [ ] 错误 URL 或无效正文显示清晰错误，保留失败阶段，且下载运行日志按钮仍可用。
- [ ] 复制日志仅在终态启用，内容包括 Job、Run ID、模式、回退情况和页面进度日志。
- [ ] 报告区域不显示 `event_type`、工具响应或 recorder 内部调试事件。
- [ ] 页面和报告使用“公司业务匹配”“A 股公司关注清单”等研究辅助表达，不含买入、卖出、目标价或收益承诺。
- [ ] 浏览器控制台无 JavaScript 错误。

## 自动化验证

运行：

```powershell
python -m pytest tests/test_app.py -q
```

该测试文件覆盖首页渲染、运行摘要字段、同一 job recorder 传递、LLM/MCP 回退、成功与失败日志下载、脱敏内容、`Content-Disposition` 和本地 HTTP handler。

## 本地人工验收步骤

1. 使用 `$env:POLICYCHAIN_PORT="8010"; python app.py` 启动服务，只等待 8010 进入 Listen。
2. 打开首页，检查政策链接、政策正文、长文本和窄屏布局。
3. 提交一条可用政策正文，观察完整阶段链路和终态运行摘要。
4. 提交错误 URL 或非政策页面，检查错误阶段、错误原因和失败日志下载。
5. 分别下载成功、失败日志，确认 JSON 可解析、敏感信息已脱敏、Run ID 与页面一致。
6. 检查浏览器控制台；验收结束后按 8010 的 `OwningProcess` 停止本地服务。

## 2026-07-21 验收记录

- `python -m pytest tests/test_app.py -q`：21 passed，0 failed。
- 8010 首页：HTTP 200；运行摘要、下载入口、720px 窄屏规则、长文本换行规则和“不构成投资建议”声明均存在。
- 政策正文任务：`done`、100%；12 条页面进度日志覆盖读取输入、URL 抓取、正文校验、相似政策检索、政策分析、行业影响、公司匹配审查、公司业务匹配、报告生成和完成。
- 成功任务终态：Run ID 存在；请求/实际模式均为 `deterministic`；四个 Agent 均为 `completed`；`fallback_used=false`；日志下载可用。
- 成功报告：包含政策分析、相似政策、行业影响、公司业务匹配和不确定性；报告正文未出现 recorder 调试事件。
- 错误 URL 任务：`error`，保留失败原因和错误阶段，不生成报告；失败日志下载可用，artifact summary 为 `failed`。
- 下载检查：job 入口和 run_id 兼容入口均返回 200；响应为 JSON，带 `Content-Disposition`，Run ID 与状态接口一致。
- 服务结束后确认 8010 不再 Listen。
- 当前限制：浏览器控制插件在本次会话初始化失败，因此未完成真实浏览器点击、像素级布局和控制台错误检查；页面结构、交互契约和 HTTP/API 链路已由自动化测试与 8010 实际请求覆盖。
