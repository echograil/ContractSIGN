# ContractSIGN v0.2 发布说明

ContractSIGN v0.2 是一个本地优先的英文合同问答流水线。用户可以上传 PDF 合同、输入中文或英文问题，系统按六个模块完成文档接入、检索、路由、生成、风险依据检查和观测导出。

## 当前可用体验

- 本地 HTML UI：上传 PDF、输入问题、运行六模块流水线。
- 可视化流程：页面展示类似生产线的六个模块工位。
- 文件搬运：每个模块都有独立 `input/` 和 `output/`，运行后可以直接查看中间产物。
- 中文问题支持：本地检索增加了中文合同问题到英文法律词的扩展层，覆盖摘要、付款、终止、违约、保密、转让、适用法、争议、不可抗力等常见问题。
- 结果导出：支持复制回答、复制运行摘要、下载完整运行 JSON 报告。
- 可选 API：默认可离线运行；启用 API 路由、API Embeddings 或 API 生成时，需要用户自己的 API Key。

## GitHub Pages 说明

GitHub Pages 只能托管静态 HTML/CSS/JS，不能在服务器侧运行 Python、接收 PDF、写入文件夹或执行 `input/` -> `output/` 文件搬运。

因此：

- GitHub Pages 适合展示项目说明、开发进度、使用指南和演示页。
- 真实 PDF 上传问答需要本地运行 `python scripts/web_server.py`。
- 如果要做公网在线处理，需要另接后端服务，例如 FastAPI、serverless 或容器部署。

## 本地启动

```powershell
python -m pip install -r requirements.txt
python scripts/web_server.py
```

打开：

```text
http://127.0.0.1:8765
```

可选 API 功能：

```powershell
python -m pip install -r requirements-api.txt
copy .env.example .env
```

然后在 `.env` 中填入自己的 `OPENAI_API_KEY`。

## 六模块进度

- 01 Document Ingestion：已可用，支持文本型 PDF 接入。
- 02 Retrieval：已可用，本地混合检索 + 中文问题扩展；API Embeddings 可选增强。
- 03 Question Router：已可用，支持 QA / Summary / Generation / Agent 保留分支。
- 04 Answer Generator：已可用，支持本地抽取式回答和 API 生成。
- 05 Risk & Evidence：已可用，基于规则输出风险和复核提示。
- 06 Observability：已可用，SQLite trace、在线指标、脱敏 eval 导出。

## 已知边界

- 本地中文适配是词典扩展和兜底，不等同于完整跨语言语义理解。
- 扫描版 PDF 暂不支持 OCR。
- 风险提示当前偏规则化，事实问答中可能显得略重。
- 更强的语义召回和自然语言回答建议启用 API Embeddings/API 生成。
