# UI 2：分析与记录

## 用户路径

首页只保留录音、现场描述与当前操作。上传完成后显示“开始分析”；生成候选后自动收起录音区，
展示当前观察问题；完成后优先展示解释。历史记录位于独立导航入口。

移除了宣传标语、英文阶段编号、侧边步骤列表、重复提示和主页面版本标签。
完整的模型分数、来源、规则、候选、问答、更正记录仍通过原生折叠区访问。
实际模型/API 错误和未完成状态不会被隐藏：错误会展开声音分析，阻止继续操作。
结果依然区分“可能是”与“暂不确定”，没有新增概率或确定性承诺。

## 代码结构

| 文件 | 责任 |
|---|---|
| `static/index.html` | 语义结构、表单标签、原生 details/summary，保留已有控件 ID |
| `static/style.css` | 颜色、间距、宽度 tokens；布局、组件、移动端和减少动态效果 |
| `static/presentation.js` | 纯展示状态；无网络、存储、DOM 或推理副作用，可独立单测 |
| `static/app.js` | 保留现有接口编排；应用展示状态、焦点、进度和记录交互 |
| `static/icons.svg` | 本地 Lucide SVG 符号集，无运行时 CDN |
| `app_version.py` | 后端、启动器、安装器和发布清单的版本来源 |

图标来自 Lucide **0.468.0**（ISC）：
https://github.com/lucide-icons/lucide/tree/0.468.0/icons
完整许可保存在 `static/vendor/LUCIDE_LICENSE.txt`。
没有引入新的前端框架、构建服务、在线字体或脚本源，也没有放宽 CSP。
对这个体量的本地应用，浏览器原生控件能减少依赖和自定义键盘交互的维护成本。

## 扩展约定

- 新状态先写进 `presentation.js`，再由页面渲染消费，避免把服务状态判断散落到 CSS。
- 新控件使用有名称的 button、显式 label 和稳定 ID；复用现有 API，不以屏幕位置作为操作契约。
- 新模型信息进入已有结构化输出与折叠报告；不要把训练元信息放回主操作区。
- 新 API 必须继续遵循服务端白名单、JSON 校验和失败状态；前端不修改模型输出。
- 问题更新与结果出现时移动键盘焦点；新分析回到上传控件。忙碌状态使用 aria-busy；错误与进度使用 live region。
- 改动后更新 `build_manifest.json`；HTML 的 data-app-version 必须与 APP_VERSION 一致（测试覆盖）。

## 验证

```powershell
python -m unittest discover -p "test_*.py"
python -m unittest discover -s tests -p "test_*.py"
node test_ui_v9.cjs
node test_presentation.cjs
node test_recorder.cjs
python build_integrity.py
python start_app.py --check
```

本次实际通过：125 项应用 Python 检查、29 项场景回归、DOM 交互检查、纯展示状态检查和 8 项录音生命周期检查。

本次检查覆盖完整 6+3+3 分数、真实 E4 子进程与模拟场景/LLM 的 HTTP 联调、未知回答、三问上限、
原始输出保留、记录/导出、录音生命周期、错误状态、原生折叠、标签关联和版本一致性。
本次没有更换或重新训练任何场景/情绪模型，没有修改候选规则及后端推断协议。

浏览器脚本已同步新布局，可在支持 Chromium 的机器上运行：

```powershell
python test_browser_six.py --real-emotion --screenshots ui_screenshots
```

需要 Playwright 及其 Chromium（或用 `--browser-executable` 指定已安装浏览器）。
本次环境的本地 Chromium 因 socket 权限限制未能启动；云端浏览器无法连接执行环境的 localhost。
因此本次未完成真实浏览器视觉验收和实机移动端验收，DOM/结构检查不能替代这两项。

## 安装

关闭原运行窗口，使用新版 `install_update.py` 将完整程序更新到原目录，然后运行 `00_START_APP.bat`。
已有 E4 环境无需重跑 08；数据、模型、密钥和历史记录由安装器保留。
版本显示在“连接状态”中：`PawTalk UI 2 · Six Scenes + Emotion E4`。
