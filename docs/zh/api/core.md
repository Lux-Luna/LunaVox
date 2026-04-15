# Core 工具

每一个 LunaVox 高层组件都依赖的小而聚焦的模块。每一个都是 `AGENT.md`
规则里强制的**单点真源**：其他代码只能从这里导入，不允许自己判断 OS、
重新解析路径、或者额外实例化 Rich Console。

> **英文 API 参考是唯一真源**：函数签名和参数详见
> [**English → Core Utilities**](../../en/api/core.md)。

## 导航

- **[`platform`](../../en/api/core.md#lunavox.core.platform)**
  ——`shared_lib_name` / `executable_suffix` / `is_windows` / `is_macos` / `is_linux`
- **[`resolve_project_root`](../../en/api/core.md#lunavox.core.project.resolve_project_root)**
  ——项目根解析（env var / ancestry / 显式参数）
- **[`DependencyPolicy` / `ensure_dependency_group`](../../en/api/core.md#lunavox.core.deps.DependencyPolicy)**
  ——按需安装 `convert` 依赖组
- **[`has_module` / `missing_modules`](../../en/api/core.md#lunavox.core.deps.has_module)**
  ——依赖检测
- **[`logging`](../../en/api/core.md#lunavox.core.logging)**
  ——`session_start` / `append` 线程安全会话日志
