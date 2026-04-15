# Core Utilities

Small, focused modules that every higher-level LunaVox component depends
on. Each one is a single-point-of-truth enforced by `AGENT.md` rules:
other code imports from here rather than probing the OS, re-reading
paths, or spinning up Rich consoles directly.

## Platform detection

::: lunavox.core.platform
    options:
      show_root_heading: true
      members:
        - shared_lib_name
        - executable_suffix
        - is_windows
        - is_macos
        - is_linux

## Project root resolution

::: lunavox.core.project.resolve_project_root
    options:
      show_root_heading: true

## Dependency management

::: lunavox.core.deps.DependencyPolicy
    options:
      show_root_heading: true

::: lunavox.core.deps.ensure_dependency_group
    options:
      show_root_heading: true

::: lunavox.core.deps.has_module
    options:
      show_root_heading: true

::: lunavox.core.deps.missing_modules
    options:
      show_root_heading: true

## Session logging

::: lunavox.core.logging
    options:
      show_root_heading: true
      members:
        - session_start
        - append
