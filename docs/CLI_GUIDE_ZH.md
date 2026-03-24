# LunaVox CLI 使用指南（中文）

## 1. 定位与命令面

LunaVox CLI 已收敛为以下命令：

- `lunavox setup`
- `lunavox build`
- `lunavox bootstrap`
- `lunavox download libs`
- `lunavox doctor`

已移除命令：

- `lunavox download model`
- `lunavox convert onnx`
- `lunavox convert gguf`
- `lunavox quantize onnx`

## 2. 安装

基础安装：

```powershell
pip install lunavox
```

包含模型转换依赖：

```powershell
pip install "lunavox[convert]"
```

本地开发安装：

```powershell
pip install -e .
```

## 3. Setup 一键转换量化（唯一入口）

`setup` 是模型准备与转换量化的唯一入口，不提供量化策略选择参数。

固定策略如下：

- `embeddings`：`fp16`
- `speaker encoder`：`fp16`
- `codec encoder`：`fp16`
- `codec decoder`：`fp16`
- `talker`：`q5_k`
- `predictor`：`q8_0`

常用示例：

```powershell
lunavox setup --model base_small
lunavox setup --model base --force
lunavox setup --model custom_small --models-dir D:\TTS\lunavox\models\custom_small
```

### 缺失源模型时的交互

当源模型缺失时，CLI 会英文询问是否下载，例如：

`Model '<name>' source files are missing. Download from HuggingFace now?`

- 交互环境：确认后自动下载并继续 `setup`。
- 非交互环境：默认不自动确认；需显式加 `--yes`。

## 4. 其他命令

### build

```powershell
lunavox build --clean --j 4 --verify
```

### bootstrap（setup + build）

```powershell
lunavox bootstrap --model base_small --clean --j 4
```

### download libs

```powershell
lunavox download libs llama win_cuda
lunavox download libs onnx win_cuda
```

### doctor

```powershell
lunavox doctor
```

## 5. 全局参数

所有命令支持：

- `--project-root PATH`：显式指定 LunaVox 项目根目录
- `--yes`：自动确认交互（下载/安装）
- `--no-install`：禁用自动安装依赖
- `--verbose`：输出详细日志

示例：

```powershell
lunavox --project-root D:\TTS\lunavox --yes setup --model base_small
```

## 6. 依赖策略

- `setup` 会检查 `convert` 依赖组。
- `build / bootstrap / download libs / doctor` 不会触发重依赖安装。

缺失依赖时会提示安装命令：

```powershell
python -m pip install "lunavox[convert]"
```

## 7. 发布流程

构建：

```powershell
python -m build
```

上传：

```powershell
twine upload dist/*
```

