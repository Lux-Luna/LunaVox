import time
import common_setup; common_setup.configure_paths()
import lunavox_tts as lunavox

# 1. 初始化 (自动加载 'luna_zh' Persona + v2/v2pp 模型)
# 若要使用 v2_pro_plus 模型，请将 version='v2' 改为 'v2_pro_plus'
# 若要强制指定运行设备，请添加参数: device='cpu' 或 device='gpu'
lunavox.initialize_tts('luna_zh', version='v2')

# 2. 合成语音
print("正在生成音频...")
lunavox.tts(
    character_name='luna_zh',
    text='你好！这是一个简化后的 LunaVox 中文教程。',
    play=True,
    language='zh'
)

# 等待播放
time.sleep(5)
