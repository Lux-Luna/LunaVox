import sys
import os
from pathlib import Path

# Add src to path
REPO_ROOT = Path(__file__).parent
REPO_SRC = REPO_ROOT / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from lunavox_tts.Utils.EnvManager import env_manager

def main():
    if len(sys.argv) < 2:
        print("Usage: python env_test_tool.py [status | switch-to-gpu | switch-to-cpu]")
        return

    cmd = sys.argv[1]
    
    print(f"--- LunaVox Environment Manager Test Tool ---")
    print(f"Configured Mode: {env_manager.get_mode()}")
    print(f"Is GPU Runtime Functional: {env_manager.is_gpu_installed()}")
    
    if cmd == "status":
        pass
    elif cmd == "switch-to-gpu":
        print("\nRequesting switch to GPU...")
        env_manager.set_mode("gpu")
        # This will trigger installation if missing
        if not env_manager.ensure_environment():
            print("\n[!] Dependency changes were made. A restart is required.")
        else:
            print("\n[+] Environment is already correct or was successfully updated.")
            
    elif cmd == "switch-to-cpu":
        print("\nRequesting switch to CPU...")
        env_manager.set_mode("cpu")
        # For the sake of this task, we explicitly uninstall GPU to confirm "cleanup"
        if env_manager.is_gpu_installed():
            env_manager.install_cpu_runtime()
            print("\n[!] GPU runtime uninstalled. A restart is required.")
        else:
            print("\n[+] Environment is already CPU-only.")
    else:
        print(f"Unknown command: {cmd}")

if __name__ == "__main__":
    main()
