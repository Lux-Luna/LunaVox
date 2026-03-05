import subprocess
import sys
import logging

logger = logging.getLogger(__name__)

class DependencyManager:
    @staticmethod
    def check_dependencies(package_names: list[str], language_name: str, auto_install: bool = False) -> bool:
        """
        Check if a list of packages are installed. 
        If not, prompt the user in the terminal to install them.
        """
        missing = []
        for pkg in package_names:
            # Handle special names (e.g. pyopenjtalk-plus is the pip name but import name is pyopenjtalk)
            import_name = pkg
            if pkg == "pyopenjtalk-plus":
                import_name = "pyopenjtalk"
            elif pkg == "nvidia-ml-py":
                import_name = "pynvml"
            
            try:
                __import__(import_name)
            except ImportError:
                missing.append(pkg)
        
        if not missing:
            return True
        
        print(f"\n[LunaVox] Missing dependencies for {language_name} support: {', '.join(missing)}")
        
        # --- AUTO INSTALL CHECK ---
        import os
        if auto_install or os.environ.get("LUNAVOX_AUTO_INSTALL") == "1":
            if not auto_install:
                 print("LUNAVOX_AUTO_INSTALL=1 detected. Auto-installing...")
            choice = 'y'
        elif not sys.stdin.isatty():
            logger.warning(f"Non-interactive environment detected. Skipping {language_name} dependency installation.")
            print("To enable automatic installation, set LUNAVOX_AUTO_INSTALL=1")
            return False
        else:
            try:
                # Use input() which is more standard than sys.stdin.readline()
                choice = input(f"Would you like to install them now? (y/n): ").strip().lower()
            except (EOFError, Exception):
                choice = 'n'
        
        if choice == 'y':
            print(f"Installing {', '.join(missing)}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
                print(f"✓ Successfully installed {language_name} dependencies. Please restart the application if needed.")
                return True
            except Exception as e:
                print(f"FAILED to install dependencies: {e}")
                return False
        else:
            print(f"Installation skipped. {language_name} features will not work.")
            return False

dependency_manager = DependencyManager()
