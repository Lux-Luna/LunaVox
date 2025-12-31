import subprocess
import sys
import logging

logger = logging.getLogger(__name__)

class DependencyManager:
    @staticmethod
    def check_dependencies(package_names: list[str], language_name: str) -> bool:
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
        print(f"Would you like to install them now? (y/n): ", end="", flush=True)
        
        # We use a simple input check. Since this is in the terminal, we can use sys.stdin
        try:
            choice = sys.stdin.readline().strip().lower()
            if choice == 'y':
                print(f"Installing {', '.join(missing)}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
                print(f"Success! Please restart the application if needed.")
                return True
            else:
                print(f"Installation skipped. {language_name} features will not work.")
                return False
        except Exception as e:
            print(f"Installation failed: {e}")
            return False

dependency_manager = DependencyManager()
