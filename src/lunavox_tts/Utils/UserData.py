import json
from pathlib import Path
from typing import Dict, Any
import logging
import importlib.resources

from .Constants import PACKAGE_NAME

logger = logging.getLogger(__name__)


class UserDataManager:
    def __init__(self, file_path: str = "./UserData.json"):
        self.file_path = Path(file_path)
        self._data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.file_path.exists():
            try:
                with self.file_path.open('r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to read user data file {self.file_path}. Using empty configuration. Error: {e}")

                return {}
        return {}

    def _save(self):
        try:
            with self.file_path.open('w', encoding='utf-8') as f:
                json.dump(self._data, f, indent=4, ensure_ascii=False)  # type: ignore
        except IOError as e:
            logger.warning(f"Failed to write user data file {self.file_path}. Error: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any):
        self._data[key] = value
        self._save()


userdata_file: str = str(importlib.resources.files(PACKAGE_NAME) / 'UserData.json')
userdata_manager = UserDataManager(file_path=userdata_file)
