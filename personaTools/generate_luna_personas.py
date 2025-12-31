import os
import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from personaTools import PersonaCreator

def main():
    creator = PersonaCreator(model_version="v2")
    
    # Task List
    tasks = [
        {
            "name": "luna_zh",
            "lang": "zh",
            "audio": REPO_ROOT / "CharacterData/audio/Chinese/大家应该都在努力着，那我们也不能偷懒，对吧？.wav"
        },
        {
            "name": "luna_en",
            "lang": "en",
            "audio": REPO_ROOT / "CharacterData/audio/English/First get into position like this, then move like that. Yep, thats it..wav"
        },
        {
            "name": "luna_ja",
            "lang": "ja",
            "audio": REPO_ROOT / "CharacterData/audio/Japanese/私たち王城堂には、二つの世界の人を満足させる責任がある。だから、依頼を受けたら、必ず最後までやり遂げるの。.wav"
        }
    ]
    
    for task in tasks:
        audio_path = str(task["audio"])
        if not os.path.exists(audio_path):
            print(f"FAILED: File not found {audio_path}")
            continue
            
        # Filename without extension as prompt text
        prompt_text = task["audio"].stem
        
        output_dir = str(REPO_ROOT / "CharacterData/character" / task["name"])
        
        creator.create(
            character_name=task["name"],
            audio_path=audio_path,
            text=prompt_text,
            language=task["lang"],
            output_dir=output_dir
        )

if __name__ == "__main__":
    main()
