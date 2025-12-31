import argparse
import sys
import logging
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from personaTools import PersonaCreator

def main():
    parser = argparse.ArgumentParser(description="LunaVox Persona Solidification Tool (High Quality)")
    parser.add_argument("--name", required=True, help="Persona name (e.g. luna_en)")
    parser.add_argument("--audio", required=True, help="Path to reference .wav file")
    parser.add_argument("--text", help="Transcript (uses filename if omitted)")
    parser.add_argument("--lang", default="auto", choices=["zh", "en", "ja", "auto"], help="Language of reference audio")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--version", default="v2", choices=["v2", "v2Pro", "v2ProPlus"], help="Model version")
    
    args = parser.parse_args()
    
    try:
        creator = PersonaCreator(model_version=args.version)
        creator.create(
            character_name=args.name,
            audio_path=args.audio,
            text=args.text,
            language=args.lang,
            output_dir=args.output
        )
    except Exception as e:
        logging.error(f"Persona creation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
