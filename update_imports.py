#!/usr/bin/env python3
"""
Safe import path updater for directory restructuring.
Updates all import statements to use new Languages/ and Resources/ structure.
"""
import os
import re
from pathlib import Path

def update_imports(file_path):
    """Update imports in a single Python file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    
    # Update Audio imports
    content = re.sub(r'from \.\.Audio\.', 'from ..Resources.Audio.', content)
    content = re.sub(r'from \.\.\.Audio\.', 'from ...Resources.Audio.', content)
    
    # Update Persona imports
    content = re.sub(r'from \.\.Persona\.', 'from ..Resources.Persona.', content)
    content = re.sub(r'from \.\.\.Persona\.', 'from ...Resources.Persona.', content)
    
    # Update Data imports
    content = re.sub(r'from \.\.Data\.', 'from ..Resources.Data.', content)
    content = re.sub(r'from \.\.\.Data\.', 'from ...Resources.Data.', content)
    
    # Update Chinese imports
    content = re.sub(r'from \.\.Chinese\.', 'from ..Languages.Chinese.', content)
    content = re.sub(r'from \.\.\.Chinese\.', 'from ...Languages.Chinese.', content)
    
    # Update English imports
    content = re.sub(r'from \.\.English\.', 'from ..Languages.English.', content)
    content = re.sub(r'from \.\.\.English\.', 'from ...Languages.English.', content)
    
    # Update Japanese imports
    content = re.sub(r'from \.\.Japanese\.', 'from ..Languages.Japanese.', content)
    content = re.sub(r'from \.\.\.Japanese\.', 'from ...Languages.Japanese.', content)
    
    # Update Symbols imports
    content = re.sub(r'from \.\.Symbols\.', 'from ..Languages.Symbols.', content)
    content = re.sub(r'from \.\.\.Symbols\.', 'from ...Languages.Symbols.', content)
    
    if content != original:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    src_dir = Path(__file__).parent.parent / 'src' / 'lunavox_tts'
    updated = 0
    
    for py_file in src_dir.rglob('*.py'):
        if update_imports(py_file):
            print(f"Updated: {py_file.relative_to(src_dir)}")
            updated += 1
    
    print(f"\nTotal files updated: {updated}")

if __name__ == '__main__':
    main()
