# coding=utf-8
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Traditional and simplified Chinese conversion, a simplified character may correspond to multiple traditional characters."""

import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _load_char_mapping() -> tuple:
    """Load character mapping from JSON file."""
    json_path = Path(__file__).parent.parent.parent.parent / "Resources" / "Data" / "zh_char_mapping.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    simplified = data['simplified']
    traditional = data['traditional']
    
    s2t_dict = {}
    t2s_dict = {}
    for i, item in enumerate(simplified):
        s2t_dict[item] = traditional[i]
        t2s_dict[traditional[i]] = item
    
    return s2t_dict, t2s_dict


def tranditional_to_simplified(text: str) -> str:
    _, t2s_dict = _load_char_mapping()
    return "".join([t2s_dict[item] if item in t2s_dict else item for item in text])


def simplified_to_traditional(text: str) -> str:
    s2t_dict, _ = _load_char_mapping()
    return "".join([s2t_dict[item] if item in s2t_dict else item for item in text])
