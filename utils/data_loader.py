

import json
from pathlib import Path
from typing import Dict, List

 
def load_jsonl(path: Path) -> List[Dict]:
    """Read a .jsonl file into a list of dicts."""
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def load_json(path: Path):
  """Read a .json file."""
  with path.open("r", encoding="utf-8") as f:
      return json.load(f)

def index_docs_by_topic(docs_list: List[Dict]) -> Dict[int, List[Dict]]:
    """
    Build a mapping: topic_id -> docs (list of dicts).
    Each item in docs_list has: {"topic_id": int, "docs": [ { ... }, ... ] }
    """
    result: Dict[int, List[Dict]] = {}
    for d in docs_list:
        tid = d["topic_id"]
        result[tid] = d["docs"]
    return result