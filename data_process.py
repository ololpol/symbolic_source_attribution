import os
from pathlib import Path
from typing import Dict, List, Tuple, Iterator, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class DataItem:
    meter: str
    key: str
    content: str


def iter_data_items(root_dir: Path) -> Iterator[DataItem]:
    root_dir = Path(root_dir)
    
    # Handle both file and directory paths
    files_to_process = []
    if root_dir.is_file():
        files_to_process = [root_dir]
    else:
        files_to_process = sorted(root_dir.glob("**/*"))
    
    for path in files_to_process:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f]

        #print("Lines:", lines)
        i = 0
        while i + 2 < len(lines):
            meter_line = lines[i].strip()
            key_line = lines[i + 1].strip()
            content_line = lines[i + 2].strip()

            if not meter_line.startswith("M:") or not key_line.startswith("K:"):
                i += 1
                continue

            meter = meter_line[len("M:") :]
            key = key_line[len("K:") :]
            #print("make dataitem (meter,key, content): ", meter, key, content_line[:20])
            yield DataItem(meter=meter, key=key, content=content_line)
            i += 3


def read_first_n_data_items(root_dir: str, n: int) -> List[DataItem]:
    items = []
    for item in iter_data_items(Path(root_dir)):
        items.append(item)
        if len(items) >= n:
            break
    return items


def group_data_by_meter_key(items: List[DataItem], group_by: str = "both") -> Dict[Union[Tuple[str, str], str], List[DataItem]]:
    """
    Group data items by meter, key, or both.
    
    Args:
        items: List of DataItem objects to group
        group_by: "both" (default), "meter", or "key"
    
    Returns:
        Dictionary with grouping keys and lists of DataItems
    """
    grouped: Dict[Union[Tuple[str, str], str], List[DataItem]] = {}
    
    for item in items:
        if group_by == "meter":
            key = item.meter
        elif group_by == "key":
            key = item.key
        else:  # "both"
            key = (item.meter, item.key)
        
        grouped.setdefault(key, []).append(item)
    return grouped


def save_all_data_to_file(items: List[DataItem], output_path: Path) -> None:
    """Save all data items to a single file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(f"M:{item.meter}\n")
            f.write(f"K:{item.key}\n")
            f.write(f"{item.content}\n")
            f.write("\n")  # Add a blank line between items

    print(f"Saved all {len(items)} items to {output_path}")


def save_grouped_data_to_files(grouped: Dict[Union[Tuple[str, str], str], List[DataItem]], output_dir: Path, group_by: str = "both") -> None:
    """Save grouped data to individual files, one per group."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, items in sorted(grouped.items()):
        # Create filename based on grouping type
        if group_by == "meter":
            safe_meter = key.replace("/", "_")
            filename = f"{safe_meter}.txt"
        elif group_by == "key":
            safe_key = key.replace("/", "_").replace(" ", "_")
            filename = f"{safe_key}.txt"
        else:  # "both"
            meter, key_val = key
            safe_meter = meter.replace("/", "_")
            safe_key = key_val.replace("/", "_").replace(" ", "_")
            filename = f"{safe_meter}-{safe_key}.txt"
        
        filepath = output_dir / filename
        
        with filepath.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(f"M:{item.meter}\n")
                f.write(f"K:{item.key}\n")
                f.write(f"{item.content}\n")
                f.write("\n")  # Add a blank line between items
        print(f"Saved {len(items)} items to {filepath}")


if __name__ == "__main__":
    data_dir = Path("data/data_v2")
    first_items = read_first_n_data_items(data_dir, 99999)
    
    # Save all data to a single file
    save_all_data_to_file(first_items, Path("data/all_data.txt"))
    
    # Group and save by both meter and key (default)
    grouped_both = group_data_by_meter_key(first_items, group_by="both")
    save_grouped_data_to_files(grouped_both, Path("data/grouped_both"), group_by="both")
    
    # Group and save by meter only
    grouped_meter = group_data_by_meter_key(first_items, group_by="meter")
    save_grouped_data_to_files(grouped_meter, Path("data/grouped_meter"), group_by="meter")
    
    # Group and save by key only
    grouped_key = group_data_by_meter_key(first_items, group_by="key")
    save_grouped_data_to_files(grouped_key, Path("data/grouped_key"), group_by="key")

    print("Keys:", list(grouped_key.keys())) 
    print("Meters:", list(grouped_meter.keys()))
    # Print summary
    print("\n=== Summary: Grouped by Meter and Key ===")
    for category, group in sorted(grouped_both.items()):
        meter, key = category
        print(f"Category: Meter={meter}, Key={key}, Count={len(group)}")



    