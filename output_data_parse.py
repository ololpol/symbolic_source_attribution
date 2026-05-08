import re
from pathlib import Path

def load_abc(data_path):
    with open(data_path, 'r') as f:
        data = f.read().strip()


    tokens_set = set(data.split())
    start_symbol, end_symbol = '<s>', '</s>'
    tokens_set.update({start_symbol, end_symbol})

    idx2token = list(tokens_set)
    vocab_size = len(idx2token)
    print('vocabulary size:', vocab_size)
    token2idx = dict(zip(idx2token, range(vocab_size)))
    tunes = data.split('\n\n')


    return tunes, idx2token, token2idx
def sanitize_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(" ", "_")


def collect_grouped_data(source_dir: Path):
    key_groups = {}
    meter_groups = {}

    pattern = re.compile(r"^folkrnn_v2-(.+?)-(.+)\.txt$")
    for file_path in source_dir.iterdir():
        if not file_path.is_file():
            continue

        match = pattern.match(file_path.name)
        if not match:
            continue

        key, meter = match.groups()
        loaded, _, _ = load_abc(str(file_path))
        text = "\n\n".join(loaded)

        key_groups.setdefault(key, []).append(text)
        meter_groups.setdefault(meter, []).append(text)

    return key_groups, meter_groups


def write_grouped_files(grouped_data, target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)

    for group_name, items in grouped_data.items():
        print(f"Writing group '{group_name}' with {len(items)} items to {target_dir}")
        sanitized_name = sanitize_name(group_name)
        out_path = target_dir / f"folkrnn_v2-{sanitized_name}.txt"
        out_path.write_text("\n\n".join(items), encoding="utf-8")

        
def main_script():
    source_dir = Path("data/folkrnn_both")
    key_dir = Path("data/folkrnn_key")
    meter_dir = Path("data/folkrnn_meter")

    key_groups, meter_groups = collect_grouped_data(source_dir)

    write_grouped_files(key_groups, key_dir)
    write_grouped_files(meter_groups, meter_dir)


if __name__ == "__main__":
    main_script()