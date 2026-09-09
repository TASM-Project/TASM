import os

def image_file_from_item(item: dict) -> str:
    """Extract standard image filename from JSONL item."""
    if item.get("image_path"):
        return os.path.basename(item["image_path"])
    img_id = str(item.get("image_id", ""))
    return img_id if img_id.lower().endswith(".jpg") else img_id + ".jpg"

def get_attr_value(row: dict, attr: str) -> float:
    """Safely get attribute value from CSV row, handling typos."""
    if attr in row:
        return float(row[attr])
    if attr == "BalancingElements" and "BalacingElements" in row:
        return float(row["BalacingElements"])
    raise KeyError(f"Attribute column not found: {attr}")