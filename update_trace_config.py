import json
import argparse
import sys

def update_json_object(json_path, obj_id, **updates):
    # Load existing data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Check data format
    if not isinstance(data, list):
        print("❌ The JSON file must contain a list of objects.")
        sys.exit(1)

    # Find and update the target object
    found = False
    for obj in data:
        if obj.get("id") == obj_id:
            for k, v in updates.items():
                obj[k] = v
            found = True
            break

    if not found:
        print(f"⚠️ Object with id={obj_id} not found.")
    else:
        print(f"✅ Updated object with id={obj_id}: {updates}")

    # Save back to file
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update a JSON object in a file by ID.")
    parser.add_argument("id", type=int, help="ID of the JSON object to update.")
    parser.add_argument("--update", nargs="+", help="Fields to update in the format key=value")

    args = parser.parse_args()

    if not args.update:
        print("⚠️ Please provide at least one --update key=value pair.")
        sys.exit(1)

    # Convert key=value pairs into dict
    updates = {}
    for kv in args.update:
        if "=" not in kv:
            print(f"Invalid format: {kv}, should be key=value")
            sys.exit(1)
        k, v = kv.split("=", 1)
        # Try to parse numbers, booleans, etc.
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass  # keep as string
        updates[k] = v

    update_json_object('./trace/trace_config_list.json', args.id, **updates)
