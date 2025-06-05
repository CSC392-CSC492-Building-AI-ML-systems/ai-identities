import json
import os


def count_objects_in_json(filepath):
    """Count the number of top-level objects in a JSON file."""
    with open(filepath, 'r') as file:
        data = json.load(file)
    return len(data)


def main():
    # Path to the results directory
    results_dir = 'results'

    # Process each JSON file in the directory
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(results_dir, filename)
            try:
                count = count_objects_in_json(filepath)
                print(f"{filename}: {count} objects")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")


if __name__ == "__main__":
    main()