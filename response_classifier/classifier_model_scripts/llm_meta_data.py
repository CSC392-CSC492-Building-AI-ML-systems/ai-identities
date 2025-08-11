import json


def _normalize_model_name(llm_name: str) -> str:
    """
    Try to normalize model names to the canonical form used in llm_set.json.
    Many pipelines use "vendor_model" (underscore), while llm_set.json uses "vendor/model" (slash).
    This function replaces the first underscore with '/' if no slash is present.
    """
    if '/' in llm_name:
        return llm_name
    if '_' in llm_name:
        parts = llm_name.split('_', 1)
        if len(parts) == 2:
            return parts[0] + '/' + parts[1]
    return llm_name


def load_llm_meta_data(model_file_path: str = '../configs/llm_set.json') -> dict:
    """
    Loads model metadata from a JSON file and creates a mapping from
    model_name to its metadata dictionary. Uses a cache for efficiency.

    :param model_file_path: The path to the model metadata JSON file.
    :return: A dict of {model_name: {"family": ..., "branch": ...}}.
    """
    try:
        with open(model_file_path, 'r') as f:
            model_list = json.load(f)

        meta_map = {
            item['model_name']: {
                'family': item.get('model_family', 'unknown'),
                'branch': item.get('model_branch', 'unknown')
            }
            for item in model_list
        }
        return meta_map
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load model metadata from {model_file_path}. Error: {e}")
        return {}


def get_llm_family_and_branch(llm_name: str, meta_map: dict) -> tuple[str, str]:
    """
    Retrieves the family and branch for a given model name from the metadata map.

    :param llm_name: The full name of the model.
    :param meta_map: The metadata map from load_model_meta_data.
    :return: A tuple containing (family, branch).
    """
    normalized_llm_name = _normalize_model_name(llm_name)
    info = meta_map.get(normalized_llm_name)
    return info.get('family', 'unknown'), info.get('branch', 'unknown')
