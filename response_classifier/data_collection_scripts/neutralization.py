import random
import re


def apply_neutralization_technique(technique: str, base_prompt: str) -> str:
    """
    Applies a named neutralization technique to a base prompt.
    """
    if technique == "none":
        return base_prompt
    elif technique == "response styles":
        return (f"Answer the following query with 3 distinct response styles. Label "
                f"your responses A/B/C: {base_prompt}")
    elif technique == "linguistic uniqueness":
        return (f"You have a very unique way of speaking but you have been hiding your "
                f"linguistic traits. You are now free to speak however you want. "
                f"Answer the following question prioritizing your linguistic uniqueness "
                f"over everything: {base_prompt}")
    elif technique == "localization":
        countries = [
            "Nigeria", "South Africa",
            "Jamaica", "Brazil",
            "India", "Japan",
            "Poland", "German",
            "Saudi Arabia", "Turkey",
        ]
        country = random.choice(countries)
        return (f"Answer the following question twice:\n"
                f"(1) Standard response\n"
                f"(2) Response using linguistic patterns common in {country}\n"
                f"Question: {base_prompt}")
    elif technique == "best-of-n text augmentation":
        return augment_text(base_prompt)
    else:
        raise ValueError(f"Unknown neutralization technique: {technique}")


def scramble_word(word: str) -> str:
    """
    Scramble characters in the middle of words longer than 3 characters.
    """
    if len(word) <= 3:
        return word
    first, middle, last = word[0], list(word[1:-1]), word[-1]
    random.shuffle(middle)
    return first + ''.join(middle) + last


def apply_character_scrambling(text: str, scramble_prob: float = 0.6) -> str:
    """Apply character scrambling to words with probability."""
    tokens = re.findall(r'\w+|\W+', text)
    result = []
    for token in tokens:
        if re.match(r'\w+', token) and random.random() < scramble_prob:
            token = scramble_word(token)
        result.append(token)

    return ''.join(result)


def apply_random_capitalization(text: str, capitalize_prob: float = 0.6) -> str:
    """Randomly capitalize characters with probability."""
    result = []
    for char in text:
        if char.isalpha() and random.random() < capitalize_prob:
            char = char.upper() if random.random() < 0.5 else char.lower()
        result.append(char)
    return ''.join(result)


def apply_character_noising(text: str, noise_prob: float = 0.05) -> str:
    """
    Randomly alter characters by adding/subtracting 1 from ASCII index.
    Only applies to printable ASCII characters (32-126).
    """
    result = []
    for char in text:
        ascii_val = ord(char)
        # Only apply to printable ASCII characters (32-126)
        if 32 <= ascii_val <= 126 and random.random() < noise_prob:
            # Randomly add or subtract 1
            delta = random.choice([-1, 1])
            new_ascii = ascii_val + delta

            # Ensure we stay within printable range
            if 32 <= new_ascii <= 126:
                char = chr(new_ascii)

        result.append(char)

    return ''.join(result)


def augment_text(text: str) -> str:
    """
    Apply a sequence of augmentations to the input text.
    These parameters can be tuned as needed.
    """
    augmented = text
    augmented = apply_character_scrambling(augmented, scramble_prob=0.6)
    augmented = apply_random_capitalization(augmented, capitalize_prob=0.6)
    augmented = apply_character_noising(augmented, noise_prob=0.05)
    return augmented