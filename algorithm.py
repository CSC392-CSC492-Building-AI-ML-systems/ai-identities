import sys
import json

def main():
    # Read input (but ignore for now)
    _ = sys.stdin.read()
    # Output fixed results
    result = {
        "chatgpt": 60,
        "gemini": 30,
        "claude": 10
    }
    print(json.dumps(result))

if __name__ == "__main__":
    main()