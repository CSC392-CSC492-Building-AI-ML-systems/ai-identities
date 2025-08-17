import json
import sys

def analyze_llm(json_file_path):
    """
    Analyze the uploaded JSON file and return LLM probabilities.
    For now, returns hardcoded values as requested.
    
    Args:
        json_file_path (str): Path to the uploaded JSON file
        
    Returns:
        dict: Dictionary containing analysis results
    """
    try:
        # Read and validate the JSON file
        with open(json_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            print(f"File content preview: {content[:200]}...", file=sys.stderr)  # Debug output
            data = json.loads(content)
        
        # For now, return hardcoded values as requested
        # TODO: Replace with actual analysis logic
        results = {
            "chatgpt": 60,
            "gemini": 30,
            "claude": 10
        }
        
        return {
            "success": True,
            "analysis": results,
            "message": "Analysis completed successfully"
        }
        
    except json.JSONDecodeError:
        return {
            "success": False,
            "error": "Invalid JSON file format",
            "analysis": None
        }
    except FileNotFoundError:
        return {
            "success": False,
            "error": "File not found",
            "analysis": None
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Analysis error: {str(e)}",
            "analysis": None
        }

def main():
    """
    Main function for command line usage
    """
    if len(sys.argv) != 2:
        print("Usage: python algorithm.py <json_file_path>")
        sys.exit(1)
    
    json_file_path = sys.argv[1]
    result = analyze_llm(json_file_path)
    
    # Output results as JSON for easy parsing
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()