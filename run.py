import sys
import os

def print_usage():
    usage = """
Usage: python run.py [benchmark_ocr | check_model | gui]

Examples:
    python run.py benchmark_ocr
    python run.py check_model
    python run.py gui
    """
    print(usage)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "benchmark_ocr":
        os.system("python src/benchmark_ocr.py")
    elif command == "check_model":
        os.system("python src/check_model.py")
    elif command == "gui":
        os.system("python src/gui.py")
    else:
        print("Unknown command:", command)
        print_usage()