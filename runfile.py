import os
import sys
import subprocess


def main():
    app_file = "app.py"

    if not os.path.exists(app_file):
        print(f"Error: {app_file} not found in the current folder.")
        sys.exit(1)

    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", app_file],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print("Failed to start Streamlit app.")
        print(e)
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
