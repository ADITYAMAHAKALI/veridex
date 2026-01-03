import subprocess
import sys
import pytest
from pathlib import Path

def run_example_script(file_path: Path):
    """
    Run an example script as a subprocess and assert success.
    
    Args:
        file_path (Path): Absolute path to the script to run.
    """
    print(f"Testing example: {file_path}")
    
    if not file_path.exists():
        pytest.fail(f"Example file not found: {file_path}")

    try:
        # Run the script
        # Capture output to avoid cluttering test logs unless there's a failure
        result = subprocess.run(
            [sys.executable, str(file_path)],
            capture_output=True,
            text=True,
            # Some models might take time to download/load, providing generous timeout
            timeout=180
        )
        
        if result.returncode != 0:
            pytest.fail(
                f"Example failed with return code {result.returncode}.\n\n"
                f"STDOUT:\n{result.stdout}\n\n"
                f"STDERR:\n{result.stderr}"
            )
            
    except subprocess.TimeoutExpired:
        pytest.fail(f"Example timed out after 180 seconds: {file_path}")
    except Exception as e:
        pytest.fail(f"Failed to run example {file_path}: {str(e)}")
