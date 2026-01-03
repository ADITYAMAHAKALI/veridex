from pathlib import Path
from tests.examples.utils import run_example_script

def test_text_detection_example():
    examples_dir = Path(__file__).parent.parent.parent / "examples"
    script = examples_dir / "text_detection_example.py"
    run_example_script(script)
