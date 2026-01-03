from pathlib import Path
from tests.examples.utils import run_example_script

def test_audio_detection_examples():
    examples_dir = Path(__file__).parent.parent.parent.parent.parent / "examples"
    script = examples_dir / "from_docs" / "tutorials" / "audio_detection_examples.py"
    run_example_script(script)
