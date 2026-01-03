from pathlib import Path
from tests.examples.utils import run_example_script

def test_forensic_analyzer():
    examples_dir = Path(__file__).parent.parent.parent.parent.parent / "examples"
    script = examples_dir / "from_docs" / "use_cases" / "forensic_analyzer.py"
    run_example_script(script)
