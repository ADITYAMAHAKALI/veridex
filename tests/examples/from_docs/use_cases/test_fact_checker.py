from pathlib import Path
from tests.examples.utils import run_example_script

def test_fact_checker():
    examples_dir = Path(__file__).parent.parent.parent.parent.parent / "examples"
    script = examples_dir / "from_docs" / "use_cases" / "fact_checker.py"
    run_example_script(script)
