"""Run live-model quality cases against the staged VSIX backend when present."""

import argparse
import importlib.util
import sys
import unittest
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-name", default="", help="Run one LiveLocalModelQualityTests method")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    packaged_root = project_root / "vscode-extension" / "runtime"
    source_root = project_root
    analyzer_root = packaged_root if (packaged_root / "backend" / "app" / "diagnose" / "agentic_investigator.py").is_file() else source_root
    test_file = project_root / "backend" / "tests" / "test_agentic_model_live.py"

    # Keep model-quality probes read-only, including Python's bytecode cache.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(analyzer_root))
    if str(project_root) not in sys.path:
        sys.path.insert(1, str(project_root))

    import backend.app.diagnose.agentic_investigator as investigator

    print(f"Using analyzer: {Path(investigator.__file__).resolve()}")
    spec = importlib.util.spec_from_file_location("rag_doctor_live_model_tests", test_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load live-model tests from {test_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    loader = unittest.defaultTestLoader
    if args.test_name:
        suite = loader.loadTestsFromName(f"LiveLocalModelQualityTests.{args.test_name}", module)
    else:
        suite = loader.loadTestsFromModule(module)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
