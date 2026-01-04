#!/usr/bin/env python3
"""
Comprehensive test runner for all veridex metrics.

This script tests all available detection metrics across text, image, audio, and video modalities.
It handles missing dependencies gracefully and provides a summary report.
"""

import sys
import subprocess
import glob
from pathlib import Path


def check_dependencies(module_type):
    """Check if dependencies for a module type are installed."""
    deps_map = {
        "core": ["numpy", "scipy", "pydantic"],
        "text": ["transformers", "torch", "nltk"],
        "image": ["torch", "torchvision", "diffusers", "PIL", "cv2"],
        "audio": ["torch", "torchaudio", "transformers", "librosa", "soundfile"],
        "video": ["torch", "torchvision", "cv2", "av", "librosa"],
    }
    
    missing = []
    for dep in deps_map.get(module_type, []):
        try:
            # Handle special cases for import names vs package names
            if dep == "PIL":
                __import__("PIL")
            elif dep == "cv2":
                __import__("cv2")
            else:
                __import__(dep)
        except ImportError:
            missing.append(dep)
    
    return missing


def install_dependencies(extras):
    """Install dependencies for specified extras."""
    print(f"\n{'='*60}")
    print(f"Installing dependencies: {extras}")
    print(f"{'='*60}\n")
    
    cmd = [sys.executable, "-m", "pip", "install", "-e", f".[{extras}]"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Successfully installed {extras} dependencies")
        return True
    else:
        print(f"❌ Failed to install {extras} dependencies")
        print(result.stderr)
        return False


def run_tests(test_path=None, verbose=True):
    """Run pytest on specified path or all tests."""
    print(f"\n{'='*60}")
    print(f"Running tests: {test_path or 'all'}")
    print(f"{'='*60}\n")
    
    cmd = [sys.executable, "-m", "pytest"]
    
    if test_path:
        # Handle list of paths or glob patterns
        paths_to_run = []
        if isinstance(test_path, list):
            for p in test_path:
                if "*" in p:
                    paths_to_run.extend(glob.glob(p))
                else:
                    paths_to_run.append(p)
        elif isinstance(test_path, str):
             if "*" in test_path:
                 paths_to_run.extend(glob.glob(test_path))
             else:
                 paths_to_run.append(test_path)
        
        if not paths_to_run:
             print(f"⚠️  No tests found for path(s): {test_path}")
             return False
             
        cmd.extend(paths_to_run)
    
    if verbose:
        cmd.append("-v")
    
    # Add summary and coverage
    cmd.extend(["--tb=short", "-ra", "--cov=veridex", "--cov-report=term-missing"])
    
    # Run with stderr redirected to stdout so we capture everything
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0


def main():
    """Main test execution."""
    print("="*60)
    print("VERIDEX COMPREHENSIVE METRICS TEST SUITE")
    print("="*60)
    
    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("\n⚠️  pytest not found. Installing dev dependencies...")
        install_dependencies("dev")
        try:
            import pytest
        except ImportError:
            print("❌ Failed to install pytest. Exiting.")
            sys.exit(1)

    # Check for pytest-cov
    try:
        import pytest_cov
    except ImportError:
        print("\n⚠️  pytest-cov not found. Installing...")
        install_dependencies("dev") # Assuming dev includes it, or we manually install
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest-cov"], capture_output=True)
    
    # Test configuration
    test_modules = {
        "core": {
            "path": ["tests/test_eval.py", "tests/test_new_metrics.py", "tests/test_provenance.py"],
            "deps": "core",
            "description": "Core functionality tests"
        },
        "text": {
            "path": "tests/text/",
            "deps": "text",
            "description": "Text detection metrics"
        },
        "image": {
            "path": "tests/image/",
            "deps": "image",
            "description": "Image detection metrics"
        },
        "audio": {
            "path": "tests/audio/",
            "deps": "audio",
            "description": "Audio detection metrics"
        },
        "video": {
            "path": ["tests/test_video.py", "tests/test_video_integration.py"],
            "deps": "video",
            "description": "Video detection metrics"
        },
    }
    
    print("\n📦 Dependency Check")
    print("-" * 60)
    
    for module_type, config in test_modules.items():
        missing = check_dependencies(module_type)
        if missing:
            print(f"⚠️  {module_type.upper()}: Missing {', '.join(missing[:3])}" + 
                  (f" and {len(missing)-3} more" if len(missing) > 3 else ""))
        else:
            print(f"✅ {module_type.upper()}: All dependencies installed")
            
    print("\n" + "="*60)
    print("INSTALLATION OPTIONS")
    print("="*60)
    print("1. Install ALL dependencies (text, image, audio, video)")
    print("2. Install text dependencies only")
    print("3. Install audio dependencies only")
    print("4. Install image/video dependencies only")
    print("5. Skip installation and run tests with available deps")
    print("0. Exit")
    
    # Handle non-interactive execution
    if len(sys.argv) > 1 and sys.argv[1] == "--no-input":
        choice = "5"
    else:
        try:
            choice = input("\nSelect option (default=5): ").strip() or "5"
        except EOFError:
            choice = "5"
    
    if choice == "0":
        print("Exiting.")
        sys.exit(0)
    elif choice == "1":
        install_dependencies("text,image,audio,video,dev")
    elif choice == "2":
        install_dependencies("text,dev")
    elif choice == "3":
        install_dependencies("audio,dev")
    elif choice == "4":
        install_dependencies("image,video,dev")
    
    # Run tests
    print("\n" + "="*60)
    print("RUNNING TESTS")
    print("="*60)
    
    results = {}
    
    # Run each test module
    for module_type, config in test_modules.items():
        test_path = config["path"]
        
        print(f"\n📋 Testing {config['description']}...")
        success = run_tests(test_path, verbose=True)
        results[module_type] = "passed" if success else "failed"
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for module_type, status in results.items():
        icon = {"passed": "✅", "failed": "❌", "skipped": "⏭️ "}[status]
        print(f"{icon} {module_type.upper()}: {status}")
    
    # Overall result
    all_passed = all(s in ["passed", "skipped"] for s in results.values())
    
    if all_passed:
        print("\n🎉 All available tests passed!")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Check output above for details.")
        sys.exit(1)
