#!/usr/bin/env python3
"""
Environment & Dependency Verification Script
Smart Parking AI Platform
"""

import sys
import os
import shutil
import subprocess
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def print_status(step: str, status: bool, details: str = ""):
    symbol = "[PASS]" if status else "[FAIL]"
    print(f"{symbol} [{step:<28}] {'OK' if status else 'FAILED'} {details}")

def main():
    print("=" * 68)
    print("     SMART PARKING AI - DEVELOPMENT ENVIRONMENT CHECK     ")
    print("=" * 68)

    all_passed = True
    is_windows = sys.platform.startswith("win")

    # 1. Python Version Check
    py_ver = sys.version_info
    py_ok = py_ver.major == 3 and py_ver.minor >= 11
    print_status(
        "Python 3.11+ Runtime",
        py_ok,
        f"(Found: Python {py_ver.major}.{py_ver.minor}.{py_ver.micro})"
    )
    if not py_ok:
        all_passed = False

    # 2. Check System CLI Tools
    tools = [
        ("Git CLI", "git", ["--version"]),
        ("Node.js Runtime", "node", ["--version"]),
        ("NPM Package Manager", "npm", ["--version"]),
        ("Docker Engine", "docker", ["--version"])
    ]

    for name, cmd, args in tools:
        executable = shutil.which(cmd)
        if executable:
            try:
                # Use shell=True on Windows to execute .cmd / .bat wrappers like npm.cmd
                res = subprocess.run(
                    [cmd] + args,
                    capture_output=True,
                    text=True,
                    check=True,
                    shell=is_windows
                )
                ver = res.stdout.strip().split("\n")[0]
                print_status(name, True, f"({ver})")
            except Exception as e:
                print_status(name, False, f"(Error executing: {e})")
                all_passed = False
        else:
            print_status(name, False, f"('{cmd}' not found in PATH)")
            if cmd != "docker":
                all_passed = False

    # 3. Directory Structure Check
    project_root = Path(__file__).resolve().parent.parent
    expected_dirs = [
        "backend", "frontend", "cv_engine", "ml_engine",
        "database", "models", "datasets", "scripts", "tests", "docs", "docker"
    ]
    missing_dirs = [d for d in expected_dirs if not (project_root / d).is_dir()]
    dir_ok = len(missing_dirs) == 0
    print_status(
        "Project Directory Tree",
        dir_ok,
        "(All directories present)" if dir_ok else f"(Missing: {missing_dirs})"
    )
    if not dir_ok:
        all_passed = False

    # 4. Crucial Files Check
    expected_files = [".gitignore", ".env.example", "docker-compose.yml", "README.md"]
    missing_files = [f for f in expected_files if not (project_root / f).is_file()]
    files_ok = len(missing_files) == 0
    print_status(
        "Core Configuration Files",
        files_ok,
        "(All files present)" if files_ok else f"(Missing: {missing_files})"
    )
    if not files_ok:
        all_passed = False

    print("=" * 68)
    if all_passed:
        print(">> SUCCESS: Phase 1 Environment baseline is fully verified and ready!")
    else:
        print(">> WARNING: Some environment requirements are missing. Review errors above.")
    print("=" * 68)

    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
