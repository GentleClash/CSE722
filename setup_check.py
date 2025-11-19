#!/usr/bin/env python
"""
Quick Setup Script for Economic Dispatch Project
Checks dependencies and environment
"""

import sys
import subprocess

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10+ required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required = [
        'pypsa',
        'gymnasium', 
        'stable_baselines3',
        'torch',
        'numpy',
        'pandas',
        'yaml'
    ]

    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"❌ {package} - NOT FOUND")
            missing.append(package)

    return len(missing) == 0, missing

def install_dependencies(missing):
    """Attempt to install missing dependencies"""
    print("\nAttempting to install missing packages...")
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Installation failed")
        print("   Please run: pip install -r requirements.txt")
        return False

def main():
    """Main setup function"""
    print("="*60)
    print("ECONOMIC DISPATCH PROJECT - SETUP CHECK")
    print("="*60)

    print("\nChecking Python version...")
    if not check_python_version():
        sys.exit(1)

    print("\nChecking dependencies...")
    deps_ok, missing = check_dependencies()

    if not deps_ok:
        print(f"\n⚠️  Missing {len(missing)} package(s)")
        response = input("\nInstall missing packages? (y/n): ")
        if response.lower() == 'y':
            if install_dependencies(missing):
                print("\n✓ Setup completed successfully!")
            else:
                sys.exit(1)
        else:
            print("\nPlease install missing packages manually:")
            print("  pip install -r requirements.txt")
            sys.exit(1)
    else:
        print("\n✓ All dependencies installed")

    print("\n" + "="*60)
    print("SETUP COMPLETE - Ready to run!")
    print("="*60)
    print("\nQuick start:")
    print("  python main.py --mode full --timesteps 100000")
    print("\nFor more options:")
    print("  python main.py --help")
    print("="*60)

if __name__ == "__main__":
    main()
