#!/usr/bin/env python3
"""
Windows Executable Builder

Creates standalone EXE file for pneumonia detection desktop app.
Uses PyInstaller to bundle everything into a single executable.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def build_windows_executable():
    """Build Windows executable using PyInstaller."""

    print("Building Windows executable for Pneumonia Detection App...")

    # Ensure we're in the right directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Check if PyInstaller is available
    try:
        import PyInstaller
        print(f"Using PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("ERROR: PyInstaller not found. Install with: pip install pyinstaller")
        return False

    # Prepare build directory
    build_dir = project_root / "build"
    dist_dir = project_root / "dist"

    # Clean previous builds
    if build_dir.exists():
        shutil.rmtree(build_dir)
    if dist_dir.exists():
        shutil.rmtree(dist_dir)

    # PyInstaller command
    app_script = "windows/pneumonia_app.py"

    cmd = [
        "pyinstaller",
        "--onefile",                    # Single executable
        "--windowed",                   # No console window
        "--name", "PneumoniaDetector",  # Executable name
        "--add-data", "windows/inference_engine.py;.",  # Include inference engine
        "--hidden-import", "onnxruntime",
        "--hidden-import", "PIL",
        "--hidden-import", "torchvision.transforms",
        "--exclude-module", "matplotlib",  # Reduce size
        "--exclude-module", "jupyter",
        "--exclude-module", "IPython",
        app_script
    ]

    print("Running PyInstaller...")
    print(" ".join(cmd))

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Build successful!")

        # Check output
        exe_path = dist_dir / "PneumoniaDetector.exe"
        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024 * 1024)
            print(f"📦 Executable created: {exe_path}")
            print(f"📊 Size: {size_mb:.1f} MB")

            # Copy model files if they exist
            model_files = list(Path("windows_exports").glob("*.onnx")) if Path("windows_exports").exists() else []
            if model_files:
                for model_file in model_files:
                    dest = dist_dir / model_file.name
                    shutil.copy(model_file, dest)
                    print(f"📋 Copied model: {model_file.name}")

            print("\n🎉 Build complete!")
            print(f"💡 Run: {exe_path}")

            return True
        else:
            print("❌ Executable not found after build")
            return False

    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def create_installer_script():
    """Create a simple installer script."""

    installer_content = """@echo off
echo Installing Pneumonia Detection App...

REM Create application directory
mkdir "%USERPROFILE%\\PneumoniaDetector" 2>nul

REM Copy executable
copy "PneumoniaDetector.exe" "%USERPROFILE%\\PneumoniaDetector\\"
copy "*.onnx" "%USERPROFILE%\\PneumoniaDetector\\" 2>nul

REM Create desktop shortcut
echo [InternetShortcut] > "%USERPROFILE%\\Desktop\\Pneumonia Detector.lnk"
echo URL=file:///%USERPROFILE%/PneumoniaDetector/PneumoniaDetector.exe >> "%USERPROFILE%\\Desktop\\Pneumonia Detector.lnk"

echo.
echo ✅ Installation complete!
echo 📱 Desktop shortcut created
echo 📂 App installed to: %USERPROFILE%\\PneumoniaDetector
echo.
pause
"""

    installer_path = Path("dist/install.bat")
    installer_path.write_text(installer_content)
    print(f"📦 Installer script created: {installer_path}")


def main():
    """Main build function."""

    print("=" * 50)
    print("Windows Executable Builder")
    print("=" * 50)

    # Build executable
    success = build_windows_executable()

    if success:
        # Create installer
        create_installer_script()

        print("\n" + "=" * 50)
        print("📦 PACKAGING COMPLETE")
        print("=" * 50)
        print("Files created:")
        print("• dist/PneumoniaDetector.exe - Main application")
        print("• dist/install.bat - Simple installer")
        print("• dist/*.onnx - Model files (if available)")
        print("\nTo distribute:")
        print("1. Copy entire dist/ folder to target machine")
        print("2. Run install.bat as administrator")
        print("3. Launch from desktop shortcut")

    else:
        print("\n❌ Build failed. Check error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()