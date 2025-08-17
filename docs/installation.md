# Installation Guide

Before diving into manual steps, note that you can **download a ready-to-use release archive** from GitHub:  
👉 [Latest Release on GitHub](https://github.com/maximsokal/Blender_to_Spine_2D_Mesh_Export_Addon/releases)

Simply download the `.zip` file from the release page and install it directly in Blender (see Method 1, Step 3 below).  
This is the easiest way for most users.

---

This addon also supports multiple installation methods to accommodate different workflows. Choose the method that best fits your environment.

## System Requirements

- **Blender Version**: 4.4.0 or newer
- **Operating System Compatibility**: This project has been tested **only on Windows**.
Functionality on macOS and Linux is **not guaranteed**.
If you encounter any issues on other operating systems, please feel free to test, fix, and submit a pull request. Contributions are welcome!
- **Storage**: 50MB free space for addon files
- **Memory**: 4GB RAM minimum (8GB+ recommended for complex meshes)

---

## Method 1: Manual Installation (Standard Users)

This method is recommended for most users who want a straightforward installation process.

### Step 1: Download Project Source
1. **Download Repository**:
   - Go to the project repository page
   - Click **Code → Download ZIP**
   - Save archive to accessible location (e.g., `Downloads` folder)

2. **Extract Archive**:
   - Extract the downloaded ZIP file
   - Navigate to the extracted project folder
   - Locate the `Blender_to_Spine_2D_Mesh_Export_Addon` subdirectory

### Step 2: Create Addon Archive
1. **Select Addon Folder**:
   - Navigate to the `Blender_to_Spine_2D_Mesh_Export_Addon` folder (NOT the root project folder)
   - This folder contains the core addon files: `__init__.py`, `main.py`, `ui.py`, etc.

2. **Create ZIP Archive**:
   - **Windows**: Right-click on `Blender_to_Spine_2D_Mesh_Export_Addon` folder → **Send to → Compressed folder**

3. **Verify Archive Contents**:
   - Open the created ZIP file
   - Ensure it contains a `Blender_to_Spine_2D_Mesh_Export_Addon` folder with Python files inside
   - Critical files must include: `__init__.py`, `blender_manifest.toml`, `main.py`, `ui.py`, etc.

### Step 3: Install in Blender
1. **Open Blender Preferences**:
   - Launch Blender 4.4.0+
   - Navigate to **Edit → Preferences**
   - Select **Add-ons** tab

2. **Install Addon**:
   - Click **Install...** button (top-right corner)
   - Browse to your created ZIP file
   - Select the ZIP and click **Install Add-on**

3. **Enable Addon**:
   - Search for "Model to Spine2D Mesh" in the addon list
   - Check the checkbox to enable the addon
   - Verify installation by looking for the panel in **3D View → Sidebar → Spine2D Mesh Exporter**

---

## Method 2: Automated Build Process (Developers)

This method is recommended for developers, contributors, or users who need optimized builds with cleaned source code.

### Prerequisites
- **Python 3.11** installed system-wide
- **Command-line access** (Terminal, Command Prompt, or PowerShell)
- **Git** (optional, for repository cloning)

### Step 1: Setup Development Environment
1. **Clone or Download Repository**:
   ```bash
   # Option A: Clone with Git
   git clone [repository-url]
   cd Blender_to_Spine_2D_Mesh_Export_Addon

   # Option B: Download and extract ZIP as in Method 1
   ```

2. **Install Python Dependencies**:
   ```bash
   # Navigate to project root directory
   cd /path/to/Blender_to_Spine_2D_Mesh_Export_Addon

   # Install required packages
   pip install libcst black isort fake-bpy-module-4.1
   ```

### Step 2: Execute Build Script
1. **Run Package Builder**:
   ```bash
   # From project root directory
   python tools/prepare_package.py
   ```

2. **Build Process Overview**:
   The script performs the following operations:
   - **File Selection**: Copies only production-ready files (excludes tests, documentation)
   - **Code Optimization**: Removes debug statements and development artifacts
   - **Validation**: Verifies syntax and structure of processed files
   - **Archive Creation**: Generates optimized ZIP file ready for distribution

3. **Monitor Build Output**:
   ```
   1. Cleaning source: __init__.py
   2. Processing file: main.py
   3. Validating final source: ui.py
   Creating archive: Model_to_Spine2D_Mesh.zip
   ✅ Successfully created Model_to_Spine2D_Mesh.zip
   Build finished.
   ```

### Step 3: Install Generated Archive
1. **Locate Build Output**:
   - Find the generated ZIP file in your project directory
   - Filename format: `Model_to_Spine2D_Mesh.zip`

2. **Install in Blender**:
   - Follow **Step 3** from Method 1 above
   - Use the generated optimized ZIP file

---

## Troubleshooting Installation Issues

### Common Problems and Solutions

#### Error: "Add-on not compatible with Blender version"
- **Cause**: Blender version below 4.4.0
- **Solution**: Upgrade to Blender 4.4.0 or newer

#### Error: "No module named 'Blender_to_Spine_2D_Mesh_Export_Addon'"
- **Cause**: Incorrect ZIP structure or missing files
- **Solution**:
  1. Recreate ZIP ensuring `Blender_to_Spine_2D_Mesh_Export_Addon` folder is the top-level directory
  2. Verify `__init__.py` and `blender_manifest.toml` are present

#### Error: "Failed to enable add-on"
- **Cause**: Missing dependencies or corrupted files
- **Solution**:
  1. Check Blender console for detailed error messages
  2. Try Method 2 (automated build) for optimized installation
  3. Verify all required files are present in ZIP

#### Build Script Errors (Method 2)
- **Missing Dependencies**: Install required Python packages
  ```bash
  pip install --upgrade libcst black isort fake-bpy-module-4.1
  ```
- **Permission Errors**: Run command prompt as administrator (Windows) or use `sudo` (Linux/macOS)
- **Path Issues**: Ensure current directory is project root when running script

### Verification Steps
1. **Check Addon Panel**: Look for "Spine2D Mesh Exporter" in 3D View sidebar
2. **Test Basic Functionality**: Select a mesh object and verify export controls appear
3. **Console Output**: Check Blender console for loading confirmation messages
4. **Preferences Validation**: Verify addon appears in **Edit → Preferences → Add-ons**

### Getting Help
If installation issues persist:
1. **Check Console Output**: Blender's console often provides detailed error information
2. **Review System Requirements**: Ensure all prerequisites are met
3. **Report Issues**: Use project's issue tracker with system details and error messages

---

## Post-Installation Configuration

### First-Time Setup
1. **Save Blender Project**: The addon requires saved .blend files for export operations
2. **Configure Output Paths**: Set default JSON and image output directories in addon preferences
3. **Test Installation**: Try exporting a simple cube to verify functionality

### Performance Optimization
1. **Adjust Texture Size**: Start with 512px textures for testing
2. **Enable File Logging**: Configure logging in addon preferences for troubleshooting
3. **Monitor System Resources**: Ensure adequate RAM and storage for export operations

The addon is now ready for use. Refer to the [Usage Guide](usage.md) for detailed operational instructions.

