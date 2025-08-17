# Usage Guide

After installation, the addon can be found in **3D View → Sidebar → Spine2D Mesh Exporter**. This comprehensive interface provides all necessary tools for converting 3D mesh objects into Spine2D-compatible skeletal animation data.

## Main Interface Overview
![ui_addon](assets/ui_addon.png)
The addon interface is organized into collapsible sections for optimal workflow management. Each section contains specific controls and configuration options for different aspects of the export pipeline.

---

## Settings Section

### Reset Button
**Function**: Restores all addon settings to their default values
**Usage**: Click to reset texture size (1024px), angle limit (30°), enable control icons, enable preview animation, and restore default output paths
**Note**: This operation affects all global settings but does not modify per-object properties

---

## Export Configuration

### Texture Size
**Parameter**: Baked texture resolution (128px - 4096px)
**Default**: 1024px
**Performance Impact**:
- 1024px: Standard processing time (~30-60 seconds)
- 2048px: Extended processing time (~2-5 minutes)
- 4096px: Maximum resolution with significantly longer processing (~10-15 minutes)

**Critical Warning**: Texture baking is the most resource-intensive operation in the export pipeline. Higher resolutions exponentially increase processing time and memory usage.

### JSON Output Path
**Function**: Destination directory for exported JSON files
**Default**: Directory containing the current .blend file
**Requirements**:
- .blend file must be saved before export
- Directory must have write permissions
- Absolute paths are recommended for consistency

### Images Subfolder Path
**Function**: Relative path within JSON directory for texture assets
**Default**: `images/`
**Behavior**: Creates subdirectory structure for organized asset management
**Example**: If JSON path is `/project/export/` and images path is `textures/`, final texture location will be `/project/export/textures/`

### Control Icons
**Function**: Adds visual bounding box indicators to exported skeleton
**Purpose**: Provides spatial reference points in Spine2D editor for easier animation setup
**Performance**: Minimal impact on export time
**Recommendation**: Enable for complex rigs requiring precise positioning

### Preview Animation
**Function**: Generates test rotation animations for immediate skeleton validation
**Output**: Creates basic X/Y/Z rotation keyframes
**Use Case**: Quality assurance for bone hierarchy and constraint functionality
**Note**: Does not affect static mesh exports

---

## Cut Configuration

### Angle Limit
**Parameter**: Edge angle threshold for automatic segmentation (0° - 180°)
**Default**: 30°
**Algorithm**: When angle between adjacent faces exceeds threshold, creates segmentation boundary
**Precision**: Lower values create more segments; higher values preserve surface continuity
**Recommendation**: 15 - 30°

### Seam Maker Mode
**Options**:
- **Auto**: Fully automated segmentation based on angle limit
- **Custom**: User-defined seams with algorithmic validation and enhancement

**Custom Mode Workflow**:
1. Enter Edit Mode on target object
2. Select edges for manual seam placement
3. Mark seams using **Edge → Mark Seam**
4. Return to Object Mode
5. Execute export with Custom mode selected

**Important**: Custom seams undergo algorithmic validation. Additional automated cuts may be applied to ensure topological correctness and UV unwrapping compatibility.

---

## Bake Configuration

### Frames for Render
**Parameter**: Total frame count for sequence baking
**Default**: 0 (single frame baking)
**Range**: 0-250 frames
**Behavior**:
- 0: Bakes current frame only
- >0: Creates animated texture sequence

### Start Frame
**Parameter**: Beginning frame for sequence baking
**Default**: 0
**Use Case**: Allows partial sequence baking for specific animation segments
**Constraint**: Cannot exceed scene frame range

### Last Frame Display
**Function**: Calculated read-only field showing final frame of baking range
**Calculation**: `Start Frame + Frames for Render - 1`
**Purpose**: Visual confirmation of baking scope

### Playback End Constraint
**Parameter**: Scene timeline limitation (250 frames maximum)
**Function**: Prevents frame range overflow beyond scene boundaries
**Automatic Adjustment**: Values exceeding scene length are clamped to maximum available frames

---

## Object Information Panel

### Refresh Button
**Function**: Updates cached object analysis data
**Necessity**: Click after modifying mesh geometry, materials, or object properties
**Performance**: Offloads expensive calculations from real-time UI updates
**Coverage**: Refreshes vertex count, face orientation analysis, and material preview icons

### Vertex Count Analysis
**Display**: Total vertex count with performance implications
**Performance Benchmarks**:
- **<1,000 vertices**: ~30-60 seconds export time
- **1,000-5,000 vertices**: ~1-5 minutes export time
- **5,000-10,000 vertices**: ~5-10 minutes export time
- **>10,000 vertices**: Extended processing with potential viewport performance issues in Spine2D

**Optimization Note**: High vertex counts may cause performance degradation in Spine2D viewport. Consider mesh decimation for complex models.

### Material Count
**Display**: Number of materials assigned to object
**Limitation**: Multi-material objects undergo material merging during baking process
**Recommendation**: Minimize material count for optimal texture baking performance

### Face Orientation Validation
**Analysis**: Detects inverted (inside-out) faces that can cause rendering artifacts
**Display**:
- "All faces oriented correctly" (green) - No issues detected
- "Inverted faces: X / Total" (red) - Manual correction required

**Resolution**: Use **Mesh → Normals → Recalculate Outside** in Edit Mode to fix orientation issues

### Scale Application Warning
**Validation**: Ensures object transforms are applied before export
**Requirement**: Object scale must be (1.0, 1.0, 1.0)
**Resolution**: **Object → Apply → All Transforms** before export
**Critical**: Unapplied transforms cause coordinate system misalignment in exported data

---

## Export Execution

### Export Current Object Button
**Function**: Initiates complete export pipeline for active object
**Prerequisites**:
- .blend file must be saved
- Object scale must be applied
- Valid output paths configured
- Sufficient disk space available

**Process Overview**:
1. Object validation and preprocessing
2. UV layout generation and optimization
3. Mesh segmentation based on cut settings
4. Texture baking (if materials present)
5. JSON data generation and export
6. Asset cleanup and finalization

---

## Addon Preferences
![ui_preferences](assets/ui_preferences.png)
Access through **Edit → Preferences → Add-ons → Model to Spine2D Mesh**

### Information & Help Section

#### Project Website Button
**Function**: Opens addon documentation and support resources
**Target**: Links to official project repository and user guides

### Logging Configuration

#### Enable File Logging
**Function**: Activates persistent log file creation
**Default**: Disabled for performance optimization
**Recommendation**: Enable only during troubleshooting or development

#### Log File Path
**Configuration**: Absolute path for log file storage
**Default**: User home directory
**Automatic Creation**: Creates directory structure if non-existent
**File Management**: Logs are appended; manual cleanup may be required for large files

#### Module Log Levels
**Granular Control**: Independent logging levels for each addon component
**Available Levels**:
- **Error**: Critical failures only (recommended for production)
- **Warning**: Potential issues and errors
- **Info**: General operational information
- **Debug**: Comprehensive diagnostic output

**Performance Warning**: Debug level logging creates additional temporary files in both scene directory and export folder. Use only for troubleshooting specific issues.

**Module Categories**:
- **ModelToSpine2D**: Core addon functionality
- **config**: Configuration management
- **ui**: User interface operations
- **main**: Export pipeline orchestration
- **plane_cut**: Mesh segmentation algorithms
- **uv_operations**: UV layout processing
- **texture_baker**: Material baking system
- **json_export**: Spine2D format generation
- **json_merger**: Multi-object data merging
- **multi_object_export**: Batch processing workflows

### Addon Management

#### Uninstall Function
**Purpose**: Complete addon removal with cleanup
**Process**: Removes all addon files, preferences, and temporary data
**Note**: Scene-specific addon properties may persist until project reload

---

## Performance Optimization Guidelines

### System Requirements
- **RAM**: Minimum 8GB, 16GB+ recommended for high-resolution textures
- **Storage**: 500MB+ free space for temporary files during processing
- **CPU**: Multi-core processor recommended for texture baking operations

### Workflow Optimization
1. **Start with low texture resolution** (128px) for testing
2. **Apply object transforms** before beginning export process
3. **Simplify complex materials** to reduce baking time
4. **Use Custom cut mode** for precise control over mesh segmentation
5. **Monitor vertex count** to maintain Spine2D viewport performance

### Troubleshooting Performance Issues
- **Reduce texture size** if Blender becomes unresponsive
- **Disable preview animation** for faster exports
- **Use Error-level logging** to minimize diagnostic overhead
- **Close unnecessary Blender windows** during processing
- **Save project frequently** before running export operations

---

## Common Workflow Patterns

### Basic Single Object Export
1. Prepare mesh with applied transforms
2. Configure texture size (1024px recommended)
3. Set output paths
4. Choose Auto cut mode with 30° angle limit
5. Execute export

### Multi-Material Complex Model
1. Reduce material count through merging where possible
2. Increase texture resolution (2048px-4096px)
3. Use Custom cut mode for material-based segmentation
4. Enable debug logging for troubleshooting
5. Monitor system performance during baking

### Animation Sequence Export
1. Configure frame range in Bake section
2. Verify scene timeline settings
3. Ensure adequate storage space for sequence files
4. Use lower texture resolution for faster processing
5. Review sequence continuity in Spine2D editor

This interface provides comprehensive control over the 3D-to-2D conversion pipeline while maintaining workflow efficiency through intelligent defaults and performance optimizations.
