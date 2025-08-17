# Example Projects Documentation

This collection provides comprehensive demonstration projects showcasing the capabilities and workflow patterns of the Spine2D Mesh Exporter addon. Each example is designed to highlight specific functionality while providing practical learning experiences for different use cases and complexity levels.

## Project Structure Overview

The examples directory contains three carefully crafted demonstration projects, each targeting specific aspects of the export pipeline:

```
examples/
├── 01_pyramid_basic/          # Fundamental workflow demonstration
├── 02_crystal_procedural/     # Advanced sequence baking showcase
├── 03_spine_text_multi/       # Complex multi-object processing
└── examples.md               # This documentation file
```

---

## Example 1: Basic Pyramid Export

### Project Overview
**File**: `01_pyramid_basic/cone_test.blend`

![example_cone](assets/example_cone.png)

**Complexity**: Beginner
**Primary Focus**: Fundamental export workflow and basic functionality validation
**Estimated Processing Time**: 30-60 seconds

### Technical Specifications
- **Geometry**: Minimalist pyramid with 5 vertices, 6 faces
- **Material Setup**: Single diffuse material with basic color assignment
- **UV Layout**: Simple planar unwrapping optimized for minimal distortion
- **Recommended Settings**:
  - Texture Size: 128px (optimal for speed)
  - Angle Limit: 30° (default)
  - Cut Mode: Auto
  - Bake Mode: Single frame

### Learning Objectives
1. **Workflow Fundamentals**: Master the complete export pipeline from mesh preparation to JSON output
2. **Settings Optimization**: Understand the relationship between texture resolution and processing time
3. **Quality Validation**: Learn to verify export integrity through Spine2D import testing
4. **Multi-Object Preparation**: Practice duplication techniques for multi-export scenarios

### Step-by-Step Workflow

#### Phase 1: Project Setup and Validation
1. **Open Project File**:
   - Load `cone_test.blend` in Blender 4.4.0+
   - Verify pyramid object is selected and active
   - Confirm material assignment and UV unwrapping

2. **Validate Prerequisites**:
   - Check that object scale is applied (should show 1.0, 1.0, 1.0)
   - Verify .blend file is saved
   - Confirm addon is properly installed and panel visible

3. **Configure Export Settings**:
   ```
   Texture Size: 128px
   JSON Path: [project_directory]/export/
   Images Path: textures/
   Control Icons: Enabled
   Preview Animation: Enabled
   ```

#### Phase 2: Single Object Export
1. **Execute Basic Export**:
   - Click "Export Current Object" button
   - Monitor console output for processing stages
   - Verify completion message and file generation

2. **Validate Output**:
   - Check JSON file structure and content
   - Verify texture file generation in specified directory
   - Confirm bone hierarchy creation

3. **Performance Benchmarking**:
   - Record processing time for baseline comparison
   - Note memory usage during export process
   - Document any warnings or optimization suggestions

#### Phase 3: Multi-Object Export Testing
1. **Prepare Multiple Objects**:
   - Duplicate pyramid object (Shift+D)
   - Position duplicates at different locations
   - Optionally modify materials for visual distinction

2. **Configure Multi-Export Settings**:
   - Select all pyramid objects
   - Configure connection settings if unified rig desired
   - Adjust output naming conventions

3. **Execute Multi-Export**:
   - Use multi-object export functionality
   - Monitor processing of each object
   - Verify merged JSON structure

### Expected Results
- **JSON Output**: Properly formatted Spine2D skeleton with bone hierarchy
- **Texture Assets**: 128px texture files with correct UV mapping
- **Performance**: Complete export in under 1 minute
- **Validation**: Successfully importable into Spine2D editor

### Troubleshooting Common Issues
- **Slow Performance**: Reduce texture size to 64px for faster testing
- **UV Distortion**: Check pyramid UV unwrapping quality
- **Export Failures**: Verify object scale application and material setup

---

## Example 2: Crystal Procedural Sequence

### Project Overview
**File**: `02_crystal_procedural/cristall.blend`

![example_crystal](assets/example_crystal.png)

**Complexity**: Advanced
**Primary Focus**: Procedural material sequence baking and temporal texture generation
**Estimated Processing Time**: 3-8 minutes (depending on frame count)

### Technical Specifications
- **Geometry**: Multi-faceted crystal with 200-500 vertices
- **Material Setup**: Complex procedural material with animated parameters
- **Animation Timeline**: 50-frame sequence demonstrating material evolution
- **Procedural Elements**:
  - Animated noise textures
  - Color ramps with keyframed progression
  - Dynamic normal mapping effects
  - Subsurface scattering variations

### Advanced Features Demonstrated
1. **Sequence Baking**: Frame-by-frame texture generation for animated materials
2. **Procedural Processing**: Complex node-based material evaluation
3. **Temporal Consistency**: Maintaining visual continuity across frame sequences
4. **Performance Optimization**: Efficient handling of computationally intensive materials

### Step-by-Step Workflow

#### Phase 1: Project Analysis and Setup
1. **Examine Procedural Material**:
   - Open Shading workspace
   - Analyze node tree complexity and animation keyframes
   - Understand parameter relationships and timing

2. **Configure Sequence Settings**:
   ```
   Frames for Render: 25 (recommended for testing)
   Start Frame: 0
   Texture Size: 512px (balance between quality and speed)
   Cut Mode: Auto with 45° angle limit
   ```

3. **Validate Animation Setup**:
   - Scrub timeline to verify material animation
   - Check keyframe placement and interpolation
   - Confirm procedural texture evaluation

#### Phase 2: Sequence Export Execution
1. **Prepare for Extended Processing**:
   - Save project before export
   - Close unnecessary applications to free system resources
   - Monitor available disk space for sequence files

2. **Execute Sequence Export**:
   - Configure bake settings for procedural materials
   - Initiate export process
   - Monitor frame-by-frame progress in console

3. **Processing Monitoring**:
   - Track individual frame completion times
   - Observe memory usage patterns
   - Note any performance degradation

#### Phase 3: Sequence Validation and Optimization
1. **Verify Frame Sequence**:
   - Check individual frame texture files
   - Validate temporal consistency
   - Confirm frame naming and numbering

2. **Quality Assessment**:
   - Import sequence into Spine2D for playback testing
   - Evaluate material quality and animation smoothness
   - Compare with original Blender viewport preview

3. **Performance Analysis**:
   - Document total processing time
   - Identify bottlenecks in procedural evaluation
   - Optimize settings for production workflows

### Advanced Configuration Options

#### Sequence Performance Tuning
- **High Quality (Production)**: 512px, 10-frame subset
- **Standard Quality (Testing)**: 256px, 5-frame subset
- **Fast Preview (Development)**: 128px, 2-frame subset

### Expected Results
- **Frame Sequence**: 25-50 individual texture files showing material evolution
- **JSON Structure**: Spine2D skeleton with first-frame material assignment
- **Processing Time**: 3-8 minutes depending on complexity and resolution
- **File Size**: 5-25MB total depending on texture resolution and frame count

---

## Example 3: Multi-Object Text Export

### Project Overview
**File**: `03_spine_text_multi/text_spine.blend`

![example_text_spine](assets/example_text_spine.png)

**Complexity**: Expert
**Primary Focus**: High-vertex-count processing and complex multi-object workflow management
**Estimated Processing Time**: 5-15 minutes

### Technical Specifications
- **Geometry**: Text objects converted to mesh with high vertex density
- **Text Content**: "SPINE" letters as individual mesh objects
- **Total Vertices**: 8,000-15,000 across all objects
- **Material Setup**: Simple but consistent material assignment
- **Object Count**: 5 individual letter objects plus optional background elements

### Performance Considerations
**Critical Warning**: This example demonstrates the performance limitations of high-vertex-count exports. Processing time scales non-linearly with vertex count and may require significant system resources.

### Vertex Count Impact Analysis
```
Individual Object Breakdown:
- S: 536 vertices
- P: 312 vertices
- I: 208 vertices
- N: 228 vertices
- E: 396 vertices
Total: 1680 vertices
```

### Step-by-Step Workflow

#### Phase 1: Scene Analysis and Preparation
1. **Examine Object Complexity**:
   - Use "Refresh" button to analyze each letter's vertex count
   - Document individual object specifications
   - Assess system resource requirements

2. **Configure for High-Performance Processing**:
   ```
   Texture Size: 128px (recommended for testing)
   Cut Mode: Custom (pre-optimized seams)
   Multi-Object Settings: Connect enabled for unified rig
   System Preparation: Close other applications
   ```

3. **Validate Object Readiness**:
   - Confirm all scales are applied
   - Verify material assignments
   - Check UV unwrapping quality

#### Phase 2: Sequential Object Processing
1. **Individual Object Export Testing**:
   - Start with simplest object (I letter)
   - Validate export quality and processing time
   - Use results to estimate total processing requirements

2. **Multi-Object Export Strategy**:
   - Configure connection settings for unified skeleton
   - Plan bone hierarchy and constraint relationships
   - Prepare for extended processing session

3. **Execute Full Scene Export**:
   - Initiate multi-object export process
   - Monitor individual object completion
   - Track system performance throughout process

#### Phase 3: Performance Optimization and Validation
1. **Processing Time Analysis**:
   - Document individual object processing times
   - Identify performance bottlenecks
   - Compare actual vs. estimated processing duration

2. **Quality Validation**:
   - Verify unified skeleton structure
   - Test bone hierarchy and constraints
   - Validate texture quality across all objects

3. **Spine2D Integration Testing**:
   - Import final JSON into Spine2D editor
   - Test viewport performance with high vertex count
   - Validate animation capabilities

### Performance Optimization Strategies

#### Pre-Export Optimization
1. **Mesh Decimation**: Reduce vertex count while preserving visual quality
2. **UV Optimization**: Simplify UV layouts to reduce processing overhead
3. **Material Consolidation**: Merge similar materials to reduce baking complexity

#### Processing Optimization
1. **Staged Export**: Process objects individually for better resource management
2. **Resolution Scaling**: Use lower texture resolutions for initial testing
3. **System Monitoring**: Track memory and CPU usage throughout process

#### Post-Export Optimization
1. **Asset Cleanup**: Remove unnecessary intermediate files
2. **Texture Compression**: Optimize final texture files for distribution
3. **JSON Validation**: Verify skeleton structure integrity

### Expected Results
- **Unified Skeleton**: Single Spine2D rig containing all letter objects
- **Complex Bone Hierarchy**: Sophisticated control structure for text animation
- **Processing Duration**: 5 minutes total processing time
- **File Output**: Large JSON file (1MB) with corresponding texture assets
- **Spine2D Performance**: May experience viewport lag with high vertex count

### System Requirements for Complex Processing
- **RAM**: 16GB+ recommended for smooth processing
- **Storage**: 500MB+ free space for temporary files
- **CPU**: Multi-core processor for optimal performance
- **Time Allocation**: Plan for extended processing session

---

## Cross-Example Learning Progression

### Beginner Path (Pyramid → Crystal → Text)
1. **Foundation**: Master basic export workflow with simple geometry
2. **Advanced Features**: Explore procedural materials and sequence baking
3. **Complex Scenarios**: Handle multi-object scenes with performance considerations

### Workflow Optimization Path
1. **Performance Baseline**: Establish processing time expectations
2. **Feature Integration**: Combine multiple addon capabilities
3. **Production Readiness**: Develop efficient workflows for real projects

### Troubleshooting Expertise Development
1. **Common Issues**: Identify and resolve typical export problems
2. **Performance Problems**: Optimize workflows for different hardware configurations
3. **Advanced Debugging**: Use logging and diagnostic tools effectively

---

## Best Practices Summary

### Project Management
- **Save Frequently**: Maintain project backups before complex exports
- **Resource Monitoring**: Track system performance during processing
- **Incremental Testing**: Validate settings with simple examples before complex exports

### Performance Optimization
- **Start Small**: Begin with low resolutions and simple settings
- **Scale Gradually**: Increase complexity based on system capabilities
- **Monitor Resources**: Watch memory usage and processing times

### Quality Assurance
- **Validate Early**: Test imports in Spine2D throughout development
- **Document Settings**: Record successful configuration parameters
- **Benchmark Performance**: Establish processing time expectations

These examples provide a comprehensive foundation for mastering the Spine2D Mesh Exporter addon, from basic functionality through advanced production workflows.

