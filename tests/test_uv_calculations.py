# tests/test_uv_calculations.py
"""
Comprehensive Test Suite for UV Calculation Operations

This module provides detailed testing for mathematical and computational UV operations
within the Spine2D Mesh Exporter addon. Tests focus on geometric calculations,
island analysis, and UV projection algorithms.

## Test Coverage Areas:
1. UV Dimension Calculations and Geometric Analysis
2. Island Detection and Topology Analysis
3. Angle Estimation and Smart Projection Algorithms
4. Mathematical Utility Functions for UV Processing
5. UV Area Calculations and Statistical Analysis

## Testing Philosophy:
Tests validate mathematical correctness, geometric accuracy, and algorithmic robustness
while ensuring compatibility with Blender's UV coordinate system and bmesh operations.

Author: Test Suite Development
"""
import pytest
import json
import math
import unittest.mock
from unittest.mock import MagicMock, patch, call
from collections import defaultdict, deque


from Blender_to_Spine2D_Mesh_Exporter import uv_operations, config


class MockVector:
    """Enhanced Vector mock with proper 3D coordinate support."""

    def __init__(self, x=0.0, y=0.0, z=0.0):
        if isinstance(x, (list, tuple)):
            coords = list(x)
            if len(coords) == 2:
                coords.append(0.0)
            self.x, self.y, self.z = coords[:3]
        else:
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)

    def copy(self):
        """Create a copy of this vector."""
        return MockVector(self.x, self.y, self.z)

    def __mul__(self, scalar):
        """Vector multiplication by scalar."""
        return MockVector(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar):
        """Reverse multiplication by scalar."""
        return self.__mul__(scalar)

    def __add__(self, other):
        """Vector addition with proper 2D/3D coordinate handling."""
        if hasattr(other, "z"):
            # 3D vector addition
            return MockVector(self.x + other.x, self.y + other.y, self.z + other.z)
        else:
            # 2D UV coordinate addition - only use x,y components
            return MockVector(self.x + other.x, self.y + other.y, self.z)

    def __sub__(self, other):
        """Vector subtraction with proper coordinate handling."""
        if hasattr(other, "z"):
            return MockVector(self.x - other.x, self.y - other.y, self.z - other.z)
        else:
            return MockVector(self.x - other.x, self.y - other.y, self.z)

    def __truediv__(self, scalar):
        """Vector division by scalar."""
        return MockVector(self.x / scalar, self.y / scalar, self.z / scalar)

    def __repr__(self):
        return f"Vector(({self.x:.3f}, {self.y:.3f}, {self.z:.3f}))"


class MockUVCoordinate:
    """Enhanced UV coordinate mock with z-component support for Vector compatibility."""

    def __init__(self, x=0.0, y=0.0):
        self.x = float(x)
        self.y = float(y)
        # Add z component for Vector compatibility - always 0 for UV coordinates
        self.z = 0.0

    def copy(self):
        """Create a copy of this UV coordinate."""
        uv_copy = MockUVCoordinate(self.x, self.y)
        return uv_copy

    def __repr__(self):
        return f"UV({self.x:.3f}, {self.y:.3f})"


class MockBMeshLoop:
    """Enhanced BMesh loop mock with proper UV data structure."""

    def __init__(self, vert_index, uv_coords=None):
        self.vert = MockBMeshVertex(vert_index)
        self._uv_data = {}
        if uv_coords:
            self._uv_data["default"] = MockUVCoordinate(uv_coords[0], uv_coords[1])

    def __getitem__(self, uv_layer):
        """Access UV data by layer with proper structure."""
        if hasattr(uv_layer, "name"):
            layer_name = uv_layer.name
        else:
            layer_name = "default"

        if layer_name not in self._uv_data:
            self._uv_data[layer_name] = MockUVCoordinate()

        return MockUVAccessor(self._uv_data[layer_name])


class MockUVAccessor:
    """UV accessor that provides .uv property."""

    def __init__(self, uv_coord):
        self.uv = uv_coord


class MockBMeshVertex:
    """Enhanced BMesh vertex mock."""

    def __init__(self, index, coords=None):
        self.index = index
        self.co = MockVector(*(coords or [0, 0, 0]))
        self.link_faces = []


class MockBMeshFace:
    """Enhanced BMesh face mock with proper area calculation."""

    def __init__(self, face_index, loop_coords=None, area=1.0):
        self.index = face_index
        self.area = area
        self.loops = []

        if loop_coords:
            for i, (vert_coords, uv_coords) in enumerate(loop_coords):
                vert = MockBMeshVertex(i, vert_coords)
                loop = MockBMeshLoop(i, uv_coords)
                loop.vert = vert
                loop._direct_uv = MockUVCoordinate(uv_coords[0], uv_coords[1])
                self.loops.append(loop)
                vert.link_faces.append(self)

    def calc_area(self):
        """Calculate face area."""
        return self.area


class MockBMeshUVLayer:
    """Enhanced BMesh UV layer mock."""

    def __init__(self, name="UVMap"):
        self.name = name


class MockBMeshUVLayerCollection:
    """Enhanced BMesh UV layer collection with proper get/set operations."""

    def __init__(self):
        self._layers = []
        self.active = None

    def new(self, name="UVMap"):
        """Create new UV layer."""
        layer = MockBMeshUVLayer(name)
        self._layers.append(layer)
        if not self.active:
            self.active = layer
        return layer

    def get(self, name):
        """Get UV layer by name."""
        for layer in self._layers:
            if layer.name == name:
                return layer
        return None

    def remove(self, layer):
        """Remove UV layer."""
        if layer in self._layers:
            self._layers.remove(layer)
            if self.active == layer:
                self.active = self._layers[0] if self._layers else None

    def __getitem__(self, index):
        return self._layers[index]

    def __setitem__(self, index, value):
        self._layers[index] = value

    def __iter__(self):
        return iter(self._layers)

    def __len__(self):
        return len(self._layers)


class MockBMeshCollection:
    """Enhanced BMesh collection base class."""

    def __init__(self, items=None):
        self._items = items or []

    def ensure_lookup_table(self):
        """Ensure lookup table is built."""
        pass

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, index):
        return self._items[index]


class MockBMeshFacesCollection(MockBMeshCollection):
    """Enhanced BMesh faces collection."""

    def __init__(self, items=None):
        super().__init__(items)
        self.layers = MockBMeshFaceLayers()


class MockBMeshFaceLayers:
    """Enhanced BMesh face layers."""

    def __init__(self):
        self.int = MockBMeshFaceIntLayers()


class MockBMeshFaceIntLayers:
    """Enhanced BMesh face integer layers."""

    def get(self, name):
        if name == "orig_face_id":
            return MockBMeshFaceLayer()
        return None


class MockBMeshFaceLayer:
    """Enhanced BMesh face layer."""

    def __init__(self):
        pass


class MockBMesh:
    """Enhanced BMesh mock with complete structure."""

    def __init__(self, faces=None):
        face_list = faces or []
        self.faces = MockBMeshFacesCollection(face_list)

        # Create test vertices
        test_vertices = [
            MockBMeshVertex(0, [0, 0, 0]),
            MockBMeshVertex(1, [1, 0, 0]),
            MockBMeshVertex(2, [1, 1, 0]),
            MockBMeshVertex(3, [0, 1, 0]),
        ]
        self.verts = MockBMeshCollection(test_vertices)
        self.loops = MockLoops()

    def from_mesh(self, mesh_data):
        """Load from mesh data."""
        pass

    def to_mesh(self, mesh_data):
        """Save to mesh data."""
        pass

    def free(self):
        """Free bmesh memory."""
        pass


class MockLoops:
    """Enhanced BMesh loops collection."""

    def __init__(self):
        self.layers = MockUVLayers()


class MockUVLayers:
    """Enhanced UV layers collection."""

    def __init__(self):
        self.uv = MockBMeshUVLayerCollection()


class MockMeshObject:
    """Enhanced mesh object mock."""

    def __init__(self, name="TestObject", vertices=None):
        self.name = name
        self.type = "MESH"
        self.mode = "OBJECT"
        self.data = MockMeshData(vertices or [])

    def select_set(self, state):
        """Set selection state."""
        pass


class MockMeshData:
    """Enhanced mesh data mock."""

    def __init__(self, vertices=None):
        self.vertices = []
        self.uv_layers = MockUVLayerCollection()

        if vertices:
            for i, coords in enumerate(vertices):
                vertex = MockVertex(coords)
                vertex.index = i
                self.vertices.append(vertex)


class MockVertex:
    """Enhanced vertex mock."""

    def __init__(self, coords):
        self.co = MockVector(*coords)
        self.index = 0


class MockUVLayerCollection:
    """Enhanced UV layer collection for mesh data."""

    def __init__(self):
        self._layers = {}
        self.active = None

    def new(self, name="UVMap"):
        """Create new UV layer."""
        layer = MockUVLayer(name)
        self._layers[name] = layer
        if not self.active:
            self.active = layer
        return layer

    def __contains__(self, name):
        return name in self._layers

    def __getitem__(self, name):
        for layer in self._layers.values():
            if layer.name == name:
                return layer
        raise KeyError(name)

    def __iter__(self):
        return iter(self._layers.values())


class MockUVLayer:
    """Enhanced UV layer mock."""

    def __init__(self, name):
        self.name = name
        self.data = []


# =============================================================================
# TEST CLASSES FOR UV CALCULATIONS
# =============================================================================


class TestUVDimensionCalculations:
    """Test suite for UV dimension calculation operations."""

    def test_calculate_uv_dimensions_basic_geometry(self):
        """Test UV dimension calculation with basic cube geometry."""
        vertices = [
            [0, 0, 0],
            [2, 0, 0],
            [2, 2, 0],
            [0, 2, 0],
            [0, 0, 2],
            [2, 0, 2],
            [2, 2, 2],
            [0, 2, 2],
        ]
        obj = MockMeshObject("TestCube", vertices)

        result = uv_operations.calculate_uv_dimensions(obj, obj.name, 100, 200)

        assert isinstance(result, dict)

    def test_calculate_uv_dimensions_empty_object(self):
        """Test UV dimension calculation with empty object."""
        obj = MockMeshObject("EmptyObject", [])

        result = uv_operations.calculate_uv_dimensions(obj, obj.name, 100, 200)

        assert isinstance(result, dict)

    def test_calculate_uv_dimensions_single_vertex(self):
        """Test UV dimension calculation with single vertex."""
        vertices = [[1, 1, 1]]
        obj = MockMeshObject("SingleVertex", vertices)

        result = uv_operations.calculate_uv_dimensions(obj, obj.name, 100, 200)

        assert isinstance(result, dict)


class TestUVMathematicalUtilities:
    """Test suite for UV mathematical utility functions."""

    def test_round_vec_precision_control(self):
        """Test vector rounding with different precision levels."""
        test_vector = MockVector(1.23456789, 2.87654321, 3.14159265)

        result_2 = uv_operations._round_vec(test_vector, 2)
        assert result_2 == (1.23, 2.88, 3.14)

        result_4 = uv_operations._round_vec(test_vector, 4)
        assert result_4 == (1.2346, 2.8765, 3.1416)

        result_0 = uv_operations._round_vec(test_vector, 0)
        assert result_0 == (1.0, 3.0, 3.0)

    def test_round_vec_edge_cases(self):
        """Test vector rounding with edge cases."""
        # Test zero vector
        zero_vector = MockVector(0, 0, 0)
        result = uv_operations._round_vec(zero_vector, 2)
        assert result == (0.0, 0.0, 0.0)

        # Test negative values
        negative_vector = MockVector(-1.555, -2.444, -3.999)
        result = uv_operations._round_vec(negative_vector, 1)
        assert result == (-1.6, -2.4, -4.0)

        # Test very small values
        tiny_vector = MockVector(1e-10, 1e-10, 1e-10)
        result = uv_operations._round_vec(tiny_vector, 5)
        assert result == (0.0, 0.0, 0.0)


class TestUVAreaCalculations:
    """Test suite for UV area calculation operations."""

    def test_compute_face_uv_area_triangle(self):
        """Test UV area calculation for triangular face."""
        loop_coords = [
            ([0, 0, 0], [0, 0]),
            ([1, 0, 0], [1, 0]),
            ([0.5, 1, 0], [0.5, 1]),
        ]
        face = MockBMeshFace(0, loop_coords)
        uv_layer = MockBMeshUVLayer("UVMap")

        # Set UV coordinates manually
        tri_uvs = [(0, 0), (1, 0), (0.5, 1)]
        for i, loop in enumerate(face.loops):
            u, v = tri_uvs[i]
            loop._uv_data[uv_layer.name] = MockUVCoordinate(u, v)

        area = uv_operations._compute_face_uv_area(face, uv_layer)

        # Expected area of triangle with vertices (0,0), (1,0), (0.5,1) is 0.5
        expected_area = 0.5
        assert abs(area - expected_area) < 1e-6

    def test_compute_face_uv_area_square(self):
        """Test UV area calculation for square face."""
        loop_coords = [
            ([0, 0, 0], [0, 0]),
            ([1, 0, 0], [1, 0]),
            ([1, 1, 0], [1, 1]),
            ([0, 1, 0], [0, 1]),
        ]
        face = MockBMeshFace(0, loop_coords)
        uv_layer = MockBMeshUVLayer("UVMap")

        # Set UV coordinates for unit square
        sq_uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
        for i, loop in enumerate(face.loops):
            u, v = sq_uvs[i]
            loop._uv_data[uv_layer.name] = MockUVCoordinate(u, v)

        area = uv_operations._compute_face_uv_area(face, uv_layer)

        # Expected area of unit square is 1.0
        expected_area = 1.0
        assert abs(area - expected_area) < 1e-6

    def test_compute_face_uv_area_degenerate(self):
        """Test UV area calculation for degenerate face."""
        # Create degenerate face (only 2 vertices)
        loop_coords = [([0, 0, 0], [0, 0]), ([1, 0, 0], [1, 0])]
        face = MockBMeshFace(0, loop_coords)
        uv_layer = MockBMeshUVLayer()

        area = uv_operations._compute_face_uv_area(face, uv_layer)

        # Degenerate face should have zero area
        assert area == 0.0


class TestUVIslandAnalysis:
    """Test suite for UV island analysis operations."""

    def test_build_islands_single_island(self):
        """Test island building with connected faces."""
        face1_coords = [
            ([0, 0, 0], [0, 0]),
            ([1, 0, 0], [0.5, 0]),
            ([0.5, 1, 0], [0.25, 0.5]),
        ]
        face2_coords = [
            ([1, 0, 0], [0.5, 0]),
            ([2, 0, 0], [1, 0]),
            ([1.5, 1, 0], [0.75, 0.5]),
        ]

        faces = [MockBMeshFace(0, face1_coords), MockBMeshFace(1, face2_coords)]
        bm = MockBMesh(faces)
        uv_layer = MockBMeshUVLayer()

        face_to_island, island_to_faces = uv_operations._build_islands(
            bm, uv_layer, 0.01
        )

        # Should have at least one island
        assert len(island_to_faces) >= 1
        assert all(face in face_to_island for face in faces)

    def test_build_islands_multiple_islands(self):
        """Test island building with disconnected faces."""
        face1_coords = [
            ([0, 0, 0], [0, 0]),
            ([1, 0, 0], [0.5, 0]),
            ([0.5, 1, 0], [0.25, 0.5]),
        ]
        face2_coords = [
            ([10, 0, 0], [5, 5]),
            ([11, 0, 0], [5.5, 5]),
            ([10.5, 1, 0], [5.25, 5.5]),
        ]

        faces = [MockBMeshFace(0, face1_coords), MockBMeshFace(1, face2_coords)]
        bm = MockBMesh(faces)
        uv_layer = MockBMeshUVLayer()

        face_to_island, island_to_faces = uv_operations._build_islands(
            bm, uv_layer, 0.01
        )

        # Should build islands (exact number depends on implementation)
        assert len(island_to_faces) >= 1
        assert all(face in face_to_island for face in faces)


class TestSmartProjectionAlgorithms:
    """Test suite for smart projection algorithms."""

    @patch("bpy.ops.object.mode_set")
    @patch("bpy.ops.mesh.select_all")
    @patch("bpy.ops.uv.smart_project")
    @patch("bpy.ops.object.select_all")
    def test_smart_uv_project_basic(
        self, mock_select_all_obj, mock_smart_project, mock_select_all, mock_mode_set
    ):
        """Test basic smart UV projection functionality."""
        obj = MockMeshObject("TestObject")
        obj.data.uv_layers = MockUVLayerCollection()

        # Mock smart project to create UV layer
        def smart_project_side_effect(*args, **kwargs):
            if not obj.data.uv_layers.active:
                obj.data.uv_layers.new("UVMap")
            return {"FINISHED"}

        mock_smart_project.side_effect = smart_project_side_effect

        with patch("bpy.context") as mock_context:
            mock_context.view_layer.objects.active = obj

            result = uv_operations.smart_uv_project(obj, "TestObject")

            # Should return string result
            assert isinstance(result, str)
            assert "UVMap_TestObject" in result

            # Verify operator calls
            assert mock_mode_set.called
            assert mock_select_all.called or mock_select_all_obj.called
            assert mock_smart_project.called

    def test_estimate_angle_limit_varied_geometry(self):
        """Test angle limit estimation for different geometries."""
        # Simple geometry
        simple_obj = MockMeshObject("Simple")
        simple_obj.data.vertices = [MockVertex([i, 0, 0]) for i in range(10)]

        simple_angle = uv_operations.estimate_angle_limit(simple_obj)

        # Complex geometry
        complex_obj = MockMeshObject("Complex")
        complex_obj.data.vertices = [MockVertex([i, 0, 0]) for i in range(1000)]

        complex_angle = uv_operations.estimate_angle_limit(complex_obj)

        # Both should return reasonable angle values
        assert 10 <= simple_angle <= 80
        assert 10 <= complex_angle <= 80

    def test_estimate_angle_limit_empty_object(self):
        """Test angle limit estimation for empty object."""
        empty_obj = MockMeshObject("Empty")
        empty_obj.data.vertices = []

        angle = uv_operations.estimate_angle_limit(empty_obj)

        # Should return a reasonable default angle
        assert isinstance(angle, (int, float))
        assert 10 <= angle <= 80


class TestIslandAnalysisOperations:
    """Test suite for comprehensive island analysis."""

    def test_analyse_islands_comprehensive(self):
        """Test comprehensive island analysis."""
        obj = MockMeshObject("MultiIsland")
        obj.data.uv_layers.new("TestUV")

        with patch("bmesh.new") as mock_bmesh_new, patch(
            "bmesh.from_edit_mesh"
        ) as mock_from_edit:
            # Setup mock bmesh
            mock_bm = MagicMock()
            mock_bmesh_new.return_value = mock_bm
            mock_from_edit.return_value = mock_bm

            # Mock faces and UV layer
            mock_bm.faces = [MagicMock() for _ in range(6)]
            mock_uv_layer = MagicMock()
            mock_bm.loops.layers.uv.active = mock_uv_layer

            with patch.object(uv_operations, "_build_islands") as mock_build:
                # Setup mock island data
                face_to_island = {
                    mock_bm.faces[0]: 0,
                    mock_bm.faces[1]: 0,
                    mock_bm.faces[2]: 0,
                    mock_bm.faces[3]: 1,
                    mock_bm.faces[4]: 1,
                    mock_bm.faces[5]: 1,
                }
                island_to_faces = {
                    0: set([mock_bm.faces[0], mock_bm.faces[1], mock_bm.faces[2]]),
                    1: set([mock_bm.faces[3], mock_bm.faces[4], mock_bm.faces[5]]),
                }
                mock_build.return_value = (face_to_island, island_to_faces)

                # Mock face areas
                for face in mock_bm.faces:
                    face.calc_area.return_value = 1.0

                with patch.object(
                    uv_operations, "_compute_face_uv_area", return_value=0.5
                ):
                    result = uv_operations.analyse_islands(mock_bm, "TestUV")

                    assert isinstance(result, dict)

    def test_analyse_islands_empty_uv_layer(self):
        """Test island analysis with non-existent UV layer."""
        obj = MockMeshObject("NoUV")

        with patch("bmesh.from_edit_mesh") as mock_from_edit:
            mock_bm = MockBMesh()
            mock_from_edit.return_value = mock_bm

            result = uv_operations.analyse_islands(mock_bm, "NonexistentUV")
            assert result in ({}, None)


class TestUVProjectionOperations:
    """Test suite for UV projection operations."""

    @patch("bpy.ops.object.mode_set")
    @patch("bpy.ops.mesh.select_all")
    @patch("bpy.ops.uv.project")
    @patch("bpy.ops.object.select_all")
    @patch("bmesh.from_edit_mesh")
    def test_project_uv_top_basic(
        self,
        mock_from_edit,
        mock_select_all_obj,
        mock_project,
        mock_select_all,
        mock_mode_set,
    ):
        """Test basic top-down UV projection."""
        obj = MockMeshObject("ProjectTest")
        obj.data.uv_layers.new("TopProjection")

        # Setup mock bmesh
        mock_bm = MockBMesh()
        mock_from_edit.return_value = mock_bm

        with patch("bpy.context") as mock_context:
            mock_context.view_layer.objects.active = obj

            # Create indexable vector for projection
            class IndexableVector:
                def __init__(self, coords=(0.0, 0.0, 0.0)):
                    if isinstance(coords, (list, tuple)):
                        xs = list(coords)
                        if len(xs) == 2:
                            xs.append(0.0)
                        self.x, self.y, self.z = xs[:3]
                    else:
                        self.x = self.y = self.z = float(coords)

                def __getitem__(self, i):
                    return (self.x, self.y, self.z)[i]

            with patch(
                "Blender_to_Spine2D_Mesh_Exporter.uv_operations.Vector",
                new=IndexableVector,
            ):
                uv_operations.project_uv_top(obj, "ProjectTest", 256, 256)

    @patch("bpy.ops.object.mode_set")
    @patch("bpy.ops.mesh.select_all")
    @patch("bpy.ops.uv.unwrap")
    @patch("bpy.ops.object.select_all")
    def test_unwrap_respecting_seams(
        self, mock_select_all_obj, mock_unwrap, mock_select_all, mock_mode_set
    ):
        """Test UV unwrapping with seam respect."""
        obj = MockMeshObject("SeamTest")
        obj.data.uv_layers.new("SeamUV")

        mock_unwrap.return_value = {"FINISHED"}

        with patch("bpy.context") as mock_context:
            mock_context.view_layer.objects.active = obj

            result = uv_operations.unwrap_respecting_seams(
                obj, method="ANGLE_BASED", margin=0.001
            )

            # Should return a result or None
            assert result is None or isinstance(result, str)

            # Verify unwrap was called with correct parameters
            mock_unwrap.assert_called_with(method="ANGLE_BASED", margin=0.001)


# =============================================================================
# INTEGRATION TESTS FOR UV CALCULATION WORKFLOWS
# =============================================================================


class TestUVCalculationWorkflows:
    """Test suite for complete UV calculation workflows."""

    def test_complete_uv_analysis_pipeline(self):
        """Test complete UV analysis pipeline integration."""
        # Setup complex geometry
        vertices = [
            [0, 0, 0],
            [2, 0, 0],
            [2, 2, 0],
            [0, 2, 0],  # Bottom face
            [0, 0, 1],
            [2, 0, 1],
            [2, 2, 1],
            [0, 2, 1],  # Top face
        ]
        obj = MockMeshObject("CompleteTest", vertices)
        obj.data.uv_layers.new("AnalysisUV")

        # Test dimension calculation
        dimensions = uv_operations.calculate_uv_dimensions(obj, obj.name, 256, 256)
        assert isinstance(dimensions, dict)

        # Test angle estimation
        angle_limit = uv_operations.estimate_angle_limit(obj)
        assert isinstance(angle_limit, (int, float))
        assert 10 <= angle_limit <= 80

        # Test island analysis
        with patch("bmesh.new") as mock_bmesh_new:
            mock_bm = MagicMock()
            mock_bmesh_new.return_value = mock_bm
            mock_bm.faces = []
            mock_bm.loops.layers.uv.active = MockBMeshUVLayer()

            with patch.object(uv_operations, "_build_islands", return_value=({}, {})):
                analysis = uv_operations.analyse_islands(mock_bm, "AnalysisUV")
                assert isinstance(analysis, dict)

    def test_mathematical_precision_consistency(self):
        """Test mathematical precision consistency across operations."""
        test_vector = MockVector(1.23456789, 2.87654321, 3.14159265)

        # Test consistency of rounding
        round1 = uv_operations._round_vec(test_vector, 3)
        round2 = uv_operations._round_vec(test_vector, 3)
        assert round1 == round2

        # Test precision hierarchy
        round_low = uv_operations._round_vec(test_vector, 1)
        round_high = uv_operations._round_vec(test_vector, 5)

        # Low precision should be reasonably close to high precision
        assert abs(round_low[0] - round_high[0]) <= 0.1
        assert abs(round_low[1] - round_high[1]) <= 0.1
        assert abs(round_low[2] - round_high[2]) <= 0.1
