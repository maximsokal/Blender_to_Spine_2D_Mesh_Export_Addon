import unittest
from unittest.mock import Mock, MagicMock, patch
import sys
import logging
import os

# Critical: Follow project's standard path setup pattern (from test_logging_integration.py)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Import the main package first, following project conventions
try:
    import Blender_to_Spine2D_Mesh_Exporter
    from Blender_to_Spine2D_Mesh_Exporter import seam_marker
except ImportError as e:
    print(f"Critical import failure: {e}")
    print(f"Project root: {project_root}")
    print(
        f"Expected addon path: {os.path.join(project_root, 'Blender_to_Spine2D_Mesh_Exporter')}"
    )
    print(
        f"Path exists: {os.path.exists(os.path.join(project_root, 'Blender_to_Spine2D_Mesh_Exporter'))}"
    )
    raise ImportError(f"Cannot import required modules from {project_root}")

# Import bpy AFTER setting up the module path (following project pattern)
import bpy


class TestSeamMarker(unittest.TestCase):
    """
    Comprehensive test suite for seam_marker module functionality.

    Following project's established testing patterns for module import and mocking,
    ensuring compatibility with the Blender_to_Spine2D_Mesh_Exporter structure.
    """

    def setUp(self):
        """Set up test environment with proper project-aware configuration."""
        # Configure logging for test debugging
        logging.basicConfig(level=logging.DEBUG)

        # Reset bpy mock to clean state (following project pattern)
        bpy.reset_mock()

        # Configure mock context and collections
        bpy.context = MagicMock()
        bpy.context.collection = MagicMock()
        bpy.context.collection.objects = MagicMock()
        bpy.context.collection.objects.link = MagicMock()

        # Setup mock utils (following project conventions)
        bpy.utils = MagicMock()

    def tearDown(self):
        """Clean up after each test following project patterns."""
        # Reset mocks to ensure test isolation
        bpy.reset_mock()

    def _create_mock_mesh_object(self, name="TestObject", mesh_data=None):
        """
        Create a properly configured mock mesh object for testing.

        Following the project's pattern for mock object creation with realistic
        Blender object structure and behaviors.

        Args:
            name (str): Name for the mock object
            mesh_data: Optional mock mesh data

        Returns:
            Mock: Configured mock object with mesh properties
        """
        mock_obj = Mock()
        mock_obj.name = name
        mock_obj.type = "MESH"
        mock_obj.data = mesh_data or Mock()

        # Configure copy behavior to return a properly structured mock
        mock_copy = Mock()
        mock_copy.name = name
        mock_copy.type = "MESH"
        mock_copy.data = Mock()
        mock_obj.copy.return_value = mock_copy

        return mock_obj

    def _create_mock_bmesh_structure(self, edges_data):
        """
        Create a comprehensive mock bmesh structure for testing.

        This method generates realistic bmesh mock objects that simulate
        actual Blender bmesh behavior for edge processing scenarios.

        Args:
            edges_data (list): List of tuples representing edges [(v1, v2), ...]

        Returns:
            Mock: Configured bmesh instance with vertices and edges
        """
        mock_bm = Mock()

        # Create mock vertices with realistic structure
        vertices = []
        max_vertex_index = max([max(edge) for edge in edges_data]) if edges_data else 0

        for i in range(max_vertex_index + 1):
            vertex = Mock()
            vertex.index = i
            vertex.co = [float(i), float(i * 2), float(i * 3)]  # Mock 3D coordinates
            vertices.append(vertex)

        # Create mock edges with proper vertex relationships
        edges = []
        for idx, (v1_idx, v2_idx) in enumerate(edges_data):
            edge = Mock()
            edge.index = idx
            edge.verts = [vertices[v1_idx], vertices[v2_idx]]
            edge.seam = False  # Initially no seam marking
            edges.append(edge)

        # Critical Fix: Create Mock collections instead of raw lists
        mock_verts_collection = Mock()
        mock_verts_collection.__iter__ = lambda self: iter(vertices)
        mock_verts_collection.__len__ = lambda self: len(vertices)
        mock_verts_collection.ensure_lookup_table = Mock()

        mock_edges_collection = Mock()
        mock_edges_collection.__iter__ = lambda self: iter(edges)
        mock_edges_collection.__len__ = lambda self: len(edges)
        mock_edges_collection.ensure_lookup_table = Mock()

        # Configure bmesh mock structure with proper collection mocks
        mock_bm.verts = mock_verts_collection
        mock_bm.edges = mock_edges_collection
        mock_bm.to_mesh = Mock()
        mock_bm.free = Mock()
        mock_bm.from_mesh = Mock()

        return mock_bm

    def test_mark_seams_on_copy_basic_functionality(self):
        """
        Test basic seam marking functionality with valid inputs.

        Validates the core workflow of object duplication, bmesh processing,
        and seam marking according to segmentation data.
        """
        # Arrange: Setup realistic test scenario
        original_obj = self._create_mock_mesh_object("OriginalMesh")
        segmentation_data = [(0, 1), (2, 3), (1, 2)]
        edges_data = [(0, 1), (1, 2), (2, 3), (3, 0)]  # Square mesh topology

        mock_bm = self._create_mock_bmesh_structure(edges_data)

        # Mock bmesh module following project patterns
        with patch(
            "Blender_to_Spine2D_Mesh_Exporter.seam_marker.bmesh"
        ) as mock_bmesh_module:
            mock_bmesh_module.new.return_value = mock_bm

            # Act: Execute seam marking operation
            result_obj, seams_info = seam_marker.mark_seams_on_copy(
                original_obj, segmentation_data, do_copy=True
            )

            # Assert: Validate object creation and copying workflow
            self.assertIsNotNone(result_obj)
            original_obj.copy.assert_called_once()
            bpy.context.collection.objects.link.assert_called_once()

            # Assert: Validate BMesh operations follow proper lifecycle
            mock_bmesh_module.new.assert_called_once()
            mock_bm.from_mesh.assert_called_once()
            mock_bm.verts.ensure_lookup_table.assert_called_once()
            mock_bm.edges.ensure_lookup_table.assert_called_once()
            mock_bm.to_mesh.assert_called_once()
            mock_bm.free.assert_called_once()

            # Assert: Validate seam marking algorithm accuracy
            marked_seams = []
            for edge in mock_bm.edges:
                if edge.seam:
                    vertex_indices = tuple(sorted([v.index for v in edge.verts]))
                    marked_seams.append(vertex_indices)

            expected_seams = {(0, 1), (1, 2), (2, 3)}
            actual_seams = set(marked_seams)
            self.assertEqual(actual_seams, expected_seams)

            # Assert: Validate seams_info data structure integrity
            self.assertEqual(len(seams_info), 3)
            for seam_info in seams_info:
                self.assertIn("edge_index", seam_info)
                self.assertIn("vertex_indices", seam_info)
                self.assertIn("vertex_coords", seam_info)
                # Validate data types for downstream processing
                self.assertIsInstance(seam_info["edge_index"], int)
                self.assertIsInstance(seam_info["vertex_indices"], tuple)
                self.assertIsInstance(seam_info["vertex_coords"], list)

    def test_mark_seams_on_copy_without_copying(self):
        """
        Test in-place seam marking functionality (do_copy=False).

        Validates that the function can modify existing objects without
        creating duplicates when copy behavior is disabled.
        """
        # Arrange: Setup for in-place modification
        original_obj = self._create_mock_mesh_object("OriginalMesh")
        segmentation_data = [(0, 1)]
        edges_data = [(0, 1), (1, 2)]

        mock_bm = self._create_mock_bmesh_structure(edges_data)

        with patch(
            "Blender_to_Spine2D_Mesh_Exporter.seam_marker.bmesh"
        ) as mock_bmesh_module:
            mock_bmesh_module.new.return_value = mock_bm

            # Act: Execute in-place seam marking
            result_obj, seams_info = seam_marker.mark_seams_on_copy(
                original_obj, segmentation_data, do_copy=False
            )

            # Assert: Validate no copying operations occurred
            original_obj.copy.assert_not_called()
            bpy.context.collection.objects.link.assert_not_called()

            # Assert: Validate object identity and name modification
            self.assertEqual(result_obj, original_obj)
            self.assertTrue(result_obj.name.endswith("_texturing"))

    def test_mark_seams_empty_segmentation_data(self):
        """
        Test handling of empty segmentation data.

        Ensures graceful processing when no seam marking is required,
        validating that the mesh processing pipeline remains stable.
        """
        # Arrange: Setup with empty segmentation data
        original_obj = self._create_mock_mesh_object("EmptySegMesh")
        segmentation_data = []
        edges_data = [(0, 1), (1, 2), (2, 0)]

        mock_bm = self._create_mock_bmesh_structure(edges_data)

        with patch(
            "Blender_to_Spine2D_Mesh_Exporter.seam_marker.bmesh"
        ) as mock_bmesh_module:
            mock_bmesh_module.new.return_value = mock_bm

            # Act: Process mesh with no segmentation requirements
            result_obj, seams_info = seam_marker.mark_seams_on_copy(
                original_obj, segmentation_data, do_copy=True
            )

            # Assert: Validate no seam marking occurred
            for edge in mock_bm.edges:
                self.assertFalse(edge.seam)

            # Assert: Validate clean processing results
            self.assertEqual(len(seams_info), 0)
            self.assertIsNotNone(result_obj)

    def test_mark_seams_invalid_segmentation_data_format(self):
        """
        Test robust handling of malformed segmentation data.

        Validates error resilience when processing mixed valid/invalid
        segmentation entries, ensuring partial success scenarios.
        """
        # Arrange: Setup with mixed valid/invalid segmentation data
        original_obj = self._create_mock_mesh_object("InvalidSegMesh")
        segmentation_data = [
            (0, 1),  # Valid tuple format
            [2, 3],  # Valid list format
            (4,),  # Invalid: single element
            "invalid",  # Invalid: string type
            (5, 6, 7),  # Invalid: three elements
            None,  # Invalid: None value
        ]
        edges_data = [(0, 1), (2, 3), (4, 5), (5, 6)]

        mock_bm = self._create_mock_bmesh_structure(edges_data)

        with patch(
            "Blender_to_Spine2D_Mesh_Exporter.seam_marker.bmesh"
        ) as mock_bmesh_module:
            mock_bmesh_module.new.return_value = mock_bm

            with patch(
                "Blender_to_Spine2D_Mesh_Exporter.seam_marker.logger"
            ) as mock_logger:
                # Act: Process mixed data formats
                result_obj, seams_info = seam_marker.mark_seams_on_copy(
                    original_obj, segmentation_data, do_copy=True
                )

                # Assert: Validate selective processing of valid data only
                marked_seams = []
                for edge in mock_bm.edges:
                    if edge.seam:
                        vertex_indices = tuple(sorted([v.index for v in edge.verts]))
                        marked_seams.append(vertex_indices)

                expected_valid_seams = {(0, 1), (2, 3)}
                actual_seams = set(marked_seams)
                self.assertEqual(actual_seams, expected_valid_seams)

                # Assert: Validate warning logging for invalid entries
                self.assertTrue(mock_logger.warning.called)
                warning_calls = mock_logger.warning.call_args_list
                self.assertGreater(len(warning_calls), 0)

    def test_mark_seams_invalid_object_type(self):
        """
        Test error handling for incompatible object types.

        Validates graceful error handling when non-mesh objects
        are passed to the seam marking function, following production
        error handling patterns that return None instead of raising exceptions.
        """
        # Arrange: Setup invalid object type
        invalid_obj = Mock()
        invalid_obj.type = "CAMERA"  # Non-mesh object type
        segmentation_data = [(0, 1)]

        # Act: Execute function with invalid object type
        result_obj, seams_info = seam_marker.mark_seams_on_copy(
            invalid_obj, segmentation_data
        )

        # Assert: Validate graceful error handling (production pattern)
        self.assertIsNone(result_obj)
        self.assertEqual(seams_info, [])

    def test_mark_seams_none_object(self):
        """
        Test error handling for None object input.

        Validates robust input validation and graceful error handling
        for null object references using production error patterns.
        """
        # Arrange: Setup None object scenario
        segmentation_data = [(0, 1)]

        # Act: Execute function with None object
        result_obj, seams_info = seam_marker.mark_seams_on_copy(None, segmentation_data)

        # Assert: Validate graceful error handling (production pattern)
        self.assertIsNone(result_obj)
        self.assertEqual(seams_info, [])

    def test_mark_seams_invalid_segmentation_data_type(self):
        """
        Test error handling for incorrect segmentation_data type.

        Validates input type validation for the segmentation_data parameter
        using production-grade error handling that returns None values.
        """
        # Arrange: Setup invalid data type
        original_obj = self._create_mock_mesh_object("ValidMesh")
        invalid_segmentation_data = "not_a_list"  # String instead of list

        # Act: Execute function with invalid data type
        result_obj, seams_info = seam_marker.mark_seams_on_copy(
            original_obj, invalid_segmentation_data
        )

        # Assert: Validate graceful error handling (production pattern)
        self.assertIsNone(result_obj)
        self.assertEqual(seams_info, [])

    def test_mark_seams_bmesh_exception_handling(self):
        """
        Test error recovery during BMesh operations.

        Validates graceful error handling when BMesh operations fail,
        ensuring proper cleanup and error reporting.
        """
        # Arrange: Setup for BMesh operation failure
        original_obj = self._create_mock_mesh_object("ExceptionMesh")
        segmentation_data = [(0, 1)]

        with patch(
            "Blender_to_Spine2D_Mesh_Exporter.seam_marker.bmesh"
        ) as mock_bmesh_module:
            # Configure bmesh to raise exception during creation
            mock_bmesh_module.new.side_effect = Exception("BMesh creation failed")

            with patch(
                "Blender_to_Spine2D_Mesh_Exporter.seam_marker.logger"
            ) as mock_logger:
                # Act: Attempt processing with simulated failure
                result_obj, seams_info = seam_marker.mark_seams_on_copy(
                    original_obj, segmentation_data, do_copy=True
                )

                # Assert: Validate graceful failure handling
                self.assertIsNone(result_obj)
                self.assertEqual(seams_info, [])

                # Assert: Validate error logging occurred
                self.assertTrue(mock_logger.error.called)
                error_calls = mock_logger.error.call_args_list
                self.assertGreater(len(error_calls), 0)

                # Validate error message contains function context
                error_message = error_calls[0][0][0]
                self.assertIn("mark_seams_on_copy", error_message)

    def test_mark_seams_complex_mesh_structure(self):
        """
        Test processing of complex mesh geometries.

        Validates seam marking accuracy on realistic mesh structures
        that simulate production mesh processing scenarios.
        """
        # Arrange: Create complex cube-like mesh topology
        original_obj = self._create_mock_mesh_object("ComplexMesh")

        # Define edges for a cube structure (8 vertices, 12 edges)
        edges_data = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),  # Bottom face edges
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),  # Top face edges
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),  # Vertical connecting edges
        ]

        # Define strategic seam placement for UV unwrapping
        segmentation_data = [(0, 1), (1, 2), (4, 5), (5, 6)]

        mock_bm = self._create_mock_bmesh_structure(edges_data)

        with patch(
            "Blender_to_Spine2D_Mesh_Exporter.seam_marker.bmesh"
        ) as mock_bmesh_module:
            mock_bmesh_module.new.return_value = mock_bm

            # Act: Process complex mesh structure
            result_obj, seams_info = seam_marker.mark_seams_on_copy(
                original_obj, segmentation_data, do_copy=True
            )

            # Assert: Validate accurate seam marking
            marked_seams = []
            for edge in mock_bm.edges:
                if edge.seam:
                    vertex_indices = tuple(sorted([v.index for v in edge.verts]))
                    marked_seams.append(vertex_indices)

            expected_seams = {(0, 1), (1, 2), (4, 5), (5, 6)}
            actual_seams = set(marked_seams)
            self.assertEqual(actual_seams, expected_seams)

            # Assert: Validate comprehensive seams_info structure
            self.assertEqual(len(seams_info), 4)
            for seam_info in seams_info:
                self.assertIsInstance(seam_info["edge_index"], int)
                self.assertIsInstance(seam_info["vertex_indices"], tuple)
                self.assertEqual(len(seam_info["vertex_indices"]), 2)
                self.assertIsInstance(seam_info["vertex_coords"], list)
                self.assertEqual(len(seam_info["vertex_coords"]), 2)

                # Validate coordinate data structure
                for coord_pair in seam_info["vertex_coords"]:
                    self.assertIsInstance(coord_pair, list)
                    self.assertEqual(len(coord_pair), 3)  # 3D coordinates

    def test_mark_seams_edge_indices_out_of_range(self):
        """
        Test handling of segmentation data with non-existent vertex indices.

        Validates robust processing when segmentation data references
        vertex indices that don't exist in the target mesh.
        """
        # Arrange: Setup small mesh with limited vertex range
        original_obj = self._create_mock_mesh_object("SmallMesh")

        # Small triangular mesh with only 3 vertices
        edges_data = [(0, 1), (1, 2), (2, 0)]

        # Segmentation data includes out-of-range vertex indices
        segmentation_data = [(0, 1), (5, 6), (10, 11)]  # Only (0,1) is valid

        mock_bm = self._create_mock_bmesh_structure(edges_data)

        with patch(
            "Blender_to_Spine2D_Mesh_Exporter.seam_marker.bmesh"
        ) as mock_bmesh_module:
            mock_bmesh_module.new.return_value = mock_bm

            # Act: Process with out-of-range indices
            result_obj, seams_info = seam_marker.mark_seams_on_copy(
                original_obj, segmentation_data, do_copy=True
            )

            # Assert: Validate selective processing of valid indices only
            marked_seams = []
            for edge in mock_bm.edges:
                if edge.seam:
                    vertex_indices = tuple(sorted([v.index for v in edge.verts]))
                    marked_seams.append(vertex_indices)

            # Only (0, 1) should be marked since other indices don't exist
            expected_seams = {(0, 1)}
            actual_seams = set(marked_seams)
            self.assertEqual(actual_seams, expected_seams)
            self.assertEqual(len(seams_info), 1)

    def test_mark_seams_duplicate_edges_in_segmentation_data(self):
        """
        Test handling of duplicate edge entries in segmentation data.

        Validates deduplication logic to ensure each edge is marked
        only once regardless of duplicate entries in segmentation data.
        """
        # Arrange: Setup with duplicate segmentation entries
        original_obj = self._create_mock_mesh_object("DuplicateMesh")
        edges_data = [(0, 1), (1, 2), (2, 0)]

        # Duplicate and reversed entries in segmentation data
        segmentation_data = [
            (0, 1),
            (1, 0),
            (0, 1),
            (1, 2),
        ]  # (0,1) appears multiple times

        mock_bm = self._create_mock_bmesh_structure(edges_data)

        with patch(
            "Blender_to_Spine2D_Mesh_Exporter.seam_marker.bmesh"
        ) as mock_bmesh_module:
            mock_bmesh_module.new.return_value = mock_bm

            # Act: Process duplicate segmentation data
            result_obj, seams_info = seam_marker.mark_seams_on_copy(
                original_obj, segmentation_data, do_copy=True
            )

            # Assert: Validate each edge marked only once
            marked_seams = []
            for edge in mock_bm.edges:
                if edge.seam:
                    vertex_indices = tuple(sorted([v.index for v in edge.verts]))
                    marked_seams.append(vertex_indices)

            expected_seams = {(0, 1), (1, 2)}
            actual_seams = set(marked_seams)
            self.assertEqual(actual_seams, expected_seams)

            # Assert: Validate seams_info contains no duplicate entries
            unique_seam_indices = set()
            for seam_info in seams_info:
                unique_seam_indices.add(seam_info["vertex_indices"])

            self.assertEqual(len(unique_seam_indices), 2)
            self.assertEqual(len(unique_seam_indices), len(seams_info))


class TestSeamMarkerModuleRegistration(unittest.TestCase):
    """
    Test module registration and unregistration functions.

    Following project conventions for testing addon lifecycle operations
    with proper logging verification and error handling.
    """

    def setUp(self):
        """Set up test environment for registration tests."""
        # Reset bpy mock following project patterns
        bpy.reset_mock()
        bpy.context = MagicMock()
        bpy.utils = MagicMock()

    def tearDown(self):
        """Clean up after registration tests."""
        bpy.reset_mock()

    def test_register_function(self):
        """
        Test the register function logging behavior.

        Validates that registration operations are properly logged
        following project's debugging and monitoring standards.
        """
        with patch(
            "Blender_to_Spine2D_Mesh_Exporter.seam_marker.logger"
        ) as mock_logger:
            # Act: Execute registration
            seam_marker.register()

            # Assert: Validate logging output
            mock_logger.debug.assert_called_with("Registration seam_marker.py")

    def test_unregister_function(self):
        """
        Test the unregister function logging behavior.

        Validates that unregistration operations are properly logged
        for debugging and cleanup verification purposes.
        """
        with patch(
            "Blender_to_Spine2D_Mesh_Exporter.seam_marker.logger"
        ) as mock_logger:
            # Act: Execute unregistration
            seam_marker.unregister()

            # Assert: Validate logging output
            mock_logger.debug.assert_called_with("Unregistration seam_marker.py")


class TestSeamMarkerPerformanceAndStress(unittest.TestCase):
    """
    Performance and stress testing for seam_marker functionality.

    Following project standards for performance validation and
    resource management verification in production scenarios.
    """

    def setUp(self):
        """Set up test environment for performance tests."""
        bpy.reset_mock()
        bpy.context = MagicMock()
        bpy.context.collection = MagicMock()
        bpy.context.collection.objects = MagicMock()
        bpy.utils = MagicMock()

    def tearDown(self):
        """Clean up after performance tests."""
        bpy.reset_mock()

    def test_large_segmentation_data_performance(self):
        """
        Test performance with enterprise-scale segmentation datasets.

        Validates algorithm scalability and resource efficiency
        when processing large mesh structures with extensive seam requirements.
        """
        # Arrange: Create large-scale dataset
        original_obj = self._create_mock_mesh_object("LargeMesh")

        # Generate extensive edge dataset (1000 edges for stress testing)
        edges_data = [(i, i + 1) for i in range(1000)]
        segmentation_data = [(i, i + 1) for i in range(0, 1000, 2)]  # Every other edge

        mock_bm = self._create_mock_bmesh_structure(edges_data)

        with patch(
            "Blender_to_Spine2D_Mesh_Exporter.seam_marker.bmesh"
        ) as mock_bmesh_module:
            mock_bmesh_module.new.return_value = mock_bm

            # Act: Process large dataset
            result_obj, seams_info = seam_marker.mark_seams_on_copy(
                original_obj, segmentation_data, do_copy=True
            )

            # Assert: Validate successful large-scale processing
            self.assertIsNotNone(result_obj)
            self.assertEqual(len(seams_info), 500)  # Half of edges marked as seams

            # Assert: Validate performance characteristics
            # Ensure all expected seams were processed
            marked_seam_count = sum(1 for edge in mock_bm.edges if edge.seam)
            self.assertEqual(marked_seam_count, 500)

    def test_memory_cleanup_verification(self):
        """
        Test BMesh memory management compliance.

        Critical validation that BMesh resources are properly released
        to prevent memory leaks in production mesh processing workflows.
        """
        # Arrange: Setup for memory management verification
        original_obj = self._create_mock_mesh_object("MemoryTestMesh")
        segmentation_data = [(0, 1), (1, 2)]
        edges_data = [(0, 1), (1, 2), (2, 0)]

        mock_bm = self._create_mock_bmesh_structure(edges_data)

        with patch(
            "Blender_to_Spine2D_Mesh_Exporter.seam_marker.bmesh"
        ) as mock_bmesh_module:
            mock_bmesh_module.new.return_value = mock_bm

            # Act: Execute operation with memory tracking
            result_obj, seams_info = seam_marker.mark_seams_on_copy(
                original_obj, segmentation_data, do_copy=True
            )

            # Assert: Critical validation of BMesh.free() call for memory cleanup
            mock_bm.free.assert_called_once()

            # Assert: Validate successful operation completion
            self.assertIsNotNone(result_obj)
            self.assertGreater(len(seams_info), 0)

    def _create_mock_mesh_object(self, name="TestObject"):
        """
        Helper method to create mock mesh objects for performance tests.

        Generates properly structured mock objects following project
        conventions for realistic Blender object simulation.
        """
        mock_obj = Mock()
        mock_obj.name = name
        mock_obj.type = "MESH"
        mock_obj.data = Mock()

        mock_copy = Mock()
        mock_copy.name = name
        mock_copy.type = "MESH"
        mock_copy.data = Mock()
        mock_obj.copy.return_value = mock_copy

        return mock_obj

    def _create_mock_bmesh_structure(self, edges_data):
        """
        Helper method to create mock bmesh structures for performance tests.

        Generates optimized mock bmesh objects for performance testing
        scenarios with realistic vertex/edge relationships and proper
        collection structure compatibility.
        """
        mock_bm = Mock()

        # Create vertices with efficient indexing
        vertices = []
        if edges_data:
            max_vertex_index = max([max(edge) for edge in edges_data])
            for i in range(max_vertex_index + 1):
                vertex = Mock()
                vertex.index = i
                vertex.co = [float(i), float(i * 2), float(i * 3)]
                vertices.append(vertex)

        # Create edges with proper vertex relationships
        edges = []
        for idx, (v1_idx, v2_idx) in enumerate(edges_data):
            edge = Mock()
            edge.index = idx
            edge.verts = [vertices[v1_idx], vertices[v2_idx]]
            edge.seam = False
            edges.append(edge)

        # Critical Fix: Create Mock collections for performance testing compatibility
        mock_verts_collection = Mock()
        mock_verts_collection.__iter__ = lambda self: iter(vertices)
        mock_verts_collection.__len__ = lambda self: len(vertices)
        mock_verts_collection.ensure_lookup_table = Mock()

        mock_edges_collection = Mock()
        mock_edges_collection.__iter__ = lambda self: iter(edges)
        mock_edges_collection.__len__ = lambda self: len(edges)
        mock_edges_collection.ensure_lookup_table = Mock()

        # Configure mock structure for performance testing
        mock_bm.verts = mock_verts_collection
        mock_bm.edges = mock_edges_collection
        mock_bm.to_mesh = Mock()
        mock_bm.free = Mock()
        mock_bm.from_mesh = Mock()

        return mock_bm


# Test runner configuration following project conventions
if __name__ == "__main__":
    # Configure test suite with comprehensive reporting
    test_suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    test_runner = unittest.TextTestRunner(verbosity=2, buffer=True)

    # Execute tests with detailed performance metrics
    print("=" * 80)
    print("COMPREHENSIVE SEAM MARKER TEST SUITE")
    print("Following Blender_to_Spine2D_Mesh_Exporter project conventions")
    print("=" * 80)

    result = test_runner.run(test_suite)

    # Generate comprehensive test report
    print("\n" + "=" * 80)
    print("TEST EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total Tests Executed: {result.testsRun}")
    print(f"Test Failures: {len(result.failures)}")
    print(f"Test Errors: {len(result.errors)}")

    if result.testsRun > 0:
        success_rate = (
            (result.testsRun - len(result.failures) - len(result.errors))
            / result.testsRun
            * 100
        )
        print(f"Success Rate: {success_rate:.1f}%")

        if success_rate == 100.0:
            print("STATUS: ✅ ALL TESTS PASSED - Production Ready")
        elif success_rate >= 90.0:
            print("STATUS: ⚠️  MOSTLY PASSING - Review Required")
        else:
            print("STATUS: ❌ SIGNIFICANT ISSUES - Immediate Attention Required")

    print("=" * 80)

    # Return appropriate exit code for CI/CD integration
    exit_code = 0 if (len(result.failures) == 0 and len(result.errors) == 0) else 1
    sys.exit(exit_code)
