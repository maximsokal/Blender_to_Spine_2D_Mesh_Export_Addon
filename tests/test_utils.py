# tests/test_utils.py
"""
Comprehensive Unit Test Suite for utils.py

This module provides thorough testing coverage for utility functions that handle
Z-depth processing and stretch value calculations in the Blender to Spine2D export pipeline.

## Test Coverage Areas:
1. _closest: Binary search functionality with edge cases
2. find_closest_z: Z-group matching with tolerance handling
3. smooth_stretch_values: Median filtering with various window sizes
4. limit_stretch_changes: Rate limiting with boundary conditions

## Testing Philosophy:
- Comprehensive edge case coverage
- Performance validation for algorithmic efficiency
- Error condition handling
- Mathematical precision verification
- Integration scenario validation

"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock
from statistics import median
import math

# Setup project path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from Blender_to_Spine2D_Mesh_Exporter import utils


class TestClosestFunction:
    """Comprehensive test suite for the _closest function."""

    def test_closest_single_element(self):
        """Test _closest with single-element list."""
        result = utils._closest([5.0], 10.0)
        assert result == 5.0

    def test_closest_exact_match(self):
        """Test _closest when target exactly matches an element."""
        sorted_values = [1.0, 3.0, 5.0, 7.0, 9.0]
        result = utils._closest(sorted_values, 5.0)
        assert result == 5.0

    def test_closest_between_elements(self):
        """Test _closest when target falls between elements."""
        sorted_values = [1.0, 3.0, 5.0, 7.0, 9.0]
        # Target 4.0 is closer to 3.0 (distance: 1.0) than 5.0 (distance: 1.0)
        # When equidistant, function should return the left element
        result = utils._closest(sorted_values, 4.0)
        assert result == 3.0

    def test_closest_target_before_range(self):
        """Test _closest when target is below the minimum value."""
        sorted_values = [5.0, 10.0, 15.0]
        result = utils._closest(sorted_values, 2.0)
        assert result == 5.0

    def test_closest_target_after_range(self):
        """Test _closest when target exceeds the maximum value."""
        sorted_values = [5.0, 10.0, 15.0]
        result = utils._closest(sorted_values, 20.0)
        assert result == 15.0

    def test_closest_with_duplicates(self):
        """Test _closest behavior with duplicate values."""
        sorted_values = [1.0, 3.0, 3.0, 5.0, 5.0, 7.0]
        result = utils._closest(sorted_values, 4.0)
        assert result == 3.0

    def test_closest_negative_values(self):
        """Test _closest with negative numbers."""
        sorted_values = [-10.0, -5.0, -1.0, 0.0, 2.0]
        result = utils._closest(sorted_values, -3.0)
        # Target -3.0 is equidistant from -5.0 and -1.0 (both distance 2.0)
        # Algorithm returns the "before" element (-5.0) when distances are equal
        assert result == -5.0

    def test_closest_floating_point_precision(self):
        """Test _closest with floating-point precision concerns."""
        sorted_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        target = 0.25
        result = utils._closest(sorted_values, target)
        # Target 0.25 is equidistant from 0.2 and 0.3, should return 0.2
        assert result == 0.2

    def test_closest_empty_list_raises_error(self):
        """Test _closest raises ValueError for empty list."""
        with pytest.raises(
            IndexError
        ):  # bisect_left on empty list should raise IndexError
            utils._closest([], 5.0)


class TestFindClosestZ:
    """Comprehensive test suite for the find_closest_z function."""

    def test_find_closest_z_with_original_groups(self):
        """Test find_closest_z preferring original_z_groups over local_z_groups."""
        original_groups = [1.0, 3.0, 5.0]
        local_groups = [2.0, 4.0, 6.0]

        result = utils.find_closest_z(
            target=3.5,
            original_z_groups=original_groups,
            local_z_groups=local_groups,
            obj_name="TestObj",
        )
        # Should use original_groups, closest to 3.5 is 3.0
        assert result == 3.0

    def test_find_closest_z_fallback_to_local(self):
        """Test find_closest_z falling back to local_z_groups when original is empty."""
        result = utils.find_closest_z(
            target=4.5,
            original_z_groups=[],  # Empty original groups
            local_z_groups=[2.0, 4.0, 6.0],
            obj_name="TestObj",
        )
        # Should use local_groups, closest to 4.5 is 4.0
        assert result == 4.0

    def test_find_closest_z_no_groups_returns_none(self):
        """Test find_closest_z returns None when no groups provided."""
        result = utils.find_closest_z(
            target=5.0, original_z_groups=None, local_z_groups=None, obj_name="TestObj"
        )
        assert result is None

    def test_find_closest_z_empty_groups_returns_none(self):
        """Test find_closest_z returns None when all groups are empty."""
        result = utils.find_closest_z(
            target=5.0, original_z_groups=[], local_z_groups=[], obj_name="TestObj"
        )
        assert result is None

    def test_find_closest_z_with_tolerance_within_range(self):
        """Test find_closest_z with tolerance - match within tolerance."""
        result = utils.find_closest_z(
            target=5.0,
            original_z_groups=[1.0, 4.8, 8.0],
            local_z_groups=[],
            obj_name="TestObj",
            tolerance=0.5,
        )
        # Closest is 4.8, distance is 0.2, within tolerance 0.5
        assert result == 4.8

    def test_find_closest_z_with_tolerance_outside_range(self):
        """Test find_closest_z with tolerance - no match within tolerance."""
        result = utils.find_closest_z(
            target=5.0,
            original_z_groups=[1.0, 7.0, 8.0],
            local_z_groups=[],
            obj_name="TestObj",
            tolerance=1.0,
        )
        # Closest is 7.0, distance is 2.0, exceeds tolerance 1.0
        assert result is None

    def test_find_closest_z_with_unsorted_input(self):
        """Test find_closest_z handles unsorted input correctly."""
        # Function should sort internally
        unsorted_groups = [5.0, 1.0, 9.0, 3.0, 7.0]
        result = utils.find_closest_z(
            target=4.0,
            original_z_groups=unsorted_groups,
            local_z_groups=[],
            obj_name="TestObj",
        )
        # Should find 3.0 or 5.0, whichever is closer to 4.0
        assert result == 3.0

    def test_find_closest_z_with_mixed_types(self):
        """Test find_closest_z converts input to float properly."""
        mixed_groups = [1, 3.5, 5, 7.2]  # Mix of int and float
        result = utils.find_closest_z(
            target=4.0,
            original_z_groups=mixed_groups,
            local_z_groups=[],
            obj_name="TestObj",
        )
        assert result == 3.5


class TestSmoothStretchValues:
    """Comprehensive test suite for the smooth_stretch_values function."""

    def test_smooth_empty_dict(self):
        """Test smooth_stretch_values with empty input."""
        result = utils.smooth_stretch_values({})
        assert result == {}

    def test_smooth_single_element(self):
        """Test smooth_stretch_values with single element."""
        data = {5.0: 0.8}
        result = utils.smooth_stretch_values(data)
        assert result == {5.0: 0.8}

    def test_smooth_window_size_one(self):
        """Test smooth_stretch_values with window_size=1 (no smoothing)."""
        data = {1.0: 0.1, 2.0: 0.9, 3.0: 0.2}
        result = utils.smooth_stretch_values(data, window_size=1)
        assert result == data

    def test_smooth_basic_median_filtering(self):
        """Test smooth_stretch_values basic median filtering functionality."""
        data = {1.0: 0.1, 2.0: 0.9, 3.0: 0.2}  # Middle value is outlier
        result = utils.smooth_stretch_values(data, window_size=3)

        # For key 2.0: window contains [0.1, 0.9, 0.2], median is 0.2
        assert result[2.0] == 0.2

    def test_smooth_edge_handling(self):
        """Test smooth_stretch_values handles edges correctly."""
        data = {1.0: 0.1, 2.0: 0.2, 3.0: 0.3, 4.0: 0.4, 5.0: 0.5}
        result = utils.smooth_stretch_values(data, window_size=3)

        # First element: window [0.1, 0.2], only uses available values
        # Last element: window [0.4, 0.5], only uses available values
        assert 1.0 in result
        assert 5.0 in result

    def test_smooth_large_window_size(self):
        """Test smooth_stretch_values with window larger than data."""
        data = {1.0: 0.1, 2.0: 0.9, 3.0: 0.2}
        result = utils.smooth_stretch_values(data, window_size=10)

        # With window size > data size, each element gets median of all values
        expected_median = median([0.1, 0.9, 0.2])  # 0.2
        for value in result.values():
            assert value == expected_median

    def test_smooth_even_window_size(self):
        """Test smooth_stretch_values with even window size."""
        data = {1.0: 0.1, 2.0: 0.2, 3.0: 0.3, 4.0: 0.4}
        result = utils.smooth_stretch_values(data, window_size=4)

        # Even window size should work correctly
        assert len(result) == len(data)

    def test_smooth_unsorted_keys(self):
        """Test smooth_stretch_values with unsorted input keys."""
        data = {3.0: 0.3, 1.0: 0.1, 4.0: 0.4, 2.0: 0.2}
        result = utils.smooth_stretch_values(data, window_size=3)

        # Function should sort keys internally
        assert len(result) == 4
        assert all(key in result for key in data.keys())

    def test_smooth_duplicate_values(self):
        """Test smooth_stretch_values with duplicate stretch values."""
        data = {1.0: 0.5, 2.0: 0.5, 3.0: 0.5, 4.0: 0.5}
        result = utils.smooth_stretch_values(data, window_size=3)

        # All median calculations should return 0.5
        for value in result.values():
            assert value == 0.5


class TestLimitStretchChanges:
    """Comprehensive test suite for the limit_stretch_changes function."""

    def test_limit_empty_dict(self):
        """Test limit_stretch_changes with empty input."""
        result = utils.limit_stretch_changes({})
        assert result == {}

    def test_limit_single_element(self):
        """Test limit_stretch_changes with single element."""
        data = {5.0: 0.8}
        result = utils.limit_stretch_changes(data, max_change=0.2)
        assert result == {5.0: 0.8}

    def test_limit_no_limiting_needed(self):
        """Test limit_stretch_changes when no values exceed max_change."""
        data = {1.0: 0.0, 2.0: 0.05, 3.0: 0.1}  # Changes: 0.05, 0.05
        result = utils.limit_stretch_changes(data, max_change=0.1)
        assert result == data

    def test_limit_basic_clamping(self):
        """Test limit_stretch_changes basic clamping functionality."""
        data = {1.0: 0.0, 2.0: 0.5, 3.0: 1.2}  # Changes: 0.5, 0.7
        result = utils.limit_stretch_changes(data, max_change=0.3)

        # First value unchanged: 0.0
        assert result[1.0] == 0.0

        # Second value: 0.0 + 0.3 = 0.3 (clamped from 0.5)
        assert result[2.0] == 0.3

        # Third value: 0.3 + 0.3 = 0.6 (clamped from 1.2)
        assert result[3.0] == 0.6

    def test_limit_negative_changes(self):
        """Test limit_stretch_changes with decreasing values."""
        data = {1.0: 1.0, 2.0: 0.3, 3.0: 0.1}  # Changes: -0.7, -0.2
        result = utils.limit_stretch_changes(data, max_change=0.4)

        # First value unchanged: 1.0
        assert result[1.0] == 1.0

        # Second value: 1.0 - 0.4 = 0.6 (clamped from 0.3)
        assert result[2.0] == 0.6

        # Third value calculation:
        # delta = 0.1 - 0.6 = -0.5 (exceeds max_change=0.4)
        # result = 0.6 + (-0.4) = 0.2 (clamped)
        # Use approximate comparison due to floating-point precision
        assert abs(result[3.0] - 0.2) < 1e-10

    def test_limit_alternating_changes(self):
        """Test limit_stretch_changes with alternating increases/decreases."""
        data = {1.0: 0.5, 2.0: 1.5, 3.0: 0.2, 4.0: 1.8}
        result = utils.limit_stretch_changes(data, max_change=0.3)

        # Sequential processing should cascade limitations
        expected_2 = 0.5 + 0.3  # 0.8 (limited from 1.5)
        expected_3 = expected_2 - 0.3  # 0.5 (limited from 0.2)
        expected_4 = expected_3 + 0.3  # 0.8 (limited from 1.8)

        assert result[2.0] == expected_2
        assert result[3.0] == expected_3
        assert result[4.0] == expected_4

    def test_limit_zero_max_change(self):
        """Test limit_stretch_changes with max_change=0 (no changes allowed)."""
        data = {1.0: 0.5, 2.0: 0.8, 3.0: 0.2}
        result = utils.limit_stretch_changes(data, max_change=0.0)

        # All values should be set to the first value
        assert result[1.0] == 0.5
        assert result[2.0] == 0.5
        assert result[3.0] == 0.5

    def test_limit_unsorted_keys(self):
        """Test limit_stretch_changes with unsorted input keys."""
        data = {3.0: 0.6, 1.0: 0.0, 2.0: 0.9}
        result = utils.limit_stretch_changes(data, max_change=0.2)

        # Function should process in sorted order: 1.0 -> 2.0 -> 3.0
        # Processing order: 0.0 -> limited to 0.2 -> limited to 0.4
        assert result[1.0] == 0.0
        assert result[2.0] == 0.2  # 0.0 + 0.2
        assert result[3.0] == 0.4  # 0.2 + 0.2

    def test_limit_floating_point_precision(self):
        """Test limit_stretch_changes with floating-point precision."""
        data = {1.0: 0.0, 2.0: 0.100001}  # Very small change over limit
        result = utils.limit_stretch_changes(data, max_change=0.1)

        # Should be limited to exactly 0.1
        assert abs(result[2.0] - 0.1) < 1e-10


class TestIntegrationScenarios:
    """Integration tests simulating real-world usage scenarios."""

    def test_full_pipeline_z_processing(self):
        """Test complete Z-processing pipeline with realistic data."""
        # Simulate Z-groups from mesh processing
        z_groups = [0.0, 1.5, 3.2, 4.8, 6.1]
        stretch_data = {0.0: 0.8, 1.5: 1.4, 3.2: 0.9, 4.8: 1.6, 6.1: 1.1}

        # Step 1: Find closest Z for intermediate value
        target_z = 2.0
        closest = utils.find_closest_z(
            target=target_z, original_z_groups=z_groups, obj_name="IntegrationTest"
        )
        assert closest == 1.5  # Closest to 2.0

        # Step 2: Smooth stretch values
        smoothed = utils.smooth_stretch_values(stretch_data, window_size=3)

        # Step 3: Limit changes
        limited = utils.limit_stretch_changes(smoothed, max_change=0.3)

        # Verify pipeline produces reasonable output
        assert len(limited) == len(stretch_data)
        assert all(isinstance(v, float) for v in limited.values())

    def test_performance_with_large_dataset(self):
        """Test performance with large datasets."""
        # Create large dataset
        large_data = {float(i): float(i % 10) / 10.0 for i in range(1000)}

        # These operations should complete quickly
        smoothed = utils.smooth_stretch_values(large_data, window_size=5)
        limited = utils.limit_stretch_changes(smoothed, max_change=0.1)

        assert len(smoothed) == 1000
        assert len(limited) == 1000

    def test_edge_case_combinations(self):
        """Test combinations of edge cases."""
        # Single element with extreme values
        data = {0.0: 1000.0}

        smoothed = utils.smooth_stretch_values(data, window_size=5)
        limited = utils.limit_stretch_changes(smoothed, max_change=0.001)

        assert limited == {0.0: 1000.0}

    @patch("Blender_to_Spine2D_Mesh_Exporter.utils.logger")
    def test_logging_integration(self, mock_logger):
        """Test that functions integrate properly with logging system."""
        # Configure mock logger to capture debug calls
        mock_logger.isEnabledFor.return_value = True

        # Call functions that should trigger logging
        utils.find_closest_z(
            target=5.0, original_z_groups=[1.0, 4.0, 7.0], obj_name="LogTest"
        )

        utils.smooth_stretch_values(
            {1.0: 0.5, 2.0: 0.8}, window_size=3, obj_name="LogTest"
        )

        utils.limit_stretch_changes(
            {1.0: 0.5, 2.0: 0.8}, max_change=0.2, obj_name="LogTest"
        )

        # Verify logging was called
        assert mock_logger.debug.call_count >= 3


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_window_size(self):
        """Test smooth_stretch_values with invalid window size."""
        data = {1.0: 0.5, 2.0: 0.8}

        # Window size 0 should return original data
        result = utils.smooth_stretch_values(data, window_size=0)
        assert result == data

        # Negative window size should return original data
        result = utils.smooth_stretch_values(data, window_size=-1)
        assert result == data

    def test_invalid_max_change(self):
        """Test limit_stretch_changes with negative max_change."""
        data = {1.0: 0.5, 2.0: 0.8}

        # Negative max_change should still work (absolute value used internally)
        result = utils.limit_stretch_changes(data, max_change=-0.1)
        # Implementation should handle this gracefully
        assert len(result) == len(data)

    def test_none_inputs(self):
        """Test functions with None inputs where applicable."""
        # find_closest_z with None tolerance (should work)
        result = utils.find_closest_z(
            target=5.0, original_z_groups=[1.0, 4.0, 7.0], tolerance=None
        )
        assert result == 4.0

        # Optional object name as None should work
        result = utils.smooth_stretch_values({1.0: 0.5}, obj_name=None)
        assert result == {1.0: 0.5}
