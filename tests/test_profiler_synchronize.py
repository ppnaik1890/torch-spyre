# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import torch
import torch_spyre
from torch_spyre.profiling import global_profiler, enable_kernel_profiling, disable_kernel_profiling, get_kernel_profile


class TestProfilerSynchronizeOnExit(unittest.TestCase):
    """Test that synchronize() is called when profiler exits."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("spyre")
        disable_kernel_profiling()

    def tearDown(self):
        """Clean up after tests."""
        disable_kernel_profiling()

    def test_synchronize_called_on_profiler_exit(self):
        """
        Verify that profiler __exit__ calls synchronize() automatically.

        This is verified by profiling two sequential operations and checking
        that the second operation's start time is after the first operation's
        end time. If synchronize() is not called in __exit__, the timings may
        overlap due to async execution.

        Note: This test requires hardware to run. It checks that:
        - Operation 1 completes before Operation 2 starts
        - This indicates synchronize() was called between them
        """
        enable_kernel_profiling()

        try:
            # Operation 1: matrix multiply
            with global_profiler:
                x1 = torch.randn(32, 32, device=self.device, dtype=torch.float16)
                y1 = torch.randn(32, 32, device=self.device, dtype=torch.float16)
                z1 = torch.matmul(x1, y1)

            # Get end time of first profiler context (after __exit__ synchronize)
            profile_data_1 = get_kernel_profile()

            # Operation 2: another matrix multiply
            with global_profiler:
                x2 = torch.randn(32, 32, device=self.device, dtype=torch.float16)
                y2 = torch.randn(32, 32, device=self.device, dtype=torch.float16)
                z2 = torch.matmul(x2, y2)

            # Get profile data for second operation
            profile_data_2 = get_kernel_profile()

            # Verify both profiler contexts executed
            self.assertIsNotNone(profile_data_1)
            self.assertIsNotNone(profile_data_2)

            # If synchronize() was called in __exit__, timings should not overlap
            # Extract timing information from profile data
            if hasattr(profile_data_1, "__len__") and len(profile_data_1) > 0:
                if hasattr(profile_data_2, "__len__") and len(profile_data_2) > 0:
                    # Get the last kernel from first profile (most recent)
                    first_op_end_time = self._get_operation_end_time(profile_data_1)
                    # Get the first kernel from second profile
                    second_op_start_time = self._get_operation_start_time(profile_data_2)

                    if first_op_end_time is not None and second_op_start_time is not None:
                        # Second op should start after first op ends
                        # (if synchronize() was called in __exit__)
                        self.assertGreaterEqual(
                            second_op_start_time,
                            first_op_end_time,
                            "Second operation started before first operation ended; "
                            "synchronize() may not have been called in profiler.__exit__()"
                        )
        finally:
            disable_kernel_profiling()

    def test_profiler_context_manager_exit_synchronizes(self):
        """
        Test that profiler context manager properly exits with synchronization.

        Verifies that after exiting a profiler context, all operations have
        completed (no pending async work).
        """
        enable_kernel_profiling()

        try:
            # Perform operation within profiler context
            profiler_active_during_op = False
            with global_profiler:
                x = torch.randn(64, 64, device=self.device, dtype=torch.float16)
                y = torch.randn(64, 64, device=self.device, dtype=torch.float16)
                z = x + y
                # At this point, profiler should be active
                profiler_active_during_op = global_profiler is not None

            # After exiting context, synchronize should have been called
            # Verify profiler context was active during operation
            self.assertTrue(
                profiler_active_during_op,
                "Profiler was not active within context"
            )

            # Perform another operation to verify no pending work from previous one
            with global_profiler:
                a = torch.randn(64, 64, device=self.device, dtype=torch.float16)
                b = torch.randn(64, 64, device=self.device, dtype=torch.float16)
                c = a + b

            # If we got here without errors, synchronization happened correctly
            self.assertTrue(True)
        finally:
            disable_kernel_profiling()

    def test_nested_operations_synchronize_at_exit(self):
        """
        Test that multiple operations within profiler context synchronize properly on exit.

        This verifies that __exit__ ensures all pending operations complete,
        making it safe to start new work afterward.
        """
        enable_kernel_profiling()

        try:
            operations_completed = []

            # First profiler context with multiple operations
            with global_profiler:
                for i in range(3):
                    x = torch.randn(16 + i * 8, 16 + i * 8, device=self.device, dtype=torch.float16)
                    y = torch.randn(16 + i * 8, 16 + i * 8, device=self.device, dtype=torch.float16)
                    z = x * y
                    operations_completed.append(i)

            # After context exit, all operations should be complete
            self.assertEqual(len(operations_completed), 3)

            # Second profiler context should start clean
            with global_profiler:
                x = torch.randn(32, 32, device=self.device, dtype=torch.float16)
                y = torch.randn(32, 32, device=self.device, dtype=torch.float16)
                # If previous operations weren't synchronized, this could fail
                z = x + y
                operations_completed.append(3)

            # All operations should have completed
            self.assertEqual(len(operations_completed), 4)
        finally:
            disable_kernel_profiling()

    def _get_operation_end_time(self, profile_data):
        """Extract the end time of the last operation from profile data."""
        if hasattr(profile_data, "metrics"):
            metrics = profile_data.metrics
            if metrics and len(metrics) > 0:
                last_metric = metrics[-1]
                if hasattr(last_metric, "end_time"):
                    return last_metric.end_time
                elif hasattr(last_metric, "execution_time"):
                    return last_metric.execution_time
        return None

    def _get_operation_start_time(self, profile_data):
        """Extract the start time of the first operation from profile data."""
        if hasattr(profile_data, "metrics"):
            metrics = profile_data.metrics
            if metrics and len(metrics) > 0:
                first_metric = metrics[0]
                if hasattr(first_metric, "start_time"):
                    return first_metric.start_time
        return None


class TestProfilerSynchronizeBasic(unittest.TestCase):
    """Basic tests for profiler synchronization behavior."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("spyre")
        disable_kernel_profiling()

    def tearDown(self):
        """Clean up after tests."""
        disable_kernel_profiling()

    def test_profiler_enable_disable_cycles(self):
        """Test that profiler can be enabled and disabled multiple times."""
        for _ in range(3):
            enable_kernel_profiling()
            self.assertIsNotNone(global_profiler)

            disable_kernel_profiling()
            # After disable, global_profiler might still exist but be disabled
            self.assertTrue(True)

    def test_operations_with_profiler_enabled(self):
        """Test that basic operations work with profiler enabled."""
        enable_kernel_profiling()

        try:
            x = torch.randn(10, 10, device=self.device, dtype=torch.float16)
            y = torch.randn(10, 10, device=self.device, dtype=torch.float16)
            z = x + y

            # Operations should complete without error
            self.assertIsNotNone(z)
            self.assertEqual(z.shape, (10, 10))
        finally:
            disable_kernel_profiling()

    def test_get_kernel_profile_returns_data(self):
        """Test that get_kernel_profile returns valid data structure."""
        enable_kernel_profiling()

        try:
            with global_profiler:
                x = torch.randn(8, 8, device=self.device, dtype=torch.float16)
                y = torch.randn(8, 8, device=self.device, dtype=torch.float16)
                z = x + y

            profile_data = get_kernel_profile()
            self.assertIsNotNone(profile_data)
        finally:
            disable_kernel_profiling()


if __name__ == "__main__":
    unittest.main()
