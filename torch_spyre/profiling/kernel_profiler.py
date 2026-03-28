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

"""Torch-Spyre kernel profiler module.

Provides profiling utilities for Spyre kernel execution on AIU accelerators.
This module is under development and currently provides stub implementations.
"""


class KernelMetrics:
  """Container for kernel execution metrics."""

  pass


class KernelProfiler:
  """Context manager for profiling kernel execution.

  Example:
      with KernelProfiler():
          x = torch.randn(32, 32, device="spyre")
          y = torch.randn(32, 32, device="spyre")
          z = torch.matmul(x, y)
  """

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    return False


# Global profiler instance
global_profiler = KernelProfiler()


def enable_kernel_profiling():
  """Enable kernel profiling globally."""
  pass


def disable_kernel_profiling():
  """Disable kernel profiling globally."""
  pass


def get_kernel_profile():
  """Get the current kernel profile data.

  Returns:
      Profiling data or None if no profile is available.
  """
  return None


__all__ = [
    "KernelMetrics",
    "KernelProfiler",
    "global_profiler",
    "enable_kernel_profiling",
    "disable_kernel_profiling",
    "get_kernel_profile",
]
