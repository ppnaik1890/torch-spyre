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

"""Profiling report generation utilities.

Provides reporters for converting profiling data to various output formats.
This module is under development and currently provides stub implementations.
"""


class ProfileReport:
  """Container for profiling report data."""

  pass


class JSONReporter:
  """Reporter that generates JSON profiling output."""

  pass


class CSVReporter:
  """Reporter that generates CSV profiling output."""

  pass


__all__ = [
    "ProfileReport",
    "JSONReporter",
    "CSVReporter",
]
