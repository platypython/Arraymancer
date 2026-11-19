# Copyright 2017 the Arraymancer contributors
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

import ../../src/arraymancer
import std / [unittest, math]

proc main() =
  suite "[ML] Metrics":
    test "Accuracy score":
      let
        y_pred = [0, 2, 1, 3].toTensor
        y_true = [0, 1, 2, 3].toTensor

      check: y_pred.accuracy_score(y_true) == 0.5

    test "Accuracy consumes int64 argmax outputs and strided native labels":
      let predictions = [[1.0, 3.0], [4.0, 2.0], [1.0, 7.0]].toTensor.argmax(1)
      let backing = [99, 1, 99, 0, 99, 1].toTensor
      check predictions.accuracy_score(backing[1..^1|2]) == 1.0
      let single = [[3.0, 1.0]].toTensor.argmax(1)
      check single.shape == toMetadata([1])
      check single.accuracy_score([0].toTensor) == 1.0

    test "Mixed label views preserve broadcasting, scalar and empty accuracy":
      let predictions = [[1'i64], [0'i64]].toTensor
      let truth = [1, 0, 1].toTensor
      let views = broadcast2(predictions, truth)
      check views.a.shape == toMetadata([2, 3])
      check views.b.shape == toMetadata([2, 3])
      check views.a.storage == predictions.storage
      check views.b.storage == truth.storage
      check predictions.accuracy_score(truth) == 0.5
      check newTensorWith[int64]([], 1'i64).accuracy_score(truth) == 2.0 / 3.0
      check newTensor[int64](0, 1).accuracy_score(truth).classify == fcNan
      expect ValueError:
        discard newTensor[int64](2, 2).accuracy_score(truth)

    test "Mean absolute error":
      var y_true = [0.9, 0.2, 0.1, 0.4, 0.9].toTensor()
      var y =      [1.0, 0.0, 0.0, 1.0, 1.0].toTensor()
      check: y.absolute_error(y_true).sum() == 1.1
      check: y.mean_absolute_error(y_true) == (1.1 / 5.0)

    test "Relative error":
      var y_true = [0.0,  0.0, -1.0, 1e-8, 1e-8].toTensor()
      var y =      [0.0, -1.0,  0.0, 0.0,  1e-7].toTensor()
      check: y.relative_error(y_true) == [0.0, 1.0, 1.0, 1.0, 0.9].toTensor()
      check: y.mean_relative_error(y_true) == 0.78

    test "Mean squared error":
      var y_true = [0.9, 0.2, 0.1, 0.4, 0.9].toTensor()
      var y =      [1.0, 0.0, 0.0, 1.0, 1.0].toTensor()
      check: y.squared_error(y_true).sum() == 0.43
      check: y.mean_squared_error(y_true) == (0.43 / 5.0)


main()
GC_fullCollect()
