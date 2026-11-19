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

## Provides label accuracy without materializing comparisons or numeric casts.
import ../../tensor


proc accuracy_score*[P,T](y_pred: Tensor[P], y_true: Tensor[T]): float =
  ## Input:
  ##   - y_true: Tensor[T] containing the ground truth (correct) labels
  ##   - y_pred: Tensor[P] containing comparable predicted labels
  ##
  ## Returns:
  ##   - The proportion of correctly classified samples (as float).

  let (prediction, truth) = broadcast2(y_pred, y_true)
  var correct = 0
  for predicted, expected in zip(prediction, truth):
    if predicted == expected: inc correct
  result = float(correct) / float(prediction.size)
