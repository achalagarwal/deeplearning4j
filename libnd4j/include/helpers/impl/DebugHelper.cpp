/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// Created by raver119 on 20/04/18.
//

#include <helpers/DebugHelper.h>
#include <NDArray.h>
#include <NDArrayFactory.h>

namespace nd4j {
    DebugHelper debugStatistics(NDArray* input) {
        DebugHelper info;

        info._minValue = 0.;
        info._maxValue = -1;
        info._meanValue = 0.;
        info._stdDevValue = 1.;
        info._zeroCount = 0;
        info._positiveCount = 0;
        info._negativeCount = 0;
        info._infCount = 0;
        info._nanCount = 0;
        if (input->lengthOf() == 1) { // scalar case
                info._minValue = input->e<double>(0);
                info._maxValue = info._minValue;
                info._meanValue = info._minValue;
                info._stdDevValue = info._minValue;
                info._zeroCount = nd4j::math::nd4j_abs(input->e<double>(0)) > 0.00001? 0: 1;
                info._positiveCount = input->e<double>(0) > 0;
                info._negativeCount = input->e<double>(0) < 0;
                info._infCount = nd4j::math::nd4j_isinf(input->e<double>(0));
                info._nanCount = nd4j::math::nd4j_isnan(input->e<double>(0));
        }
        else if (input->lengthOf() > 0) {
                // TO DO: here processing for all elements with array
        }
        // else - no statistics for empty

        return info;
    }
}
