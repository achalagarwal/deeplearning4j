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

package org.nd4j.linalg.learning.config;

import lombok.Builder;
import lombok.Data;
import lombok.experimental.SuperBuilder;
import lombok.val;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.AMSGradUpdater;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.PadamUpdater;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Arrays;

/**
 * The Padam updater<br>
 * Reference: On the Convergence of Adam and Beyond - https://openreview.net/forum?id=ryQu7f-RZ
 *
 * @author Achal Agarwal
 */
@Data
@SuperBuilder
public class Padam extends AMSGrad {

    public static final double DEFAULT_PADAM_PARTIAL_PARAM = 1.0 / 8.0;
    public static final double DEFAULT_PADAM_EPSILON = 1e-8;
    public static final double DEFAULT_PADAM_BETA1_MEAN_DECAY = 0.9;
    public static final double DEFAULT_PADAM_BETA2_VAR_DECAY = 0.999;
    public static final double DEFAULT_PADAM_LEARNING_RATE = 1e-3;


    @lombok.Builder.Default
    private double param = DEFAULT_PADAM_PARTIAL_PARAM;

    public Padam() {
        this(DEFAULT_PADAM_LEARNING_RATE, DEFAULT_PADAM_BETA1_MEAN_DECAY, DEFAULT_PADAM_BETA2_VAR_DECAY,
                DEFAULT_PADAM_EPSILON, DEFAULT_PADAM_PARTIAL_PARAM);
    }

    public Padam(double learningRate) {
        this(learningRate, null, DEFAULT_PADAM_BETA1_MEAN_DECAY, DEFAULT_PADAM_BETA2_VAR_DECAY, DEFAULT_PADAM_EPSILON, DEFAULT_PADAM_PARTIAL_PARAM);
    }

    public Padam(double learningRate, double partial_param) {
        this(learningRate, null, DEFAULT_PADAM_BETA1_MEAN_DECAY, DEFAULT_PADAM_BETA2_VAR_DECAY, DEFAULT_PADAM_EPSILON, partial_param);
    }

    public Padam(ISchedule learningRateSchedule) {
        this(Double.NaN, learningRateSchedule, DEFAULT_AMSGRAD_BETA1_MEAN_DECAY, DEFAULT_AMSGRAD_BETA2_VAR_DECAY, DEFAULT_AMSGRAD_EPSILON, DEFAULT_PADAM_PARTIAL_PARAM);
    }

    public Padam(double learningRate, double beta1, double beta2, double epsilon, double param) {
        this(learningRate, null, beta1, beta2, epsilon, param);
    }

    private Padam(@JsonProperty("learningRate") double learningRate,
                  @JsonProperty("learningRateSchedule") ISchedule learningRateSchedule,
                  @JsonProperty("beta1") double beta1,
                  @JsonProperty("beta2") double beta2,
                  @JsonProperty("epsilon") double epsilon,
                  @JsonProperty("param") double param) {
        super(learningRate, learningRateSchedule, beta1, beta2, epsilon);
        this.param = param;
    }

//    @Override
//    public long stateSize(long numParams) {
//        return 3 * numParams;
//    }
//
//    @Override
//    public GradientUpdater instantiate(INDArray viewArray, boolean initializeViewArray) {
//        PadamUpdater u = new PadamUpdater(this);
//        long[] gradientShape = viewArray.shape();
//        gradientShape = Arrays.copyOf(gradientShape, gradientShape.length);
//        gradientShape[1] /= 3;
//        u.setStateViewArray(viewArray, gradientShape, viewArray.ordering(), initializeViewArray);
//        return u;
//    }

    @Override
    public Padam clone() {
        return new Padam(learningRate, learningRateSchedule, beta1, beta2, epsilon, param);
    }
}
//    @Override
//    public double getLearningRate(int iteration, int epoch) {
//        if (learningRateSchedule != null) {
//            return learningRateSchedule.valueAt(iteration, epoch);
//        }
//        return learningRate;
//    }

//
//    @Override
//    public boolean hasLearningRate() {
//        return true;
//    }

//    @Override
//    public void setLrAndSchedule(double lr, ISchedule lrSchedule) {
//        this.learningRate = lr;
//        this.learningRateSchedule = lrSchedule;
//    }

    //Partial builder implementation to give public no-arg constructor
//    public static class Builder {
//        public Builder(){ }
//    }
//
