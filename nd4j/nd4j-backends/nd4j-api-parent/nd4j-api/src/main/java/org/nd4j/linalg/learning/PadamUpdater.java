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

package org.nd4j.linalg.learning;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.math3.util.FastMath;
import org.joda.time.IllegalFieldValueException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.scalar.Pow;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Padam;
import org.nd4j.linalg.ops.transforms.Transforms;
/**
 * The PADAM updater<br>
 * Reference: Closing the Generalization Gap of Adaptive Gradient
 * Methods in Training Deep Neural Networks - https://arxiv.org/pdf/1806.06763.pdf
 *
 * @author Achal Agarwal
 */
@Data
@Slf4j
public class PadamUpdater extends AMSGradUpdater {

    private Padam config;

    public PadamUpdater(Padam config){
        super(config);
        this.config = config;
    }


    @Override
    public void applyUpdater(INDArray gradient, int iteration, int epoch) {
        if (m == null || v == null || vHat == null)
            throw new IllegalStateException("Updater has not been initialized with view state");

        double param = config.getParam();

        if(param>0.5 || param<=0.0)
            log.warn("Partial Parameter should be within (0,1/2] to ensure convergence");
           // throw new IllegalArgumentException("Partial Parameter should be within (0,1/2] to ensure convergence");

        double beta1 = config.getBeta1();
        double beta2 = config.getBeta2();
        double learningRate = config.getLearningRate(iteration, epoch);
        double epsilon = config.getEpsilon();

        //m_t = b_1 * m_{t-1} + (1-b_1) * g_t
        INDArray oneMinusBeta1Grad = gradient.mul(1.0 - beta1);
        m.muli(beta1).addi(oneMinusBeta1Grad);

        //v_t = b_2 * v_{t-1} + (1-b_2) * (g_t)^2
        INDArray oneMinusBeta2GradSquared = gradient.mul(gradient).muli(1 - beta2);
        v.muli(beta2).addi(oneMinusBeta2GradSquared);

        double beta1t = FastMath.pow(beta1, iteration + 1);
        double beta2t = FastMath.pow(beta2, iteration + 1);

        //vHat_t = max(vHat_{t-1}, v_t)
        Transforms.max(vHat, v, false);

        double alphat = learningRate * FastMath.sqrt(1 - beta2t) / (1 - beta1t);
        if (Double.isNaN(alphat) || alphat == 0.0)
            alphat = epsilon;

        //gradient array contains: pow(vHat,p) + eps
        Nd4j.getExecutioner().exec(new Pow(vHat, gradient, param)).addi(epsilon);

        //gradient = alphat * m_t / (pow(vHat,p) + eps)
        gradient.rdivi(m).muli(alphat);
    }

}
