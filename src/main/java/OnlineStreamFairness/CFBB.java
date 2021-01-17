/*
 *    OnlineSmoothBoost.java
 *    Copyright (C) 2013 University of Waikato, Hamilton, New Zealand
 *    @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 */
package OnlineStreamFairness;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.options.ClassOption;
import org.apache.commons.collections4.queue.CircularFifoQueue;

import java.util.ArrayList;
import java.util.Collections;

import static java.lang.Math.abs;


/**
 * Incremental on-line boosting with Theoretical Justifications of Shang-Tse Chen,
 * Hsuan-Tien Lin and Chi-Jen Lu.
 * <p>
 * <p>See details in:<br /> </p>
 * <p>
 * <p>Parameters:</p> <ul> <li>-l : Classiﬁer to train</li> <li>-s : The number
 * of models to boost</li>
 * </ul>
 *
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @version $Revision: 7 $
 */
public class CFBB extends AbstractClassifier implements MultiClassClassifier {

    public double threshold = 0.5;
    private static final long serialVersionUID = 1L;

    private int indexOfTargetClass; // sensitive attribute: female
    private int indexOfDeprived; // sensitive attribute: female
    private int saIndex; // index of sensitive attribute
    private String OPT; // index of sensitive attribute

    public CFBB(int indexOfDeprived, int saIndex, int targetClass, String OPT) {
        this.indexOfDeprived = indexOfDeprived;
        this.saIndex = saIndex;
        this.OPT = OPT;
        this.indexOfTargetClass = targetClass;
    }

    @Override
    public String getPurposeString() {
        return "Incremental on-line boosting of Shang-Tse Chen, Hsuan-Tien Lin and Chi-Jen Lu.";
    }

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train.", Classifier.class, "trees.HoeffdingTree");

    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 's',
            "The number of models to boost.", 10, 1, Integer.MAX_VALUE);

    //public FlagOption pureBoostOption = new FlagOption("pureBoost", 'p',
    //        "Boost with weights only; no poisson.");

    public FloatOption gammaOption = new FloatOption("gamma",
            'g',
            "The value of the gamma parameter.",
            0.1, 0.0, 1.0);

    protected Classifier[] ensemble;

    protected double[] alpha;

    protected double gamma;

    protected double theta;

    @Override
    public void resetLearningImpl() {
        this.ensemble = new Classifier[this.ensembleSizeOption.getValue()];
        Classifier baseLearner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
        baseLearner.resetLearning();
        this.alpha = new double[this.ensemble.length];
        for (int i = 0; i < this.ensemble.length; i++) {
            this.ensemble[i] = baseLearner.copy();
            this.alpha[i] = 1.0 / (double) this.ensemble.length;
        }
        this.gamma = this.gammaOption.getValue();
        this.theta = this.gamma / (2.0 + this.gamma);
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        double zt = 0.0;
        double weight = 1.0;
        for (int i = 0; i < this.ensemble.length; i++) {
            zt += (this.ensemble[i].correctlyClassifies(inst) ? 1 : -1) - theta;
            // normalized_predict(ex.x) * ex.y - theta;
            Instance weightedInst = (Instance) inst.copy();
            weightedInst.setWeight(weight);
            this.ensemble[i].trainOnInstance(weightedInst);
            weight = (zt <= 0) ? 1.0 : Math.pow(1.0 - gamma, zt / 2.0);
        }
    }


    public void trainInstanceImbalance(Instance inst, boolean targetClass, double imbalanceRate) {
        double zt = 0.0;
        double weight = 1.0;
        for (int i = 0; i < this.ensemble.length; i++) {
            zt += (this.ensemble[i].correctlyClassifies(inst) ? 1 : -1) - theta;
            Instance weightedInst = (Instance) inst.copy();
            weightedInst.setWeight(weight);
            this.ensemble[i].trainOnInstance(weightedInst);

            if (imbalanceRate >= 0 ){
                if (targetClass) {
                    weight = weight / (1 - imbalanceRate);
                }
                if (!targetClass) {
                    weight = weight / (1 + imbalanceRate);
                }
            } else if( imbalanceRate < 0){
                if (targetClass) {
                    weight = weight / (1 - imbalanceRate);
                }
                if (!targetClass) {
                    weight = weight / (1 + imbalanceRate);
                }
            }

        }
    }


    protected double getEnsembleMemberWeight(int i) {
        return this.alpha[i];
    }

    public double[] getVotesForInstance(Instance inst) {

        DoubleVector combinedVote = new DoubleVector();
        for (int i = 0; i < this.ensemble.length; i++) {
            double memberWeight = getEnsembleMemberWeight(i);
            if (memberWeight > 0.0) {
                DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(inst));
                if (vote.sumOfValues() > 0.0) {
                    vote.normalize();
                    vote.scaleValues(memberWeight);
                    combinedVote.addValues(vote);
                }
            } else {
                break;
            }
        }
        combinedVote.normalize();
        double targetConf;
        if (OPT.equals("EQOP") || OPT.equals("SP")) {
            try {
                targetConf = combinedVote.getArrayCopy()[indexOfTargetClass];
            } catch (Exception e) {
                targetConf = -1;
            }
            if (inst.value(saIndex) == indexOfDeprived && targetConf > threshold)
                combinedVote.setValue(indexOfTargetClass, 1);

            return combinedVote.getArrayRef();
        }
        try {
            targetConf = combinedVote.getArrayCopy()[indexOfTargetClass];
        } catch (Exception e) {
            targetConf = -1;
        }
        if (inst.value(saIndex) != indexOfDeprived && targetConf > threshold)
            combinedVote.setValue(indexOfTargetClass, 1);

        return combinedVote.getArrayRef();

    }

    public boolean isRandomizable() {
        return true;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        // TODO Auto-generated method stub
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[]{new Measurement("ensemble size",
                this.ensemble != null ? this.ensemble.length : 0)};
    }

    @Override
    public Classifier[] getSubClassifiers() {
        return this.ensemble.clone();
    }


    public double tweak_boundary_reverse(Instances buffer, int sdb) {

        if (sdb < 0) {
            threshold -= (threshold - 0.5) / (2);
            return threshold;
        }
        ArrayList<Double> predictions = new ArrayList<Double>();
        for (int j = 0; j < buffer.size(); j++) {
            Instance inst = buffer.get(j);

            DoubleVector combinedVote = new DoubleVector();
            for (int i = 0; i < this.ensemble.length; i++) {
                double memberWeight = getEnsembleMemberWeight(i);
                if (memberWeight > 0.0) {
                    DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(inst));

                    if (vote.sumOfValues() > 0.0) {
                        vote.normalize();
                        vote.scaleValues(memberWeight);
                        combinedVote.addValues(vote);
                    }

                } else {
                    continue;
                }
            }
            combinedVote.normalize();
            double targetConf;
            try {
                targetConf = combinedVote.getArrayCopy()[indexOfTargetClass];
            } catch (Exception e) {
                targetConf = 1;

            }

            if (inst.value(saIndex) != indexOfDeprived && inst.classValue() == indexOfTargetClass && targetConf <= 0.5 && sdb >= 0)
                predictions.add(targetConf);
        }

        Collections.sort(predictions);
        if (sdb > predictions.size())
            sdb = predictions.size() - 1;

        try {
            threshold = predictions.get(sdb);
        } catch (Exception e) {
            threshold = 0.5;
        }

        return threshold;
    }

    public double tweak_boundary(Instances buffer, int sdb, double disc) {
        if (disc < 0 || sdb < 0) {
            threshold += (0.5 - threshold) / (2);
            return threshold;
        }

        ArrayList<Double> predictions = new ArrayList<Double>();
        for (int j = 0; j < buffer.size(); j++) {
            Instance inst = buffer.get(j);

            DoubleVector combinedVote = new DoubleVector();
            for (int i = 0; i < this.ensemble.length; i++) {
                double memberWeight = getEnsembleMemberWeight(i);
                if (memberWeight > 0.0) {
                    DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(inst));

                    if (vote.sumOfValues() > 0.0) {
                        vote.normalize();
                        vote.scaleValues(memberWeight);
                        combinedVote.addValues(vote);
                    }

                } else {
                    continue;
                }
            }
            combinedVote.normalize();
            double targetConf;
            try {
                targetConf = combinedVote.getArrayCopy()[indexOfTargetClass];
            } catch (Exception e) {
                targetConf = 1;

            }

            if (inst.value(saIndex) == indexOfDeprived && inst.classValue() == indexOfTargetClass && targetConf <= 0.5 && sdb >= 0)
                predictions.add(targetConf);
        }
        sdb = abs(sdb);
        Collections.sort(predictions, Collections.reverseOrder());
        if (sdb >= predictions.size())
            sdb = predictions.size() - 1;

        try {
            threshold = predictions.get(sdb);
        } catch (Exception e) {
            threshold = 0.5;
        }
        return threshold;

    }


    public void optimize_for_equal_opportunity(Instances buffer, int sdb, double disc) {
        if (disc < 0 || sdb < 0) {
            threshold = 0.5;
            return;
        }
        if (disc == 0)
            return;

        ArrayList<Double> predictions = new ArrayList<Double>();
        for (int j = 0; j < buffer.size(); j++) {
            Instance inst = buffer.get(j);

            DoubleVector combinedVote = new DoubleVector();
            for (int i = 0; i < this.ensemble.length; i++) {
                double memberWeight = getEnsembleMemberWeight(i);
                if (memberWeight > 0.0) {
                    DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(inst));

                    if (vote.sumOfValues() > 0.0) {
                        vote.normalize();
                        vote.scaleValues(memberWeight);
                        combinedVote.addValues(vote);
                    }

                } else {
                    continue;
                }
            }
            combinedVote.normalize();
            double targetConf;
            try {
                targetConf = combinedVote.getArrayCopy()[indexOfTargetClass];
            } catch (Exception e) {
                targetConf = 0;
            }

            if (inst.value(saIndex) == indexOfDeprived && inst.classValue() == indexOfTargetClass && targetConf <= 0.5)
                predictions.add(targetConf);
        }
        Collections.sort(predictions, Collections.reverseOrder());
        if (sdb >= predictions.size())
            sdb = predictions.size() - 1;
        if (predictions.size() == 1)
            sdb = 0;
        if (predictions.size() == 0) {
            threshold = 0.5;
            return;
        }

        try {
            threshold = predictions.get(sdb);
        } catch (Exception e) {
            System.out.println("sdb = " + sdb);
            System.out.println(predictions);
        }


    }
}
