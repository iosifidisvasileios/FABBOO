package OnlineStreamFairness;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.classifiers.trees.HoeffdingAdaptiveTree;
import moa.core.InstanceExample;
import org.apache.commons.collections4.queue.CircularFifoQueue;
import org.apache.log4j.Logger;

import java.util.ArrayList;

import static java.lang.Math.abs;
import static java.lang.Math.log;

/**
 * Created by iosifidis on 26.07.19.
 */
public class OFIB_Template {
    private int saIndex;

    private int weakL;
    private int indexOfDenied;
    private int indexOfGranted;
    private int indexOfDeprived;
    private  double delayed_discrimination;
    private String OPT;

    private static ArrayList<Double> Gmean = new ArrayList<Double>();
    private static ArrayList<Double> F1Score = new ArrayList<Double>();
    private static ArrayList<Double> Accuracy = new ArrayList<Double>();
    private static ArrayList<Double> Kappa = new ArrayList<Double>();
    private static ArrayList<Double> StParity = new ArrayList<Double>();
    private static ArrayList<Double> Thresholds = new ArrayList<Double>();
    private static ArrayList<Double> EQOP = new ArrayList<Double>();
    private static ArrayList<Double> PredPar = new ArrayList<Double>();
    private static ArrayList<Double> time = new ArrayList<Double>();
    private static ArrayList<Double> BACC = new ArrayList<Double>();
    private static ArrayList<Double> Recall = new ArrayList<Double>();


    public ArrayList<Double> getBACC() {
        return BACC;
    }

    public ArrayList<Double> getRecall() {
        return Recall;
    }

    private static final CircularFifoQueue<Double> buf_predictions = new CircularFifoQueue<Double>(2000);

    public ArrayList<Double> getGmean() {
        return Gmean;
    }

    public ArrayList<Double> getF1Score() {
        return F1Score;
    }

    public ArrayList<Double> getAccuracy() {
        return Accuracy;
    }

    public ArrayList<Double> getKappa() {
        return Kappa;
    }

    public ArrayList<Double> getStParity() {
        return StParity;
    }

    public ArrayList<Double> getThresholds() {
        return Thresholds;
    }

    public ArrayList<Double> getEQOP() {
        return EQOP;
    }
    public ArrayList<Double> getPredPar() {
        return PredPar;
    }

    private final static Logger logger = Logger.getLogger(OFIB_Template.class.getName());


    public OFIB_Template(int weakL, int saIndex, int indexOfDenied, int indexOfGranted, int indexOfDeprived, String OPT) {
        this.weakL = weakL;
        this.saIndex = saIndex;
        this.indexOfGranted = indexOfGranted;
        this.indexOfDeprived = indexOfDeprived;
        this.indexOfDenied = indexOfDenied;
        this.OPT = OPT;
    }


    private  void static_monitor_fairness(double prot_pos, double non_prot_pos,
                                                double prot_neg, double non_prot_neg) {

        double temp_Wdp = prot_pos / (prot_pos + prot_neg + 1);
        double temp_Wfp = non_prot_pos / (non_prot_pos + non_prot_neg + 1);
        delayed_discrimination = temp_Wfp - temp_Wdp;
    }

    private static double equal_opportunity(double tp_protected, double fn_protected, double tp_non_protected, double fn_non_protected) {
        return tp_non_protected / (tp_non_protected + fn_non_protected) - tp_protected / (tp_protected + fn_protected);
    }


    public void deploy(Instances buffer) throws Exception {

        FABBOO fairImbaBoosting = new FABBOO(indexOfDeprived, saIndex, indexOfGranted, indexOfDenied, OPT);
        fairImbaBoosting.ensembleSizeOption.setValue(weakL);
        fairImbaBoosting.baseLearnerOption.setCurrentObject(new HoeffdingAdaptiveTree());
        fairImbaBoosting.setModelContext(new InstancesHeader(buffer));
        fairImbaBoosting.prepareForUse();


        WindowAUCImbalancedPerformanceEvaluator evaluator = new WindowAUCImbalancedPerformanceEvaluator();
        evaluator.widthOption.setValue(buffer.size());
        evaluator.setIndex(saIndex);
        evaluator.prepareForUse();

        Thresholds.clear();
        buf_predictions.clear();

        double classified_prot_pos = 0.0;
        double classified_prot_neg = 0.0;

        double classified_non_prot_pos = 0.0;
        double classified_non_prot_neg = 0.0;

        double tp_protected = 0;
        double fn_protected = 0;
        double fp_protected = 0;
        double tp_non_protected = 0;
        double fn_non_protected = 0;
        double fp_non_protected = 0;

        long timeStart = System.nanoTime();
        for (int i = 0; i < buffer.size(); i++) {
            if (i == 0)
                Thresholds.add(0.5);

             Instance inst = buffer.get(i);

            double[] votes = fairImbaBoosting.getVotesForInstance(inst);
            double label = 0;
            try {
                label = (votes[indexOfDenied] < votes[indexOfGranted]) ? indexOfGranted : indexOfDenied;
            } catch (Exception e) {

                try {
                    if (!Double.isNaN(votes[indexOfDenied]))
                        label = indexOfDenied;
                } catch (Exception e1) {
                    label = indexOfGranted;
                }
            }

            evaluator.addResult(new InstanceExample(inst), votes, indexOfGranted);
            if (inst.value(saIndex) == indexOfDeprived) {
                if (label == indexOfGranted) {
                    classified_prot_pos++;
                } else {
                    classified_prot_neg++;
                }

                if (label == indexOfGranted && inst.classValue() == indexOfGranted) {
                    tp_protected += 1;

                } else if (inst.classValue() == indexOfGranted && label != indexOfGranted) {
                    fn_protected += 1;

                    if (!OPT.equals("PredPar")) {
                        // misclassifed positive protected instance
                        try {
                            buf_predictions.add(votes[indexOfGranted]);
                        } catch (ArrayIndexOutOfBoundsException e) {
                            // has predicted negative class 100%
                            buf_predictions.add(0.);
                        }
                    }
                } else if (inst.classValue() != indexOfGranted && label == indexOfGranted) {
                    fp_protected += 1;
                }

            } else {
                if (label == indexOfGranted) {
                    classified_non_prot_pos++;
                } else {
                    classified_non_prot_neg++;
                }

                if (label == indexOfGranted && inst.classValue() == indexOfGranted) {
                    tp_non_protected += 1;
                    if (OPT.equals("PredPar")) {
                        // misclassifed positive protected instance
                        try {
                            buf_predictions.add(votes[indexOfGranted]);
                        } catch (ArrayIndexOutOfBoundsException e) {
                            // has predicted negative class 100%
                            buf_predictions.add(1.0);
                        }
                    }
                } else if (inst.classValue() == indexOfGranted && label != indexOfGranted) {
                    fn_non_protected += 1;
                } else if (inst.classValue() != indexOfGranted && label == indexOfGranted) {
                    fp_non_protected += 1;
                }
            }


            if (OPT.equals("SP")) {

                static_monitor_fairness(classified_prot_pos, classified_non_prot_pos, classified_prot_neg, classified_non_prot_neg);

                if (abs(delayed_discrimination) >= 0.001) {
                    int position = shifted_location(classified_prot_pos, classified_non_prot_pos, classified_prot_neg, classified_non_prot_neg);
                    Thresholds.add(fairImbaBoosting.tweak_boundary(buf_predictions, position));

                } else {
                    Thresholds.add(Thresholds.get(Thresholds.size() - 1));
                }
            } else if (OPT.equals("EQOP")) {
                double delayed_EQOP = equal_opportunity(tp_protected, fn_protected, tp_non_protected, fn_non_protected);
//                logger.info(delayed_EQOP);
                if (abs(delayed_EQOP) >= 0.001) {
                    int position = shifted_location(tp_protected, tp_non_protected, fn_protected, fn_non_protected);

                    Thresholds.add(fairImbaBoosting.tweak_boundary(buf_predictions, position));
//                    logger.info(i + "\t" + delayed_EQOP + ", " + position + ", "+  Thresholds.get(Thresholds.size() - 1));

                } else {
                    Thresholds.add(Thresholds.get(Thresholds.size() - 1));
                }
            } else if (OPT.equals("PredPar")) {
                double predParity = predictive_parity(tp_protected, fp_protected, tp_non_protected, fp_non_protected);
                int position = shifted_location_pred_parity(tp_protected, fp_protected, tp_non_protected, fp_non_protected);
                if (abs(predParity) >= 0.001) {
                    Thresholds.add(fairImbaBoosting.tweak_boundary_reverse(buf_predictions, position));
                } else {
                    Thresholds.add(Thresholds.get(Thresholds.size() - 1));
                }
//                logger.info(i + ","+ predParity + "," + position+ "," + Thresholds.get(Thresholds.size() - 1) );
            }

            fairImbaBoosting.trainOnInstanceImpl(inst);

        }

        if (OPT.equals("SP")) {
            StParity.add(delayed_discrimination);
        }

        if (OPT.equals("EQOP")) {
            EQOP.add(equal_opportunity(tp_protected, fn_protected, tp_non_protected, fn_non_protected));
        }

        Accuracy.add(evaluator.getErrorRate());
        Kappa.add(evaluator.getKappa());
        Gmean.add(evaluator.getGmean());
        F1Score.add(evaluator.getF1Score());
        BACC.add(evaluator.getBACC());
        Recall.add(evaluator.getRecall());
    }
    private double predictive_parity(double tp_protected, double fp_protected, double tp_non_protected, double fp_non_protected) {
        return (tp_non_protected / (tp_non_protected + fp_non_protected)) - (tp_protected / (tp_protected + fp_protected));
    }

    private int shifted_location_pred_parity(double tp_protected, double fp_protected, double tp_non_protected, double fp_non_protected) {
        return (int) ((tp_protected*(tp_non_protected+fp_non_protected))/(tp_non_protected*(tp_protected+fp_protected))+.5);
    }

    private static int shifted_location(double classified_prot_pos, double classified_non_prot_pos, double classified_prot_neg, double classified_non_prot_neg) {
        return (int) ((classified_prot_neg + classified_prot_pos) * ((classified_non_prot_pos) / (classified_non_prot_pos + classified_non_prot_neg)) - classified_prot_pos);
    }


}
