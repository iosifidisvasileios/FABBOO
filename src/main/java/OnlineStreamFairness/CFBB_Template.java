package OnlineStreamFairness;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.classifiers.trees.HoeffdingAdaptiveTree;
import moa.core.InstanceExample;
import org.apache.log4j.Logger;

import java.util.ArrayList;

import static java.lang.Math.abs;

/**
 * Created by iosifidis on 26.07.19.
 */
public class CFBB_Template {
    private static int saIndex;

    private double overall_disc = 0;

    private static int windowSize;
    private static int weakL;
    private static int indexOfDenied;
    private static int indexOfGranted;
    private static int indexOfDeprived;
    private double window_disc;
    private boolean StreamEval = false;
    private static String OPT;

    public ArrayList<Double> getBACC() {
        return BACC;
    }

    public ArrayList<Double> getRecall() {
        return Recall;
    }

    private static double class_lamda = 0.9;

    public static double Wp = 0;
    public static double Wn = 0;

    private static ArrayList<Double> Gmean = new ArrayList<Double>();
    private static ArrayList<Double> F1Score = new ArrayList<Double>();
    private static ArrayList<Double> Accuracy = new ArrayList<Double>();
    private static ArrayList<Double> Kappa = new ArrayList<Double>();
    private static ArrayList<Double> StParity = new ArrayList<Double>();
    private static ArrayList<Double> Thresholds = new ArrayList<Double>();
    private static ArrayList<Double> EQOP = new ArrayList<Double>();
    private static ArrayList<Double> BACC = new ArrayList<Double>();
    private static ArrayList<Double> Recall = new ArrayList<Double>();

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

    private final static Logger logger = Logger.getLogger(CFBB_Template.class.getName());


    public CFBB_Template(int weakL, int windowSize, int saIndex, int indexOfDenied, int indexOfGranted, int indexOfDeprived, String OPT) {
        this.weakL = weakL;
        this.windowSize = windowSize;
        this.saIndex = saIndex;
        this.indexOfGranted = indexOfGranted;
        this.indexOfDeprived = indexOfDeprived;
        this.indexOfDenied = indexOfDenied;
        this.OPT = OPT;
    }

    public CFBB_Template(int weakL, double lamda, int window, int saIndex, int indexOfDenied, int indexOfGranted, int indexOfDeprived, String OPT, boolean StreamEval) {
        this.StreamEval = StreamEval;
        this.weakL = weakL;
        this.windowSize = window;
        class_lamda = lamda;
        this.saIndex = saIndex;
        this.indexOfGranted = indexOfGranted;
        this.indexOfDeprived = indexOfDeprived;
        this.indexOfDenied = indexOfDenied;
        this.OPT = OPT;
    }


    private void static_monitor_fairness(double prot_pos, double non_prot_pos,
                                         double prot_neg, double non_prot_neg) {

        double temp_Wdp = prot_pos / (prot_pos + prot_neg + 1);
        double temp_Wfp = non_prot_pos / (non_prot_pos + non_prot_neg + 1);
        window_disc = temp_Wfp - temp_Wdp;
    }

    private static double equal_opportunity(double tp_protected, double fn_protected, double tp_non_protected, double fn_non_protected) {
        return tp_non_protected / (tp_non_protected + fn_non_protected) - tp_protected / (tp_protected + fn_protected);
    }


    public void deploy(Instances buffer) throws Exception {
        CFBB fairChunkBoosting = new CFBB(indexOfDeprived, saIndex, indexOfGranted, OPT);
        fairChunkBoosting.ensembleSizeOption.setValue(weakL);
        fairChunkBoosting.baseLearnerOption.setCurrentObject(new HoeffdingAdaptiveTree());
        fairChunkBoosting.setModelContext(new InstancesHeader(buffer));
        fairChunkBoosting.prepareForUse();
        Thresholds.clear();


        WindowAUCImbalancedPerformanceEvaluator evaluator = new WindowAUCImbalancedPerformanceEvaluator();
        evaluator.widthOption.setValue(buffer.size());
        evaluator.setIndex(saIndex);
        evaluator.prepareForUse();

        double window_classified_prot_pos = 0.0;
        double window_classified_prot_neg = 0.0;

        double window_classified_non_prot_pos = 0.0;
        double window_classified_non_prot_neg = 0.0;

        double classified_prot_pos = 0.0;
        double classified_prot_neg = 0.0;

        double classified_non_prot_pos = 0.0;
        double classified_non_prot_neg = 0.0;

        double window_EQOP = 0;

        double window_tp_protected = 0;
        double window_fn_protected = 0;
        double window_fp_protected = 0;
        double window_tp_non_protected = 0;
        double window_fn_non_protected = 0;
        double window_fp_non_protected = 0;

        double tp_protected = 0;
        double fp_protected = 0;
        double fn_protected = 0;
        double tp_non_protected = 0;
        double fp_non_protected = 0;
        double fn_non_protected = 0;

        int pos = 0;
        int neg = 0;

        Instances windowData = new Instances(buffer, 0);

        for (int i = 0; i < buffer.size(); i++) {
            boolean targetClass = false;
            Instance inst = buffer.get(i);
            windowData.add(inst);

            if (inst.classValue() == indexOfGranted) {
                targetClass = true;
                pos++;
            } else {
                neg++;
            }


            if (i == 0)
                Thresholds.add(0.5);


            update_class_rates(pos, neg);

            double[] votes = fairChunkBoosting.getVotesForInstance(inst);
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
                    window_classified_prot_pos++;
                } else {
                    classified_prot_neg++;
                    window_classified_prot_neg++;
                }

                if (label == indexOfGranted && inst.classValue() == indexOfGranted) {
                    tp_protected += 1;
                    window_tp_protected += 1;
                } else if (inst.classValue() == indexOfGranted && label != indexOfGranted) {
                    fn_protected += 1;
                    window_fn_protected += 1;
                } else if (inst.classValue() != indexOfGranted && label == indexOfGranted) {
                    window_fp_protected += 1;
                    fp_protected += 1;
                }
            } else {
                if (label == indexOfGranted) {
                    classified_non_prot_pos++;
                    window_classified_non_prot_pos++;
                } else {
                    classified_non_prot_neg++;
                    window_classified_non_prot_neg++;
                }

                if (label == indexOfGranted && inst.classValue() == indexOfGranted) {
                    tp_non_protected += 1;
                    window_tp_non_protected += 1;
                } else if (inst.classValue() == indexOfGranted && label != indexOfGranted) {
                    fn_non_protected += 1;
                    window_fn_non_protected += 1;
                } else if (inst.classValue() != indexOfGranted && label == indexOfGranted) {
                    window_fp_non_protected += 1;
                    fp_non_protected += 1;
                }
            }

            if (OPT.equals("SP")) {
                statistical_parity(classified_prot_pos, classified_non_prot_pos, classified_prot_neg, classified_non_prot_neg);
                if (StreamEval) {
                    static_monitor_fairness(window_classified_prot_pos, window_classified_non_prot_pos, window_classified_prot_neg, window_classified_non_prot_neg);
                    StParity.add(overall_disc);
                }
            }

            if (OPT.equals("EQOP")) {
                window_EQOP = equal_opportunity(window_tp_protected, window_fn_protected, window_tp_non_protected, window_fn_non_protected);
                if (StreamEval) {
                    EQOP.add(equal_opportunity(tp_protected, fn_protected, tp_non_protected, fn_non_protected));
                }
            }


            if ((i + 1) % windowSize == 0) {

                if (OPT.equals("SP")) {
                    if (Math.abs(window_disc) > .001) {
                        int position = shifted_location(window_classified_prot_pos, window_classified_non_prot_pos, window_classified_prot_neg, window_classified_non_prot_neg);
                        Thresholds.add(fairChunkBoosting.tweak_boundary(windowData, position, window_disc));
                    }
                }

                if (OPT.equals("EQOP")) {
                    if (abs(window_EQOP) >= 0.001) {
                        int position = shifted_location(window_tp_protected, window_tp_non_protected, window_fn_protected, window_fn_non_protected);
                        Thresholds.add(fairChunkBoosting.tweak_boundary(windowData, position, window_EQOP));
                    } else {
                        Thresholds.add(Thresholds.get(Thresholds.size() - 1));
                    }
                }

                windowData.delete();
                window_classified_prot_pos = 0.0;
                window_classified_prot_neg = 0.0;
                window_classified_non_prot_pos = 0.0;
                window_classified_non_prot_neg = 0.0;

                window_tp_protected = 0;
                window_fn_protected = 0;
                window_tp_non_protected = 0;
                window_fn_non_protected = 0;

            } else {
                Thresholds.add(Thresholds.get(Thresholds.size() - 1));
            }
            if (StreamEval) {
                Accuracy.add(evaluator.getErrorRate());
                Gmean.add(evaluator.getGmean());
                Kappa.add(evaluator.getKappa());
                F1Score.add(evaluator.getF1Score());
                BACC.add(evaluator.getBACC());
                Recall.add(evaluator.getRecall());

            }
            fairChunkBoosting.trainInstanceImbalance(inst, targetClass, Wn - Wp);
        }
        if (!StreamEval) {
            Accuracy.add(evaluator.getErrorRate());
            Gmean.add(evaluator.getGmean());
            Kappa.add(evaluator.getKappa());
            F1Score.add(evaluator.getF1Score());
            BACC.add(evaluator.getBACC());
            Recall.add(evaluator.getRecall());
            if (OPT.equals("EQOP")) {
                EQOP.add(equal_opportunity(tp_protected, fn_protected, tp_non_protected, fn_non_protected));
            }

            if (OPT.equals("SP")) {
                static_monitor_fairness(classified_prot_pos, classified_non_prot_pos, classified_prot_neg, classified_non_prot_neg);
                StParity.add(window_disc);
            }
        }
    }

    private double statistical_parity(double prot_pos, double non_prot_pos,
                                      double prot_neg, double non_prot_neg) {

        double temp_Wdp = prot_pos / (prot_pos + prot_neg + 1);
        double temp_Wfp = non_prot_pos / (non_prot_pos + non_prot_neg + 1);
        overall_disc = temp_Wfp - temp_Wdp;
        return overall_disc;
    }

    private static int shifted_location(double classified_prot_pos, double classified_non_prot_pos, double classified_prot_neg, double classified_non_prot_neg) {
        return (int) ((classified_prot_neg + classified_prot_pos) * ((classified_non_prot_pos) / (classified_non_prot_pos + classified_non_prot_neg)) - classified_prot_pos);
    }

    private double predictive_parity(double tp_protected, double fp_protected, double tp_non_protected, double fp_non_protected) {
        return tp_non_protected / (tp_non_protected + fp_non_protected) - tp_protected / (tp_protected + fp_protected);
    }

    private int shifted_location_pred_parity(double tp_protected, double fp_protected, double tp_non_protected, double fp_non_protected) {
        return (int) (tp_protected * (tp_non_protected + fp_non_protected) / tp_non_protected * (tp_protected + fp_protected));
    }

    private static void update_class_rates(double positives, double negatives) {
        Wp = class_lamda * Wp + (1 - class_lamda) * positives;
        Wn = class_lamda * Wn + (1 - class_lamda) * negatives;
        double sum = Wp + Wn;
        Wp = Wp / (sum);
        Wn = Wn / (sum);
    }

}
