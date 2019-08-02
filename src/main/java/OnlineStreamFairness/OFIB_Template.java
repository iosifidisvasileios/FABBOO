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

/**
 * Created by iosifidis on 26.07.19.
 */
public class OFIB_Template {
    private static int saIndex;
    public static OFBB FairImbaBoosting;

    private static int weakL;
    private static int indexOfDenied;
    private static int indexOfGranted;
    private static int indexOfDeprived;
    private static double delayed_discrimination;
    private static String OPT;

    private static ArrayList<Double> Gmean = new ArrayList<Double>();
    private static ArrayList<Double> F1Score = new ArrayList<Double>();
    private static ArrayList<Double> Accuracy = new ArrayList<Double>();
    private static ArrayList<Double> Kappa = new ArrayList<Double>();
    private static ArrayList<Double> StParity = new ArrayList<Double>();
    private static ArrayList<Double> Thresholds = new ArrayList<Double>();
    private static ArrayList<Double> EQOP = new ArrayList<Double>();
    private static ArrayList<Double> BACC = new ArrayList<Double>();
    private static ArrayList<Double> Recall = new ArrayList<Double>();

    public ArrayList<Double> getBACC() {
        return BACC;
    }

    public ArrayList<Double> getRecall() {
        return Recall;
    }

    private static final CircularFifoQueue<Double> buf_predictions = new CircularFifoQueue<Double>(5000);

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

    private final static Logger logger = Logger.getLogger(OFIB_Template.class.getName());


    public OFIB_Template(int weakL, int saIndex, int indexOfDenied, int indexOfGranted, int indexOfDeprived, String OPT) {
        this.weakL = weakL;
        this.saIndex = saIndex;
        this.indexOfGranted = indexOfGranted;
        this.indexOfDeprived = indexOfDeprived;
        this.indexOfDenied = indexOfDenied;
        this.OPT = OPT;
    }


    private static void static_monitor_fairness(double prot_pos, double non_prot_pos,
                                                double prot_neg, double non_prot_neg) {

        double temp_Wdp = prot_pos / (prot_pos + prot_neg + 1);
        double temp_Wfp = non_prot_pos / (non_prot_pos + non_prot_neg + 1);
        delayed_discrimination = temp_Wfp - temp_Wdp;
    }

    private static double equal_opportunity(double tp_protected, double fn_protected, double tp_non_protected, double fn_non_protected) {
        return tp_non_protected / (tp_non_protected + fn_non_protected) - tp_protected / (tp_protected + fn_protected);
    }


    public void deploy(Instances buffer) throws Exception {

        FairImbaBoosting = new OFBB(indexOfDeprived, saIndex, indexOfGranted);
        FairImbaBoosting.ensembleSizeOption.setValue(weakL);
        FairImbaBoosting.baseLearnerOption.setCurrentObject(new HoeffdingAdaptiveTree());
        FairImbaBoosting.setModelContext(new InstancesHeader(buffer));
        FairImbaBoosting.prepareForUse();


        WindowAUCImbalancedPerformanceEvaluator evaluator = new WindowAUCImbalancedPerformanceEvaluator();
        evaluator.widthOption.setValue(buffer.size());
        evaluator.setIndex(saIndex);
        evaluator.prepareForUse();

        double tp_protected = 0;
        double fn_protected = 0;
        double tp_non_protected = 0;
        double fn_non_protected = 0;

        double classified_prot_pos = 0.0;
        double classified_prot_neg = 0.0;

        double classified_non_prot_pos = 0.0;
        double classified_non_prot_neg = 0.0;

        Thresholds.clear();
        buf_predictions.clear();

        for (int i = 0; i < buffer.size(); i++) {


            boolean targetClass = false;
            Instance inst = buffer.get(i);


            if (i == 0)
                Thresholds.add(0.5);


            double[] votes = FairImbaBoosting.getVotesForInstance(inst);
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

            evaluator.addResult(new InstanceExample(inst), votes);

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
                    // misclassifed positive protected instance
                    try {
                        buf_predictions.add(votes[indexOfGranted]);
                    } catch (ArrayIndexOutOfBoundsException e) {
                        // has predicted negative class 100%
                        buf_predictions.add(0.);
                    }
                } else if (inst.classValue() != indexOfGranted && label != indexOfGranted) {
                    // correctly negative protected instance
//                    try {
//                        buf_predictions.add(votes[indexOfGranted]);
//                    } catch (ArrayIndexOutOfBoundsException e) {
//                        // has predicted negative class 100%
//                        buf_predictions.add(0.);
//                    }
                }
            } else {
                if (label == indexOfGranted) {
                    classified_non_prot_pos++;
                } else {
                    classified_non_prot_neg++;
                }

                if (label == indexOfGranted && inst.classValue() == indexOfGranted) {
                    tp_non_protected += 1;
                } else if (inst.classValue() == indexOfGranted && label != indexOfGranted) {
                    fn_non_protected += 1;
                }
            }


            if (OPT.equals("SP")) {
                static_monitor_fairness(classified_prot_pos, classified_non_prot_pos, classified_prot_neg, classified_non_prot_neg);
                if (abs(delayed_discrimination) >= 0.001) {
                    int count_for_sp = shifted_location(classified_prot_pos, classified_non_prot_pos, classified_prot_neg, classified_non_prot_neg);
                    Thresholds.add(FairImbaBoosting.tweak_boundary(buf_predictions, count_for_sp));
                } else {
                    Thresholds.add(Thresholds.get(Thresholds.size() - 1));
                }
            }

            if (OPT.equals("EQOP")) {
                double delayed_EQOP = equal_opportunity(tp_protected, fn_protected, tp_non_protected, fn_non_protected);
                if (abs(delayed_EQOP) >= 0.001) {
                    int position = shifted_location(tp_protected, tp_non_protected,fn_protected,  fn_non_protected);
                    Thresholds.add(FairImbaBoosting.tweak_boundary(buf_predictions, position));
                } else {
                    Thresholds.add(Thresholds.get(Thresholds.size() - 1));
                }
            }


            FairImbaBoosting.trainOnInstanceImpl(inst);

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


    private static int shifted_location(double classified_prot_pos, double classified_non_prot_pos, double classified_prot_neg, double classified_non_prot_neg) {
        return (int) ((classified_prot_neg + classified_prot_pos) * ((classified_non_prot_pos) / (classified_non_prot_pos + classified_non_prot_neg)) - classified_prot_pos);
    }


}
