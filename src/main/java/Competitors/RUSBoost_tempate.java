package Competitors;

import OnlineStreamFairness.WindowAUCImbalancedPerformanceEvaluator;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.classifiers.meta.imbalanced.OnlineAdaC2;
import moa.classifiers.meta.imbalanced.OnlineRUSBoost;
import moa.classifiers.trees.HoeffdingAdaptiveTree;
import moa.core.InstanceExample;
import org.apache.log4j.Logger;

import java.util.ArrayList;

/**
 * Created by iosifidis on 26.07.19.
 */
public class RUSBoost_tempate {
    private static int saIndex;
    private static int weakL;
    private static int indexOfDenied;
    private static int indexOfGranted;
    private static int indexOfDeprived;
    private double delayed_discrimination;
    private static String  OPT;

    public ArrayList<Double> getBACC() {
        return BACC;
    }

    public ArrayList<Double> getRecall() {
        return Recall;
    }

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


    private final static Logger logger = Logger.getLogger(RUSBoost_tempate.class.getName());


    public RUSBoost_tempate(int weakL , int saIndex, int indexOfDenied, int indexOfGranted, int indexOfDeprived, String OPT) {
        this.weakL= weakL;
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
        delayed_discrimination = temp_Wfp - temp_Wdp;
    }
    private static double equal_opportunity(double tp_protected, double fn_protected, double tp_non_protected, double fn_non_protected) {
        return tp_non_protected / (tp_non_protected + fn_non_protected) - tp_protected / (tp_protected + fn_protected);
    }
    private static double predictive_parity(double tp_protected, double fp_protected, double tp_non_protected, double fp_non_protected) {
        return tp_non_protected / (tp_non_protected + fp_non_protected) - tp_protected / (tp_protected + fp_protected);
    }


    public void deploy(Instances buffer) throws Exception {


        OnlineRUSBoost rusBoost = new OnlineRUSBoost();
        rusBoost.ensembleSizeOption.setValue(weakL);
        rusBoost.baseLearnerOption.setCurrentObject(new HoeffdingAdaptiveTree());
        rusBoost.setModelContext(new InstancesHeader(buffer));
        rusBoost.prepareForUse();

        WindowAUCImbalancedPerformanceEvaluator evaluator = new WindowAUCImbalancedPerformanceEvaluator();
        evaluator.widthOption.setValue(buffer.size());
        evaluator.setIndex(saIndex);
        evaluator.prepareForUse();


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

        for (int i = 0; i < buffer.size(); i++) {
            Instance inst = buffer.get(i);

            double[] votes = rusBoost.getVotesForInstance(inst);
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
                } else if (inst.classValue() == indexOfGranted && label != indexOfGranted) {
                    fn_non_protected += 1;
                } else if (inst.classValue() != indexOfGranted && label == indexOfGranted) {
                    fp_non_protected += 1;
                }

            }

               rusBoost.trainOnInstanceImpl(inst);
        }
        if (OPT.equals("SP")) {
            static_monitor_fairness(classified_prot_pos, classified_non_prot_pos, classified_prot_neg, classified_non_prot_neg);
            StParity.add(delayed_discrimination);
        }

        if (OPT.equals("EQOP")) {
            EQOP.add(equal_opportunity(tp_protected, fn_protected, tp_non_protected, fn_non_protected));
        }


        if (OPT.equals("PredPar")) {
            PredPar.add(predictive_parity(tp_protected, fp_protected, tp_non_protected, fp_non_protected));
        }

        Accuracy.add(evaluator.getErrorRate());
        Gmean.add(evaluator.getGmean());
        Kappa.add(evaluator.getKappa());
        F1Score.add(evaluator.getF1Score());
        BACC.add(evaluator.getBACC());
        Recall.add(evaluator.getRecall());
    }

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

    public ArrayList<Double> getEQOP() {
        return EQOP;
    }
}
