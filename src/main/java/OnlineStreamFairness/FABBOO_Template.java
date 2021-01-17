package OnlineStreamFairness;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.classifiers.trees.HoeffdingAdaptiveTree;
import moa.core.InstanceExample;
import org.apache.commons.collections4.queue.CircularFifoQueue;
import org.apache.log4j.Logger;
import weka.core.stemmers.Stemmer;

import java.util.ArrayList;
import java.util.Arrays;

import static java.lang.Math.abs;

/**
 * Created by iosifidis on 26.07.19.
 */
public class FABBOO_Template {
    private int saIndex;

    private int weakL;
    private int indexOfDenied;
    private int indexOfGranted;
    private int indexOfDeprived;
    private double delayed_discrimination;
    private String OPT;
    private boolean StreamEval = false;


    public ArrayList<Double> getBACC() {
        return BACC;
    }

    public ArrayList<Double> getRecall() {
        return Recall;
    }

    private double class_lamda = 0.8;

    private double Wp = 0.0;
    private double Wn = 0.0;

    public double getClass_lamda() {
        return class_lamda;
    }

    public void setClass_lamda(double class_lamda) {
        this.class_lamda = class_lamda;
    }

    public ArrayList<Double> getAccuracy() {
        return Accuracy;
    }

    public ArrayList<Double> getF1Score() {
        return F1Score;
    }

    public ArrayList<Double> getGmean() {
        return Gmean;
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

    public ArrayList<Double> getTime() {
        return time;
    }


    private CircularFifoQueue<Double> buf_predictions = new CircularFifoQueue<Double>(2000);


    private static ArrayList<Double> Gmean = new ArrayList<Double>();
    private static ArrayList<Double> F1Score = new ArrayList<Double>();
    private static ArrayList<Double> Accuracy = new ArrayList<Double>();
    private static ArrayList<Double> Kappa = new ArrayList<Double>();
    private static ArrayList<Double> StParity = new ArrayList<Double>();
    private static ArrayList<Double> Thresholds = new ArrayList<Double>();
    private static ArrayList<Double> EQOP = new ArrayList<Double>();
    private static ArrayList<Double> time = new ArrayList<Double>();
    private static ArrayList<Double> BACC = new ArrayList<Double>();
    private static ArrayList<Double> Recall = new ArrayList<Double>();

//    private final static Logger logger = Logger.getLogger(FABBOO_Template.class.getName());


    public FABBOO_Template(int weakL, int saIndex, int indexOfDenied, int indexOfGranted, int indexOfDeprived, String OPT) {
        this.weakL = weakL;
        this.saIndex = saIndex;
        this.indexOfGranted = indexOfGranted;
        this.indexOfDeprived = indexOfDeprived;
        this.indexOfDenied = indexOfDenied;
        this.OPT = OPT;
    }

    public FABBOO_Template(int weakL, double lambda, int saIndex, int indexOfDenied, int indexOfGranted, int indexOfDeprived, String OPT) {
        this.weakL = weakL;
        setClass_lamda(lambda);
        this.saIndex = saIndex;
        this.indexOfGranted = indexOfGranted;
        this.indexOfDeprived = indexOfDeprived;
        this.indexOfDenied = indexOfDenied;
        this.OPT = OPT;
    }

    public FABBOO_Template(int weakL, double lambda, int window, int saIndex, int indexOfDenied, int indexOfGranted, int indexOfDeprived, String OPT) {
        this.buf_predictions = new CircularFifoQueue<Double>(window);
        this.weakL = weakL;
        setClass_lamda(lambda);
        this.saIndex = saIndex;
        this.indexOfGranted = indexOfGranted;
        this.indexOfDeprived = indexOfDeprived;
        this.indexOfDenied = indexOfDenied;
        this.OPT = OPT;
    }
    public FABBOO_Template(int weakL, double lambda, int window, int saIndex, int indexOfDenied, int indexOfGranted, int indexOfDeprived, String OPT, boolean StreamEval) {
        this.StreamEval = StreamEval;
        this.buf_predictions = new CircularFifoQueue<Double>(window);
        this.weakL = weakL;
        setClass_lamda(lambda);
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
        this.delayed_discrimination = temp_Wfp - temp_Wdp;
    }

    private static double equal_opportunity(double tp_protected, double fn_protected, double tp_non_protected, double fn_non_protected) {
        return tp_non_protected / (tp_non_protected + fn_non_protected) - tp_protected / (tp_protected + fn_protected);
    }


    public void deploy(Instances buffer) {

        FABBOO fairBoosting = new FABBOO(indexOfDeprived, saIndex, indexOfGranted, indexOfDenied, OPT);
        fairBoosting.ensembleSizeOption.setValue(weakL);
        fairBoosting.baseLearnerOption.setCurrentObject(new HoeffdingAdaptiveTree());
        fairBoosting.setModelContext(new InstancesHeader(buffer));
        fairBoosting.prepareForUse();

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
        int pos = 0;
        int neg = 0;
        double tp_protected = 0;
        double fn_protected = 0;
        double tp_non_protected = 0;
        double fn_non_protected = 0;

        long timeStart = System.nanoTime();
        for (int i = 0; i < buffer.size(); i++) {
            if (i == 0)
                Thresholds.add(0.5);

            boolean targetClass = false;
            Instance inst = buffer.get(i);

            if (inst.classValue() == indexOfGranted) {
                targetClass = true;
                pos++;
            } else {
                neg++;
            }

            update_class_rates(pos, neg);
            pos = neg = 0;

            double[] votes = fairBoosting.getVotesForInstance(inst);
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
                        // misclassifed positive protected instance
                        try {
                            buf_predictions.add(votes[indexOfGranted]);
                        } catch (ArrayIndexOutOfBoundsException e) {
                            // has predicted negative class 100%
                            buf_predictions.add(0.);
                        }
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
                    int position = shifted_location(classified_prot_pos, classified_non_prot_pos, classified_prot_neg, classified_non_prot_neg);
                    Thresholds.add(fairBoosting.tweak_boundary(buf_predictions, position));
                } else {
                    Thresholds.add(Thresholds.get(Thresholds.size() - 1));
                }

                if(StreamEval){
                    StParity.add(delayed_discrimination);
                }


            }

            if (OPT.equals("EQOP")) {
                double delayed_EQOP = equal_opportunity(tp_protected, fn_protected, tp_non_protected, fn_non_protected);
                if (abs(delayed_EQOP) >= 0.001) {
                    int position = shifted_location(tp_protected, tp_non_protected, fn_protected, fn_non_protected);
                    Thresholds.add(fairBoosting.tweak_boundary(buf_predictions, position));
                } else {
                    Thresholds.add(Thresholds.get(Thresholds.size() - 1));
                }
                if(StreamEval){
                    EQOP.add(delayed_EQOP);
                }
            }
            if(StreamEval){
                Accuracy.add(evaluator.getErrorRate());
                Gmean.add(evaluator.getGmean());
                Kappa.add(evaluator.getKappa());
                F1Score.add(evaluator.getF1Score());
                BACC.add(evaluator.getBACC());
                Recall.add(evaluator.getRecall());

            }
            fairBoosting.trainInstanceImbalance(inst, targetClass, Wn - Wp);

        }
        if(!StreamEval) {
            if (OPT.equals("SP")) {
                StParity.add(delayed_discrimination);
            }
        }
        if(!StreamEval) {

            if (OPT.equals("EQOP")) {
                EQOP.add(equal_opportunity(tp_protected, fn_protected, tp_non_protected, fn_non_protected));
            }
        }

        double elapsedTime = (double) (System.nanoTime() - timeStart) / 1000000000;

        time.add(elapsedTime);
        if(!StreamEval) {

            Accuracy.add(evaluator.getErrorRate());
            Gmean.add(evaluator.getGmean());
            Kappa.add(evaluator.getKappa());
            F1Score.add(evaluator.getF1Score());
            BACC.add(evaluator.getBACC());
            Recall.add(evaluator.getRecall());
        }
    }
    private static int shifted_location(double classified_prot_pos, double classified_non_prot_pos, double classified_prot_neg, double classified_non_prot_neg) {
        return (int) ((classified_prot_neg + classified_prot_pos) *
                (
                        (classified_non_prot_pos) / (classified_non_prot_pos + classified_non_prot_neg)
                )
                - classified_prot_pos);
    }

    private void update_class_rates(double positives, double negatives) {
        Wp = class_lamda * Wp + (1 - class_lamda) * positives;
        Wn = class_lamda * Wn + (1 - class_lamda) * negatives;
        double sum = Wp + Wn + 0.1;
        Wp = Wp / (sum);
        Wn = Wn / (sum);
    }

/*
    private double predictive_parity(double tp_protected, double fp_protected, double tp_non_protected, double fp_non_protected) {
        return tp_non_protected / (tp_non_protected + fp_non_protected) - tp_protected / (tp_protected + fp_protected);
    }
*/

}
