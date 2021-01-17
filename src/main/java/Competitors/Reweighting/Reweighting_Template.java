package Competitors.Reweighting;

import OnlineStreamFairness.WindowAUCImbalancedPerformanceEvaluator;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import com.yahoo.labs.samoa.instances.SamoaToWekaInstanceConverter;
import moa.classifiers.trees.HoeffdingAdaptiveTree;
import moa.core.InstanceExample;
import org.apache.log4j.Logger;

import java.util.ArrayList;

import static java.lang.Math.abs;

/**
 * Created by iosifidis on 26.07.19.
 */
public class Reweighting_Template {
    private static int saIndex;
    private static int windowSize;
    private static int indexOfDenied;
    private static int indexOfGranted;
    private static int indexOfDeprived;
    private static double delayed_discrimination;
    private static double window_disc = 0;



    private static int saPos;
    private static int saNeg;
    private static int nSaPos;
    private static int nSaNeg;

    public static HoeffdingAdaptiveTree rwLearner;

    private static ArrayList<Double> Gmean = new ArrayList<Double>();
    private static ArrayList<Double> F1Score = new ArrayList<Double>();
    private static ArrayList<Double> Accuracy = new ArrayList<Double>();
    private static ArrayList<Double> Kappa = new ArrayList<Double>();
    private static ArrayList<Double> StParity = new ArrayList<Double>();
    private static ArrayList<Double> BACC = new ArrayList<Double>();
    private static ArrayList<Double> Recall = new ArrayList<Double>();

    public ArrayList<Double> getBACC() {
        return BACC;
    }

    public ArrayList<Double> getRecall() {
        return Recall;
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

    private final static Logger logger = Logger.getLogger(Reweighting_Template.class.getName());


    public Reweighting_Template(int windowSize, int saIndex, int indexOfDenied, int indexOfGranted, int indexOfDeprived) {
        this.windowSize = windowSize;
        this.saIndex = saIndex;
        this.indexOfGranted = indexOfGranted;
        this.indexOfDeprived = indexOfDeprived;
        this.indexOfDenied = indexOfDenied;
    }


    private static void static_monitor_fairness(double prot_pos, double non_prot_pos,
                                                double prot_neg, double non_prot_neg) {

        double temp_Wdp = prot_pos / (prot_pos + prot_neg + 1);
        double temp_Wfp = non_prot_pos / (non_prot_pos + non_prot_neg + 1);
        delayed_discrimination = temp_Wfp - temp_Wdp;
    }




    public void deploy(Instances buffer) throws Exception {

        delayed_discrimination = 0;
        window_disc = 0;

        rwLearner = new HoeffdingAdaptiveTree();
        rwLearner.setModelContext(new InstancesHeader(buffer));
        rwLearner.prepareForUse();

        double window_classified_prot_pos = 0.0;
        double window_classified_prot_neg = 0.0;

        double window_classified_non_prot_pos = 0.0;
        double window_classified_non_prot_neg = 0.0;

        double tp_protected = 0;
        double fn_protected = 0;
        double tp_non_protected = 0;
        double fn_non_protected = 0;

        double tn_protected = 0;
        double fp_protected = 0;
        double tn_non_protected = 0;
        double fp_non_protected = 0;


        double classified_prot_pos = 0.0;
        double classified_prot_neg = 0.0;

        double classified_non_prot_pos = 0.0;
        double classified_non_prot_neg = 0.0;

        SamoaToWekaInstanceConverter converter = new SamoaToWekaInstanceConverter();
        WindowAUCImbalancedPerformanceEvaluator evaluator = new WindowAUCImbalancedPerformanceEvaluator();
        evaluator.widthOption.setValue(buffer.size());
        evaluator.setIndex(saIndex);
        evaluator.prepareForUse();

        int numberSamples = 0;
        InstanceExample[] windowList = new InstanceExample[windowSize];

        for (int i = 0; i < buffer.size(); i++) {

            Instance trainInst = buffer.get(i);
            InstanceExample trainInstanceExample = new InstanceExample(trainInst);

            windowList[numberSamples % windowSize] = trainInstanceExample;
            numberSamples++;


            double[] votes = rwLearner.getVotesForInstance(trainInst);
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

            evaluator.addResult(trainInstanceExample, votes, indexOfGranted);

            if (trainInst.value(saIndex) == indexOfDeprived) {
                if (label == indexOfGranted) {
                    window_classified_prot_pos++;
                } else {
                    window_classified_prot_neg++;
                }

                if (label == indexOfGranted && trainInst.classValue() == indexOfGranted) {
                    tp_protected += 1;
                } else if (label != indexOfGranted && trainInst.classValue() == indexOfGranted) {
                    fn_protected += 1;
                } else if (label == indexOfDenied && trainInst.classValue() == indexOfDenied) {
                    tn_protected += 1;
                } else if (label != indexOfDenied && trainInst.classValue() == indexOfDenied) {
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

                if (label == indexOfGranted && trainInst.classValue() == indexOfGranted) {
                    tp_non_protected += 1;
                } else if (label != indexOfGranted && trainInst.classValue() == indexOfGranted) {
                    fn_non_protected += 1;
                } else if (label == indexOfDenied && trainInst.classValue() == indexOfDenied) {
                    tn_non_protected += 1;
                } else if (label != indexOfDenied && trainInst.classValue() == indexOfDenied) {
                    fp_non_protected += 1;
                }

            }

            if (numberSamples < 1000)
                rwLearner.trainOnInstance(trainInst);


            saPos = (int) (tp_protected + fn_protected);
            saNeg = (int) (tn_protected + fp_protected);
            nSaPos = (int) (tp_non_protected + fn_non_protected);
            nSaNeg = (int) (tn_non_protected + fp_non_protected);


            statistical_parity(window_classified_prot_pos, window_classified_non_prot_pos, window_classified_prot_neg, window_classified_non_prot_neg);



            if (numberSamples % windowSize == 0) {
                if (abs(window_disc) > 0.001) {
                    windowList = ApplyReweighing(windowList);
                }

                for (int k = 0; k < windowSize - 1; k++)
                    rwLearner.trainOnInstance(windowList[k].instance);

                tp_protected = 0;
                tn_protected = 0;
                tp_non_protected = 0;
                tn_non_protected = 0;
                fn_protected = 0;
                fp_protected = 0;
                fn_non_protected = 0;
                fp_non_protected = 0;
                window_classified_prot_pos = 0;
                window_classified_prot_neg = 0;
                window_classified_non_prot_pos = 0;
                window_classified_non_prot_neg = 0;
            }
        }

        static_monitor_fairness(classified_prot_pos, classified_non_prot_pos, classified_prot_neg, classified_non_prot_neg);
        Accuracy.add(evaluator.getErrorRate());
        Gmean.add(evaluator.getGmean());
        Kappa.add(evaluator.getKappa());
        F1Score.add(evaluator.getF1Score());
        StParity.add(delayed_discrimination);
        BACC.add(evaluator.getBACC());
        Recall.add(evaluator.getRecall());
    }

    public static InstanceExample[] ApplyReweighing(InstanceExample[] windowList) {
        //weight calculation
        double savPos;
        double savNeg;
        double favPos;
        double favNeg;
        if (saPos != 0)
            savPos = (double) (saPos + saNeg) * (double) (saPos + nSaPos) / (double) (windowSize * saPos);
        else
            savPos = 1;
        if (saNeg != 0)
            savNeg = (double) (saPos + saNeg) * (double) (saNeg + nSaNeg) / (double) (windowSize * saNeg);
        else
            savNeg = 1;
        if (nSaPos != 0)
            favPos = (double) (nSaPos + nSaNeg) * (double) (saPos + nSaPos) / (double) (windowSize * nSaPos);
        else
            favPos = 1;
        if (nSaNeg != 0)
            favNeg = (double) (nSaPos + nSaNeg) * (double) (saNeg + nSaNeg) / (double) (windowSize * nSaNeg);
        else
            favNeg = 1;

//        logger.info(savPos +"," +savNeg+"," + favPos+"," +favNeg);
        //apply new weight for the current window
        for (int i = 0; i < windowSize - 1; i++) {
            double cl = windowList[i].instance.classValue();
            if (windowList[i].instance.value(saIndex) == indexOfDeprived) {//Deprived
                if (cl == indexOfGranted)//Positive class
                    windowList[i].instance.setWeight(savPos);
                else
                    windowList[i].instance.setWeight(savNeg);
            } else {
                if (cl == indexOfGranted)//Positive class
                    windowList[i].instance.setWeight(favPos);
                else
                    windowList[i].instance.setWeight(favNeg);
            }
        }
        return windowList;
    }


    private static double statistical_parity(double prot_pos, double non_prot_pos,
                                             double prot_neg, double non_prot_neg) {

        double temp_Wdp = prot_pos / (prot_pos + prot_neg + 1);
        double temp_Wfp = non_prot_pos / (non_prot_pos + non_prot_neg + 1);
        window_disc = temp_Wfp - temp_Wdp;
        return window_disc;
    }


}
