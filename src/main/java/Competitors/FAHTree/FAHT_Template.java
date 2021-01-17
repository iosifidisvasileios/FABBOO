package Competitors.FAHTree;

import OnlineStreamFairness.WindowAUCImbalancedPerformanceEvaluator;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.SamoaToWekaInstanceConverter;
import moa.core.InstanceExample;
import org.apache.log4j.Logger;

import java.util.ArrayList;

/**
 * Created by iosifidis on 26.07.19.
 */
public class FAHT_Template {
    private static int saIndex;
    private static int indexOfDenied;
    private static int indexOfGranted;
    private static int indexOfDeprived;
    private static double delayed_discrimination;
    public static Competitors.FAHTree.HoeffdingTree WenBinHT;


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

    private final static Logger logger = Logger.getLogger(FAHT_Template.class.getName());


    public FAHT_Template(int saIndex, int indexOfDenied, int indexOfGranted, int indexOfDeprived) {
        this.saIndex = saIndex;
        this.indexOfGranted = indexOfGranted;
        this.indexOfDeprived = indexOfDeprived;
        this.indexOfDenied = indexOfDenied;
    }


    public void deploy(Instances buffer) throws Exception {
        WenBinHT = new Competitors.FAHTree.HoeffdingTree();
        delayed_discrimination = 0;

        SamoaToWekaInstanceConverter converter = new SamoaToWekaInstanceConverter();
        WindowAUCImbalancedPerformanceEvaluator evaluator = new WindowAUCImbalancedPerformanceEvaluator();
        evaluator.widthOption.setValue(buffer.size());
        evaluator.setIndex(saIndex);
        evaluator.prepareForUse();

        WenBinHT.buildClassifier(new weka.core.Instances(converter.wekaInstances(buffer), 0));


        double classified_prot_pos = 0.0;
        double classified_prot_neg = 0.0;

        double classified_non_prot_pos = 0.0;
        double classified_non_prot_neg = 0.0;

        for (int i = 0; i < buffer.size(); i++) {
            Instance inst = buffer.get(i);
            double[] votes = WenBinHT.distributionForInstance(converter.wekaInstance(inst));

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


            } else {
                if (label == indexOfGranted) {
                    classified_non_prot_pos++;
                } else {
                    classified_non_prot_neg++;
                }
            }


            WenBinHT.updateClassifier(converter.wekaInstance(inst));

        }
        static_monitor_fairness(classified_prot_pos, classified_non_prot_pos, classified_prot_neg, classified_non_prot_neg);

        Accuracy.add(evaluator.getErrorRate());
        Gmean.add(evaluator.getGmean());
        Kappa.add(evaluator.getKappa());
        StParity.add(delayed_discrimination);
        F1Score.add(evaluator.getF1Score());
        BACC.add(evaluator.getBACC());
        Recall.add(evaluator.getRecall());
    }


    private static void static_monitor_fairness(double prot_pos, double non_prot_pos,
                                                double prot_neg, double non_prot_neg) {

        double temp_Wdp = prot_pos / (prot_pos + prot_neg + 1);
        double temp_Wfp = non_prot_pos / (non_prot_pos + non_prot_neg + 1);
        delayed_discrimination = temp_Wfp - temp_Wdp;
    }


}
