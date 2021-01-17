package Competitors.Massaging;

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
public class Massaging_Template {
    private static int saIndex;
    private static int windowSize;
    private static int indexOfDenied;
    private static int indexOfGranted;
    private static int indexOfDeprived;
    private static double delayed_discrimination;
    private static double window_disc = 0;

    private static String saValue;

    private static int saPos;
    private static int saNeg;
    private static int nSaPos;
    private static int nSaNeg;

    public static HoeffdingAdaptiveTree masLearner;


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
    private final static Logger logger = Logger.getLogger(Massaging_Template.class.getName());


    public Massaging_Template(int windowSize, int saIndex, int indexOfDenied, int indexOfGranted, int indexOfDeprived, String saValue) {
        this.windowSize = windowSize;
        this.saIndex = saIndex;
        this.indexOfGranted = indexOfGranted;
        this.indexOfDeprived = indexOfDeprived;
        this.indexOfDenied = indexOfDenied;
        this.saValue = saValue;
    }


    private static void static_monitor_fairness(double prot_pos, double non_prot_pos,
                                                double prot_neg, double non_prot_neg) {

        double temp_Wdp = prot_pos / (prot_pos + prot_neg + 1);
        double temp_Wfp = non_prot_pos / (non_prot_pos + non_prot_neg + 1);
        delayed_discrimination = temp_Wfp - temp_Wdp;
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

    public void deploy(Instances buffer) throws Exception {
        delayed_discrimination = 0;
        window_disc = 0;

        masLearner = new HoeffdingAdaptiveTree();
        masLearner.setModelContext(new InstancesHeader(buffer));
        masLearner.prepareForUse();


        double window_classified_prot_pos = 0.0;
        double window_classified_prot_neg = 0.0;

        double window_classified_non_prot_pos = 0.0;
        double window_classified_non_prot_neg = 0.0;


        double classified_prot_pos = 0.0;
        double classified_prot_neg = 0.0;

        double classified_non_prot_pos = 0.0;
        double classified_non_prot_neg = 0.0;

        double tp_protected = 0;
        double fn_protected = 0;
        double tp_non_protected = 0;
        double fn_non_protected = 0;

        double tn_protected = 0;
        double fp_protected = 0;
        double tn_non_protected = 0;
        double fp_non_protected = 0;

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

            double[] votes = masLearner.getVotesForInstance(trainInst);
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

            numberSamples++;
//            evaluator.addResult(trainInstanceExample, votes);
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
                masLearner.trainOnInstance(trainInst);

            saPos = (int) (tp_protected + fn_protected);
            saNeg = (int) (tn_protected + fp_protected);
            nSaPos = (int) (tp_non_protected + fn_non_protected);
            nSaNeg = (int) (tn_non_protected + fp_non_protected);

            statistical_parity(window_classified_prot_pos, window_classified_non_prot_pos, window_classified_prot_neg, window_classified_non_prot_neg);

            double changes = 0;

            if (numberSamples % windowSize == 0) {
                //deploy
                if (abs(window_disc) > 0.001) {
                    int saNum = saPos + saNeg;
                    int nSaNum = nSaPos + nSaNeg;

                    changes = ((double) nSaPos * (double) saNum - (double) saPos * (double) nSaNum
                            - (double) (0 / 100) * (double) saNum * (double) nSaNum)
                            / (double) (windowSize);

                    if (changes > 0) { //deploy taking place
                        //ranker
                        weka.classifiers.bayes.NaiveBayes ranker = new weka.classifiers.bayes.NaiveBayes();
                        ranker.buildClassifier(converter.wekaInstances(buffer));

                        WindowAUCImbalancedPerformanceEvaluator ranker_evaluator = new WindowAUCImbalancedPerformanceEvaluator();
                        ranker_evaluator.widthOption.setValue(windowSize);
                        ranker_evaluator.setIndex(saIndex);
                        ranker_evaluator.prepareForUse();

                        for (int k = 0; k < windowList.length; k++) {
                            double[] ranker_votes = ranker.distributionForInstance(converter.wekaInstance(windowList[k].instance));
                            ranker_evaluator.addResult(new InstanceExample(buffer.get(i)), ranker_votes, indexOfGranted);
                        }

                        int[] posWindow = ranker_evaluator.getAucEstimator().getPosWindowFromSortedScores();
                        int[] sortedLabels = ranker_evaluator.getAucEstimator().getsortedLabels();

                        double[] sortedScores = ranker_evaluator.getAucEstimator().getsortedScores();
                        String[] saValFromSortedScores = new String[windowSize];

                        for (int k = 0; k < windowList.length; k++) {
                            saValFromSortedScores[k] = converter.wekaInstance(windowList[posWindow[k] % windowSize].instance).stringValue(saIndex);
                        }

                        windowList = rankingWithSA(posWindow, saValFromSortedScores, sortedLabels, sortedScores, changes, windowList);

                    }
                }

                for (int k = 0; k < windowList.length - 1; k++)
                    masLearner.trainOnInstance(windowList[k].instance);

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
        BACC.add(evaluator.getBACC());
        Recall.add(evaluator.getRecall());
        StParity.add(delayed_discrimination);

    }


    public static InstanceExample[] rankingWithSA(int[] posWindow, String[] saValFromSortedScores, int[] sortedLabels, double[] sortedScores, double changes, InstanceExample[] windowList) {
        double[][] promotionList = new double[windowSize][2];
        double[][] demotionList = new double[windowSize][2];
        int demote = 0, promote = 0;
        for (int i = 0; i < posWindow.length; i++) {
            String sa = saValFromSortedScores[i];
            int classVal = sortedLabels[i];

            if (sa.equals(saValue) && classVal == indexOfDenied) {
                promotionList[promote][0] = posWindow[i] % windowSize;
                promotionList[promote++][1] = sortedScores[i];
            } else if (!sa.equals(saValue) && classVal == indexOfGranted) {
                demotionList[demote][0] = posWindow[i] % windowSize;
                demotionList[demote++][1] = sortedScores[i];
            }
        }//end of for i
        double[][] sortedPromotionList = sorting(promotionList, promote, 1);
        double[][] sortedDemotionList = sorting(demotionList, demote, 2);

        if (changes > sortedDemotionList.length || changes > sortedPromotionList.length) {
            changes = Math.min(sortedDemotionList.length, sortedPromotionList.length);
        }

        for (int i = 0; i < changes; i++) {
            int index = 0;
            index = (int) sortedPromotionList[i][0];
            windowList[index].instance.setClassValue(indexOfGranted);

            index = (int) sortedDemotionList[i][0];
            windowList[index].instance.setClassValue(indexOfDenied);
        }


        return windowList;
    }

    public static double[][] sorting(double[][] arrayToSort, int length, int type) {
        int max = length;
        double val1 = 0, val2 = 0;
        double[][] sortedArray = new double[length][2];
        double[][] temp = new double[1][2];
        for (int index = 0; index < length; index++)
            for (int i = 0; i < max - 1; i++) {
                try {
                    val1 = arrayToSort[i][1];
                    val2 = arrayToSort[i + 1][1];

                    if (val1 < val2 && type == 1) {  //swapping for sort descending
                        System.arraycopy(arrayToSort[i], 0, temp[0], 0, 2);
                        System.arraycopy(arrayToSort[i + 1], 0, arrayToSort[i], 0, 2);
                        System.arraycopy(temp[0], 0, arrayToSort[i + 1], 0, 2);//System.out.println("val1 = "+val1+" new value of rec[] "+rec[i+1][20]+" i = "+i);
                    }     //end of  if
                    else if (val1 > val2 && type == 2) {  //swapping for sort ascending
                        System.arraycopy(arrayToSort[i], 0, temp[0], 0, 2);
                        System.arraycopy(arrayToSort[i + 1], 0, arrayToSort[i], 0, 2);
                        System.arraycopy(temp[0], 0, arrayToSort[i + 1], 0, 2);//System.out.println("val1 = "+val1+" new value of rec[] "+rec[i+1][20]+" i = "+i);
                    }     //end of else if

                } catch (NumberFormatException e) {
                    System.out.println(" Probelme with sorting during Massaging_Template");
                }

            }//end of out for-i loop
        for (int i = 0; i < length; i++)
            System.arraycopy(arrayToSort[i], 0, sortedArray[i], 0, 2);
        return sortedArray;
    }   // End of sorting function


    private static double statistical_parity(double prot_pos, double non_prot_pos,
                                             double prot_neg, double non_prot_neg) {

        double temp_Wdp = prot_pos / (prot_pos + prot_neg + 1);
        double temp_Wfp = non_prot_pos / (non_prot_pos + non_prot_neg + 1);
        window_disc = temp_Wfp - temp_Wdp;
        return window_disc;
    }


}
