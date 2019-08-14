package Experiments;

import Competitors.HellingerTree.GHVFDT;
import OnlineStreamFairness.CFBB;
import OnlineStreamFairness.OFBB;
import OnlineStreamFairness.WindowAUCImbalancedPerformanceEvaluator;
import com.yahoo.labs.samoa.instances.*;
import moa.classifiers.meta.OnlineSmoothBoost;
import moa.classifiers.trees.HoeffdingAdaptiveTree;
import moa.core.InstanceExample;
import org.apache.commons.collections4.queue.CircularFifoQueue;
import org.apache.log4j.Logger;
import weka.core.converters.ArffLoader;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import static java.lang.Math.abs;

/**
 * Created by iosifidis on 01.07.19.
 */
public class OutputStreamExperiments {

    public static OFBB FairBoosting;
    public static CFBB FairChunkBoosting;
    public static OFBB FairImbaBoosting;
    public static OnlineSmoothBoost OnlineBoost;
    public static HoeffdingAdaptiveTree rwLearner;
    public static HoeffdingAdaptiveTree masLearner;
    public static Competitors.FAHTree.HoeffdingTree WenBinHT;


    private static ArrayList<Double> accuracyFair = new ArrayList<Double>();
    private static ArrayList<Double> F1Fair = new ArrayList<Double>();
    private static ArrayList<Double> gmeanFair = new ArrayList<Double>();
    private static ArrayList<Double> kappaFair = new ArrayList<Double>();
    private static ArrayList<Double> StatParFair = new ArrayList<Double>();
    private static ArrayList<Double> thresholdFair = new ArrayList<Double>();
    private static ArrayList<Double> EQOPFair = new ArrayList<Double>();
    private static ArrayList<Double> recallFair = new ArrayList<Double>();
    private static ArrayList<Double> balaccFair = new ArrayList<Double>();

    private static ArrayList<Double> gmeanFairChunk = new ArrayList<Double>();
    private static ArrayList<Double> F1FairChunk = new ArrayList<Double>();
    private static ArrayList<Double> accuracyFairChunk = new ArrayList<Double>();
    private static ArrayList<Double> kappaFairChunk = new ArrayList<Double>();
    private static ArrayList<Double> StatParFairChunk = new ArrayList<Double>();
    private static ArrayList<Double> thresholdFairChunk = new ArrayList<Double>();
    private static ArrayList<Double> EQOPFairChunk = new ArrayList<Double>();
    private static ArrayList<Double> recallFairChunk = new ArrayList<Double>();
    private static ArrayList<Double> balaccFairChunk = new ArrayList<Double>();

    private static ArrayList<Double> gmeanFairImb = new ArrayList<Double>();
    private static ArrayList<Double> F1FairImb = new ArrayList<Double>();
    private static ArrayList<Double> accuracyFairImb = new ArrayList<Double>();
    private static ArrayList<Double> kappaFairImb = new ArrayList<Double>();
    private static ArrayList<Double> StatParFairImb = new ArrayList<Double>();
    private static ArrayList<Double> thresholdFairImb = new ArrayList<Double>();
    private static ArrayList<Double> EQOPFairImb = new ArrayList<Double>();
    private static ArrayList<Double> recallFairImb = new ArrayList<Double>();
    private static ArrayList<Double> balaccFairImb = new ArrayList<Double>();

    private static ArrayList<Double> gmeanSimpleBoost = new ArrayList<Double>();
    private static ArrayList<Double> F1SimpleBoost = new ArrayList<Double>();
    private static ArrayList<Double> accuracySimpleBoost = new ArrayList<Double>();
    private static ArrayList<Double> kappaSimpleBoost = new ArrayList<Double>();
    private static ArrayList<Double> StatParSimpleBoost = new ArrayList<Double>();
    private static ArrayList<Double> EQOPSimpleBoost = new ArrayList<Double>();
    private static ArrayList<Double> recallSimpleBoost = new ArrayList<Double>();
    private static ArrayList<Double> balaccSimpleBoost = new ArrayList<Double>();

    private static ArrayList<Double> gmeanWBTree = new ArrayList<Double>();
    private static ArrayList<Double> F1WBTree = new ArrayList<Double>();
    private static ArrayList<Double> accuracyWBTree = new ArrayList<Double>();
    private static ArrayList<Double> kappaWBTree = new ArrayList<Double>();
    private static ArrayList<Double> StatParWBTree = new ArrayList<Double>();
    private static ArrayList<Double> recallWBTree = new ArrayList<Double>();
    private static ArrayList<Double> balaccWBTree = new ArrayList<Double>();

    private static ArrayList<Double> gmeanRW = new ArrayList<Double>();
    private static ArrayList<Double> F1RW = new ArrayList<Double>();
    private static ArrayList<Double> accuracyRW = new ArrayList<Double>();
    private static ArrayList<Double> kappaRW = new ArrayList<Double>();
    private static ArrayList<Double> StatParRW = new ArrayList<Double>();
    private static ArrayList<Double> recallRW = new ArrayList<Double>();
    private static ArrayList<Double> balaccRW = new ArrayList<Double>();

    private static ArrayList<Double> gmeanMAS = new ArrayList<Double>();
    private static ArrayList<Double> F1MAS = new ArrayList<Double>();
    private static ArrayList<Double> accuracyMAS = new ArrayList<Double>();
    private static ArrayList<Double> kappaMAS = new ArrayList<Double>();
    private static ArrayList<Double> StatParMAS = new ArrayList<Double>();
    private static ArrayList<Double> recallMAS = new ArrayList<Double>();
    private static ArrayList<Double> balaccMAS = new ArrayList<Double>();

    private static ArrayList<Double> gmeanGHTree = new ArrayList<Double>();
    private static ArrayList<Double> F1GHTree = new ArrayList<Double>();
    private static ArrayList<Double> accuracyGHTree = new ArrayList<Double>();
    private static ArrayList<Double> kappaGHTree = new ArrayList<Double>();
    private static ArrayList<Double> StatParGHTree = new ArrayList<Double>();
    private static ArrayList<Double> EQOPGHTree = new ArrayList<Double>();
    private static ArrayList<Double> recallGHTree = new ArrayList<Double>();
    private static ArrayList<Double> balaccGHTree = new ArrayList<Double>();

    private static int saPos;
    private static int saNeg;
    private static int nSaPos;
    private static int nSaNeg;

    private static int windowSize = 1000;


    private static final CircularFifoQueue<Double> buf_predictions = new CircularFifoQueue<Double>(5000);

    private final static Logger logger = Logger.getLogger(OutputStreamExperiments.class.getName());

    private static String OPT; // OPT for Statistical Parity "SP" or Equal Opportunity "EQOP"
    private static String saName; // sensitive attribute name
    private static String saValue; // sensitive attribute value

    //    private static String saValue = "Female"; // sensitive attribute value
    private static int saIndex; // index of sensitive attribute

    private static ArrayList<Double> SP_rewe_ephimeral = new ArrayList<Double>();
    private static ArrayList<Double> SP_fair_ephimeral = new ArrayList<Double>();
    private static ArrayList<Double> SP_fair_imb_ephimeral = new ArrayList<Double>();
    private static ArrayList<Double> SP_masa_ephimeral = new ArrayList<Double>();
    private static ArrayList<Double> SP_fair_chunk_ephimeral = new ArrayList<Double>();
    private static ArrayList<Double> SP_wen_ephimeral = new ArrayList<Double>();
    private static ArrayList<Double> SP_boost_ephimeral = new ArrayList<Double>();
/*
    private static ArrayList<Double> EQOP_rewe_ephimeral = new ArrayList<Double>();
    private static ArrayList<Double> EQOP_fair_ephimeral = new ArrayList<Double>();
    private static ArrayList<Double> EQOP_fair_imb_ephimeral = new ArrayList<Double>();
    private static ArrayList<Double> EQOP_masa_ephimeral = new ArrayList<Double>();
    private static ArrayList<Double> EQOP_fair_chunk_ephimeral = new ArrayList<Double>();
    private static ArrayList<Double> EQOP_wen_ephimeral = new ArrayList<Double>();
    private static ArrayList<Double> EQOP_boost_ephimeral = new ArrayList<Double>();*/


    public static double current_sliding_disc = 0;
    public static double EQOP = 0;
    private static double window_disc = 0;

    public static double delayed_discrimination = 0;


    public static int global_counter = 0;

    public static double Wp = 0;
    public static double Wn = 0;

/*    public static double Wdp = 0;
    public static double Wfp = 0;
    public static double Wdn = 0;
    public static double Wfn = 0;*/

    public static double pos = 0.0;
    public static double neg = 0.0;

    public static double original_prot_pos = 0.0;
    public static double original_non_prot_pos = 0.0;

    public static double classified_prot_pos = 0.0;
    public static double classified_prot_neg = 0.0;

    public static double classified_non_prot_pos = 0.0;
    public static double classified_non_prot_neg = 0.0;


    public static String favored;
    public static String targetClass;
    public static String otherClass;
    public static String arffInputFileName;
    public static String outputFileName;

    private static int indexOfDeprived; // sensitive attribute: female
    private static int indexOfUndeprived; // sensitive attribute: male
    private static int indexOfDenied; // class label: income <=50k
    private static int indexOfGranted; // class label: income > 50k
    private static double class_lamda = 0.9;

    private static GHVFDT GHvfdt;

    private static void stats(weka.core.Instances stream) {
        int pos_cnt = 0;
        int neg_cnt = 0;
        int total = 0;
        for(weka.core.Instance iii : stream){
            total+=iii.classValue();
            if (iii.classValue()==indexOfGranted){
                pos_cnt+=1;
            }else{
                neg_cnt+=1;
            }
        }


        logger.info("positives = " + pos_cnt);
        logger.info("negatives = " + neg_cnt);
        logger.info("total = " + total);

        logger.info("dataset size  = " + stream.size());
        logger.info("numAttributes = " + stream.numAttributes());
    }

    public static void main(String[] args) throws Exception {
//        String datasetString = args[0];
//        windowSize = Integer.valueOf(args[1]);
//        OPT = args[2];
//        boolean synth = Boolean.valueOf(args[3]);
        String datasetString = "nypd";
        logger.info("dataset = " + datasetString);
        OPT = "SP";
        init_dataset(datasetString);

        outputFileName = "Tables/Stream/" + datasetString + "_tune_" + OPT + "_";


        ArffLoader.ArffReader arffReader = new ArffLoader.ArffReader(new FileReader(arffInputFileName));
        weka.core.Instances stream = arffReader.getData();

        if (datasetString.equals("nypd"))
            stream.setClassIndex(3);
        else
            stream.setClassIndex(stream.numAttributes() - 1);

        saIndex = stream.attribute(saName).index();
        indexOfDeprived = stream.attribute(saName).indexOfValue(saValue); // M:0 F:1
        indexOfUndeprived = stream.attribute(saName).indexOfValue(favored);
        indexOfDenied = stream.classAttribute().indexOfValue(otherClass); // <=50K: 0, >50K: 1
        indexOfGranted = stream.classAttribute().indexOfValue(targetClass);



        Instances dataset = new WekaToSamoaInstanceConverter().samoaInstances(stream);
        Instances currentWindow = new Instances(dataset, 0);
        if (!datasetString.equals("synthetic") && !datasetString.equals("nypd"))
            dataset.randomize(new Random(0));

        init_models(20, currentWindow);
        stats(stream);
//        if (synth) {
//            weka.core.Instances synethtic = swapLabels(stream);
//            for (weka.core.Instance inst : synethtic)
//                dataset.add(converter.samoaInstance(inst));
//        }

        reset_parameters();

        if (OPT.equals("SP")) {
            logger.info("WenBin Tree");
            runWenBinTree(dataset);
            reset_parameters();

            logger.info("Massaging_Template");
            massaging(dataset);
            reset_parameters();

            logger.info("Reweighting_Template");
            reweighting(dataset);
            reset_parameters();
        }


        logger.info("Fair & balanced Boosting");
        runFairModel(dataset);
        reset_parameters();

        logger.info("Fair & IM-balanced Boosting");
        runFairImbalancedModel(dataset);
        reset_parameters();
//
        logger.info("Fair Chunk Based Boosting");
        runChunkFairModel(dataset);
        reset_parameters();

        logger.info("No Fairness Boosting");
        runSimpleBoosting(dataset);
        reset_parameters();

        logger.info("No Fairness GHTree");
        runGHTree(dataset);
        reset_parameters();

        if (OPT.equals("SP"))
            flushToFiles(outputFileName);
        else
            flushToFilesForEQOP(outputFileName);

    }


    private static void flushToFilesForEQOP(String outputFileName) throws IOException {
        BufferedWriter br = new BufferedWriter(new FileWriter(new File(outputFileName + "accuracy.csv")));
        br.write("OnlineBoost, GHVFDT, FairBoost, FairImbaBoost, FairChunkBoost\n");
        for (int i = 0; i < accuracySimpleBoost.size(); i++)
            br.write(accuracySimpleBoost.get(i) + "," + accuracyGHTree.get(i) + "," + accuracyFair.get(i) + "," + accuracyFairImb.get(i) + "," + accuracyFairChunk.get(i) + "\n");
        br.close();

        br = new BufferedWriter(new FileWriter(new File(outputFileName + "kappa.csv")));
        br.write("OnlineBoost, GHVFDT, FairBoost, FairImbaBoost, FairChunkBoost\n");
        for (int i = 0; i < kappaSimpleBoost.size(); i++)
            br.write(kappaSimpleBoost.get(i) + "," + kappaGHTree.get(i) + "," + kappaFair.get(i) + "," + kappaFairImb.get(i) + "," + kappaFairChunk.get(i) + "\n");
        br.close();


        br = new BufferedWriter(new FileWriter(new File(outputFileName + "F1Score.csv")));
        br.write("OnlineBoost, GHVFDT, FairBoost, FairImbaBoost, FairChunkBoost\n");
        for (int i = 0; i < F1SimpleBoost.size(); i++)
            br.write(F1SimpleBoost.get(i) + "," + F1GHTree.get(i) + "," + F1Fair.get(i) + "," + F1FairImb.get(i) + "," + F1FairChunk.get(i) + "\n");
        br.close();


        br = new BufferedWriter(new FileWriter(new File(outputFileName + "gmean.csv")));
        br.write("OnlineBoost, GHVFDT, FairBoost, FairImbaBoost, FairChunkBoost\n");
        for (int i = 0; i < gmeanFairChunk.size(); i++)
            br.write(gmeanSimpleBoost.get(i) + "," + gmeanGHTree.get(i) + "," + gmeanFair.get(i) + "," + gmeanFairImb.get(i) + "," + gmeanFairChunk.get(i) + "\n");
        br.close();

        br = new BufferedWriter(new FileWriter(new File(outputFileName + "recall.csv")));
        br.write("OnlineBoost, GHVFDT, FairBoost, FairImbaBoost, FairChunkBoost\n");
        for (int i = 0; i < recallFairChunk.size(); i++)
            br.write(recallSimpleBoost.get(i) + "," + recallGHTree.get(i) + "," + recallFair.get(i) + "," + recallFairImb.get(i) + "," + recallFairChunk.get(i) + "\n");
        br.close();

        br = new BufferedWriter(new FileWriter(new File(outputFileName + "bacc.csv")));
        br.write("OnlineBoost, GHVFDT, FairBoost, FairImbaBoost, FairChunkBoost\n");
        for (int i = 0; i < balaccFairChunk.size(); i++)
            br.write(balaccSimpleBoost.get(i) + "," + balaccGHTree.get(i) + "," + balaccFair.get(i) + "," + balaccFairImb.get(i) + "," + balaccFairChunk.get(i) + "\n");
        br.close();


        br = new BufferedWriter(new FileWriter(new File(outputFileName + "boundary.csv")));
        br.write("FairBoost, FairImbaBoost, FairChunkBoost\n");
        for (int i = 0; i < thresholdFair.size() - 1; i++)
            br.write(thresholdFair.get(i) + "," + thresholdFairImb.get(i) + "," + thresholdFairChunk.get(i) + "\n");
        br.close();

        br = new BufferedWriter(new FileWriter(new File(outputFileName + "discrimination_equal_opportunity.csv")));
        br.write("OnlineBoost, GHVFDT, FairBoost, FairImbaBoost, FairChunkBoost\n");
        for (int i = 0; i < EQOPSimpleBoost.size(); i++)
            br.write(EQOPSimpleBoost.get(i) + "," + EQOPGHTree.get(i) + "," + EQOPFair.get(i) + "," + EQOPFairImb.get(i) + "," + EQOPFairChunk.get(i) + "\n");
        br.close();

    }

    private static void flushToFiles(String outputFileName) throws IOException {
        BufferedWriter br = new BufferedWriter(new FileWriter(new File(outputFileName + "accuracy.csv")));
        br.write("OnlineBoost, GHVFDT, WenBin, Massaging_Template, Reweighting_Template, FairBoost, FairImbaBoost, FairChunkBoost\n");
        for (int i = 0; i < accuracySimpleBoost.size(); i++)
            br.write(accuracySimpleBoost.get(i) + "," + accuracyGHTree.get(i) + "," + accuracyWBTree.get(i) + "," + accuracyMAS.get(i) + "," +
                    accuracyRW.get(i) + "," + accuracyFair.get(i) + "," + accuracyFairImb.get(i) + "," + accuracyFairChunk.get(i) + "\n");
        br.close();

        br = new BufferedWriter(new FileWriter(new File(outputFileName + "kappa.csv")));
        br.write("OnlineBoost, GHVFDT, WenBin, Massaging_Template, Reweighting_Template, FairBoost, FairImbaBoost, FairChunkBoost\n");
        for (int i = 0; i < kappaSimpleBoost.size(); i++)
            br.write(kappaSimpleBoost.get(i) + "," + kappaGHTree.get(i) + "," + kappaWBTree.get(i) + "," + kappaMAS.get(i) + "," +
                    kappaRW.get(i) + "," + kappaFair.get(i) + "," + kappaFairImb.get(i) + "," + kappaFairChunk.get(i) + "\n");
        br.close();

        br = new BufferedWriter(new FileWriter(new File(outputFileName + "F1Score.csv")));
        br.write("OnlineBoost, GHVFDT, WenBin, Massaging_Template, Reweighting_Template, FairBoost, FairImbaBoost, FairChunkBoost\n");
        for (int i = 0; i < F1SimpleBoost.size(); i++)
            br.write(F1SimpleBoost.get(i) + "," + F1GHTree.get(i) + "," + F1WBTree.get(i) + "," + F1MAS.get(i) + "," +
                    F1RW.get(i) + "," + F1Fair.get(i) + "," + F1FairImb.get(i) + "," + F1FairChunk.get(i) + "\n");
        br.close();

        br = new BufferedWriter(new FileWriter(new File(outputFileName + "gmean.csv")));
        br.write("OnlineBoost, GHVFDT, WenBin, Massaging_Template, Reweighting_Template, FairBoost, FairImbaBoost, FairChunkBoost\n");
        for (int i = 0; i < gmeanSimpleBoost.size(); i++)
            br.write(gmeanSimpleBoost.get(i) + "," + gmeanGHTree.get(i) + "," + gmeanWBTree.get(i) + "," + gmeanMAS.get(i) + "," +
                    gmeanRW.get(i) + "," + gmeanFair.get(i) + "," + gmeanFairImb.get(i) + "," + gmeanFairChunk.get(i) + "\n");
        br.close();

        br = new BufferedWriter(new FileWriter(new File(outputFileName + "recall.csv")));
        br.write("OnlineBoost, GHVFDT, WenBin, Massaging_Template, Reweighting_Template, FairBoost, FairImbaBoost, FairChunkBoost\n");
        for (int i = 0; i < recallSimpleBoost.size(); i++)
            br.write(recallSimpleBoost.get(i) + "," + recallGHTree.get(i) + "," + recallWBTree.get(i) + "," + recallMAS.get(i) + "," +
                    recallRW.get(i) + "," + recallFair.get(i) + "," + recallFairImb.get(i) + "," + recallFairChunk.get(i) + "\n");
        br.close();


        br = new BufferedWriter(new FileWriter(new File(outputFileName + "bacc.csv")));
        br.write("OnlineBoost, GHVFDT, WenBin, Massaging_Template, Reweighting_Template, FairBoost, FairImbaBoost, FairChunkBoost\n");
        for (int i = 0; i < balaccSimpleBoost.size(); i++)
            br.write(balaccSimpleBoost.get(i) + "," + balaccGHTree.get(i) + "," + balaccWBTree.get(i) + "," + balaccMAS.get(i) + "," +
                    balaccRW.get(i) + "," + balaccFair.get(i) + "," + balaccFairImb.get(i) + "," + balaccFairChunk.get(i) + "\n");
        br.close();

        br = new BufferedWriter(new FileWriter(new File(outputFileName + "boundary.csv")));
        br.write("FairBoost, FairImbaBoost, FairChunkBoost\n");
        for (int i = 0; i < thresholdFair.size() - 1; i++)
            br.write(thresholdFair.get(i) + "," + thresholdFairImb.get(i) + "," + thresholdFairChunk.get(i) + "\n");
        br.close();

        br = new BufferedWriter(new FileWriter(new File(outputFileName + "discrimination_statistical_parity.csv")));
        br.write("OnlineBoost, GHVFDT, WenBin, Massaging_Template, Reweighting_Template, FairBoost, FairImbaBoost, FairChunkBoost\n");
        for (int i = 0; i < StatParSimpleBoost.size(); i++)
            br.write(StatParSimpleBoost.get(i) + "," + StatParGHTree.get(i) + "," + StatParWBTree.get(i) + "," + StatParMAS.get(i) + "," +
                    StatParRW.get(i) + "," + StatParFair.get(i) + "," + StatParFairImb.get(i) + "," + StatParFairChunk.get(i) + "\n");
        br.close();
    }

    public static void reweighting(Instances buffer) throws Exception {

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

            evaluator.addResultForEvaluation(trainInstanceExample, votes, trainInstanceExample.instance.classValue() == indexOfGranted);

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


            SP_rewe_ephimeral.add(statistical_parity(window_classified_prot_pos, window_classified_non_prot_pos, window_classified_prot_neg, window_classified_non_prot_neg));

            static_monitor_fairness(classified_prot_pos, classified_non_prot_pos, classified_prot_neg, classified_non_prot_neg);


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
            accuracyRW.add(evaluator.getErrorRate());
            gmeanRW.add(evaluator.getGmean());
            kappaRW.add(evaluator.getKappa());
            F1RW.add(evaluator.getF1Score());
            StatParRW.add(delayed_discrimination);
            recallRW.add(evaluator.getRecall());
            balaccRW.add(evaluator.getBACC());

        }
        logger.info("tp " + evaluator.getAucEstimator().getCorrectPosPred() + ", positives = " + evaluator.getAucEstimator().getNumPos());
        logger.info("recall = " + evaluator.getRecall());
//        logger.info("ephimeral parity = " + Stats.meanOf(SP_rewe_ephimeral));


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


    private static void init_dataset(String datasetString) {
        if (datasetString.equals("adult-gender")) {
            arffInputFileName = "Data/adult.arff";
            saValue = " Female";
            favored = " Male";
            saName = "sex";
            targetClass = " >50K";
            otherClass = " <=50K";
        } else if (datasetString.equals("adult-race")) {
            arffInputFileName = "Data/adult.arff";
            saValue = " Minorities";
            favored = " White";
            saName = "race";
            targetClass = " >50K";
            otherClass = " <=50K";
        } else if (datasetString.equals("kdd")) {
            arffInputFileName = "Data/kdd.arff";
            saValue = "Female";
            saName = "sex";
            favored = "Male";
            targetClass = "1";
            otherClass = "0";
        } else if (datasetString.equals("bank")) {
            arffInputFileName = "Data/bank-full.arff";
            targetClass = "yes";
            otherClass = "no";
            saName = "marital";
            saValue = "married";
            favored = "single";
        } else if (datasetString.equals("synthetic")) {
            arffInputFileName = "Data/synthetic.arff";
            targetClass = "0";
            otherClass = "1";
            saName = "SA";
            saValue = "Female";
            favored = "Male";
        } else if (datasetString.equals("default")) {
            arffInputFileName = "Data/DefaultDataset.arff";
            saValue = "female";
            favored = "male";
            saName = "SEX";
            targetClass = "1";
            otherClass = "0";
        } else if (datasetString.equals("dutch")) {
            arffInputFileName = "Data/dutch.arff";
            saValue = "2";
            favored = "1";
            saName = "sex";
            targetClass = "2_1"; // high level ?
            otherClass = "5_4_9";
        } else if (datasetString.equals("compass")) {
            arffInputFileName = "Data/compass_zafar.arff";
            saName = "sex";
            saValue = "0";
            favored = "1";
            targetClass = "1";
            otherClass = "-1";
        } else if (datasetString.equals("nypd")) {
            arffInputFileName = "Data/NYPD_COMPLAINT.arff";
            saName = "SUSP_SEX";
            saValue = "F";
            favored = "M";
            targetClass = "FELONY";
            otherClass = "MISDEMEANOR";
        }
    }

    private static void runFairModel(Instances buffer) throws Exception {
        WindowAUCImbalancedPerformanceEvaluator evaluator = new WindowAUCImbalancedPerformanceEvaluator();
        evaluator.widthOption.setValue(buffer.size());
        evaluator.setIndex(saIndex);
        evaluator.prepareForUse();

        double tp_protected = 0;
        double fn_protected = 0;
        double tp_non_protected = 0;
        double fn_non_protected = 0;

        int fairnessCase = 0;
        for (int i = 0; i < buffer.size(); i++) {
            global_counter += 1;

            if (i == 0)
                thresholdFair.add(0.5);

            boolean targetClass = false;
            Instance inst = buffer.get(i);

            if (inst.classValue() == indexOfGranted) {
                targetClass = true;
                pos++;
            } else {
                neg++;
            }

            update_class_rates(pos, neg);

            double[] votes = FairBoosting.getVotesForInstance(inst);
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

//            evaluator.addResult(new InstanceExample(inst), votes);
            evaluator.addResultForEvaluation(new InstanceExample(inst), votes, inst.classValue() == indexOfGranted);

            if (inst.value(saIndex) == indexOfDeprived) {
                if (label == indexOfGranted) {
                    fairnessCase = 1;
                    classified_prot_pos++;
                } else {
                    fairnessCase = 2;
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
                }

            } else {
                if (label == indexOfGranted) {
                    fairnessCase = 3;
                    classified_non_prot_pos++;
                } else {
                    fairnessCase = 4;
                    classified_non_prot_neg++;
                }

                if (label == indexOfGranted && inst.classValue() == indexOfGranted) {
                    tp_non_protected += 1;

                } else if (inst.classValue() == indexOfGranted && label != indexOfGranted) {
                    fn_non_protected += 1;

                }
            }
/*            double decayedFairness =0;
            if (fairnessCase == 1)
                decayedFairness = update_decayed_fairness(1,0,0,0);
            else if (fairnessCase == 2)
                decayedFairness = update_decayed_fairness(0,0,1,0);
            else if (fairnessCase == 3)
                decayedFairness = update_decayed_fairness(0,1,0,0);
            else
                decayedFairness = update_decayed_fairness(0,0,0,1);*/

            if (OPT.equals("SP")) {
                SP_fair_ephimeral.add(statistical_parity(classified_prot_pos, classified_non_prot_pos, classified_prot_neg, classified_non_prot_neg));
                static_monitor_fairness(classified_prot_pos, classified_non_prot_pos, classified_prot_neg, classified_non_prot_neg);

                if (abs(delayed_discrimination) >= 0.001) {
//                if (abs(decayedFairness) >= 0.001) {
                    int position = shifted_location(classified_prot_pos, classified_non_prot_pos, classified_prot_neg, classified_non_prot_neg);
                    thresholdFair.add(FairBoosting.tweak_boundary(buf_predictions, position));
                } else {
                    thresholdFair.add(thresholdFair.get(thresholdFair.size() - 1));
                }
                StatParFair.add(delayed_discrimination);
            }

            if (OPT.equals("EQOP")) {
                double delayed_EQOP = equal_opportunity(tp_protected, fn_protected, tp_non_protected, fn_non_protected);

                if (abs(delayed_EQOP) >= 0.001) {
                    int position = shifted_location(tp_protected, fn_protected, tp_non_protected, fn_non_protected);
                    thresholdFair.add(FairBoosting.tweak_boundary(buf_predictions, position));
                } else {
                    thresholdFair.add(thresholdFair.get(thresholdFair.size() - 1));
                }
                EQOPFair.add(delayed_EQOP);
            }

            accuracyFair.add(evaluator.getErrorRate());
            gmeanFair.add(evaluator.getGmean());
            kappaFair.add(evaluator.getKappa());
            F1Fair.add(evaluator.getF1Score());
            recallFair.add(evaluator.getRecall());
            balaccFair.add(evaluator.getBACC());

            FairBoosting.trainInstanceImbalance(inst, targetClass, Wn - Wp);

        }
        logger.info("tp " + evaluator.getAucEstimator().getCorrectPosPred() + ", positives = " + evaluator.getAucEstimator().getNumPos());

        logger.info("recall = " + evaluator.getRecall());

//        logger.info(thresholdFair);

    }

    private static void runFairImbalancedModel(Instances buffer) throws Exception {

        WindowAUCImbalancedPerformanceEvaluator evaluator = new WindowAUCImbalancedPerformanceEvaluator();
        evaluator.widthOption.setValue(buffer.size());
        evaluator.setIndex(saIndex);
        evaluator.prepareForUse();

        double tp_protected = 0;
        double fn_protected = 0;
        double tp_non_protected = 0;
        double fn_non_protected = 0;


        for (int i = 0; i < buffer.size(); i++) {


            boolean targetClass = false;
            Instance inst = buffer.get(i);

            if (inst.classValue() == indexOfGranted) {
                targetClass = true;
                pos++;
            } else {
                neg++;
            }

            if (i == 0)
                thresholdFairImb.add(0.5);

            update_class_rates(pos, neg);

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

//            evaluator.addResult(new InstanceExample(inst), votes);
            evaluator.addResultForEvaluation(new InstanceExample(inst), votes, inst.classValue() == indexOfGranted);

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

                SP_fair_imb_ephimeral.add(statistical_parity(classified_prot_pos, classified_non_prot_pos, classified_prot_neg, classified_non_prot_neg));
                static_monitor_fairness(classified_prot_pos, classified_non_prot_pos, classified_prot_neg, classified_non_prot_neg);
                if (abs(delayed_discrimination) >= 0.001) {
                    int count_for_sp = shifted_location(classified_prot_pos, classified_non_prot_pos, classified_prot_neg, classified_non_prot_neg);
                    thresholdFairImb.add(FairImbaBoosting.tweak_boundary(buf_predictions, count_for_sp));
                } else {
                    thresholdFairImb.add(thresholdFairImb.get(thresholdFairImb.size() - 1));
                }
                StatParFairImb.add(delayed_discrimination);
            }

            if (OPT.equals("EQOP")) {
                double delayed_EQOP = equal_opportunity(tp_protected, fn_protected, tp_non_protected, fn_non_protected);
                if (abs(delayed_EQOP) >= 0.001) {
                    int position = shifted_location(tp_protected, fn_protected, tp_non_protected, fn_non_protected);
                    thresholdFairImb.add(FairImbaBoosting.tweak_boundary(buf_predictions, position));
                } else {
                    thresholdFairImb.add(thresholdFairImb.get(thresholdFairImb.size() - 1));
                }
                EQOPFairImb.add(delayed_EQOP);
            }

            accuracyFairImb.add(evaluator.getErrorRate());
            kappaFairImb.add(evaluator.getKappa());
            gmeanFairImb.add(evaluator.getGmean());
            F1FairImb.add(evaluator.getF1Score());
            recallFairImb.add(evaluator.getRecall());
            balaccFairImb.add(evaluator.getBACC());

            FairImbaBoosting.trainOnInstanceImpl(inst);

        }
        logger.info("tp " + evaluator.getAucEstimator().getCorrectPosPred() + ", positives = " + evaluator.getAucEstimator().getNumPos());

        logger.info("recall = " + evaluator.getRecall());

//        logger.info("ephimeral parity = " + Stats.meanOf(SP_fair_imb_ephimeral));

    }

    private static double statistical_parity(double prot_pos, double non_prot_pos,
                                             double prot_neg, double non_prot_neg) {

        double temp_Wdp = prot_pos / (prot_pos + prot_neg + 1);
        double temp_Wfp = non_prot_pos / (non_prot_pos + non_prot_neg + 1);
        window_disc = temp_Wfp - temp_Wdp;
        return window_disc;
    }

    private static double equal_opportunity(double tp_protected, double fn_protected, double tp_non_protected, double fn_non_protected) {
        return tp_non_protected / (tp_non_protected + fn_non_protected) - tp_protected / (tp_protected + fn_protected);
    }

    private static int shifted_location(double classified_prot_pos, double classified_non_prot_pos, double classified_prot_neg, double classified_non_prot_neg) {
        return (int) ((classified_prot_neg + classified_prot_pos) * ((classified_non_prot_pos) / (classified_non_prot_pos + classified_non_prot_neg)) - classified_prot_pos);
    }

    private static void update_class_rates(double positives, double negatives) {
        Wp = class_lamda * Wp + (1 - class_lamda) * positives;
        Wn = class_lamda * Wn + (1 - class_lamda) * negatives;
        double sum = Wp + Wn;
        Wp = Wp / (sum);
        Wn = Wn / (sum);
    }


/*    private static double update_decayed_fairness(int dp, int fp, int dn, int fn) {
        Wdp = fair_lamda * Wdp + (1 - fair_lamda) * dp;
        Wfp = fair_lamda * Wfp + (1 - fair_lamda) * fp;

        Wfn = fair_lamda * Wfn + (1 - fair_lamda) * fn;
        Wdn = fair_lamda * Wdn + (1 - fair_lamda) * dn;

        return Wfp / (Wfp + Wfn + 1) - Wdp / (Wdp + Wdn + 1);
    }*/


    private static void static_monitor_fairness(double prot_pos, double non_prot_pos,
                                                double prot_neg, double non_prot_neg) {

        double temp_Wdp = prot_pos / (prot_pos + prot_neg + 1);
        double temp_Wfp = non_prot_pos / (non_prot_pos + non_prot_neg + 1);
        delayed_discrimination = temp_Wfp - temp_Wdp;
    }


    private static weka.core.Instances generateSynthetic(weka.core.Instances stream, int multiplier) {
        weka.core.Instances synethtic = new weka.core.Instances(stream);

        HashMap<Integer, Double> attrInfo = new HashMap<Integer, Double>();
        for (int i = 0; i < stream.numAttributes(); i++) {
            if (stream.attribute(i).type() != 1 && i != saIndex && i != stream.classIndex()) {
                attrInfo.put(i, stream.attributeStats(i).numericStats.mean * multiplier);
            }
        }

        for (int i = 0; i < synethtic.numAttributes(); i++) {
            if (attrInfo.containsKey(i)) {
                for (weka.core.Instance inst : synethtic) {
                    inst.setValue(i, inst.value(i) + attrInfo.get(i));
                }
            }
        }

        return synethtic;
    }

    private static weka.core.Instances swapLabels(weka.core.Instances stream) {
        weka.core.Instances synethtic = new weka.core.Instances(stream);

        for (weka.core.Instance inst : synethtic) {
            inst.setValue(inst.classIndex(), abs(1 - inst.classValue()));
        }

        return synethtic;
    }

    private static void init_models(int weakL, Instances currentWindow) {

        FairBoosting = new OFBB(indexOfDeprived, saIndex, indexOfGranted);
        FairBoosting.ensembleSizeOption.setValue(weakL);
        FairBoosting.baseLearnerOption.setCurrentObject(new HoeffdingAdaptiveTree());
        FairBoosting.setModelContext(new InstancesHeader(currentWindow));
        FairBoosting.prepareForUse();

        FairImbaBoosting = new OFBB(indexOfDeprived, saIndex, indexOfGranted);
        FairImbaBoosting.ensembleSizeOption.setValue(weakL);
        FairImbaBoosting.baseLearnerOption.setCurrentObject(new HoeffdingAdaptiveTree());
        FairImbaBoosting.setModelContext(new InstancesHeader(currentWindow));
        FairImbaBoosting.prepareForUse();

        GHvfdt = new GHVFDT();
        GHvfdt.setModelContext(new InstancesHeader(currentWindow));
        GHvfdt.prepareForUse();

        OnlineBoost = new OnlineSmoothBoost();
        OnlineBoost.ensembleSizeOption.setValue(weakL);
        OnlineBoost.baseLearnerOption.setCurrentObject(new HoeffdingAdaptiveTree());
        OnlineBoost.setModelContext(new InstancesHeader(currentWindow));
        OnlineBoost.prepareForUse();

        rwLearner = new HoeffdingAdaptiveTree();
        rwLearner.setModelContext(new InstancesHeader(currentWindow));
        rwLearner.prepareForUse();

        masLearner = new HoeffdingAdaptiveTree();
        masLearner.setModelContext(new InstancesHeader(currentWindow));
        masLearner.prepareForUse();

        WenBinHT = new Competitors.FAHTree.HoeffdingTree();

        FairChunkBoosting = new CFBB(indexOfDeprived, saIndex, indexOfGranted);
        FairChunkBoosting.ensembleSizeOption.setValue(weakL);
        FairChunkBoosting.baseLearnerOption.setCurrentObject(new HoeffdingAdaptiveTree());
        FairChunkBoosting.setModelContext(new InstancesHeader(currentWindow));
        FairChunkBoosting.prepareForUse();
    }

    private static void reset_parameters() {


        buf_predictions.clear();

        delayed_discrimination = 0;
        current_sliding_disc = 0;
        window_disc = 0;
        Wp = 0;
        Wn = 0;

/*        Wdp = 0;
        Wfp = 0;*/

        pos = 0.0;
        neg = 0.0;

        original_prot_pos = 0.0;
        original_non_prot_pos = 0.0;

        classified_prot_pos = 0.0;
        classified_prot_neg = 0.0;

        classified_non_prot_pos = 0.0;
        classified_non_prot_neg = 0.0;
    }


    private static void runChunkFairModel(Instances buffer) throws Exception {

        WindowAUCImbalancedPerformanceEvaluator evaluator = new WindowAUCImbalancedPerformanceEvaluator();
        evaluator.widthOption.setValue(buffer.size());
        evaluator.setIndex(saIndex);
        evaluator.prepareForUse();

        double window_classified_prot_pos = 0.0;
        double window_classified_prot_neg = 0.0;

        double window_classified_non_prot_pos = 0.0;
        double window_classified_non_prot_neg = 0.0;
        double window_EQOP = 0;
        double tp_protected = 0;
        double fn_protected = 0;
        double tp_non_protected = 0;
        double fn_non_protected = 0;

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
                thresholdFairChunk.add(0.5);


            update_class_rates(pos, neg);

            double[] votes = FairChunkBoosting.getVotesForInstance(inst);
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


//            evaluator.addResult(new InstanceExample(inst), votes);
            evaluator.addResultForEvaluation(new InstanceExample(inst), votes, inst.classValue() == indexOfGranted);

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
                } else if (inst.classValue() == indexOfGranted && label != indexOfGranted) {
                    fn_protected += 1;
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
                } else if (inst.classValue() == indexOfGranted && label != indexOfGranted) {
                    fn_non_protected += 1;
                }
            }

            if (OPT.equals("SP")) {
                SP_fair_chunk_ephimeral.add(statistical_parity(window_classified_prot_pos, window_classified_non_prot_pos, window_classified_prot_neg, window_classified_non_prot_neg));
                static_monitor_fairness(classified_prot_pos, classified_non_prot_pos, classified_prot_neg, classified_non_prot_neg);
            }

            if (OPT.equals("EQOP")) {
                window_EQOP = equal_opportunity(tp_protected, fn_protected, tp_non_protected, fn_non_protected);
            }

            if ((i + 1) % windowSize == 0) {

                if (OPT.equals("SP")) {
                    if (Math.abs(window_disc) > .001) {
                        int position = shifted_location(window_classified_prot_pos, window_classified_non_prot_pos, window_classified_prot_neg, window_classified_non_prot_neg);
                        thresholdFairChunk.add(FairChunkBoosting.tweak_boundary(windowData, position, window_disc));
                    }
                }

                if (OPT.equals("EQOP")) {
                    if (abs(window_EQOP) >= 0.001) {
                        int position = shifted_location(tp_protected, fn_protected, tp_non_protected, fn_non_protected);
                        thresholdFairChunk.add(FairChunkBoosting.tweak_boundary(windowData, position, window_EQOP));
                    } else {
                        thresholdFairChunk.add(thresholdFairChunk.get(thresholdFairChunk.size() - 1));
                    }
                }

                windowData.delete();
                window_classified_prot_pos = 0.0;
                window_classified_prot_neg = 0.0;
                window_classified_non_prot_pos = 0.0;
                window_classified_non_prot_neg = 0.0;

                tp_protected = 0;
                fn_protected = 0;
                tp_non_protected = 0;
                fn_non_protected = 0;

            } else {
                thresholdFairChunk.add(thresholdFairChunk.get(thresholdFairChunk.size() - 1));
            }

            accuracyFairChunk.add(evaluator.getErrorRate());
            gmeanFairChunk.add(evaluator.getGmean());
            kappaFairChunk.add(evaluator.getKappa());
            F1FairChunk.add(evaluator.getF1Score());
            recallFairChunk.add(evaluator.getRecall());
            balaccFairChunk.add(evaluator.getBACC());

            if (OPT.equals("EQOP")) {
                EQOPFairChunk.add(window_EQOP);
            }

            if (OPT.equals("SP")) {
                StatParFairChunk.add(delayed_discrimination);
            }
            FairChunkBoosting.trainInstanceImbalance(inst, targetClass, Wn - Wp);

        }
        logger.info("tp " + evaluator.getAucEstimator().getCorrectPosPred() + ", positives = " + evaluator.getAucEstimator().getNumPos());

        logger.info("recall = " + evaluator.getRecall());

//        logger.info("ephimeral parity = " + Stats.meanOf(SP_fair_chunk_ephimeral));

    }

    public static void massaging(Instances buffer) throws Exception {


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
            evaluator.addResultForEvaluation(trainInstanceExample, votes, trainInstanceExample.instance.classValue() == indexOfGranted);

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

            SP_masa_ephimeral.add(statistical_parity(window_classified_prot_pos, window_classified_non_prot_pos, window_classified_prot_neg, window_classified_non_prot_neg));
            static_monitor_fairness(classified_prot_pos, classified_non_prot_pos, classified_prot_neg, classified_non_prot_neg);

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
                            ranker_evaluator.addResult(new InstanceExample(buffer.get(i)), ranker_votes);
//                            evaluator.addResultForEvaluation(new InstanceExample(buffer.get(i)), ranker_votes, buffer.get(i).classValue() == indexOfGranted);

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
            accuracyMAS.add(evaluator.getErrorRate());
            gmeanMAS.add(evaluator.getGmean());
            kappaMAS.add(evaluator.getKappa());
            F1MAS.add(evaluator.getF1Score());
            recallMAS.add(evaluator.getRecall());
            balaccMAS.add(evaluator.getBACC());

            StatParMAS.add(delayed_discrimination);
        }
        logger.info("tp " + evaluator.getAucEstimator().getCorrectPosPred() + ", positives = " + evaluator.getAucEstimator().getNumPos());

        logger.info("recall = " + evaluator.getRecall());

//        logger.info("ephimeral parity = " + Stats.meanOf(SP_masa_ephimeral));


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


    private static void runWenBinTree(Instances buffer) throws Exception {

        SamoaToWekaInstanceConverter converter = new SamoaToWekaInstanceConverter();
        WindowAUCImbalancedPerformanceEvaluator evaluator = new WindowAUCImbalancedPerformanceEvaluator();
        evaluator.widthOption.setValue(buffer.size());
        evaluator.setIndex(saIndex);
        evaluator.prepareForUse();

        WenBinHT.buildClassifier(new weka.core.Instances(converter.wekaInstances(buffer), 0));

        double window_classified_prot_pos = 0.0;
        double window_classified_prot_neg = 0.0;

        double window_classified_non_prot_pos = 0.0;
        double window_classified_non_prot_neg = 0.0;

        double tp_protected = 0;
        double fn_protected = 0;
        double tp_non_protected = 0;
        double fn_non_protected = 0;
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


//            evaluator.addResult(new InstanceExample(inst), votes);
            evaluator.addResultForEvaluation(new InstanceExample(inst), votes, inst.classValue() == indexOfGranted);

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
                } else if (inst.classValue() == indexOfGranted && label != indexOfGranted) {
                    fn_protected += 1;
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
                } else if (inst.classValue() == indexOfGranted && label != indexOfGranted) {
                    fn_non_protected += 1;
                }

            }
            SP_wen_ephimeral.add(statistical_parity(window_classified_prot_pos, window_classified_non_prot_pos, window_classified_prot_neg, window_classified_non_prot_neg));
            static_monitor_fairness(classified_prot_pos, classified_non_prot_pos, classified_prot_neg, classified_non_prot_neg);

            accuracyWBTree.add(evaluator.getErrorRate());
            gmeanWBTree.add(evaluator.getGmean());
            kappaWBTree.add(evaluator.getKappa());
            StatParWBTree.add(delayed_discrimination);
            F1WBTree.add(evaluator.getF1Score());
            recallWBTree.add(evaluator.getRecall());
            balaccWBTree.add(evaluator.getBACC());

            WenBinHT.updateClassifier(converter.wekaInstance(inst));
        }
        logger.info("tp " + evaluator.getAucEstimator().getCorrectPosPred() + ", positives = " + evaluator.getAucEstimator().getNumPos());

        logger.info("recall = " + evaluator.getRecall());

    }


    private static void runSimpleBoosting(Instances buffer) throws Exception {

        WindowAUCImbalancedPerformanceEvaluator evaluator = new WindowAUCImbalancedPerformanceEvaluator();
        evaluator.widthOption.setValue(buffer.size());
        evaluator.setIndex(saIndex);
        evaluator.prepareForUse();

        double window_classified_prot_pos = 0.0;
        double window_classified_prot_neg = 0.0;

        double window_classified_non_prot_pos = 0.0;
        double window_classified_non_prot_neg = 0.0;

        double tp_protected = 0;
        double fn_protected = 0;
        double tp_non_protected = 0;
        double fn_non_protected = 0;

        for (int i = 0; i < buffer.size(); i++) {
            Instance inst = buffer.get(i);

            double[] votes = OnlineBoost.getVotesForInstance(inst);
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

//            evaluator.addResult(new InstanceExample(inst), votes);
            evaluator.addResultForEvaluation(new InstanceExample(inst), votes, inst.classValue() == indexOfGranted);

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
                } else if (inst.classValue() == indexOfGranted && label != indexOfGranted) {
                    fn_protected += 1;
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
                } else if (inst.classValue() == indexOfGranted && label != indexOfGranted) {
                    fn_non_protected += 1;
                }

            }

            if (OPT.equals("SP")) {
                static_monitor_fairness(classified_prot_pos, classified_non_prot_pos, classified_prot_neg, classified_non_prot_neg);
                StatParSimpleBoost.add(delayed_discrimination);
            }

            if (OPT.equals("EQOP")) {
                EQOPSimpleBoost.add(equal_opportunity(tp_protected, fn_protected, tp_non_protected, fn_non_protected));
            }

            accuracySimpleBoost.add(evaluator.getErrorRate());
            gmeanSimpleBoost.add(evaluator.getGmean());
            kappaSimpleBoost.add(evaluator.getKappa());
            F1SimpleBoost.add(evaluator.getF1Score());
            recallSimpleBoost.add(evaluator.getRecall());
            balaccSimpleBoost.add(evaluator.getBACC());

            OnlineBoost.trainOnInstanceImpl(inst);
        }
        logger.info("tp " + evaluator.getAucEstimator().getCorrectPosPred() + ", positives = " + evaluator.getAucEstimator().getNumPos());

        logger.info("recall = " + evaluator.getRecall());

    }

    private static void runGHTree(Instances buffer) throws Exception {

        WindowAUCImbalancedPerformanceEvaluator evaluator = new WindowAUCImbalancedPerformanceEvaluator();
        evaluator.widthOption.setValue(buffer.size());
        evaluator.setIndex(saIndex);
        evaluator.prepareForUse();


        double window_classified_prot_pos = 0.0;
        double window_classified_prot_neg = 0.0;

        double window_classified_non_prot_pos = 0.0;
        double window_classified_non_prot_neg = 0.0;

        double tp_protected = 0;
        double fn_protected = 0;
        double tp_non_protected = 0;
        double fn_non_protected = 0;

        for (int i = 0; i < buffer.size(); i++) {
            Instance inst = buffer.get(i);

            double[] votes = GHvfdt.getVotesForInstance(inst);
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

//            evaluator.addResult(new InstanceExample(inst), votes);
            evaluator.addResultForEvaluation(new InstanceExample(inst), votes, inst.classValue() == indexOfGranted);

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
                } else if (inst.classValue() == indexOfGranted && label != indexOfGranted) {
                    fn_protected += 1;
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
                } else if (inst.classValue() == indexOfGranted && label != indexOfGranted) {
                    fn_non_protected += 1;
                }

            }

            if (OPT.equals("SP")) {
                static_monitor_fairness(classified_prot_pos, classified_non_prot_pos, classified_prot_neg, classified_non_prot_neg);
                StatParGHTree.add(delayed_discrimination);
            }

            if (OPT.equals("EQOP")) {
                EQOPGHTree.add(equal_opportunity(tp_protected, fn_protected, tp_non_protected, fn_non_protected));
            }

            accuracyGHTree.add(evaluator.getErrorRate());
            gmeanGHTree.add(evaluator.getGmean());
            kappaGHTree.add(evaluator.getKappa());
            F1GHTree.add(evaluator.getF1Score());
            recallGHTree.add(evaluator.getRecall());
            balaccGHTree.add(evaluator.getBACC());

            GHvfdt.trainOnInstanceImpl(inst);
        }
        logger.info("tp " + evaluator.getAucEstimator().getCorrectPosPred() + ", positives = " + evaluator.getAucEstimator().getNumPos());

        logger.info("recall = " + evaluator.getRecall());

    }


}

