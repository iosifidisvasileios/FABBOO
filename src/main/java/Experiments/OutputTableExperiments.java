package Experiments;

import Competitors.FAHTree.FAHT_Template;
import Competitors.HellingerTree.GHTree_Template;
import Competitors.Massaging.Massaging_Template;
import Competitors.Reweighting.Reweighting_Template;
import Competitors.VanillaOnlineBoosting_Template;
import OnlineStreamFairness.CFBB_Template;
import OnlineStreamFairness.OFBB_Template;
import OnlineStreamFairness.OFIB_Template;
import com.google.common.math.Stats;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.WekaToSamoaInstanceConverter;
import org.apache.log4j.Logger;
import weka.core.converters.ArffLoader;

import java.io.*;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import static java.lang.Math.abs;

/**
 * Created by iosifidis on 01.07.19.
 */
public class OutputTableExperiments {

    private static ArrayList<Double> accuracyFair = new ArrayList<Double>();
    private static ArrayList<Double> F1Fair = new ArrayList<Double>();
    private static ArrayList<Double> gmeanFair = new ArrayList<Double>();
    private static ArrayList<Double> kappaFair = new ArrayList<Double>();
    private static ArrayList<Double> StatParFair = new ArrayList<Double>();
    private static ArrayList<Double> thresholdFair = new ArrayList<Double>();
    private static ArrayList<Double> EQOPFair = new ArrayList<Double>();

    private static ArrayList<Double> gmeanFairChunk = new ArrayList<Double>();
    private static ArrayList<Double> F1FairChunk = new ArrayList<Double>();
    private static ArrayList<Double> accuracyFairChunk = new ArrayList<Double>();
    private static ArrayList<Double> kappaFairChunk = new ArrayList<Double>();
    private static ArrayList<Double> StatParFairChunk = new ArrayList<Double>();
    private static ArrayList<Double> thresholdFairChunk = new ArrayList<Double>();
    private static ArrayList<Double> EQOPFairChunk = new ArrayList<Double>();

    private static ArrayList<Double> gmeanFairImb = new ArrayList<Double>();
    private static ArrayList<Double> F1FairImb = new ArrayList<Double>();
    private static ArrayList<Double> accuracyFairImb = new ArrayList<Double>();
    private static ArrayList<Double> kappaFairImb = new ArrayList<Double>();
    private static ArrayList<Double> StatParFairImb = new ArrayList<Double>();
    private static ArrayList<Double> thresholdFairImb = new ArrayList<Double>();
    private static ArrayList<Double> EQOPFairImb = new ArrayList<Double>();

    private static ArrayList<Double> gmeanSimpleBoost = new ArrayList<Double>();
    private static ArrayList<Double> F1SimpleBoost = new ArrayList<Double>();
    private static ArrayList<Double> accuracySimpleBoost = new ArrayList<Double>();
    private static ArrayList<Double> kappaSimpleBoost = new ArrayList<Double>();
    private static ArrayList<Double> StatParSimpleBoost = new ArrayList<Double>();
    private static ArrayList<Double> EQOPSimpleBoost = new ArrayList<Double>();

    private static ArrayList<Double> gmeanWBTree = new ArrayList<Double>();
    private static ArrayList<Double> F1WBTree = new ArrayList<Double>();
    private static ArrayList<Double> accuracyWBTree = new ArrayList<Double>();
    private static ArrayList<Double> kappaWBTree = new ArrayList<Double>();
    private static ArrayList<Double> StatParWBTree = new ArrayList<Double>();
    private static ArrayList<Double> EQOPWBTree = new ArrayList<Double>();

    private static ArrayList<Double> gmeanRW = new ArrayList<Double>();
    private static ArrayList<Double> F1RW = new ArrayList<Double>();
    private static ArrayList<Double> accuracyRW = new ArrayList<Double>();
    private static ArrayList<Double> kappaRW = new ArrayList<Double>();
    private static ArrayList<Double> StatParRW = new ArrayList<Double>();
    private static ArrayList<Double> EQOPRW = new ArrayList<Double>();

    private static ArrayList<Double> gmeanMAS = new ArrayList<Double>();
    private static ArrayList<Double> F1MAS = new ArrayList<Double>();
    private static ArrayList<Double> accuracyMAS = new ArrayList<Double>();
    private static ArrayList<Double> kappaMAS = new ArrayList<Double>();
    private static ArrayList<Double> StatParMAS = new ArrayList<Double>();
    private static ArrayList<Double> EQOPMAS = new ArrayList<Double>();

    private static ArrayList<Double> gmeanGHTree = new ArrayList<Double>();
    private static ArrayList<Double> F1GHTree = new ArrayList<Double>();
    private static ArrayList<Double> accuracyGHTree = new ArrayList<Double>();
    private static ArrayList<Double> kappaGHTree = new ArrayList<Double>();
    private static ArrayList<Double> StatParGHTree = new ArrayList<Double>();
    private static ArrayList<Double> EQOPGHTree = new ArrayList<Double>();


    private static int windowSize = 1000;


    private final static Logger logger = Logger.getLogger(OutputTableExperiments.class.getName());

    private static String OPT; // OPT for Statistical Parity "SP" or Equal Opportunity "EQOP"
    private static String saName; // sensitive attribute name
    private static String saValue; // sensitive attribute value
    private static int saIndex; // index of sensitive attribute
    private static String favored;
    private static String targetClass;
    private static String otherClass;
    private static String arffInputFileName;
    private static String outputFileName;

    private static int indexOfDeprived; // sensitive attribute: female
    private static int indexOfUndeprived; // sensitive attribute: male
    private static int indexOfDenied; // class label: income <=50k
    private static int indexOfGranted; // class label: income > 50k


    public static void main(String[] args) throws Exception {
        String datasetString = args[0];
        OPT = "SP";
        boolean synth = false;

        int iterations = 10;

        DecimalFormat mean = new DecimalFormat();
        mean.setMaximumFractionDigits(4);
        DecimalFormat deviation = new DecimalFormat();
        deviation.setMaximumFractionDigits(3);

//        String datasetString = "default";
        logger.info("dataset = " + datasetString);
//        OPT = "SP";
//        boolean synth = false;
        init_dataset(datasetString);

        if (!synth)
            outputFileName = "IEEE_BIG_DATA_RESULTS/Tables/original_" + datasetString + "_tune_" + OPT + "_";
        else
            outputFileName = "IEEE_BIG_DATA_RESULTS/Tables/semi_synthetic_" + datasetString + "_tune_" + OPT + "_";


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

        WekaToSamoaInstanceConverter converter = new WekaToSamoaInstanceConverter();


        Instances dataset = new WekaToSamoaInstanceConverter().samoaInstances(stream);

        for (int k = 0; k < iterations; k++) {
            dataset.randomize(new Random(k));

            if (synth) {
                weka.core.Instances synethtic = swapLabels(stream);
                for (weka.core.Instance inst : synethtic)
                    dataset.add(converter.samoaInstance(inst));
            }


            logger.info("FAHT Tree");
            final FAHT_Template faht = new FAHT_Template(saIndex, indexOfDenied, indexOfGranted, indexOfDeprived);
            faht.deploy(new Instances(dataset));

            logger.info("Massaging_Template");
            final Massaging_Template massaging = new Massaging_Template(windowSize, saIndex, indexOfDenied, indexOfGranted, indexOfDeprived, saValue);
            massaging.deploy(new Instances(dataset));

            logger.info("Reweighting_Template");
            final Reweighting_Template reweighting = new Reweighting_Template(windowSize, saIndex, indexOfDenied, indexOfGranted, indexOfDeprived);
            reweighting.deploy(new Instances(dataset));

            logger.info("Hellinger Tree");
            final GHTree_Template GHTree = new GHTree_Template(saIndex, indexOfDenied, indexOfGranted, indexOfDeprived, OPT);
            GHTree.deploy(new Instances(dataset));

            logger.info("Vanilla Online Boosting");
            final VanillaOnlineBoosting_Template VOB = new VanillaOnlineBoosting_Template(20, saIndex, indexOfDenied, indexOfGranted, indexOfDeprived, OPT);
            VOB.deploy(new Instances(dataset));

            logger.info("Fair & IM-balanced Boosting");
            final OFIB_Template OFIB = new OFIB_Template(20, saIndex, indexOfDenied, indexOfGranted, indexOfDeprived, OPT);
            OFIB.deploy(new Instances(dataset));

            logger.info("Fair Chunk Based Boosting");
            final CFBB_Template CFBB = new CFBB_Template(20, windowSize, saIndex, indexOfDenied, indexOfGranted, indexOfDeprived, OPT);
            CFBB.deploy(new Instances(dataset));

            logger.info("Fair & balanced Boosting");
            final OFBB_Template OFBB = new OFBB_Template(20, saIndex, indexOfDenied, indexOfGranted, indexOfDeprived, OPT);
            OFBB.deploy(new Instances(dataset));

            if (OPT.equals("SP") && k == iterations - 1) {

                String fahtResult = mean.format(Stats.meanOf(faht.getAccuracy())) + "\u00B1" + deviation.format(Stats.of(faht.getAccuracy()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(faht.getBACC())) + "\u00B1" + deviation.format(Stats.of(faht.getBACC()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(faht.getF1Score())) + "\u00B1" + deviation.format(Stats.of(faht.getF1Score()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(faht.getGmean())) + "\u00B1" + deviation.format(Stats.of(faht.getGmean()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(faht.getKappa())) + "\u00B1" + deviation.format(Stats.of(faht.getKappa()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(faht.getStParity())) + "\u00B1" + deviation.format(Stats.of(faht.getStParity()).populationStandardDeviation()) + "\n";

                String masResult = mean.format(Stats.meanOf(massaging.getAccuracy())) + "\u00B1" + deviation.format(Stats.of(massaging.getAccuracy()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(massaging.getBACC())) + "\u00B1" + deviation.format(Stats.of(massaging.getBACC()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(massaging.getF1Score())) + "\u00B1" + deviation.format(Stats.of(massaging.getF1Score()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(massaging.getGmean())) + "\u00B1" + deviation.format(Stats.of(massaging.getGmean()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(massaging.getKappa())) + "\u00B1" + deviation.format(Stats.of(massaging.getKappa()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(massaging.getStParity())) + "\u00B1" + deviation.format(Stats.of(massaging.getStParity()).populationStandardDeviation()) + "\n";

                String rwResult = mean.format(Stats.meanOf(reweighting.getAccuracy())) + "\u00B1" + deviation.format(Stats.of(reweighting.getAccuracy()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(reweighting.getBACC())) + "\u00B1" + deviation.format(Stats.of(reweighting.getBACC()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(reweighting.getF1Score())) + "\u00B1" + deviation.format(Stats.of(reweighting.getF1Score()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(reweighting.getGmean())) + "\u00B1" + deviation.format(Stats.of(reweighting.getGmean()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(reweighting.getKappa())) + "\u00B1" + deviation.format(Stats.of(reweighting.getKappa()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(reweighting.getStParity())) + "\u00B1" + deviation.format(Stats.of(reweighting.getStParity()).populationStandardDeviation()) + "\n";


                String GHResult = mean.format(Stats.meanOf(GHTree.getAccuracy())) + "\u00B1" + deviation.format(Stats.of(GHTree.getAccuracy()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(GHTree.getBACC())) + "\u00B1" + deviation.format(Stats.of(GHTree.getBACC()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(GHTree.getF1Score())) + "\u00B1" + deviation.format(Stats.of(GHTree.getF1Score()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(GHTree.getGmean())) + "\u00B1" + deviation.format(Stats.of(GHTree.getGmean()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(GHTree.getKappa())) + "\u00B1" + deviation.format(Stats.of(GHTree.getKappa()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(GHTree.getStParity())) + "\u00B1" + deviation.format(Stats.of(GHTree.getStParity()).populationStandardDeviation()) + "\n";

                String VOBResult = mean.format(Stats.meanOf(VOB.getAccuracy())) + "\u00B1" + deviation.format(Stats.of(VOB.getAccuracy()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(VOB.getBACC())) + "\u00B1" + deviation.format(Stats.of(VOB.getBACC()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(VOB.getF1Score())) + "\u00B1" + deviation.format(Stats.of(VOB.getF1Score()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(VOB.getGmean())) + "\u00B1" + deviation.format(Stats.of(VOB.getGmean()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(VOB.getKappa())) + "\u00B1" + deviation.format(Stats.of(VOB.getKappa()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(VOB.getStParity())) + "\u00B1" + deviation.format(Stats.of(VOB.getStParity()).populationStandardDeviation()) + "\n";

                String OFIBResult = mean.format(Stats.meanOf(OFIB.getAccuracy())) + "\u00B1" + deviation.format(Stats.of(OFIB.getAccuracy()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(OFIB.getBACC())) + "\u00B1" + deviation.format(Stats.of(OFIB.getBACC()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(OFIB.getF1Score())) + "\u00B1" + deviation.format(Stats.of(OFIB.getF1Score()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(OFIB.getGmean())) + "\u00B1" + deviation.format(Stats.of(OFIB.getGmean()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(OFIB.getKappa())) + "\u00B1" + deviation.format(Stats.of(OFIB.getKappa()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(OFIB.getStParity())) + "\u00B1" + deviation.format(Stats.of(OFIB.getStParity()).populationStandardDeviation()) + "\n";

                String CFBBResult = mean.format(Stats.meanOf(CFBB.getAccuracy())) + "\u00B1" + deviation.format(Stats.of(CFBB.getAccuracy()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(CFBB.getBACC())) + "\u00B1" + deviation.format(Stats.of(CFBB.getBACC()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(CFBB.getF1Score())) + "\u00B1" + deviation.format(Stats.of(CFBB.getF1Score()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(CFBB.getGmean())) + "\u00B1" + deviation.format(Stats.of(CFBB.getGmean()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(CFBB.getKappa())) + "\u00B1" + deviation.format(Stats.of(CFBB.getKappa()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(CFBB.getStParity())) + "\u00B1" + deviation.format(Stats.of(CFBB.getStParity()).populationStandardDeviation()) + "\n";

                String OFBBResult = mean.format(Stats.meanOf(OFBB.getAccuracy())) + "\u00B1" + deviation.format(Stats.of(OFBB.getAccuracy()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(OFBB.getBACC())) + "\u00B1" + deviation.format(Stats.of(OFBB.getBACC()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(OFBB.getF1Score())) + "\u00B1" + deviation.format(Stats.of(OFBB.getF1Score()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(OFBB.getGmean())) + "\u00B1" + deviation.format(Stats.of(OFBB.getGmean()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(OFBB.getKappa())) + "\u00B1" + deviation.format(Stats.of(OFBB.getKappa()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(OFBB.getStParity())) + "\u00B1" + deviation.format(Stats.of(OFBB.getStParity()).populationStandardDeviation()) + "\n";
                BufferedWriter br = new BufferedWriter(new FileWriter(new File(outputFileName + "table_results.csv")));

                br.write("Method,Accuracy,B.Accuracy,F1Score,Gmean,Kappa,St.Parity\n");
                br.write("FAHT," + fahtResult);
                br.write("Massaging," + masResult);
                br.write("Reweighting," + rwResult);
                br.write("GHTree," + GHResult);
                br.write("VOB," + VOBResult);
                br.write("OFIB," + OFIBResult);
                br.write("CFBB," + CFBBResult);
                br.write("OFBB," + OFBBResult);
                br.close();

            }
        }
    }


    private static void init_dataset(String datasetString) {
        if (datasetString.equals("adult-gender")) {
            arffInputFileName = "/home/iosifidis/ImbalancedStreamFairness/Data/adult.arff";
            saValue = " Female";
            favored = " Male";
            saName = "sex";
            targetClass = " >50K";
            otherClass = " <=50K";
        } else if (datasetString.equals("adult-race")) {
            arffInputFileName = "/home/iosifidis/ImbalancedStreamFairness/Data/adult.arff";
            saValue = " Minorities";
            favored = " White";
            saName = "race";
            targetClass = " >50K";
            otherClass = " <=50K";
        } else if (datasetString.equals("kdd")) {
            arffInputFileName = "/home/iosifidis/ImbalancedStreamFairness/Data/kdd.arff";
            saValue = "Female";
            saName = "sex";
            favored = "Male";
            targetClass = "1";
            otherClass = "0";
        } else if (datasetString.equals("bank")) {
            arffInputFileName = "/home/iosifidis/ImbalancedStreamFairness/Data/bank-full.arff";
            targetClass = "yes";
            otherClass = "no";
            saName = "marital";
            saValue = "married";
            favored = "single";
        } else if (datasetString.equals("synthetic")) {
            arffInputFileName = "/home/iosifidis/ImbalancedStreamFairness/Data/synthetic.arff";
            targetClass = "0";
            otherClass = "1";
            saName = "SA";
            saValue = "Female";
            favored = "Male";
        } else if (datasetString.equals("default")) {
            arffInputFileName = "/home/iosifidis/ImbalancedStreamFairness/Data/DefaultDataset.arff";
            saValue = "female";
            favored = "male";
            saName = "SEX";
            targetClass = "1";
            otherClass = "0";
        } else if (datasetString.equals("dutch")) {
            arffInputFileName = "/home/iosifidis/ImbalancedStreamFairness/Data/dutch.arff";
            saValue = "2";
            favored = "1";
            saName = "sex";
            targetClass = "2_1"; // high level ?
            otherClass = "5_4_9";
        } else if (datasetString.equals("compass")) {
            arffInputFileName = "/home/iosifidis/ImbalancedStreamFairness/Data/compass_zafar.arff";
            saName = "sex";
            saValue = "0";
            favored = "1";
            targetClass = "1";
            otherClass = "-1";
        } else if (datasetString.equals("nypd")) {
            arffInputFileName = "/home/iosifidis/ImbalancedStreamFairness/Data/NYPD_COMPLAINT.arff";
            saName = "SUSP_SEX";
            saValue = "F";
            favored = "M";
            targetClass = "FELONY";
            otherClass = "MISDEMEANOR";
        }
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

}

