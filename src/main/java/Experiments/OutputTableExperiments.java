package Experiments;

import Competitors.CSMOTE_Template;
import Competitors.FAHTree.FAHT_Template;
import Competitors.Massaging.Massaging_Template;
import Competitors.Reweighting.Reweighting_Template;
import Competitors.VanillaOnlineBoosting_Template;
import OnlineStreamFairness.CFBB_Template;
import OnlineStreamFairness.FABBOO_Template;
import OnlineStreamFairness.OFIB_Template;
import com.google.common.math.Stats;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.WekaToSamoaInstanceConverter;
import org.apache.log4j.Logger;
import weka.core.Instance;
import weka.core.converters.ArffLoader;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.text.DecimalFormat;
import java.util.Random;

/**
 * Created by iosifidis on 01.07.19.
 */
public class OutputTableExperiments {

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
//        String datasetString = "compass";
        String datasetString = args[0];
//        OPT = "SP";
        OPT = args[1];

        int iterations = 10;

        DecimalFormat mean = new DecimalFormat();
        mean.setMaximumFractionDigits(4);
        DecimalFormat deviation = new DecimalFormat();
        deviation.setMaximumFractionDigits(3);

        DecimalFormat mean_latex = new DecimalFormat();
        mean_latex.setMaximumFractionDigits(2);
        DecimalFormat deviation_latex = new DecimalFormat();
        deviation_latex.setMaximumFractionDigits(1);


        logger.info("dataset = " + datasetString);
        init_dataset(datasetString);

        if (datasetString.equals("kdd"))
            iterations =3;

        File dir = new File("Tables");
        if (!dir.exists())
            dir.mkdirs();
        dir = new File("Tables/EqualOpportunityResults");
        if (!dir.exists())
            dir.mkdirs();

        dir = new File("Tables/StatisticalParityResults");
        if (!dir.exists())
            dir.mkdirs();

        if (OPT.equals("SP")) {
            outputFileName = "Tables/StatisticalParityResults/" + datasetString + "_" + OPT;
        }
        if (OPT.equals("EQOP")) {
            outputFileName = "Tables/EqualOpportunityResults/" + datasetString + "_" + OPT;
        }
        ArffLoader.ArffReader arffReader = new ArffLoader.ArffReader(new FileReader("Data/" + arffInputFileName));
        weka.core.Instances stream = arffReader.getData();

        if (datasetString.equals("nypd"))
            stream.setClassIndex(3);
        else if (datasetString.equals("loan"))
            stream.setClassIndex(stream.numAttributes() - 2);
        else
            stream.setClassIndex(stream.numAttributes() - 1);

        saIndex = stream.attribute(saName).index();
        indexOfDeprived = stream.attribute(saName).indexOfValue(saValue); // M:0 F:1
        indexOfUndeprived = stream.attribute(saName).indexOfValue(favored);
        indexOfDenied = stream.classAttribute().indexOfValue(otherClass); // <=50K: 0, >50K: 1
        indexOfGranted = stream.classAttribute().indexOfValue(targetClass);

        stats(stream);

        Instances dataset = new WekaToSamoaInstanceConverter().samoaInstances(stream);

        for (int k = 0; k < iterations; k++) {
            if (datasetString.equals("nypd") || datasetString.equals("synthetic") || datasetString.equals("loan")) {
                k = iterations - 1;
            } else {
                dataset.randomize(new Random(System.currentTimeMillis()));
            }

            final FAHT_Template faht = new FAHT_Template(saIndex, indexOfDenied, indexOfGranted, indexOfDeprived);
            final Massaging_Template massaging = new Massaging_Template(windowSize, saIndex, indexOfDenied, indexOfGranted, indexOfDeprived, saValue);
            final Reweighting_Template reweighting = new Reweighting_Template(windowSize, saIndex, indexOfDenied, indexOfGranted, indexOfDeprived);

            if (OPT.equals("SP")) {
                logger.info("FAHT Tree");
                faht.deploy(new Instances(dataset));

                logger.info("Massaging_Template");
                massaging.deploy(new Instances(dataset));

                logger.info("Reweighting_Template");
                reweighting.deploy(new Instances(dataset));
            }


            logger.info("Vanilla Online Boosting");
            final VanillaOnlineBoosting_Template VOB = new VanillaOnlineBoosting_Template(20, saIndex, indexOfDenied, indexOfGranted, indexOfDeprived, OPT);
            VOB.deploy(new Instances(dataset));

            logger.info("CSMOTE");
            final CSMOTE_Template csmote = new CSMOTE_Template(20, saIndex, indexOfDenied, indexOfGranted, indexOfDeprived, OPT);
            csmote.deploy(new Instances(dataset));

            logger.info("Fair & IM-balanced Boosting");
            final OFIB_Template OFIB = new OFIB_Template(20, saIndex, indexOfDenied, indexOfGranted, indexOfDeprived, OPT);
            OFIB.deploy(new Instances(dataset));

            logger.info("Fair Chunk Based Boosting");
            final CFBB_Template CFBB = new CFBB_Template(20, windowSize, saIndex, indexOfDenied, indexOfGranted, indexOfDeprived, OPT);
            CFBB.deploy(new Instances(dataset));


            logger.info("Fair & balanced Boosting");
            final FABBOO_Template FABBOO = new FABBOO_Template(20, saIndex, indexOfDenied, indexOfGranted, indexOfDeprived, OPT);
            FABBOO.deploy(new Instances(dataset));

            if (OPT.equals("SP") && k == iterations - 1) {
                String CSMOTEResult = mean.format(Stats.meanOf(csmote.getAccuracy())) + "\u00B1" + deviation.format(Stats.of(csmote.getAccuracy()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(csmote.getBACC())) + "\u00B1" + deviation.format(Stats.of(csmote.getBACC()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(csmote.getF1Score())) + "\u00B1" + deviation.format(Stats.of(csmote.getF1Score()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(csmote.getGmean())) + "\u00B1" + deviation.format(Stats.of(csmote.getGmean()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(csmote.getKappa())) + "\u00B1" + deviation.format(Stats.of(csmote.getKappa()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(csmote.getRecall())) + "\u00B1" + deviation.format(Stats.of(csmote.getRecall()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(csmote.getStParity())) + "\u00B1" + deviation.format(Stats.of(csmote.getStParity()).populationStandardDeviation()) + "\n";

                String fahtResult = mean.format(Stats.meanOf(faht.getAccuracy())) + "\u00B1" + deviation.format(Stats.of(faht.getAccuracy()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(faht.getBACC())) + "\u00B1" + deviation.format(Stats.of(faht.getBACC()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(faht.getF1Score())) + "\u00B1" + deviation.format(Stats.of(faht.getF1Score()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(faht.getGmean())) + "\u00B1" + deviation.format(Stats.of(faht.getGmean()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(faht.getKappa())) + "\u00B1" + deviation.format(Stats.of(faht.getKappa()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(faht.getRecall())) + "\u00B1" + deviation.format(Stats.of(faht.getRecall()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(faht.getStParity())) + "\u00B1" + deviation.format(Stats.of(faht.getStParity()).populationStandardDeviation()) + "\n";

                String masResult = mean.format(Stats.meanOf(massaging.getAccuracy())) + "\u00B1" + deviation.format(Stats.of(massaging.getAccuracy()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(massaging.getBACC())) + "\u00B1" + deviation.format(Stats.of(massaging.getBACC()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(massaging.getF1Score())) + "\u00B1" + deviation.format(Stats.of(massaging.getF1Score()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(massaging.getGmean())) + "\u00B1" + deviation.format(Stats.of(massaging.getGmean()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(massaging.getKappa())) + "\u00B1" + deviation.format(Stats.of(massaging.getKappa()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(massaging.getRecall())) + "\u00B1" + deviation.format(Stats.of(massaging.getRecall()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(massaging.getStParity())) + "\u00B1" + deviation.format(Stats.of(massaging.getStParity()).populationStandardDeviation()) + "\n";

                String rwResult = mean.format(Stats.meanOf(reweighting.getAccuracy())) + "\u00B1" + deviation.format(Stats.of(reweighting.getAccuracy()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(reweighting.getBACC())) + "\u00B1" + deviation.format(Stats.of(reweighting.getBACC()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(reweighting.getF1Score())) + "\u00B1" + deviation.format(Stats.of(reweighting.getF1Score()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(reweighting.getGmean())) + "\u00B1" + deviation.format(Stats.of(reweighting.getGmean()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(reweighting.getKappa())) + "\u00B1" + deviation.format(Stats.of(reweighting.getKappa()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(reweighting.getRecall())) + "\u00B1" + deviation.format(Stats.of(reweighting.getRecall()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(reweighting.getStParity())) + "\u00B1" + deviation.format(Stats.of(reweighting.getStParity()).populationStandardDeviation()) + "\n";

                String VOBResult = mean.format(Stats.meanOf(VOB.getAccuracy())) + "\u00B1" + deviation.format(Stats.of(VOB.getAccuracy()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(VOB.getBACC())) + "\u00B1" + deviation.format(Stats.of(VOB.getBACC()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(VOB.getF1Score())) + "\u00B1" + deviation.format(Stats.of(VOB.getF1Score()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(VOB.getGmean())) + "\u00B1" + deviation.format(Stats.of(VOB.getGmean()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(VOB.getKappa())) + "\u00B1" + deviation.format(Stats.of(VOB.getKappa()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(VOB.getRecall())) + "\u00B1" + deviation.format(Stats.of(VOB.getRecall()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(VOB.getStParity())) + "\u00B1" + deviation.format(Stats.of(VOB.getStParity()).populationStandardDeviation()) + "\n";

                String OFIBResult = mean.format(Stats.meanOf(OFIB.getAccuracy())) + "\u00B1" + deviation.format(Stats.of(OFIB.getAccuracy()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(OFIB.getBACC())) + "\u00B1" + deviation.format(Stats.of(OFIB.getBACC()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(OFIB.getF1Score())) + "\u00B1" + deviation.format(Stats.of(OFIB.getF1Score()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(OFIB.getGmean())) + "\u00B1" + deviation.format(Stats.of(OFIB.getGmean()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(OFIB.getKappa())) + "\u00B1" + deviation.format(Stats.of(OFIB.getKappa()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(OFIB.getRecall())) + "\u00B1" + deviation.format(Stats.of(OFIB.getRecall()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(OFIB.getStParity())) + "\u00B1" + deviation.format(Stats.of(OFIB.getStParity()).populationStandardDeviation()) + "\n";

                String CFBBResult = mean.format(Stats.meanOf(CFBB.getAccuracy())) + "\u00B1" + deviation.format(Stats.of(CFBB.getAccuracy()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(CFBB.getBACC())) + "\u00B1" + deviation.format(Stats.of(CFBB.getBACC()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(CFBB.getF1Score())) + "\u00B1" + deviation.format(Stats.of(CFBB.getF1Score()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(CFBB.getGmean())) + "\u00B1" + deviation.format(Stats.of(CFBB.getGmean()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(CFBB.getKappa())) + "\u00B1" + deviation.format(Stats.of(CFBB.getKappa()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(CFBB.getRecall())) + "\u00B1" + deviation.format(Stats.of(CFBB.getRecall()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(CFBB.getStParity())) + "\u00B1" + deviation.format(Stats.of(CFBB.getStParity()).populationStandardDeviation()) + "\n";

                String OFBBResult = mean.format(Stats.meanOf(FABBOO.getAccuracy())) + "\u00B1" + deviation.format(Stats.of(FABBOO.getAccuracy()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(FABBOO.getBACC())) + "\u00B1" + deviation.format(Stats.of(FABBOO.getBACC()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(FABBOO.getF1Score())) + "\u00B1" + deviation.format(Stats.of(FABBOO.getF1Score()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(FABBOO.getGmean())) + "\u00B1" + deviation.format(Stats.of(FABBOO.getGmean()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(FABBOO.getKappa())) + "\u00B1" + deviation.format(Stats.of(FABBOO.getKappa()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(FABBOO.getRecall())) + "\u00B1" + deviation.format(Stats.of(FABBOO.getRecall()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(FABBOO.getStParity())) + "\u00B1" + deviation.format(Stats.of(FABBOO.getStParity()).populationStandardDeviation()) + "\n";

                BufferedWriter br = new BufferedWriter(new FileWriter(new File(outputFileName + ".csv")));

                br.write("Method,Accuracy,B.Accuracy,F1Score,Gmean,Kappa,Recall,St.Parity\n");
                br.write("FAHT," + fahtResult);
                br.write("Massaging," + masResult);
                br.write("Reweighting," + rwResult);
                br.write("VOB," + VOBResult);
                br.write("OFIB," + OFIBResult);
                br.write("CFBB," + CFBBResult);
                br.write("FABBOO," + OFBBResult);
                br.write("CSMOTE," + CSMOTEResult);

                br.write("LATEX TABLE EDITIONS\n\n");
                String temp = " & " + mean_latex.format(100 * Stats.meanOf(faht.getBACC())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(faht.getBACC()).populationStandardDeviation()) + " & " +
                        mean_latex.format(100 * Stats.meanOf(faht.getGmean())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(faht.getGmean()).populationStandardDeviation()) + " & " +
                        mean_latex.format(100 * Stats.meanOf(faht.getKappa())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(faht.getKappa()).populationStandardDeviation()) + " & " +
                        mean_latex.format(100 * Stats.meanOf(faht.getRecall())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(faht.getRecall()).populationStandardDeviation()) + " & " +
                        mean.format(Stats.meanOf(faht.getStParity())) + "$\\pm$" + deviation.format(Stats.of(faht.getStParity()).populationStandardDeviation()) + "\\\\ \n";
                br.write("& FAHT " + temp);

                temp = " & " + mean_latex.format(100 * Stats.meanOf(massaging.getBACC())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(massaging.getBACC()).populationStandardDeviation()) + " & " +
                        mean_latex.format(100 * Stats.meanOf(massaging.getGmean())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(massaging.getGmean()).populationStandardDeviation()) + " & " +
                        mean_latex.format(100 * Stats.meanOf(massaging.getKappa())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(massaging.getKappa()).populationStandardDeviation()) + " & " +
                        mean_latex.format(100 * Stats.meanOf(massaging.getRecall())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(massaging.getRecall()).populationStandardDeviation()) + " & " +
                        mean.format(Stats.meanOf(massaging.getStParity())) + "$\\pm$" + deviation.format(Stats.of(massaging.getStParity()).populationStandardDeviation()) + "\\\\ \n";
                br.write("& MS " + temp);

                temp = " & " + mean_latex.format(100 * Stats.meanOf(VOB.getBACC())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(VOB.getBACC()).populationStandardDeviation()) + " & " +
                        mean_latex.format(100 * Stats.meanOf(VOB.getGmean())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(VOB.getGmean()).populationStandardDeviation()) + " & " +
                        mean_latex.format(100 * Stats.meanOf(VOB.getKappa())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(VOB.getKappa()).populationStandardDeviation()) + " & " +
                        mean_latex.format(100 * Stats.meanOf(VOB.getRecall())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(VOB.getRecall()).populationStandardDeviation()) + " & " +
                        mean.format(Stats.meanOf(VOB.getStParity())) + "$\\pm$" + deviation.format(Stats.of(VOB.getStParity()).populationStandardDeviation()) + "\\\\ \n";
                br.write("& OSBoost " + temp);

                temp = " & " + mean_latex.format(100 * Stats.meanOf(csmote.getBACC())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(csmote.getBACC()).populationStandardDeviation()) + " & " +
                        mean_latex.format(100 * Stats.meanOf(csmote.getGmean())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(csmote.getGmean()).populationStandardDeviation()) + " & " +
                        mean_latex.format(100 * Stats.meanOf(csmote.getKappa())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(csmote.getKappa()).populationStandardDeviation()) + " & " +
                        mean_latex.format(100 * Stats.meanOf(csmote.getRecall())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(csmote.getRecall()).populationStandardDeviation()) + " & " +
                        mean.format(Stats.meanOf(csmote.getStParity())) + "$\\pm$" + deviation.format(Stats.of(csmote.getStParity()).populationStandardDeviation()) + "\\\\ \n";
                br.write("& CSMOTE " + temp);

                temp = " & " + mean_latex.format(100 * Stats.meanOf(OFIB.getBACC())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(OFIB.getBACC()).populationStandardDeviation()) + " & " +
                        mean_latex.format(100 * Stats.meanOf(OFIB.getGmean())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(OFIB.getGmean()).populationStandardDeviation()) + " & " +
                        mean_latex.format(100 * Stats.meanOf(OFIB.getKappa())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(OFIB.getKappa()).populationStandardDeviation()) + " & " +
                        mean_latex.format(100 * Stats.meanOf(OFIB.getRecall())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(OFIB.getRecall()).populationStandardDeviation()) + " & " +
                        mean.format(Stats.meanOf(OFIB.getStParity())) + "$\\pm$" + deviation.format(Stats.of(OFIB.getStParity()).populationStandardDeviation()) + "\\\\ \n";
                br.write("& OFIB " + temp);

                temp = " & " + mean_latex.format(100 * Stats.meanOf(CFBB.getBACC())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(CFBB.getBACC()).populationStandardDeviation()) + " & " +
                        mean_latex.format(100 * Stats.meanOf(CFBB.getGmean())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(CFBB.getGmean()).populationStandardDeviation()) + " & " +
                        mean_latex.format(100 * Stats.meanOf(CFBB.getKappa())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(CFBB.getKappa()).populationStandardDeviation()) + " & " +
                        mean_latex.format(100 * Stats.meanOf(CFBB.getRecall())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(CFBB.getRecall()).populationStandardDeviation()) + " & " +
                        mean.format(Stats.meanOf(CFBB.getStParity())) + "$\\pm$" + deviation.format(Stats.of(CFBB.getStParity()).populationStandardDeviation()) + "\\\\ \n";
                br.write("& CFBB " + temp);

                temp = " & " + mean_latex.format(100 * Stats.meanOf(FABBOO.getBACC())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(FABBOO.getBACC()).populationStandardDeviation()) + " & " +
                        mean_latex.format(100 * Stats.meanOf(FABBOO.getGmean())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(FABBOO.getGmean()).populationStandardDeviation()) + " & " +
                        mean_latex.format(100 * Stats.meanOf(FABBOO.getKappa())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(FABBOO.getKappa()).populationStandardDeviation()) + " & " +
                        mean_latex.format(100 * Stats.meanOf(FABBOO.getRecall())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(FABBOO.getRecall()).populationStandardDeviation()) + " & " +
                        mean.format(Stats.meanOf(FABBOO.getStParity())) + "$\\pm$" + deviation.format(Stats.of(FABBOO.getStParity()).populationStandardDeviation()) + "\\\\ \n";
                br.write("& FABBOO " + temp);
                br.close();
            }

            if (OPT.equals("EQOP") && k == iterations - 1) {
                String CSMOTEResult = mean.format(Stats.meanOf(csmote.getAccuracy())) + "\u00B1" + deviation.format(Stats.of(csmote.getAccuracy()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(csmote.getBACC())) + "\u00B1" + deviation.format(Stats.of(csmote.getBACC()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(csmote.getF1Score())) + "\u00B1" + deviation.format(Stats.of(csmote.getF1Score()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(csmote.getGmean())) + "\u00B1" + deviation.format(Stats.of(csmote.getGmean()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(csmote.getKappa())) + "\u00B1" + deviation.format(Stats.of(csmote.getKappa()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(csmote.getRecall())) + "\u00B1" + deviation.format(Stats.of(csmote.getRecall()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(csmote.getEQOP())) + "\u00B1" + deviation.format(Stats.of(csmote.getEQOP()).populationStandardDeviation()) + "\n";

                String VOBResult = mean.format(Stats.meanOf(VOB.getAccuracy())) + "\u00B1" + deviation.format(Stats.of(VOB.getAccuracy()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(VOB.getBACC())) + "\u00B1" + deviation.format(Stats.of(VOB.getBACC()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(VOB.getF1Score())) + "\u00B1" + deviation.format(Stats.of(VOB.getF1Score()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(VOB.getGmean())) + "\u00B1" + deviation.format(Stats.of(VOB.getGmean()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(VOB.getKappa())) + "\u00B1" + deviation.format(Stats.of(VOB.getKappa()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(VOB.getRecall())) + "\u00B1" + deviation.format(Stats.of(VOB.getRecall()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(VOB.getEQOP())) + "\u00B1" + deviation.format(Stats.of(VOB.getEQOP()).populationStandardDeviation()) + "\n";

                String OFIBResult = mean.format(Stats.meanOf(OFIB.getAccuracy())) + "\u00B1" + deviation.format(Stats.of(OFIB.getAccuracy()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(OFIB.getBACC())) + "\u00B1" + deviation.format(Stats.of(OFIB.getBACC()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(OFIB.getF1Score())) + "\u00B1" + deviation.format(Stats.of(OFIB.getF1Score()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(OFIB.getGmean())) + "\u00B1" + deviation.format(Stats.of(OFIB.getGmean()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(OFIB.getKappa())) + "\u00B1" + deviation.format(Stats.of(OFIB.getKappa()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(OFIB.getRecall())) + "\u00B1" + deviation.format(Stats.of(OFIB.getRecall()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(OFIB.getEQOP())) + "\u00B1" + deviation.format(Stats.of(OFIB.getEQOP()).populationStandardDeviation()) + "\n";

                String CFBBResult = mean.format(Stats.meanOf(CFBB.getAccuracy())) + "\u00B1" + deviation.format(Stats.of(CFBB.getAccuracy()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(CFBB.getBACC())) + "\u00B1" + deviation.format(Stats.of(CFBB.getBACC()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(CFBB.getF1Score())) + "\u00B1" + deviation.format(Stats.of(CFBB.getF1Score()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(CFBB.getGmean())) + "\u00B1" + deviation.format(Stats.of(CFBB.getGmean()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(CFBB.getKappa())) + "\u00B1" + deviation.format(Stats.of(CFBB.getKappa()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(CFBB.getRecall())) + "\u00B1" + deviation.format(Stats.of(CFBB.getRecall()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(CFBB.getEQOP())) + "\u00B1" + deviation.format(Stats.of(CFBB.getEQOP()).populationStandardDeviation()) + "\n";

                String OFBBResult = mean.format(Stats.meanOf(FABBOO.getAccuracy())) + "\u00B1" + deviation.format(Stats.of(FABBOO.getAccuracy()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(FABBOO.getBACC())) + "\u00B1" + deviation.format(Stats.of(FABBOO.getBACC()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(FABBOO.getF1Score())) + "\u00B1" + deviation.format(Stats.of(FABBOO.getF1Score()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(FABBOO.getGmean())) + "\u00B1" + deviation.format(Stats.of(FABBOO.getGmean()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(FABBOO.getKappa())) + "\u00B1" + deviation.format(Stats.of(FABBOO.getKappa()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(FABBOO.getRecall())) + "\u00B1" + deviation.format(Stats.of(FABBOO.getRecall()).populationStandardDeviation()) + "," +
                        mean.format(Stats.meanOf(FABBOO.getEQOP())) + "\u00B1" + deviation.format(Stats.of(FABBOO.getEQOP()).populationStandardDeviation()) + "\n";

                BufferedWriter br = new BufferedWriter(new FileWriter(new File(outputFileName + ".csv")));
                br.write("Method,Accuracy,B.Accuracy,F1Score,Gmean,Kappa,Recall,EQOP\n");
                br.write("VOB," + VOBResult);
                br.write("OFIB," + OFIBResult);
                br.write("CFBB," + CFBBResult);
                br.write("FABBOO," + OFBBResult);
                br.write("CSMOTE," + CSMOTEResult);

                br.write("LATEX TABLE EDITIONS\n\n");

                String temp = " & " + mean_latex.format(100 * Stats.meanOf(VOB.getBACC())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(VOB.getBACC()).populationStandardDeviation()) + " & " +
                        mean_latex.format(100 * Stats.meanOf(VOB.getGmean())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(VOB.getGmean()).populationStandardDeviation()) + " & " +
                        mean_latex.format(100 * Stats.meanOf(VOB.getKappa())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(VOB.getKappa()).populationStandardDeviation()) + " & " +
                        mean_latex.format(100 * Stats.meanOf(VOB.getRecall())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(VOB.getRecall()).populationStandardDeviation()) + " & " +
                        mean.format(Stats.meanOf(VOB.getEQOP())) + "$\\pm$" + deviation.format(Stats.of(VOB.getEQOP()).populationStandardDeviation()) + "\\\\ \n";
                br.write("& OSBoost" + temp);

                temp = " & " + mean_latex.format(100 * Stats.meanOf(csmote.getBACC())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(csmote.getBACC()).populationStandardDeviation()) + " & " +
                        mean_latex.format(100 * Stats.meanOf(csmote.getGmean())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(csmote.getGmean()).populationStandardDeviation()) + " & " +
                        mean_latex.format(100 * Stats.meanOf(csmote.getKappa())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(csmote.getKappa()).populationStandardDeviation()) + " & " +
                        mean_latex.format(100 * Stats.meanOf(csmote.getRecall())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(csmote.getRecall()).populationStandardDeviation()) + " & " +
                        mean.format(Stats.meanOf(csmote.getEQOP())) + "$\\pm$" + deviation.format(Stats.of(csmote.getEQOP()).populationStandardDeviation()) + "\\\\ \n";
                br.write("& CSMOTE" + temp);

                temp = " & " + mean_latex.format(100 * Stats.meanOf(OFIB.getBACC())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(OFIB.getBACC()).populationStandardDeviation()) + " & " +
                        mean_latex.format(100 * Stats.meanOf(OFIB.getGmean())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(OFIB.getGmean()).populationStandardDeviation()) + " & " +
                        mean_latex.format(100 * Stats.meanOf(OFIB.getKappa())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(OFIB.getKappa()).populationStandardDeviation()) + " & " +
                        mean_latex.format(100 * Stats.meanOf(OFIB.getRecall())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(OFIB.getRecall()).populationStandardDeviation()) + " & " +
                        mean.format(Stats.meanOf(OFIB.getEQOP())) + "$\\pm$" + deviation.format(Stats.of(OFIB.getEQOP()).populationStandardDeviation()) + "\\\\ \n";
                br.write("& OFIB" + temp);

                temp = " & " + mean_latex.format(100 * Stats.meanOf(CFBB.getBACC())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(CFBB.getBACC()).populationStandardDeviation()) + " & " +
                        mean_latex.format(100 * Stats.meanOf(CFBB.getGmean())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(CFBB.getGmean()).populationStandardDeviation()) + " & " +
                        mean_latex.format(100 * Stats.meanOf(CFBB.getKappa())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(CFBB.getKappa()).populationStandardDeviation()) + " & " +
                        mean_latex.format(100 * Stats.meanOf(CFBB.getRecall())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(CFBB.getRecall()).populationStandardDeviation()) + " & " +
                        mean.format(Stats.meanOf(CFBB.getEQOP())) + "$\\pm$" + deviation.format(Stats.of(CFBB.getEQOP()).populationStandardDeviation()) + "\\\\ \n";
                br.write("& CFBB" + temp);

                temp = " & " + mean_latex.format(100 * Stats.meanOf(FABBOO.getBACC())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(FABBOO.getBACC()).populationStandardDeviation()) + " & " +
                        mean_latex.format(100 * Stats.meanOf(FABBOO.getGmean())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(FABBOO.getGmean()).populationStandardDeviation()) + " & " +
                        mean_latex.format(100 * Stats.meanOf(FABBOO.getKappa())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(FABBOO.getKappa()).populationStandardDeviation()) + " & " +
                        mean_latex.format(100 * Stats.meanOf(FABBOO.getRecall())) + "$\\pm$" + deviation_latex.format(100 * Stats.of(FABBOO.getRecall()).populationStandardDeviation()) + " & " +
                        mean.format(Stats.meanOf(FABBOO.getEQOP())) + "$\\pm$" + deviation.format(Stats.of(FABBOO.getEQOP()).populationStandardDeviation()) + "\\\\ \n";
                br.write("& FABBOO" + temp);


                br.close();
            }
        }
    }

    private static void stats(weka.core.Instances stream) {
        int pos_cnt = 0;
        int neg_cnt = 0;

        for (Instance iii : stream) {
            if (iii.classValue() == indexOfGranted) {
                pos_cnt += 1;
            } else {
                neg_cnt += 1;
            }
        }

        logger.info("positives = " + pos_cnt);
        logger.info("negatives = " + neg_cnt);

        logger.info("dataset size  = " + stream.size());
        logger.info("numAttributes = " + stream.numAttributes());
    }


    private static void init_dataset(String datasetString) {
        if (datasetString.equals("adult-gender")) {
            arffInputFileName = "adult.arff";
            saValue = " Female";
            favored = " Male";
            saName = "sex";
            targetClass = " >50K";
            otherClass = " <=50K";
        } else if (datasetString.equals("adult-race")) {
            arffInputFileName = "adult.arff";
            saValue = " Minorities";
            favored = " White";
            saName = "race";
            targetClass = " >50K";
            otherClass = " <=50K";
        } else if (datasetString.equals("kdd")) {
            arffInputFileName = "kdd.arff";
            saValue = "Female";
            saName = "sex";
            favored = "Male";
            targetClass = "1";
            otherClass = "0";
        } else if (datasetString.equals("bank")) {
            arffInputFileName = "bank-full.arff";
            targetClass = "yes";
            otherClass = "no";
            saName = "marital";
            saValue = "married";
            favored = "single";
        } else if (datasetString.equals("synthetic")) {
            arffInputFileName = "synthetic.arff";
            targetClass = "0";
            otherClass = "1";
            saName = "SA";
            saValue = "Female";
            favored = "Male";
        } else if (datasetString.equals("default")) {
            arffInputFileName = "DefaultDataset.arff";
            saValue = "female";
            favored = "male";
            saName = "SEX";
            targetClass = "1";
            otherClass = "0";
        } else if (datasetString.equals("dutch")) {
            arffInputFileName = "dutch.arff";
            saValue = "2";
            favored = "1";
            saName = "sex";
            targetClass = "2_1"; // high level ?
            otherClass = "5_4_9";
        } else if (datasetString.equals("compass")) {
            arffInputFileName = "compass_zafar.arff";
            saName = "sex";
            saValue = "0";
            favored = "1";
            targetClass = "1";
            otherClass = "-1";
        } else if (datasetString.equals("compass-race")) {
            arffInputFileName = "compass_zafar.arff";
            saName = "race";
            saValue = "1";
            favored = "0";
            targetClass = "1";
            otherClass = "-1";
        } else if (datasetString.equals("nypd")) {
            arffInputFileName = "NYPD_COMPLAINT.arff";
            saName = "SUSP_SEX";
            saValue = "F";
            favored = "M";
            targetClass = "FELONY";
            otherClass = "MISDEMEANOR";
        } else if (datasetString.equals("loan")) {
            arffInputFileName = "LoanDataProcessed.arff";
            saName = "Gender";
            saValue = "female";
            favored = "male";
            targetClass = "true";
            otherClass = "false";
        } else if (datasetString.equals("law")) {
            arffInputFileName = "law_dataset.arff";
            saName = "male";
            saValue = "1.00";
            favored = "0.00";
            targetClass = "0";
            otherClass = "1";
        }
    }
}

