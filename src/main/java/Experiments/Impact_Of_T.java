package Experiments;

import OnlineStreamFairness.FABBOO_Template;
import OnlineStreamFairness.FABBOO_Template_for_T_Exp;
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

import static java.lang.System.currentTimeMillis;

/**
 * Created by iosifidis on 01.07.19.
 */
public class Impact_Of_T {


    private final static Logger logger = Logger.getLogger(Impact_Of_T.class.getName());

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
        String datasetString = "compass-race";
        OPT = "SP";
//        String datasetString = args[0];
//        OPT = args[1];

        int iterations = 10;

        DecimalFormat mean = new DecimalFormat();
        mean.setMaximumFractionDigits(4);
        DecimalFormat deviation = new DecimalFormat();
        deviation.setMaximumFractionDigits(3);

        logger.info("dataset = " + datasetString);
        init_dataset(datasetString);

        if (datasetString.equals("kdd"))
            iterations = 3;

        File dir = new File("Tables");
        if (!dir.exists())
            dir.mkdirs();

        outputFileName = "Tables/Impact_of_T/" + datasetString + "_" + OPT + "_";


        int weakLearnerList[] = {1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100};

        BufferedWriter br = new BufferedWriter(new FileWriter(new File(outputFileName + "impact_of_T.csv")));
        br.write("Iterations,Accuracy,B.Accuracy,F1Score,Gmean,Kappa,Recall,Time," + OPT + "\n");
        ArffLoader.ArffReader arffReader = new ArffLoader.ArffReader(new FileReader(arffInputFileName));
        weka.core.Instances stream = arffReader.getData();

        stream.setClassIndex(stream.numAttributes() - 1);
        if (datasetString.equals("loan"))
            stream.setClassIndex(stream.numAttributes() - 2);
        else if (datasetString.equals("nypd"))
            stream.setClassIndex(3);

        saIndex = stream.attribute(saName).index();
        indexOfDeprived = stream.attribute(saName).indexOfValue(saValue); // M:0 F:1
        indexOfUndeprived = stream.attribute(saName).indexOfValue(favored);
        indexOfDenied = stream.classAttribute().indexOfValue(otherClass); // <=50K: 0, >50K: 1
        indexOfGranted = stream.classAttribute().indexOfValue(targetClass);

        stats(stream);
        Instances dataset = new WekaToSamoaInstanceConverter().samoaInstances(stream);


        for (int weakL : weakLearnerList) {

            final FABBOO_Template_for_T_Exp FABBOO = new FABBOO_Template_for_T_Exp(weakL, saIndex, indexOfDenied, indexOfGranted, indexOfDeprived, OPT);

            for (int k = 0; k < iterations; k++) {
                if (datasetString.equals("nypd") || datasetString.equals("synthetic") || datasetString.equals("loan")){
                    k = iterations - 1;
                } else {
                    dataset.randomize(new Random(currentTimeMillis()));
                }
                logger.info("weak learners: " + weakL + ", iteration: " + k);

                FABBOO.deploy(new Instances(dataset));

                if (OPT.equals("SP") && k == iterations - 1) {
                    String OFBBResult = mean.format(Stats.meanOf(FABBOO.getAccuracy())) + "\u00B1" + deviation.format(Stats.of(FABBOO.getAccuracy()).populationStandardDeviation()) + "," +
                            mean.format(Stats.meanOf(FABBOO.getBACC())) + "\u00B1" + deviation.format(Stats.of(FABBOO.getBACC()).populationStandardDeviation()) + "," +
                            mean.format(Stats.meanOf(FABBOO.getF1Score())) + "\u00B1" + deviation.format(Stats.of(FABBOO.getF1Score()).populationStandardDeviation()) + "," +
                            mean.format(Stats.meanOf(FABBOO.getGmean())) + "\u00B1" + deviation.format(Stats.of(FABBOO.getGmean()).populationStandardDeviation()) + "," +
                            mean.format(Stats.meanOf(FABBOO.getKappa())) + "\u00B1" + deviation.format(Stats.of(FABBOO.getKappa()).populationStandardDeviation()) + "," +
                            mean.format(Stats.meanOf(FABBOO.getRecall())) + "\u00B1" + deviation.format(Stats.of(FABBOO.getRecall()).populationStandardDeviation()) + "," +
                            mean.format(Stats.meanOf(FABBOO.getTime())) + "\u00B1" + deviation.format(Stats.of(FABBOO.getTime()).populationStandardDeviation()) + "," +
                            mean.format(Stats.meanOf(FABBOO.getStParity())) + "\u00B1" + deviation.format(Stats.of(FABBOO.getStParity()).populationStandardDeviation()) + "\n";
                    br.write(weakL + "," + OFBBResult);
                }

                if (OPT.equals("EQOP") && k == iterations - 1) {
                    logger.info("log data now!" + k);
                    String OFBBResult = mean.format(Stats.meanOf(FABBOO.getAccuracy())) + "\u00B1" + deviation.format(Stats.of(FABBOO.getAccuracy()).populationStandardDeviation()) + "," +
                            mean.format(Stats.meanOf(FABBOO.getBACC())) + "\u00B1" + deviation.format(Stats.of(FABBOO.getBACC()).populationStandardDeviation()) + "," +
                            mean.format(Stats.meanOf(FABBOO.getF1Score())) + "\u00B1" + deviation.format(Stats.of(FABBOO.getF1Score()).populationStandardDeviation()) + "," +
                            mean.format(Stats.meanOf(FABBOO.getGmean())) + "\u00B1" + deviation.format(Stats.of(FABBOO.getGmean()).populationStandardDeviation()) + "," +
                            mean.format(Stats.meanOf(FABBOO.getKappa())) + "\u00B1" + deviation.format(Stats.of(FABBOO.getKappa()).populationStandardDeviation()) + "," +
                            mean.format(Stats.meanOf(FABBOO.getRecall())) + "\u00B1" + deviation.format(Stats.of(FABBOO.getRecall()).populationStandardDeviation()) + "," +
                            mean.format(Stats.meanOf(FABBOO.getTime())) + "\u00B1" + deviation.format(Stats.of(FABBOO.getTime()).populationStandardDeviation()) + "," +
                            mean.format(Stats.meanOf(FABBOO.getEQOP())) + "\u00B1" + deviation.format(Stats.of(FABBOO.getEQOP()).populationStandardDeviation()) + "\n";
                    br.write(weakL + "," + OFBBResult);
                }

            }
        }
        br.close();
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
            arffInputFileName = "C:\\Users\\bill\\IdeaProjects\\FABBOO\\Data\\adult.arff";
            saValue = " Female";
            favored = " Male";
            saName = "sex";
            targetClass = " >50K";
            otherClass = " <=50K";
        } else if (datasetString.equals("adult-race")) {
            arffInputFileName = "C:\\Users\\bill\\IdeaProjects\\FABBOO\\Data\\adult.arff";
            saValue = " Minorities";
            favored = " White";
            saName = "race";
            targetClass = " >50K";
            otherClass = " <=50K";
        } else if (datasetString.equals("kdd")) {
            arffInputFileName = "C:\\Users\\bill\\IdeaProjects\\FABBOO\\Data\\kdd.arff";
            saValue = "Female";
            saName = "sex";
            favored = "Male";
            targetClass = "1";
            otherClass = "0";
        } else if (datasetString.equals("bank")) {
            arffInputFileName = "C:\\Users\\bill\\IdeaProjects\\FABBOO\\Data\\bank-full.arff";
            targetClass = "yes";
            otherClass = "no";
            saName = "marital";
            saValue = "married";
            favored = "single";
        } else if (datasetString.equals("synthetic")) {
            arffInputFileName = "C:\\Users\\bill\\IdeaProjects\\FABBOO\\Data\\synthetic.arff";
            targetClass = "0";
            otherClass = "1";
            saName = "SA";
            saValue = "Female";
            favored = "Male";
        } else if (datasetString.equals("default")) {
            arffInputFileName = "C:\\Users\\bill\\IdeaProjects\\FABBOO\\Data\\DefaultDataset.arff";
            saValue = "female";
            favored = "male";
            saName = "SEX";
            targetClass = "1";
            otherClass = "0";
        } else if (datasetString.equals("dutch")) {
            arffInputFileName = "C:\\Users\\bill\\IdeaProjects\\FABBOO\\Data\\dutch.arff";
            saValue = "2";
            favored = "1";
            saName = "sex";
            targetClass = "2_1"; // high level ?
            otherClass = "5_4_9";
        } else if (datasetString.equals("compass")) {
            arffInputFileName = "C:\\Users\\bill\\IdeaProjects\\FABBOO\\Data\\compass_zafar.arff";
            saName = "sex";
            saValue = "0";
            favored = "1";
            targetClass = "1";
            otherClass = "-1";
        } else if (datasetString.equals("compass-race")) {
            arffInputFileName = "Data\\compass_zafar.arff";
            saName = "race";
            saValue = "1";
            favored = "0";
            targetClass = "1";
            otherClass = "-1";
        } else if (datasetString.equals("nypd")) {
            arffInputFileName = "C:\\Users\\bill\\IdeaProjects\\FABBOO\\Data\\NYPD_COMPLAINT.arff";
            saName = "SUSP_SEX";
            saValue = "F";
            favored = "M";
            targetClass = "FELONY";
            otherClass = "MISDEMEANOR";
        } else if (datasetString.equals("loan")) {
            arffInputFileName = "C:\\Users\\bill\\IdeaProjects\\FABBOO\\Data\\LoanDataProcessed.arff";
            saName = "Gender";
            saValue = "female";
            favored = "male";
            targetClass = "true";
            otherClass = "false";
        }else if (datasetString.equals("law")) {
            arffInputFileName = "C:\\Users\\bill\\IdeaProjects\\FABBOO\\Data\\law_dataset.arff";
            saName = "male";
            saValue = "1.00";
            favored = "0.00";
            targetClass = "0";
            otherClass = "1";
        }
    }
}

