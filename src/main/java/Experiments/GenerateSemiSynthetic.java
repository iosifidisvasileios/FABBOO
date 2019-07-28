package Experiments;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.FileReader;
import java.util.HashMap;

/**
 * Created by iosifidis on 01.07.19.
 */
public class GenerateSemiSynthetic {

//    private final static Logger logger = Logger.getLogger(GenerateSemiSynthetic.class.getName());

    public static int saIndex;
    public static String saName;


    public static void main(String[] args) throws Exception {
        // Import data
        String datasetString = "adult-gender";
        String arffInputFileName = "";

        if (datasetString.equals("adult-gender")) {
            arffInputFileName = "/home/iosifidis/BiasForStreams/out/artifacts/BiasForStreams_jar/adult.arff";
            saName = "sex";
        } else if (datasetString.equals("adult-race")) {
            arffInputFileName = "/home/iosifidis/BiasForStreams/out/artifacts/BiasForStreams_jar/adult.arff";
            saName = "race";
        } else if (datasetString.equals("dutch")) {
            arffInputFileName = "/home/iosifidis/BiasForStreams/out/artifacts/BiasForStreams_jar/dutch.arff";
            saName = "sex";
        } else if (datasetString.equals("kdd")) {
            arffInputFileName = "/home/iosifidis/BiasForStreams/out/artifacts/BiasForStreams_jar/kdd.arff";
            saName = "sex";
        } else if (datasetString.equals("bank")) {
            arffInputFileName = "/home/iosifidis/BiasForStreams/bank-full.arff";
            saName = "marital";
        } else if (datasetString.equals("synthetic")) {
            arffInputFileName = "/home/iosifidis/BiasForStreams/synthetic_data.arff";
            saName = "SA";
        } else if (datasetString.equals("no_drift_no_imbalance")) {
            arffInputFileName = "/home/iosifidis/PycharmProjects/MyScripts/CIKM_Scripts/no_drifts.arff";
            saName = "SA";
        } else if (datasetString.equals("communities")) {
            arffInputFileName = "/home/iosifidis/BiasForStreams/communities.arff";
            saName = "black";
        } else if (datasetString.equals("default")) {
            arffInputFileName = "/home/iosifidis/BiasForStreams/DefaultDataset.arff";
            saName = "SEX";
        }

        ArffLoader.ArffReader arffReader = new ArffLoader.ArffReader(new FileReader(arffInputFileName));
        weka.core.Instances stream = arffReader.getData();
        stream.setClassIndex(stream.numAttributes() - 1);
        saIndex = stream.attribute(saName).index();
        Instances synethtic = generateSynthetic(stream, 5);

        for (int i = 0; i < synethtic.numAttributes(); i++) {
            if (synethtic.attribute(i).type() != 1 && i != saIndex && i != synethtic.classIndex()) {
//                logger.info(synethtic.attribute(i).name());
//                logger.info(synethtic.attributeStats(i).numericStats.mean);
//                logger.info(synethtic.attributeStats(i).numericStats.stdDev);
            }
        }
    }

    private static Instances generateSynthetic(Instances stream, int multiplier) {
        Instances synethtic = new Instances(stream);

        HashMap<Integer, Double> attrInfo = new HashMap<Integer, Double>();
//        logger.info("load data successfully!");
        for (int i = 0; i < stream.numAttributes(); i++) {
            if (stream.attribute(i).type() != 1 && i != saIndex && i != stream.classIndex()) {
//                logger.info(stream.attribute(i).name());
//                logger.info(stream.attributeStats(i).numericStats.mean);
//                logger.info(stream.attributeStats(i).numericStats.stdDev);
                attrInfo.put(i, stream.attributeStats(i).numericStats.stdDev * 5);
            }
        }

        for (int i = 0; i < synethtic.numAttributes(); i++) {
            if (attrInfo.containsKey(i)){
                for (Instance inst : synethtic){
                    inst.setValue(i, inst.value(i) + attrInfo.get(i));
                }
            }
        }

        return synethtic;
    }
}

