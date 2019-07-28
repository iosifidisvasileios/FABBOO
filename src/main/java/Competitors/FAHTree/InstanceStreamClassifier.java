package Competitors.FAHTree;/*
 *  Instance based classifier, which predicts and updates instance one by one,
 *  in contrast to window based stream classifier processes window based instances.
 *  
 * * ***The hoeffding WenBinHT here is the fair splitting criteria embedded based on weka.***
 */

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;

public class InstanceStreamClassifier {

	private static int windowSize = 1000;
	private static String saName = "sex"; // sensitive attribute name
	private static String saValue = " Female"; // sensitive attribute value
	private static int saIndex; // sensitive attribute index of the stream

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		// Import data
		String arffInputFileName= "/home/iosifidis/BiasForStreams/out/artifacts/BiasForStreams_jar/adult.arff";
		ArffReader arffReader= new ArffReader(new FileReader( arffInputFileName));
		Instances stream = arffReader.getData();
		
		/** stream= stream.stream() */
		stream.setClassIndex(stream.numAttributes() - 1);
		System.out.println("load data successfully!");

		InstanceStreamClassifier streamClassifier = new InstanceStreamClassifier();
		streamClassifier.trainClassifier(stream);

		System.out.println("task finished!");

	}

	protected void trainClassifier(Instances stream) throws Exception {

		saIndex= stream.attribute(saName).index();

		// counts
		int numCorrectClassified = 0;
		int numUndeprived = 0, numDeprived = 0;
		int undeprivedPredictedCount = 0, deprivedPredictedCount = 0;
		int numOfInstances = 0;

		int indexOfUndeprived = stream.attribute(saName).indexOfValue(" Male");
		int indexOfDeprived = stream.attribute(saName).indexOfValue(saValue); // M:0 F:1
		int indexOfGranted= 1;
		int indexOfDenied= 0;
		
		
		int trueClassLabel = -1;
		int predictedClassLabel = -1;
		double grantedProb= 0;
		double deniedProb= 0;
		boolean correctClassification = false;
		
		//Store predicted and actual labels
		int []predictedClassLabelArr= new int[stream.numInstances()];
		int []trueClassLabelArr= new int[stream.numInstances()];
		int []saValueArr= new int[stream.numInstances()];
		
		//periodically store the number of nodes of the WenBinHT
		int len= stream.numInstances()/5000 +1; //5000 instances per time
		int curIndexOfNumOfNodes= 0;
		int [] numOfNodesArr= new int[len];

		HoeffdingTree FAHT = new HoeffdingTree();
		
		Instances twoData= new Instances(stream, 2);
		for (int i = 0; i < 2; i++) {
			twoData.add(stream.instance(i));
		}
		FAHT.buildClassifier(twoData);

		for (int i = 0; i < stream.numInstances(); i++) {
			numOfInstances++;
			Instance inst = stream.instance(i);
			trueClassLabel = (int) inst.classValue();
			
			grantedProb= FAHT.distributionForInstance(inst)[indexOfGranted];
			deniedProb= FAHT.distributionForInstance(inst)[indexOfDenied];
			
			if (grantedProb>= deniedProb) {
				predictedClassLabel= indexOfGranted;
			} else {
				predictedClassLabel= indexOfDenied;
			}
			
			
			//update label array
			predictedClassLabelArr[i]= predictedClassLabel;
			trueClassLabelArr[i]= trueClassLabel;
			saValueArr[i]= (int) inst.value(saIndex);
			
			if (predictedClassLabel== trueClassLabel) {
				correctClassification= true;
			}
			
			// calculate accuracy
			if (correctClassification) {
				numCorrectClassified++;
			}

			// calculate discrimination
			if (inst.value(saIndex) == indexOfUndeprived) {
				numUndeprived++;
				if (predictedClassLabel == indexOfGranted) {
					undeprivedPredictedCount++;
				}
			}

			if (inst.value(saIndex) == indexOfDeprived) {
				numDeprived++;
				if (predictedClassLabel == indexOfGranted) {
					deprivedPredictedCount++;
				}

			}
			
			// periodically record the number of nodes of the WenBinHT
			if (numOfInstances%5000 == 0) {
				numOfNodesArr[curIndexOfNumOfNodes]= FAHT.getNumOfNodes();
				curIndexOfNumOfNodes++;
			}
			
			//record the last number of the nodes of the WenBinHT
			if (numOfInstances== stream.numInstances()) {
				numOfNodesArr[curIndexOfNumOfNodes]= FAHT.getNumOfNodes();
			}
			
			correctClassification= false;
			FAHT.updateClassifier(inst);
		}

		System.out.println("undeprivedCount:" + numUndeprived + ", deprivedCount:" + numDeprived);
		System.out.println("deprivedPredictedCount:"+ deprivedPredictedCount+", undeprivedPredictedCount:"+undeprivedPredictedCount);

		double accuracy = 100 * (double) numCorrectClassified / numOfInstances;
		double discrimination = 100 * ((double) undeprivedPredictedCount / numUndeprived)
				- ((double) deprivedPredictedCount / numDeprived);

		System.out.println(FAHT);
		System.out.println(numOfInstances + " instances processed with " + accuracy + "% accuracy");
		System.out.println(numOfInstances + " instances processed with " + discrimination + "% discrimination");
		System.out.println(numOfInstances + " instances processed with " + (accuracy-discrimination) + "% fairaccuracy");
		
		
		//print number of nodes
		for (int i = 0; i < numOfNodesArr.length; i++) {
			System.out.print(numOfNodesArr[i]+",");
		}
		
		
		
		//print senAttVal, predClassLabel, trueClassLabel
		String outputFileName = "fair-labels" + ".csv";
		BufferedWriter br = new BufferedWriter(new FileWriter(new File(outputFileName)));
		br.write("sensitiveAttVal, predictedClassLabel, trueClassLabel\n");
		
		for (int j = 0; j < stream.numInstances(); j++) {
			if (saValueArr[j]== indexOfDeprived) {
				br.write(saValueArr[j]+","+predictedClassLabelArr[j]+","+trueClassLabelArr[j]+"\n");
			}
			
		}
		
		br.close();
		
	}

}
