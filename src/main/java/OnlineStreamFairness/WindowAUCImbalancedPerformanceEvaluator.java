package OnlineStreamFairness;/*
 *    ImbalancedPerformanceEvaluator.java
 *    Copyright (C) 2016 Poznan University of Technology
 *    @author Dariusz Brzezinski (dbrzezinski@cs.put.poznan.pl)
 *    @author Tomasz Pewinski
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstanceImpl;
import com.yahoo.labs.samoa.instances.Prediction;
import moa.core.Example;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.core.Utils;
import moa.evaluation.ClassificationPerformanceEvaluator;
import moa.options.AbstractOptionHandler;
import moa.tasks.TaskMonitor;

import java.util.TreeSet;

/**
 * Classification evaluator that updates evaluation results using a sliding
 * window. Performance measures designed for class imbalance problems.
 * Only to be used for binary classification problems with unweighted instances.
 * Class 0 - majority/negative examples, class 1 - minority, positive examples.
 * Prequential AUC calculation as described and analyzed in: D. Brzezinski,
 * J. Stefanowski, "Prequential AUC: Properties of the Area Under the ROC
 * Curve for Data Streams with Concept Drift", Knowledge and Information
 * Systems, 2017.
 *
 * @author Dariusz Brzezinski (dbrzezinski at cs.put.poznan.pl)
 * @author Tomasz Pewinski
 */
public class WindowAUCImbalancedPerformanceEvaluator extends
		AbstractOptionHandler implements ClassificationPerformanceEvaluator {

	private static final long serialVersionUID = 1L;

	public IntOption widthOption = new IntOption("width", 'w',
			"Size of Window", 500);

	protected double totalObservedInstances = 0;
	private Estimator aucEstimator;
	private SimpleEstimator weightMajorityClassifier;
	protected int numClasses;

	protected int saIndex;//HanTTN added
    //HanTTN added
    public void setIndex(int index){
    	this.saIndex = index;
    }


	//end of HanTTN added

    public class SimpleEstimator {
        protected double len;

        protected double sum;

        public void add(double value) {
            sum += value;
            len++;
        }

        public double estimation(){
            return sum/len;
        }
    }

	public class Estimator {

		public class Score implements Comparable<Score> {
			/**
			 * Predicted score of the example
			 */
			protected double value;

			/**
			 * Age of example - position in the window where the example was
			 * added
			 */
			protected int posWindow;

			/**
			 * True if example's true label is positive
			 */
			protected boolean isPositive;

			//HanTTN added
			protected String sensitiveVal;
			//end of HanTTN added
			/**
			 * Constructor.
			 *
			 * @param value
			 *            score value
			 * @param position
			 *            score position in window (defines its age)
			 * @param isPositive
			 *            true if the example's true label is positive
			 */
			public Score(double value, int position, boolean isPositive, String sensitiveVal) {
				this.value = value;
				this.posWindow = position;
				this.isPositive = isPositive;
				this.sensitiveVal = sensitiveVal;
			}

			/**
			 * Sort descending based on score value.
			 */
			@Override
			public int compareTo(Score o) {
				if (o.value < this.value) {
					return -1;
				} else if (o.value > this.value){
					return 1;
				} else {
					if (!o.isPositive && this.isPositive) {
						return -1;
					} else if (o.isPositive && !this.isPositive){
						return 1;
					} else {
						if (o.posWindow > this.posWindow) {
							return -1;
						} else if (o.posWindow < this.posWindow){
							return 1;
						} else {
							return 0;
						}
					}
				}
			}

			@Override
			public boolean equals(Object o) {
				return (o instanceof Score) && ((Score)o).posWindow == this.posWindow;
			}
		}

		protected TreeSet<Score> sortedScores;

		protected TreeSet<Score> holdoutSortedScores;

		protected Score[] window;

		protected double[] predictions;

		protected int posWindow;

		protected int size;

		protected double numPos;

		protected double numNeg;

		protected double holdoutNumPos;

		protected double holdoutNumNeg;

		protected double correctPredictions;

		protected double correctPositivePredictions;

//		protected double falsePositivePredictions;

	    protected double[] columnKappa;

	    protected double[] rowKappa;


		public Estimator(int sizeWindow) {
			this.sortedScores = new TreeSet<Score>();
			this.holdoutSortedScores = new TreeSet<Score>();
			this.size = sizeWindow;
			this.window = new Score[sizeWindow];
			this.predictions = new double[sizeWindow];

	        this.rowKappa = new double[numClasses];
	        this.columnKappa = new double[numClasses];
	        for (int i = 0; i < numClasses; i++) {
	            this.rowKappa[i] = 0.0;
	            this.columnKappa[i] = 0.0;
	        }

			this.posWindow = 0;
			this.numPos = 0;
			this.numNeg = 0;
			this.holdoutNumPos = 0;
			this.holdoutNumNeg = 0;
			this.correctPredictions = 0;
			this.correctPositivePredictions = 0;
//			this.falsePositivePredictions = 0;
		}

		public void add(double score, boolean isPositive, boolean correctPrediction, String sensitiveVal) {
            // // periodically update holdout evaluation
			if (size > 0 && posWindow % this.size == 0) {
				this.holdoutSortedScores = new TreeSet<Score>();

				for (Score s : this.sortedScores) {
					this.holdoutSortedScores.add(s);
				}

				this.holdoutNumPos = this.numPos;
				this.holdoutNumNeg = this.numNeg;
			}

			// // if the window is used and it's full
			if (size > 0 && posWindow >= this.size) {
				// // remove the oldest example
				sortedScores.remove(window[posWindow % size]);
				correctPredictions -= predictions[posWindow % size];
				correctPositivePredictions -= window[posWindow % size].isPositive ? predictions[posWindow % size] : 0;

				if (window[posWindow % size].isPositive) {
					numPos--;
				} else {
					numNeg--;
				}

				int oldestExampleTrueClass = window[posWindow % size].isPositive ? 1 : 0;
	            int oldestExamplePredictedClass = predictions[posWindow % size] == 1.0 ? oldestExampleTrueClass : Math.abs(oldestExampleTrueClass - 1);

				this.rowKappa[oldestExamplePredictedClass] -= 1;
	            this.columnKappa[oldestExampleTrueClass] -= 1;
			}

			// // add new example
			Score newScore = new Score(score, posWindow, isPositive, sensitiveVal);
			sortedScores.add(newScore);
			correctPredictions += correctPrediction ? 1 : 0;
			correctPositivePredictions += correctPrediction && isPositive ? 1 : 0;
//			falsePositivePredictions += !correctPrediction && !isPositive ? 1 : 0;

            int trueClass = isPositive ? 1 : 0;
            int predictedClass = correctPrediction ? trueClass : Math.abs(trueClass - 1);
            this.rowKappa[predictedClass] += 1;
            this.columnKappa[trueClass] += 1;

			if (newScore.isPositive) {
				numPos++;
			} else {
				numNeg++;
			}

			if (size > 0) {
				window[posWindow % size] = newScore;
				predictions[posWindow % size] = correctPrediction ? 1 : 0;
			}

			//// posWindow needs to be always incremented to differentiate between examples in the red-black WenBinHT
			posWindow++;
		}

		public double getAUC() {
			double AUC = 0;
			double c = 0;
			double prevc = 0;
			double lastPosScore = Double.MAX_VALUE;

			if (numPos == 0 || numNeg == 0) {
				return 1;
			}

			for (Score s : sortedScores){
				if(s.isPositive) {
					if (s.value != lastPosScore) {
						prevc = c;
						lastPosScore = s.value;
					}

					c += 1;
				} else {
					if (s.value == lastPosScore) {
						// tie
						AUC += ((double)(c + prevc))/2.0;
					} else {
						AUC += c;
					}
				}
			}

			return AUC / (numPos * numNeg);
		}

		public double getHoldoutAUC() {
			double AUC = 0;
			double c = 0;
			double prevc = 0;
			double lastPosScore = Double.MAX_VALUE;

			if (holdoutSortedScores.isEmpty()) {
				return 0;
			}

			if (holdoutNumPos == 0 || holdoutNumNeg == 0) {
				return 1;
			}

			for (Score s : holdoutSortedScores){
				if(s.isPositive) {
					if (s.value != lastPosScore) {
						prevc = c;
						lastPosScore = s.value;
					}

					c += 1;
				} else {
					if (s.value == lastPosScore) {
						// tie
						AUC += ((double)(c + prevc))/2.0;
					} else {
						AUC += c;
					}
				}
			}

			return AUC / (holdoutNumPos * holdoutNumNeg);
		}

		public double getScoredAUC() {
			double AOC = 0;
			double AUC = 0;
			double r = 0;
			double prevr = 0;
			double c = 0;
			double prevc = 0;
			double R_plus, R_minus;
			double lastPosScore = Double.MAX_VALUE;
			double lastNegScore = Double.MAX_VALUE;

			if (numPos == 0 || numNeg == 0) {
				return 1;
			}

			for (Score s : sortedScores){
				if(s.isPositive) {
					if (s.value != lastPosScore) {
						prevc = c;
						lastPosScore = s.value;
					}

					c += s.value;

					if (s.value == lastNegScore) {
						// tie
						AOC += ((double)(r + prevr))/2.0;
					} else {
						AOC += r;
					}
				} else {
					if (s.value != lastNegScore) {
						prevr = r;
						lastNegScore = s.value;
					}

					r += s.value;

					if (s.value == lastPosScore) {
						// tie
						AUC += ((double)(c + prevc))/2.0;
					} else {
						AUC += c;
					}
				}
			}

			R_minus = (numPos*r - AOC)/(numPos * numNeg);
			R_plus = (AUC)/(numPos * numNeg);
			return R_plus - R_minus;
		}

		public double getRatio() {
			if(numNeg == 0) {
				return Double.MAX_VALUE;
			} else {
				return numPos/numNeg;
			}
		}

		public double getAccuracy() {
			if (size > 0) {
				return totalObservedInstances > 0.0 ? correctPredictions / Math.min(size, totalObservedInstances) : 0.0;
			} else {
				return totalObservedInstances > 0.0 ? correctPredictions / totalObservedInstances : 0.0;
			}
		}

		public double getKappa() {
            double p0 = getAccuracy();
            double pc = 0.0;

            if (size > 0) {
	            for (int i = 0; i < numClasses; i++) {
	                pc += (this.rowKappa[i]/Math.min(size, totalObservedInstances)) * (this.columnKappa[i]/Math.min(size, totalObservedInstances));
	            }
            } else {
            	for (int i = 0; i < numClasses; i++) {
	                pc += (this.rowKappa[i]/totalObservedInstances) * (this.columnKappa[i]/totalObservedInstances);
	            }
            }
            return (p0 - pc) / (1.0 - pc);
	    }

	    public double getKappaM() {//HanTTN changed from private to public
            double p0 = getAccuracy();
            double pc = weightMajorityClassifier.estimation();

            return (p0 - pc) / (1.0 - pc);
	    }

		public double getGMean() {
			double positiveAccuracy = correctPositivePredictions / numPos;
			double negativeAccuracy = (correctPredictions - correctPositivePredictions) / numNeg;
			return Math.sqrt(positiveAccuracy * negativeAccuracy);
		}

		public double getF1Score(){
			double recall = this.getRecall();
			double precision = this.getPrecision();
			return 2*((precision*recall)/(recall+precision));
		}

		public double getBalancedError() {
			double positiveAccuracy = correctPositivePredictions / numPos;
			double negativeAccuracy = (correctPredictions - correctPositivePredictions) / numNeg;
			return 0.5*(positiveAccuracy + negativeAccuracy);
		}

		public double getRecall() {
			return correctPositivePredictions / numPos;
		}
		//HanTTN added
		public double getCorrectPosPred(){
			return this.correctPositivePredictions;
		}
		public double getCorrectPred(){
			return this.correctPredictions;
		}
		public double getNumPos(){
			return this.numPos;
		}
		public double getNumNeg(){
			return this.numNeg;
		}

		public double getPrecision(){
			double correctNegativePredictions=correctPredictions-correctPositivePredictions;
			double falsePos=numNeg - correctNegativePredictions;
			if (correctPositivePredictions==0) return 0;
			return correctPositivePredictions/(correctPositivePredictions+falsePos);
		}
		public double prArea(){
			double prArea=0;
			double prevRecall=0;
			double recall=0;
			int cnt=0;
			int pos=0;
			double precision=0;
			for (Score s:sortedScores){
				cnt++;
				if (s.isPositive){
					pos++;
					precision=(double)pos/(double)cnt;
					recall=(double)pos/(double)numPos;
					prArea+=precision*(recall-prevRecall);
				}else{

				}
				prevRecall=recall;
			}

			return prArea;
		}
		public int[] getsortedLabels(){
			int [] rankedLabels = new int[size];
			int i=0;
			for (Score s:sortedScores){
				rankedLabels[i]=(s.isPositive)?1:0;
				i++;
			}
			return rankedLabels;
		}

		public double[] getsortedScores(){
			double [] rankedLabels = new double[size];
			int i=0;
			for (Score s:sortedScores){
				rankedLabels[i]=s.value;
				i++;
			}
			return rankedLabels;
		}

		public String[] getSAValFromSortedScores(){
			String[] list = new String[size];
			int i=0;
			for (Score s:sortedScores){
				list[i]=s.sensitiveVal;
				i++;
			}
			return list;
		}

		public int[] getPosWindowFromSortedScores(){
			int[] list = new int[size];
			int i=0;
			for (Score s:sortedScores){
				list[i]=s.posWindow;
				i++;
			}
			return list;
		}

		//get from window (not sorted)
		public String[] getSAVal(){
			String[] list = new String[size];
			for (int i=0; i<size; i++){
				list[i]=window[i].sensitiveVal;
			}
			return list;
		}
		public double[] getPredictions(){
			return predictions;
		}
		public int[] getTrueLabel(){
			int[] list=new int[size];
			for (int i=0; i<size; i++){
				list[i]=(window[i].isPositive)?1:0;
			}
			return list;
		}
		///end of HanTTN modified
	}

	@Override
	public void reset() {
		reset(this.numClasses);
	}

	public void reset(int numClasses) {
		if (numClasses != 2) {
			throw new RuntimeException(
					"Too many classes ("
							+ numClasses
							+ "). AUC evaluation can be performed only for two-class problems!");
		}

		this.numClasses = numClasses;

		this.aucEstimator = new Estimator(this.widthOption.getValue());
		this.weightMajorityClassifier = new SimpleEstimator();
		this.totalObservedInstances = 0;
	}

	@Override
	public void addResult(Example<Instance> exampleInstance, double[] classVotes) {
		InstanceImpl inst = (InstanceImpl)exampleInstance.getData();
		double weight = inst.weight();
		//HanTTN added
		String[] splits = inst.toString().split(",");
		String sa = splits[saIndex];
		//end of HanTTN
		if (inst.classIsMissing() == false){
			int trueClass = (int) inst.classValue();

			if (weight > 0.0) {
				// // initialize evaluator
				if (totalObservedInstances == 0) {
					reset(inst.dataset().numClasses());
				}
				this.totalObservedInstances += 1;

				//// if classVotes has length == 1, then the negative (0) class got all the votes
				Double normalizedVote = 0.0;

				//// normalize and add score
				if(classVotes.length == 2) {
					normalizedVote = classVotes[1]/(classVotes[0] + classVotes[1]);
				}

				if(normalizedVote.isNaN()){
					normalizedVote = 0.0;
				}

				this.aucEstimator.add(normalizedVote, trueClass == 1, Utils.maxIndex(classVotes) == trueClass, sa);
				this.weightMajorityClassifier.add((this.aucEstimator.getRatio() <= 1 ? 0 : 1) == trueClass ? weight: 0);
			}
		}
	}

	@Override
	public Measurement[] getPerformanceMeasurements() {
		return new Measurement[] {
				new Measurement("classified instances",
						this.totalObservedInstances),
				new Measurement("AUC", this.aucEstimator.getAUC()),
				new Measurement("sAUC", this.aucEstimator.getScoredAUC()),
				new Measurement("Accuracy", this.aucEstimator.getAccuracy()),
				new Measurement("Kappa", this.aucEstimator.getKappa()),
				new Measurement("Periodical holdout AUC", this.aucEstimator.getHoldoutAUC()),
				new Measurement("Pos/Neg ratio", this.aucEstimator.getRatio()),
				new Measurement("G-Mean", this.aucEstimator.getGMean()),
				new Measurement("Recall", this.aucEstimator.getRecall()),
				new Measurement("KappaM", this.aucEstimator.getKappaM())};

	}


	public double getErrorRate() {
		return this.aucEstimator.getAccuracy();
	}

	public double getKappa() {
		return this.aucEstimator.getKappa();
	}

	public double getGmean() {
		return this.aucEstimator.getGMean();
	}
	public double getF1Score() {return this.aucEstimator.getF1Score();}
	public double getBACC() {return this.aucEstimator.getBalancedError();	}
	public double getRecall() {return this.aucEstimator.getRecall();	}


	@Override
	public void getDescription(StringBuilder sb, int indent) {
		Measurement.getMeasurementsDescription(getPerformanceMeasurements(),
				sb, indent);
	}

	@Override
	public void prepareForUseImpl(TaskMonitor monitor,
			ObjectRepository repository) {
	}

	public Estimator getAucEstimator() {
		return aucEstimator;
	}

	@Override
	public void addResult(Example<Instance> arg0, Prediction arg1) {
		throw new RuntimeException("Designed for scoring classifiers");
	}
}
