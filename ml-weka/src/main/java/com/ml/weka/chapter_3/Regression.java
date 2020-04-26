package com.ml.weka.chapter_3;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.SMOreg;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


public class Regression {
	public static void main(String[] args) throws Exception{
		Instances dataSet = loadTrainData();

		buildLinearRegression(dataSet);

		buildSMORegression(dataSet);

	}

	private static void buildSMORegression(Instances dataSet) throws Exception {
		SMOreg smoreg = new SMOreg();
		smoreg.buildClassifier(dataSet);
		Evaluation evaluation = new Evaluation(dataSet);
		evaluation.evaluateModel(smoreg, dataSet);
		System.out.println(evaluation.toSummaryString());
	}

	private static void buildLinearRegression(Instances dataSet) throws Exception {
		LinearRegression lr = new LinearRegression();
		lr.buildClassifier(dataSet);

		Evaluation evaluation = new Evaluation(dataSet);
		evaluation.evaluateModel(lr, dataSet);
		System.out.println(evaluation.toSummaryString());
	}

	private static Instances loadTrainData() throws Exception {
		DataSource source = new DataSource(Regression.class.getClassLoader().getResource("chapter_3/house.arff").getPath());
		Instances dataSet = source.getDataSet();
		dataSet.setClassIndex(dataSet.numAttributes()-1);
		return dataSet;
	}
}
