package com.ml.weka.chapter_3;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMOreg;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


public class SaveWEKAModelForFurtherUsage {
	public static void main(String[] args) throws Exception{
		Instances trainData = loadTranData();

		SMOreg smo = createModel(trainData);

		saveModel(smo);

		loadModel(trainData);

	}

	private static void loadModel(Instances trainData) throws Exception {
		SMOreg smo2 = (SMOreg) weka.core.SerializationHelper.read("smo.model");
		Evaluation evol = new Evaluation(trainData);
		evol.evaluateModel(smo2, trainData);
		System.out.println(evol.toSummaryString());
	}

	private static void saveModel(SMOreg smo) throws Exception {
		weka.core.SerializationHelper.write("smo.model", smo);
	}

	private static SMOreg createModel(Instances trainData) throws Exception {
		SMOreg smo = new SMOreg();
		smo.buildClassifier(trainData);
		return smo;
	}

	private static Instances loadTranData() throws Exception {
		DataSource source = new DataSource(SaveWEKAModelForFurtherUsage.class.getClassLoader().getResource("chapter_3/house.arff").getPath());
		Instances trainData = source.getDataSet();
		trainData.setClassIndex(trainData.numAttributes()-1);
		return trainData;
	}
}
