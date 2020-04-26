package com.ml.weka.chapter_3;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

import java.io.File;
import java.io.IOException;


public class Normalization {
	public static void main(String[] args) throws Exception{
		Instances dataSet = loadHousingData();

		Instances normalizedData = normalizeFeatureValuesTo0To1Range(dataSet);

		evaluateDataUsingLinearRegression(normalizedData);

		saveNormalizedHousingData(normalizedData);
	}

	private static void saveNormalizedHousingData(Instances newdata) throws IOException {
		ArffSaver saver = new ArffSaver();
		saver.setInstances(newdata);
		saver.setFile(new File(Normalization.class.getClassLoader().getResource("chapter_3/housenormlize.arff").getPath()));
		saver.writeBatch();
	}

	private static void evaluateDataUsingLinearRegression(Instances newdata) throws Exception {
		LinearRegression lr = new LinearRegression();
		lr.buildClassifier(newdata);
		Evaluation lreval = new Evaluation(newdata);
		lreval.evaluateModel(lr, newdata);
		System.out.println(lreval.toSummaryString());
	}

	private static Instances normalizeFeatureValuesTo0To1Range(Instances dataSet) throws Exception {
		Normalize normalize = new Normalize();
		normalize.setInputFormat(dataSet);
		return Filter.useFilter(dataSet, normalize);
	}

	private static Instances loadHousingData() throws Exception {
		DataSource source = new DataSource(Normalization.class.getClassLoader().getResource("chapter_3/house.arff").getPath());
		Instances dataset = source.getDataSet();
		dataset.setClassIndex(dataset.numAttributes()-1);
		return dataset;
	}
}
