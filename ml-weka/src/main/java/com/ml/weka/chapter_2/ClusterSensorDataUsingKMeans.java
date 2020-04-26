package com.ml.weka.chapter_2;

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


public class ClusterSensorDataUsingKMeans {
	public static void main(String[] args) throws Exception{
		Instances trainData = loadTrainData();

		SimpleKMeans kMeans = createKMeansModel(trainData);

		ClusterEvaluation eval = trainKMeansModel(trainData, kMeans);

		printClusterInfo(eval);
	}

	private static void printClusterInfo(ClusterEvaluation eval) {
		System.out.println(eval.clusterResultsToString());
	}

	private static ClusterEvaluation trainKMeansModel(Instances trainData, SimpleKMeans kMeans) throws Exception {
		ClusterEvaluation eval = new ClusterEvaluation();
		eval.setClusterer(kMeans);
		eval.evaluateClusterer(trainData);
		return eval;
	}

	private static SimpleKMeans createKMeansModel(Instances trainData) throws Exception {
		SimpleKMeans kMeans = new SimpleKMeans();
		kMeans.setNumClusters(4);
		kMeans.buildClusterer(trainData);
		return kMeans;
	}

	private static Instances loadTrainData() throws Exception {
		String path = ClusterSensorDataUsingKMeans.class.getClassLoader().getResource("chapter_2/weather.arff").getPath();
		DataSource source = new DataSource(path);
		return source.getDataSet();
	}
}
