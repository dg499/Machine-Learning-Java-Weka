package com.ml.weka.chapter_0.clustering;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.clusterers.SimpleKMeans;
import weka.clusterers.ClusterEvaluation;
public class ClusterEval {

  public static void main(String[] args) throws Exception {

    DataSource source = new DataSource("src/main/resources/chapter_0/weather.arff");
    Instances data = source.getDataSet();

    SimpleKMeans model = new SimpleKMeans();
    model.setNumClusters(3);
    model.buildClusterer(data);

    System.out.println(model);
    
    ClusterEvaluation eval = new ClusterEvaluation();
    DataSource source1 = new DataSource("src/main/resources/chapter_0/weather.test.arff");
    Instances tdt = source1.getDataSet();
    eval.setClusterer(model);
    eval.evaluateClusterer(tdt);
    
    System.out.println(eval.clusterResultsToString());
    System.out.println("# of clusters: " + eval.getNumClusters());
    

  }
  
}
