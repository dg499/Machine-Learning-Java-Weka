package com.ml.weka.chapter_0.clustering;


import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.clusterers.SimpleKMeans;

public class Clustering {
  public static void main(String[] args) throws Exception {

    DataSource source = new DataSource("src/main/resources/chapter_0/weather.arff");
    Instances data = source.getDataSet();

    SimpleKMeans model = new SimpleKMeans();
    model.setNumClusters(3);
    model.buildClusterer(data);

    System.out.println(model.toString());

  }
}
