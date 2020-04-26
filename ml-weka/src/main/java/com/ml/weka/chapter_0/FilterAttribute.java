package com.ml.weka.chapter_0;

import java.io.File;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;


public class FilterAttribute {

  public static void main(String[] args) throws Exception {

    DataSource source2 = new DataSource("src/main/resources/chapter_0/weather.arff");
    Instances dt = source2.getDataSet();

    String[] op = new String[] {"-R", "2"};
    Remove rmv = new Remove();
    rmv.setOptions(op);
    rmv.setInputFormat(dt);
    Instances nd = Filter.useFilter(dt, rmv);

    ArffSaver as = new ArffSaver();
    as.setInstances(nd);
    as.setFile(new File("src/main/resources/chapter_0/fw.arff"));
    as.writeBatch();


  }

}
