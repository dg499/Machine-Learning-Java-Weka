package com.ml.weka.chapter_0;

import java.io.File;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;


public class DiscretizeAttribute {

  public static void main(String[] args) throws Exception {

    DataSource source2 = new DataSource("src/main/resources/chapter_0/weather.arff");
    Instances dt = source2.getDataSet();

    String[] op = new String[4];
    op[0] = "-B"; // specifies the max no of bins to divide numeric attributes into default 10
    op[1] = "2";
    op[2] = "-R"; // specifies list of columns to discretize, first and last are valid indexes
                  // (default: first-last)
    op[3] = "2-3";
    Discretize dis = new Discretize();
    dis.setOptions(op);
    dis.setInputFormat(dt);

    Instances nd = Filter.useFilter(dt, dis);

    ArffSaver as = new ArffSaver();
    as.setInstances(nd);
    as.setFile(new File("src/main/resources/chapter_0/dis.arff"));
    as.writeBatch();


  }

}
