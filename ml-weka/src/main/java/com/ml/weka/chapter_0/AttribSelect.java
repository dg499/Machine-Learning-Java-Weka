package com.ml.weka.chapter_0;

import java.io.File;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;


public class AttribSelect {

  public static void main(String[] args) throws Exception {

    DataSource source2 = new DataSource("src/main/resources/chapter_0/weather.arff");
    Instances instanceInfo = source2.getDataSet();

    AttributeSelection asel = new AttributeSelection();

    CfsSubsetEval evaluator = new CfsSubsetEval();
    GreedyStepwise search = new GreedyStepwise();

    asel.setEvaluator(evaluator);
    asel.setSearch(search);

    asel.setInputFormat(instanceInfo);


    Instances nd = Filter.useFilter(instanceInfo, asel);

    ArffSaver as = new ArffSaver();
    as.setInstances(nd);
    as.setFile(new File("src/main/resources/chapter_0/sel.arff"));
    as.writeBatch();


  }

}
