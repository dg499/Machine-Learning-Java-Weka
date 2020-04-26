package com.ml.weka.chapter_0.clustering;

import weka.classifiers.collective.functions.LLGC;
import weka.classifiers.collective.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class SemiSupervised {

  public static void main(String[] args) throws Exception {

    DataSource source = new DataSource("src/main/resources/chapter_0/weather.arff");
    Instances dt = source.getDataSet();
    dt.setClassIndex(dt.numAttributes() - 1);

    LLGC model = new LLGC();
    model.buildClassifier(dt);
    System.out.println(model.getCapabilities());
    
    Evaluation eval = new Evaluation(dt);

    DataSource src1 = new DataSource("src/main/resources/chapter_0/weather-test.arff");
    Instances tdt = src1.getDataSet();
    tdt.setClassIndex(tdt.numAttributes() - 1);
    
    eval.evaluateModel(model, tdt);
    
    System.out.println(eval.toSummaryString("Evaluation Results:\n", false));

    System.out.println("Correct % = " + eval.pctCorrect());
    System.out.println("Incorrect % = " + eval.pctIncorrect());
    System.out.println("kappa = " + eval.kappa());
    System.out.println("MAE = " + eval.meanAbsoluteError());
    System.out.println("RMSE = " + eval.rootMeanSquaredError());
    System.out.println("RAE = " + eval.relativeAbsoluteError());
    System.out.println("Precision = " + eval.precision(1));
    System.out.println("Recall = " + eval.recall(1));
    System.out.println("fMeasure = " + eval.fMeasure(1));
    System.out.println("Error Rate = " + eval.errorRate());
    System.out.println(eval.toMatrixString("Overall Confusion Matrix"));
  }
}

