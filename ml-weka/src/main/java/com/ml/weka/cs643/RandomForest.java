package com.ml.weka.cs643;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.Random;

public class RandomForest {
    public static void main(String[] args) throws Exception {
        DataSource src = new DataSource("src/main/resources/chapter_0/solar-flare_2.arff");
        Instances dt = src.getDataSet();
        dt.setClassIndex(dt.numAttributes() - 1);
        Classifier cls = new weka.classifiers.trees.RandomForest();
        Evaluation eval = new Evaluation(dt);
        int folds = 10;
        System.out.println("=== Run information ===\n\n");
        System.out.println("Scheme:     " + cls.getClass().getName());
        System.out.println("Relation:   " + dt.relationName());
        System.out.println("Instances:  " + dt.numInstances());
        System.out.println("Attributes: " + dt.numAttributes());
        for (int i = 0; i < dt.numAttributes(); i++)
            System.out.println(" " + (i + 1) + " " + dt.attribute(i).name());
        System.out.printf("Test mode:    %d -fold cross-validation\n", folds);
        System.out.println("Started evaluation for " + cls.getClass().getName());
        eval.crossValidateModel(cls, dt, folds, new Random(1));
        System.out.println("Summary...");
        System.out.println("\n" + eval.toSummaryString("=== Cross-validation Evaluation Results ===\n", false));
        System.out.println("\n" + eval.toClassDetailsString());
        System.out.println("\n" + eval.toMatrixString("Overall Confusion Matrix"));
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
    }
}
