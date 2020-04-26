package com.ml.weka.chapter_3;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

import java.io.File;
import java.io.IOException;



public class CSVtoWEKAFormatConverter {
	public static void main(String[] args) throws IOException{
		Instances data = loadCsvDataFormat();

		saveToWEKAFormat(data);
	}

	private static void saveToWEKAFormat(Instances data) throws IOException {
		ArffSaver saver = new ArffSaver();
		saver.setInstances(data);
		saver.setFile(new File(CSVtoWEKAFormatConverter.class.getClassLoader().getResource("chapter_3/house-new.arff").getPath()));
		saver.writeBatch();
	}

	private static Instances loadCsvDataFormat() throws IOException {
		CSVLoader loader = new CSVLoader();
		loader.setSource(new File(CSVtoWEKAFormatConverter.class.getClassLoader().getResource("chapter_3/house.csv").getPath()));
		return loader.getDataSet();
	}
}
