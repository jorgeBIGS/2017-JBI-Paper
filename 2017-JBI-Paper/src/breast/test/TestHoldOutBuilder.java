package breast.test;

import java.io.File;
import java.util.Random;

import breast.utils.ComparativeUtils;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class TestHoldOutBuilder {
	private static final Integer NUM_FOLDS = 2;
	
	private static String PATH_ORIGINAL = "data/original_immuno.csv";
	

	public static void main(String[] args) throws Exception {
		CSVLoader loader = new CSVLoader();
		File f = new File(PATH_ORIGINAL);
		loader.setNoHeaderRowPresent(false);
		loader.setSource(f);
		Instances original = loader.getDataSet();
		
		NumericToNominal filter = new NumericToNominal();
		filter.setInputFormat(original);
		filter.setAttributeIndicesArray(new int[] {
				original.numAttributes() - 1, original.numAttributes() - 2 });
		original = Filter.useFilter(original, filter);

		for (int i = 0; i < ComparativeUtils.NUM_REP; i++) {
			original.setClassIndex(original.numAttributes() - 1);
			original.randomize(new Random());
			original.stratify(NUM_FOLDS);

			Instances training = original.trainCV(NUM_FOLDS, 0);
			Instances test = original.testCV(NUM_FOLDS, 0);
			
			training.deleteAttributeAt(training.numAttributes()-2);
			training.deleteAttributeAt(training.numAttributes()-3);
			test.deleteAttributeAt(test.numAttributes()-2);
			test.deleteAttributeAt(test.numAttributes()-3);


			ArffSaver saver = new ArffSaver();
			saver.setInstances(training);
			saver.setFile(new File("mortality/training" + i + ".arff"));
			saver.writeBatch();
			
			saver = new ArffSaver();
			saver.setInstances(test);
			saver.setFile(new File("mortality/test" + i + ".arff"));
			saver.writeBatch();
			
			original.setClassIndex(original.numAttributes() - 2);
			original.randomize(new Random());
			original.stratify(NUM_FOLDS);

			training = original.trainCV(NUM_FOLDS, 0);
			test = original.testCV(NUM_FOLDS, 0);

			training.deleteAttributeAt(training.numAttributes()-1);
			training.deleteAttributeAt(training.numAttributes()-2);
			test.deleteAttributeAt(test.numAttributes()-1);
			test.deleteAttributeAt(test.numAttributes()-2);
			
			saver = new ArffSaver();
			saver.setInstances(training);
			saver.setFile(new File("recidive/training" + i + ".arff"));
			saver.writeBatch();
			
			saver = new ArffSaver();
			saver.setInstances(test);
			saver.setFile(new File("recidive/test" + i + ".arff"));
			saver.writeBatch();
		}
	}

}
