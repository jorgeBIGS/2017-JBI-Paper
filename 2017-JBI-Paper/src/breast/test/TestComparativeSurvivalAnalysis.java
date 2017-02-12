package breast.test;

import static breast.utils.ComparativeUtils.*;
import java.io.File;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.util.Pair;

import weka.core.Instances;
import weka.core.converters.ArffLoader;
import breast.preprocessing.AutoencodingPreprocessing;
import breast.preprocessing.AutomaticPreprocessing;
import breast.preprocessing.ManualPreprocessing;
import breast.preprocessing.PreprocessingTechnique;

public class TestComparativeSurvivalAnalysis {
	private static final String MORTALITY_DATA = "mortality/";
	private static final String RECIDIVE_DATA = "recidive/";

	public static void main(String[] args) throws Exception {
		testOnData(MORTALITY_DATA);
		testOnData(RECIDIVE_DATA);
	}

	private static void testOnData(String path) throws Exception {
		Map<String, List<Pair<Double, Double>>> mapManual = new HashMap<>();
		Map<String, List<Pair<Double, Double>>> mapAutomatico = new HashMap<>();
		Map<String, List<Pair<Double, Double>>> mapAutoencoded = new HashMap<>();

		ArffLoader loader = new ArffLoader();
		PreprocessingTechnique manual = new ManualPreprocessing();
		PreprocessingTechnique automatic = new AutomaticPreprocessing();
		PreprocessingTechnique autoencoder = new AutoencodingPreprocessing(
				null, null);

		for (int i = 0; i < NUM_REP; i++) {
			loader.setFile(new File(path + "training" + i + ".arff"));
			Instances training = loader.getDataSet();

			loader.setFile(new File(path + "test" + i + ".arff"));
			Instances test = loader.getDataSet();
			List<Integer> indexes = getInconsistentInstances(test);

			manual.buildTechnique(training);
			updateStatistics(mapManual, manual.preprocess(test), indexes);

			automatic.buildTechnique(training);
			updateStatistics(mapAutomatico, automatic.preprocess(test), indexes);

			autoencoder.buildTechnique(training);
			automatic.buildTechnique(autoencoder.preprocess(training));
			updateStatistics(mapAutoencoded,
					automatic.preprocess(autoencoder.preprocess(test)), indexes);
			System.out.println("Params: " + autoencoder);

		}

		printResults(mapManual, "Manual - " + path, NUM_REP / 2);
		printResults(mapAutomatico, "Automatico - " + path, NUM_REP / 2);
		printResults(mapAutoencoded, "Autoencoder - " + path, NUM_REP / 2);

	}

}
