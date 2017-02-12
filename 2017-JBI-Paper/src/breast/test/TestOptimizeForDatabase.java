package breast.test;

import static breast.utils.ComparativeUtils.printResults;
import static breast.utils.ComparativeUtils.updateStatistics;

import java.io.File;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.util.Pair;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import breast.preprocessing.AutoencodingPreprocessing;
import breast.preprocessing.PreprocessingTechnique;

public class TestOptimizeForDatabase {
	private static final String DATA = "data/original_immuno.csv";
	private static final PreprocessingTechnique PREPROCESS = new AutoencodingPreprocessing(0.01, 2);

	public static void main(String[] args) throws Exception {
		Map<String, List<Pair<Double, Double>>> mapMort = new HashMap<>();
		Map<String, List<Pair<Double, Double>>> mapRec = new HashMap<>();

		CSVLoader loader = new CSVLoader();
		loader.setNoHeaderRowPresent(false);
		loader.setFile(new File(DATA));
		Instances all = loader.getDataSet();

		Instances mortality = new Instances(all);
		mortality.deleteAttributeAt(mortality.numAttributes() - 2);
		mortality.deleteAttributeAt(mortality.numAttributes() - 3);

		Instances recidive = new Instances(all);
		recidive.deleteAttributeAt(recidive.numAttributes() - 1);
		recidive.deleteAttributeAt(recidive.numAttributes() - 2);

		PreprocessingTechnique p1 = PREPROCESS;
		p1.buildTechnique(recidive);
		recidive = p1.preprocess(recidive);
		updateStatistics(mapRec, recidive);
		printResults(mapRec, "Recidive", 0);
		System.out.println(p1);

		p1.buildTechnique(mortality);
		mortality = p1.preprocess(mortality);
		updateStatistics(mapMort, mortality);
		printResults(mapMort, "Mortality", 0);
		System.out.println(p1);
	}

}
