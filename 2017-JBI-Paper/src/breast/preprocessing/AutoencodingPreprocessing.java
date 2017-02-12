package breast.preprocessing;

import static breast.utils.ComparativeUtils.discretize;
import static breast.utils.ComparativeUtils.updateStatistics;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.util.Pair;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.MLPAutoencoder;
import weka.filters.unsupervised.attribute.Normalize;

public class AutoencodingPreprocessing implements PreprocessingTechnique {

	private static final int FILTER = MLPAutoencoder.FILTER_NORMALIZE;
	private MLPAutoencoder filtro;
	private Double lambda;
	private Integer numFunctions;

	public AutoencodingPreprocessing(Double lambd, Integer numFunc) {
		lambda = lambd;
		numFunctions = numFunc;
		filtro = new MLPAutoencoder();
	}

	public void buildTechnique(Instances training) throws Exception {

		Instances copy = new Instances(training);
		copy.deleteAttributeAt(0);
		copy.deleteAttributeAt(copy.numAttributes() - 1);
		copy.deleteAttributeAt(copy.numAttributes() - 1);
		copy.setClassIndex(-1);

		if (lambda == null || numFunctions == null) {
			Double min = Double.MAX_VALUE;
			for (int n = 2; n <= copy.numAttributes() + 3; n += 3) {
				for (double l = 0.001; l <= 0.03; l += 0.005) {
					Map<String, List<Pair<Double, Double>>> map = new HashMap<>();
					filtro.setFilterType(new SelectedTag(FILTER,
							MLPAutoencoder.TAGS_FILTER));
					filtro.setOutputInOriginalSpace(true);
					filtro.setNumThreads(Runtime.getRuntime()
							.availableProcessors() - 1);
					filtro.setPoolSize(Runtime.getRuntime()
							.availableProcessors() - 1);
					filtro.setLambda(l);
					filtro.setNumFunctions(n);
					filtro.setUseCGD(true);
					filtro.setInputFormat(copy);
					Instances copy2 = Filter.useFilter(copy, filtro);

					Normalize normalizer = new Normalize();
					normalizer.setScale(1.0);
					normalizer.setTranslation(0.);
					normalizer.setInputFormat(copy2);

					copy2 = getUpdatedInstances(
							discretize(Filter.useFilter(copy2, normalizer)),
							training);

					updateStatistics(map, copy2);

					if (map.containsKey("ER") && map.containsKey("Her2")
							&& map.containsKey("Ki67") && map.containsKey("PR")) {
						double her2 = map.get("Her2").get(0).getKey();
						double ki67 = map.get("Ki67").get(0).getKey();
						double er = map.get("ER").get(0).getKey();
						double pr = map.get("PR").get(0).getKey();
						double total = her2 + ki67 + er + pr;

						if (min > total / 4) {
							min = total / 4;
							lambda = l;
							numFunctions = n;
						}
					}
				}
			}
		}

		filtro.setFilterType(new SelectedTag(FILTER, MLPAutoencoder.TAGS_FILTER));
		filtro.setNumThreads(Runtime.getRuntime().availableProcessors() - 1);
		filtro.setPoolSize(Runtime.getRuntime().availableProcessors() - 1);
		filtro.setNumFunctions(numFunctions);
		filtro.setLambda(lambda);
		filtro.setOutputInOriginalSpace(true);
		filtro.setInputFormat(copy);

	}

	public Instances preprocess(Instances test) throws Exception {
		Instances copy = new Instances(test);

		copy.deleteAttributeAt(0);
		copy.deleteAttributeAt(copy.numAttributes() - 1);
		copy.deleteAttributeAt(copy.numAttributes() - 1);
		copy.setClassIndex(-1);
		
		copy = Filter.useFilter(copy, filtro);

		Normalize normalizer = new Normalize();
		normalizer.setScale(1.0);
		normalizer.setTranslation(0.);
		normalizer.setInputFormat(copy);

		copy = getUpdatedInstances(
				discretize(Filter.useFilter(copy, normalizer)), test);
		// copy = getUpdatedInstances(discretize(copy), test);
		return copy;
	}

	private Instances getUpdatedInstances(Instances copy, Instances test) {
		Instances result = new Instances(test);
		for (int i = 0; i < copy.numInstances(); i++) {
			Instance original = result.get(i);
			Instance copia = copy.get(i);
			for (int j = 0; j < copy.numAttributes(); j++) {
				original.setValue(j + 1, copia.value(j));
			}
		}
		return result;
	}

	public String toString() {
		return "AutoencodingPreprocessing [lambda=" + lambda
				+ ", numFunctions=" + numFunctions + "]";
	}

}
