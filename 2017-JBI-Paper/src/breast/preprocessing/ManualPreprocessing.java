package breast.preprocessing;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class ManualPreprocessing implements PreprocessingTechnique {
	private static final String MANUAL_FILE = "data/manually_binarized_immuno.csv";
	private Instances manual;
	private List<Integer> indexes;

	public ManualPreprocessing() throws IOException {
		CSVLoader loader = new CSVLoader();
		loader.setFile(new File(MANUAL_FILE));
		manual = loader.getDataSet();
		indexes = new ArrayList<>();
	}


	public void buildTechnique(Instances training) throws Exception {
		indexes.clear();
		Set<String> attrManual = getAttributes(manual);
		Set<String> attrTraining = getAttributes(training);
		attrManual.removeAll(attrTraining);

		for (String s : attrManual) {
			indexes.add(manual.attribute(s).index());
		}
	}

	private Set<String> getAttributes(Instances instances) {
		Set<String> set = new HashSet<>();
		for (int i = 0; i < instances.numAttributes(); i++) {
			set.add(instances.attribute(i).name().trim());
		}
		return set;
	}

	
	public Instances preprocess(Instances test) throws Exception {
		Instances result = new Instances(manual, 0);
		List<Double> aux = toList(test.attributeToDoubleArray(0));
		for (int i = 0; i < test.numInstances(); i++) {
			Instance insManual = manual.get(i);
			if (aux.contains(insManual.value(0))) {
				result.add(insManual);
			}
		}

		if (!indexes.isEmpty()) {
			Collections.sort(indexes);
			Remove remove = new Remove();
			remove.setAttributeIndicesArray(getIntArray(indexes));
			remove.setInputFormat(result);
			result = Filter.useFilter(result, remove);
		}
		return result;
	}

	private int[] getIntArray(List<Integer> indexes2) {
		int[] result = new int[indexes2.size()];
		for (int i = 0; i < result.length; i++) {
			result[i] = indexes2.get(i);
		}
		return result;
	}

	private List<Double> toList(double[] ids) {
		List<Double> result = new ArrayList<Double>();
		for (double d : ids) {
			result.add(d);
		}
		return result;
	}

	
}
