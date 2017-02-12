package breast.preprocessing;

import weka.core.Instance;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import static breast.utils.ComparativeUtils.discretize;

public class AutomaticPreprocessing implements PreprocessingTechnique {

	private Normalize normalizer;

	public AutomaticPreprocessing() {
		normalizer = new Normalize();
		normalizer.setIgnoreClass(true);
	}

	public void buildTechnique(Instances training) throws Exception {
		normalizer.setScale(1.);
		normalizer.setTranslation(0.);
		normalizer.setInputFormat(new Instances(training));

	}

	public Instances preprocess(Instances test) throws Exception {

		Instances result = Filter.useFilter(test, normalizer);

		return getUpdatedAttributes(discretize(result), test);
	}

	private Instances getUpdatedAttributes(Instances discretize, Instances test) {
		Instances result = new Instances(discretize);
		result.setClassIndex(discretize.classIndex());
		for (int i = 0; i < test.size(); i++) {
			Instance aux = result.instance(i);
			aux.setValue(0, test.instance(i).value(0));
			aux.setValue(result.numAttributes() - 2,
					test.instance(i).value(test.numAttributes() - 2));
			aux.setValue(result.numAttributes() - 1,
					test.instance(i).value(test.numAttributes() - 1));
		}
		return result;
	}

}
