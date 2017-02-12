package breast.preprocessing;

import weka.core.Instances;

public interface PreprocessingTechnique {
	void buildTechnique(Instances training) throws Exception;

	Instances preprocess(Instances test) throws Exception;
}
