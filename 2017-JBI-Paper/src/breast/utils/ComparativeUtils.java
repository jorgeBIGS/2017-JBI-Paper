package breast.utils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import javastat.survival.regression.CoxRegression;

import org.apache.commons.math3.util.Pair;

import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveRange;

public class ComparativeUtils {
	public static Integer NUM_REP = 10;
	public static Double ALPHA = 0.05;

	public static void updateStatistics(
			Map<String, List<Pair<Double, Double>>> map, Instances preprocess)
			throws Exception {
		updateStatistics(map, preprocess, new ArrayList<>());
	}

	public static void updateStatistics(
			Map<String, List<Pair<Double, Double>>> map, Instances done,
			List<Integer> filtered) throws Exception {

		// To remove ID attribute.
		done.deleteAttributeAt(0);

		// Remove inconsistent data (missing values in TREC, TMORT, MORT or REC)
		done = removeInconsistency(done, filtered);

		double[] taim = done.attributeToDoubleArray(done.numAttributes() - 2);
		double[] aim = done.attributeToDoubleArray(done.numAttributes() - 1);

		double[][] attributes = new double[done.numAttributes() - 2][done
				.numInstances()];

		for (int i = 0; i < done.numAttributes() - 2; i++) {
			attributes[i] = done.attributeToDoubleArray(i);
		}

		double[] pValues = new double[done.numAttributes() - 2];
		double[] exps = new double[pValues.length];

		for (int i = 0; i < done.numAttributes() - 2; i++) {
			try {

				CoxRegression testclass1 = new CoxRegression(taim, aim,
						attributes[i]);
				pValues[i] = testclass1.pValue[0];
				exps[i] = Math.pow(Math.E, testclass1.coefficients[0]);
			} catch (RuntimeException e) {
			}
		}

		for (int i = 0; i < pValues.length; i++) {
			String key = done.attribute(i).name().trim();
			if (map.containsKey(key)) {
				map.get(key).add(new Pair<Double, Double>(pValues[i], exps[i]));
			} else {
				List<Pair<Double, Double>> list = new ArrayList<Pair<Double, Double>>();
				list.add(new Pair<Double, Double>(pValues[i], exps[i]));
				map.put(key, list);
			}

		}
	}

	public static List<Integer> getInconsistentInstances(Instances data) {
		// We registered index + 1 to use RemoveRange filter from Weka.
		List<Integer> result = new ArrayList<Integer>();
		for (int i = 0; i < data.numInstances(); i++) {
			Instance aux = data.instance(i);

			if (aux.isMissing(aux.numAttributes() - 1)
					|| aux.isMissing(aux.numAttributes() - 2)
					|| aux.isMissing(aux.numAttributes() - 3)
					|| aux.isMissing(aux.numAttributes() - 4)) {
				result.add(i + 1);
			}
		}

		return result;
	}

	private static Instances removeInconsistency(Instances data,
			List<Integer> filtered) throws Exception {
		Instances result = new Instances(data);
		RemoveRange remove = new RemoveRange();
		remove.setInstancesIndices(filtered.toString().replace("[", "")
				.replace("]", ""));
		remove.setInputFormat(result);
		return filtered.isEmpty() ? result : Filter.useFilter(result, remove);
	}

	public static void printResults(
			Map<String, List<Pair<Double, Double>>> map, String mensaje,
			Integer pos) {
		System.out.println(mensaje);
		List<Pair<Double, Double>> par = map.get("ER");

		List<Double> pvalues = par.stream().map(x -> x.getKey())
				.collect(Collectors.toList());
		Collections.sort(pvalues);
		System.out.println("Marcador - Mediana - Media");
		System.out.println("ER - " + pvalues.get(pos).toString() + " - "
				+ pvalues.stream().collect(Collectors.averagingDouble(x -> x)));
		par = map.get("PR");
		pvalues = par.stream().map(x -> x.getKey())
				.collect(Collectors.toList());
		Collections.sort(pvalues);
		System.out.println("PR - " + pvalues.get(pos).toString() + " - "
				+ pvalues.stream().collect(Collectors.averagingDouble(x -> x)));
		par = map.get("Ki67");
		pvalues = par.stream().map(x -> x.getKey())
				.collect(Collectors.toList());
		Collections.sort(pvalues);
		System.out.println("Ki67 - " + pvalues.get(pos).toString() + " - "
				+ pvalues.stream().collect(Collectors.averagingDouble(x -> x)));
		par = map.get("Her2");
		pvalues = par.stream().map(x -> x.getKey())
				.collect(Collectors.toList());
		Collections.sort(pvalues);
		System.out.println("HER2 - " + pvalues.get(pos).toString() + " - "
				+ pvalues.stream().collect(Collectors.averagingDouble(x -> x)));
	}
	
	public static Instances discretize(Instances done) {
		Instances result = new Instances(done);
		result.setClassIndex(done.classIndex());
		for (int i = 0; i < done.size(); i++) {
			Instance aux = result.instance(i);
			for (int j = 0; j < aux.numAttributes(); j++) {
				if (aux.classIndex() != j) {
					aux.setValue(j, Math.round(aux.value(j)));
				}
			}
		}
		return result;
	}

}
