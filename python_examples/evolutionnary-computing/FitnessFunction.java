package lab2;

import org.uncommons.watchmaker.framework.FitnessEvaluator;

import java.util.List;
import java.util.Random;

public class FitnessFunction implements FitnessEvaluator<double[]> {
    // don't change!!!!

    public FitnessFunction(int dimension) {
        this.dimension = dimension;
    }

    private int dimension;
    private int limit = 1000000;
    private double best_result = 0.0;

    public double getFitness(double[] solution, List<? extends double[]> list) {
        // check for limit
        if (limit == 0) {
            System.out.println("Number of fitness evaluations is exceeded!");
            System.out.println("___Your final best result = " + best_result + "!___");
            System.exit(0);
        }
        limit--;
        // start evaluations
        int n = dimension;
        double pi = Math.PI;
        double dn = 1.0 / n;
        double a = 10;
        double b = 0.2;
        double c = 2 * pi;
        double s1 = 0.0;
        double s2 = 0.0;
        Random noise = new Random(1);
        for (int i = 0; i < dimension; i++) {
            double val = solution[i] + noise.nextDouble();
            s1 += val * val;
            s2 += Math.cos(c * val);
        }
        s1 = -a * Math.exp(-b * Math.sqrt(dn * s1));
        s2 = -Math.exp(dn * s2);
        double result = s1 + s2 + a + Math.exp(1);
        result = -result;
        result = result + a;
        result = Math.abs(result);

        if (best_result < result) {
            best_result = result;
        }
        return result;
    }


    public boolean isNatural() {
        return true;
    }
}
