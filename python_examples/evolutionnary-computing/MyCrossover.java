package lab2;

import org.uncommons.watchmaker.framework.operators.AbstractCrossover;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class MyCrossover extends AbstractCrossover<double[]> {
    protected MyCrossover() {
        super(1);
    }
    public int c;
    public int cbis;
    protected List<double[]> mate(double[] p1, double[] p2, int i, Random random) {
        ArrayList children = new ArrayList();

        // your implementation:
        double[] c1 = new double[p1.length];
        double[] c2 = new double[p2.length];
        int m = Math.min(p1.length, p2.length);
        c = random.nextInt(m); // index of the crossover
        while(random.nextInt(m)==c) {
            cbis = random.nextInt(m); // index of the second crossover
        }

        for(int k=0;k<Math.min(c, cbis);k++){
            c1[k]=p1[k];
            c2[k]=p2[k];
        }
        for(int k=Math.min(c, cbis);k<Math.max(c, cbis);k++){
            c1[k]=p2[k];
            c2[k]=p1[k];
        }
        for(int k=Math.max(c, cbis);k<Math.min(c, cbis);k++){
            c1[k]=p1[k];
            c2[k]=p2[k];
        }
        children.add(c1);
        children.add(c2);
        return children;
    }
}
