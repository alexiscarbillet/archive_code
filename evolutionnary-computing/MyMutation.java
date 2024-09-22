package lab2;

import org.uncommons.watchmaker.framework.EvolutionaryOperator;

import java.util.List;
import java.util.Random;

public class MyMutation implements EvolutionaryOperator<double[]> {
   int count = 0;
   int gene = 0;
    public MyMutation(int generations) {
        this.gene = generations;
        }

    public List<double[]> apply(List<double[]> population, Random random) {
        // initial population
        // need to change individuals, but not their number!
        count++;
        double p,p2;
        double sigma = 0.7-0.6*(count/gene);
        // double sigma = 0.7;
        // your implementation:
    	for(int i=0;i<population.size();i++) {
            p2 = random.nextInt(100)/100;
            if(p2<0.5) {  // 50% of mutation
                for (int l = 0; l < population.get(i).length; l++) {
                    p = random.nextInt(100) / 100;
                    if (p < 0.1) { // 10% of probability of mutation
                        population.get(i)[l] += random.nextGaussian() * sigma;
                    }
                    if(population.get(i)[l]>5 || population.get(i)[l]<(-5)){
                        population.get(i)[l]=0;
                    }
                }
            }
    	}
        //result population
        return population;
    }
}
