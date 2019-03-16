package ann.optimizer;

import java.util.Arrays;

public class Individ {
    private Gene g;
    private double fit;

    public Individ(Gene g, double fit) {
        this.g = g;
        this.fit = fit;
    }

    /**
     * @return the g
     */
    public Gene getG() {
        return g;
    }

    /**
     * @param g the Gene to set
     */
    public void setG(Gene g) {
        this.g = g;
    }

    /**
     * @return the fit
     */
    public double getFit() {
        return fit;
    }

    /**
     * @param fit the fitness to set
     */
    public void setFit(double fit) {
        this.fit = fit;
    }
    
    public static void Sort(Individ[] Ind){
        Arrays.sort(Ind, (Individ o1, Individ o2) -> {
            if (o1.fit > o2.fit) return -1;
            if (o1.fit < o2.fit) return 1;
            return 0;
        });
    }
}
