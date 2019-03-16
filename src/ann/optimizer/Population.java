package ann.optimizer;

import org.neuroph.core.NeuralNetwork;
//import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.nnet.learning.MomentumBackpropagation;

public class Population {
    public Gene[] G;
    public CompData CD;
    public Individ Ind[];
    
    Population(Gene[] G, CompData CD) {
        this.G=G;
        this.CD=CD;
    }
    
    Population(Gene Init, int Size, CompData CD){
        G = new Gene[Size];
        G[0] = new Gene(Init);
        for(int i=1; i<Size; i++) {
            G[i] = new Gene(Init);
            G[i].Mutate();
        }
        this.CD=CD;
    }
    
    public double getFitness(int Num){
        NeuralNetwork net = this.G[Num].TransformToNet();
        net.reset();
        //BackPropagation train = new BackPropagation();
        MomentumBackpropagation train = new MomentumBackpropagation();
        net.setLearningRule(train);
        train.setLearningRate(this.G[Num].getLearRate());
        train.setMomentum(this.G[Num].getMomentum());
        train.learn(CD.getTrainingSet(),CD.getFitError(),CD.getMAX_ITERATIONS());
        if(train.getCurrentIteration()==CD.getMAX_ITERATIONS())
            System.out.println("Maximum of iterations reached!");
        System.out.print("Net #"+Num+
                         " LR: "+String.format(java.util.Locale.ENGLISH,
                         "%(.2f",G[Num].getLearRate())+
                         " Mmntm: "+String.format(java.util.Locale.ENGLISH,
                         "%(.2f",G[Num].getMomentum())+
                         " LS: "+G[Num].getLinearSlope()+
                         " SS: "+G[Num].getSigmoidSlope()+" ");
        return -1*Func.calculateError(net,CD.getControlSet());   
    }
    
    private void InitFit() {
        /* The fitness function is evaluated for each individual, 
         * providing fitness values, which are then normalized
         */        
        Ind = new Individ[this.G.length];
        double sumFit=0;
        for(int i=0; i<this.G.length; i++){
            Ind[i] = new Individ(this.G[i],this.getFitness(i));
            sumFit+=Ind[i].getFit();
        }
        for(int i=0; i<Ind.length; i++)
            Ind[i].setFit(Ind[i].getFit()/sumFit);
        
        /* The population is sorted by descending fitness values*/
        Individ.Sort(Ind);

        /* Accumulated normalized fitness values are computed*/
        for(int i=Ind.length-1; i>=0; i--) {
            for(int j=0; j<i; j++)
                Ind[i].setFit(Ind[i].getFit()+Ind[j].getFit());
        }
    }
 
    private Gene GetParent() {
        /* A random number rnd between 0 and 1 is chosen*/
        double rnd = Math.random();

        /* The selected individual is the first one whose 
         * accumulated normalized value is greater than R
         */     
        int i=0;
        for(; i<Ind.length; i++)
            if(Ind[i].getFit()>rnd)
                break;
        return Ind[i].getG();
    }
    
    private Gene[] NewGen() {
        Gene[] NG = this.G;
        InitFit();
        NG[0] = Ind[Ind.length-1].getG();
        for(int i=1; i<this.G.length; i++){
            NG[i] = Gene.Crossover(GetParent(), GetParent());
            NG[i].Mutate();
        }
        return NG;
    }
    
    public void Iterate() {
        this.G = NewGen();
    }
    
    public void PrintGene(int Num) {
        for(int i=0; i<Gene.MAX_LAYERS; i++) {
            System.out.print("L# "+i+" = "+(G[Num].getConnections()[i][0][0] ?1:0)+"\n");
            for(int j=1; j<Gene.MAX_NEUR_IN_LAY+1; j++) {
                System.out.print("\tN# "+j+" = "+(G[Num].getConnections()[i][j][0] ?1:0)+"\n\t\t");
                for(int k=1; k<Gene.MAX_NEUR_IN_LAY+1; k++){
                    System.out.print((G[Num].getConnections()[i][j][k] ?1:0)+" ");
                }
                System.out.print("\n");
            }
        }        
    }
}
