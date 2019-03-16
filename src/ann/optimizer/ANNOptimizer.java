package ann.optimizer;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.exceptions.NeurophException;
import org.neuroph.nnet.learning.BackPropagation;

public class ANNOptimizer {
    public static void main(String[] args) {
        String file = "NNETS/P5_N2.nnet";
        DataSet trainingSet = Func.GetData("RomTr.dset");
        DataSet controlSet = Func.GetData("RomCr.dset");
        
        //Func.GenerateNet(trainingSet, 1, 7, file);
        /*NeuralNetwork net = NeuralNetwork.createFromFile(file);
        BackPropagation train = new BackPropagation();
        net.setLearningRule(train);
        train.learn(trainingSet,0.1);
        Func.calculateError(net, trainingSet);
        Func.calculateError(net, controlSet);*/        
        
        double fitError = 0.01;
        int maxIter = 100000;
        CompData CD = new CompData(trainingSet, controlSet, fitError, maxIter);
        int PopSize = 3; int Iterations=3;
        
        //Func.GenerateNet(trainingSet, 1, 5, file);
        try {
            NeuralNetwork.createFromFile(file);
        }
        catch(NeurophException ne) {
            System.err.println(ne);
            System.exit(1);
        }
        
        Gene g = new Gene(Gene.TransformFromNet(NeuralNetwork.createFromFile(file)));
        Population pop = new Population(g,PopSize,CD);
        int i=0;
        for(; i<Iterations; i++) {
            for(int j=0; j<PopSize; j++)
                pop.G[j].TransformToNet().save("NNETS/P"+i+"_N"+j+".nnet");               
            System.out.print("Pop #"+i+"\n");
            pop.Iterate();         
        }
        
        System.out.print("Pop #"+i+"\n");
        for(int j=0; j<PopSize; j++) {
            pop.getFitness(j);
            pop.G[j].TransformToNet().save("NNETS/P"+i+"_N"+j+".nnet"); 
        }
    }
    
}
