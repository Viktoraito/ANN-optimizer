package ann.optimizer;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Locale;
import java.util.Scanner;
import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.input.WeightedSum;
import org.neuroph.core.transfer.Linear;
import org.neuroph.core.transfer.Sigmoid;
import org.neuroph.nnet.comp.neuron.InputNeuron;
import org.neuroph.util.ConnectionFactory;
import org.neuroph.util.NeuronProperties;

public class Func {
    public static int rnd(int min, int max)
        {
                max -= min;
                return (int) (Math.random() * ++max) + min;
        } 
    
    public static double calculateError(NeuralNetwork net, DataSet controlSet) {
        double error = 0;
        for (int i = 0; i < controlSet.size(); i++) {
            net.setInput(controlSet.getRowAt(i).getInput());
            net.calculate();
            for (int j = 0; j < controlSet.getRowAt(i).getDesiredOutput().length; j++) {
                double delta = controlSet.getRowAt(i).getDesiredOutput()[j] - net.getOutput()[j];
                error += Math.pow(delta, 2);
            }
        }
        System.out.print("Err: "+error/controlSet.size()+"\n");
        return error / controlSet.size();
    }    
    
    public static DataSet GetData(String filePath) {
        try {
            Scanner sc = new Scanner(new File(filePath));
            int inp=sc.nextInt(), out=sc.nextInt();
            DataSet Data = new DataSet(inp,out);
            int rows=sc.nextInt();
            sc.useLocale(Locale.US);
            for(int i=0; i<rows; i++) {
                double InpSet[] = new double[inp];
                for(int j=0; j<inp; j++)
                    InpSet[j]=sc.nextDouble();
                double OutSet[] = new double[out];
                for(int j=0; j<out; j++)
                    OutSet[j]=sc.nextDouble();
                Data.addRow(InpSet, OutSet);
            }
            return Data;
        } catch (FileNotFoundException ex) {
            System.err.println(ex);
            System.exit(1);
        }
        return null;
    }
    
    public static void GenerateNet(DataSet trainingSet, int hidLayNum, int NeurInLay, String filePath) {
        NeuralNetwork net = new NeuralNetwork();
        net.addLayer(0, new Layer(trainingSet.getInputSize(),new NeuronProperties(InputNeuron.class, Linear.class))); 
        net.setInputNeurons(net.getLayerAt(0).getNeurons());
        int i=1;
        for(; i<hidLayNum+1; i++)
            net.addLayer(i,new Layer(NeurInLay,new NeuronProperties(Neuron.class,WeightedSum.class,Sigmoid.class))); 
        net.addLayer(i,new Layer(trainingSet.getOutputSize(),new NeuronProperties(Neuron.class,WeightedSum.class,Sigmoid.class)));
        net.setOutputNeurons(net.getLayerAt(i).getNeurons());
        for(int j=0; j<i; j++)
            ConnectionFactory.fullConnect(net.getLayerAt(j),net.getLayerAt(j+1));
        net.save(filePath);
    }
}
