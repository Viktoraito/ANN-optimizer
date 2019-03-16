package ann.optimizer;

import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.input.*;
import org.neuroph.core.transfer.*;
import org.neuroph.nnet.comp.neuron.InputNeuron;
import org.neuroph.util.ConnectionFactory;
import org.neuroph.util.NeuronProperties;

public class Gene {
    final static int MAX_LAYERS = 10;
    final static int MAX_NEUR_IN_LAY = 30;
    private boolean[][][] Connections;    
    private int InputNum;
    private int OutputNum;
    private double LearRate;
    private double Momentum;
    //private Class<? extends TransferFunction> FTrans;
    //private Class<? extends InputFunction> FInp;
    private double LinearSlope;
    private double SigmoidSlope;
    
    Gene(Gene G) {
        this.Connections = new boolean[MAX_LAYERS][1+MAX_NEUR_IN_LAY][1+MAX_NEUR_IN_LAY];
        for(int i=0; i<MAX_LAYERS; i++)
            for(int j=0; j<1+MAX_NEUR_IN_LAY; j++)
                for(int k=0; k<1+MAX_NEUR_IN_LAY; k++)
                    this.Connections[i][j][k]=G.getConnections()[i][j][k];
        this.InputNum = G.getInputNum();
        this.OutputNum = G.getOutputNum();
        this.LearRate=0.2; this.Momentum=0.0;
        this.LinearSlope=1d; this.SigmoidSlope=1d;
    }
    
    Gene(boolean[][][] Connections, int InputNum, int OutputNum, 
            double LearRate, double Momentum, 
            double LinearSlope,
            double SigmoidSlope) {
        this.Connections = new boolean[MAX_LAYERS][1+MAX_NEUR_IN_LAY][1+MAX_NEUR_IN_LAY];
        for(int i=0; i<MAX_LAYERS; i++)
            this.Connections[i]=Connections[i];
        this.InputNum = InputNum;
        this.OutputNum = OutputNum;
        this.LearRate=LearRate; this.Momentum=Momentum;
        this.LinearSlope=LinearSlope; this.SigmoidSlope=SigmoidSlope;
    }
    
    Gene(boolean[][][] Connections, DataSet trainingSet) {
        this.Connections = new boolean[MAX_LAYERS][1+MAX_NEUR_IN_LAY][1+MAX_NEUR_IN_LAY];
        for(int i=0; i<MAX_LAYERS; i++)
            this.Connections[i]=Connections[i];
        InputNum = trainingSet.getInputSize();
        OutputNum = trainingSet.getOutputSize();
        this.LearRate=0.2; this.Momentum=0.0;
        this.LinearSlope=1d; this.SigmoidSlope=1d;
    }

    public boolean isConnDeletable(int LayerNum) {
        int ActiveConn=0;
        for(int i=0; i<MAX_NEUR_IN_LAY; i++)
            for(int j=0; j<MAX_NEUR_IN_LAY; j++)
                if(LayerNum > 0) {
                    //for(int k=0; k<MaxNeurInLay; k++) {
                        if( getConnections()[LayerNum]   [1+i][0]&
                            getConnections()[LayerNum]   [1+i][1+j]&
                            getConnections()[LayerNum-1] [1+j/*k*/][0]==true)
                            ActiveConn++;
                    //}
                }
                else {
                    if( getConnections()[LayerNum][1+i][0]&
                        getConnections()[LayerNum][1+i][1+j]==true)
                        ActiveConn++;
                } 
        return ActiveConn>1;
    }
    
    public boolean isNeurDeletable(int LayerNum) {
        int ActiveNeur = 0;
        for(int i=0; i<getLastActNeur(LayerNum); i++)
                if(getConnections()[LayerNum][1+i][0]==true)
                    ActiveNeur++;
        return ActiveNeur>1;
    }
    
    public boolean isLaySplitable (int LayerNum) {
        int ActiveNeur = 0;
            for(int i=getLastActNeur(LayerNum)/2; i<getLastActNeur(LayerNum); i++)
                if(getConnections()[LayerNum][1+i][0])
                    ActiveNeur++;
        return ActiveNeur>1;
    }
    
    public int getLayerCount() {
        int LayerCount = 0;
        for(int i=0; i<MAX_LAYERS; i++)
            if(getConnections()[i][0][0]==true)
                LayerCount++;
        return LayerCount;
    }
    
    public int getLastActNeur(int LayerNum) {
        int LastNeur = 0;
        for(int i=1; i<1+MAX_NEUR_IN_LAY; i++)
            if(getConnections()[LayerNum][i][0] == true)
                LastNeur = i;
        return LastNeur;
    }
    
    public static Gene Crossover(Gene a, Gene b){
        boolean[][][] Conn_c = new boolean[Gene.MAX_LAYERS][1+Gene.MAX_NEUR_IN_LAY][1+Gene.MAX_NEUR_IN_LAY];
        for(int i=0; i<Gene.MAX_LAYERS; i++)
            for(int j=0; j<Gene.MAX_NEUR_IN_LAY+1; j++)
                for(int k=0; k<Gene.MAX_NEUR_IN_LAY+1; k++)
                    if(i!=(a.getLayerCount()>b.getLayerCount()?
                            a.getLayerCount():
                            b.getLayerCount())) {
                        if(a.getConnections()[i][j][k]==b.getConnections()[i][j][k])
                            Conn_c[i][j][k]=a.getConnections()[i][j][k];
                        else {
                            int rnd = Func.rnd(0, 1);
                            Conn_c[i][j][k]=(boolean)(rnd==1);
                        }
                    }
                    else {
                        if(a.getLayerCount()>b.getLayerCount())
                            Conn_c[i][j][k]=a.getConnections()[i][j][k];
                        else
                            Conn_c[i][j][k]=b.getConnections()[i][j][k];
                    }
        double retLSlope=(a.getLinearSlope()+b.getLinearSlope())/2.0;
        double retSSlope=(a.getSigmoidSlope()+b.getSigmoidSlope())/2.0;
        return new Gene(Conn_c,a.getInputNum(),a.getOutputNum(),
                (a.getLearRate()+b.getLearRate())/2,
                (a.getMomentum()+b.getMomentum())/2,
                retLSlope, retSSlope);
    }
    
    public void Mutate() {
        int rnd = Func.rnd(1, 100);
        if(rnd>0  && rnd<=46)
            Mutation.ConnInv(this);
        if(rnd>46 && rnd<=92)
            Mutation.NeurInv(this);
        if(rnd>92 && rnd<=94)
            Mutation.LayerCopy(this);
        if(rnd>94 && rnd<=96)
            Mutation.LayerSplit(this);
        if(rnd>96 && rnd<=98)
            Mutation.LayerConvlt(this);
        if(rnd>98 && rnd<=100)
            Mutation.LayerMerge(this);
        Mutation.LRChange(this);
        Mutation.MomentChange(this);
        Mutation.LSChange(this);
        Mutation.SSChange(this);
    }
    
    public static Gene TransformFromNet(NeuralNetwork net) {
        boolean[][][] Conn = new boolean[MAX_LAYERS][1+MAX_NEUR_IN_LAY][1+MAX_NEUR_IN_LAY]; 
        int IN = net.getLayerAt(0).getNeuronsCount();
        int ON = net.getLayerAt(net.getLayersCount()-1).getNeuronsCount();
        for(int i=0; i<net.getLayersCount()-1; i++) {
            Conn[i][0][0] = true;
            for(int j=0; j<net.getLayerAt(i+1).getNeuronsCount(); j++) {
                Conn[i][1+j][0] = true;
                for(int k=0; k<net.getLayerAt(i).getNeuronsCount(); k++)
                if(net.getLayerAt(i+1).getNeuronAt(j).
                        hasInputConnectionFrom(net.getLayerAt(i).getNeuronAt(k)))
                    Conn[i][1+j][1+k] = true;
            }
        }
        Gene g = new Gene(Conn,IN,ON,0.2,0.0,1.0,1.0);
        return g;
    }
    
    public NeuralNetwork TransformToNet() {
        NeuralNetwork net = new NeuralNetwork();
        net.addLayer(0, new Layer(getInputNum(),new NeuronProperties(InputNeuron.class)));
        Linear L = new Linear(this.LinearSlope);
        for(int n=0; n<getInputNum(); n++)
            net.getLayerAt(0).getNeuronAt(n).setTransferFunction(L);
        net.setInputNeurons(net.getLayerAt(0).getNeurons());
        int i=1;
        Sigmoid S = new Sigmoid(this.SigmoidSlope);
        for(; i<getLayerCount(); i++) {
            net.addLayer(i,new Layer(MAX_NEUR_IN_LAY,new NeuronProperties(Neuron.class))); 
            for(int n=0; n<MAX_NEUR_IN_LAY; n++)
                net.getLayerAt(i).getNeuronAt(n).setTransferFunction(S);
        }
        net.addLayer(i,new Layer(getOutputNum(),new NeuronProperties(Neuron.class)));
        for(int n=0; n<getOutputNum(); n++)
            net.getLayerAt(i).getNeuronAt(n).setTransferFunction(S);
        net.setOutputNeurons(net.getLayerAt(i).getNeurons());
        int LayerNum=0;
        for(; LayerNum<getLayerCount()-1; LayerNum++) {
            for(i=0; i<MAX_NEUR_IN_LAY; i++)
                for(int j=0; j<MAX_NEUR_IN_LAY; j++)
                    if(LayerNum > 0) {
                        if( getConnections()[LayerNum]   [1+i][0]&
                            getConnections()[LayerNum]   [1+i][1+j]&
                            getConnections()[LayerNum-1] [1+j][0]==true)
                            ConnectionFactory.createConnection(
                                net.getLayerAt(LayerNum).getNeuronAt(j),
                                net.getLayerAt(LayerNum+1).getNeuronAt(i));
                    }
                    else {
                    if( getConnections()[LayerNum][1+i][0]&
                        getConnections()[LayerNum][1+i][1+j] &
                        (j<InputNum) ==true) {
                        ConnectionFactory.createConnection(
                            net.getLayerAt(LayerNum).getNeuronAt(j),
                            net.getLayerAt(LayerNum+1).getNeuronAt(i));
                    }
                }         
        }
        for(i=0; i<MAX_NEUR_IN_LAY; i++)
            for(int j=0; j<MAX_NEUR_IN_LAY; j++)
                if( getConnections()[LayerNum]  [1+i][0]&
                    getConnections()[LayerNum]  [1+i][1+j] &
                    getConnections()[LayerNum-1][1+j][0] &
                    (i<OutputNum) ==true)
                    ConnectionFactory.createConnection(
                        net.getLayerAt(LayerNum).getNeuronAt(j),
                        net.getLayerAt(LayerNum+1).getNeuronAt(i));
        for(i=0; i<getLayerCount()-1; i++)
            for(int j=MAX_NEUR_IN_LAY-1; j>=0; j--)
                //if(getConnections()[i][1+j][0] == false)
                if(!net.getLayerAt(i+1).getNeuronAt(j).hasInputConnections() ||
                    net.getLayerAt(i+1).getNeuronAt(j).getOutConnections().isEmpty())
                    net.getLayerAt(i+1).removeNeuronAt(j);
        return net;
    }

    /**
     * @return the Connections
     */
    public boolean[][][] getConnections() {
        return Connections;
    }

    /**
     * @param Connections the Connections to set
     */
    public void setConnections(boolean[][][] Connections) {
        this.Connections = Connections;
    }
    
    /**
     * Get the value of InputNum
     *
     * @return the value of InputNum
     */
    public int getInputNum() {
        return InputNum;
    }

    /**
     * Set the value of InputNum
     *
     * @param InputNum new value of InputNum
     */
    public void setInputNum(int InputNum) {
        this.InputNum = InputNum;
    }    

    /**
     * Get the value of OutputNum
     *
     * @return the value of OutputNum
     */
    public int getOutputNum() {
        return OutputNum;
    }

    /**
     * Set the value of OutputNum
     *
     * @param OutputNum new value of OutputNum
     */
    public void setOutputNum(int OutputNum) {
        this.OutputNum = OutputNum;
    }

    /**
     * Get the value of LearRate
     * 
     * @return the LearRate
     */
    public double getLearRate() {
        return LearRate;
    }

    /**
     * Set the value of LearRate
     * 
     * @param LearRate the LearRate to set
     */
    public void setLearRate(double LearRate) {
        this.LearRate = LearRate;
    }

    /**
     * Get the value of Momentum
     * 
     * @return the Momentum
     */
    public double getMomentum() {
        return Momentum;
    }

    /**
     * Set the value of Momentum
     * 
     * @param Momentum the Momentum to set
     */
    public void setMomentum(double Momentum) {
        this.Momentum = Momentum;
    }

    /**
     * @return the LinearSlope
     */
    public double getLinearSlope() {
        return LinearSlope;
    }

    /**
     * @param LinearSlope the LinearSlope to set
     */
    public void setLinearSlope(double LinearSlope) {
        this.LinearSlope = LinearSlope;
    }

    /**
     * @return the SigmoidSlope
     */
    public double getSigmoidSlope() {
        return SigmoidSlope;
    }

    /**
     * @param SigmoidSlope the SigmoidSlope to set
     */
    public void setSigmoidSlope(double SigmoidSlope) {
        this.SigmoidSlope = SigmoidSlope;
    }

    
}
