package ann.optimizer;

import org.neuroph.core.input.*;
import org.neuroph.core.transfer.*;

/**
 * Contains static functions which implement mutations of Gene.
 * @author M. Kofman
 */
public class Mutation {
    /** 
    * Inverse value of random connection between two neurons.
    * If there are less than two connections with value 1, 
    * it set random connection to 1.
    * @param g Gene to mutate
    */
    public static void ConnInv(Gene g){
        System.out.println("MUT:ConnInv");
        /*random layer*/
        int rnd = Func.rnd(0, g.getLayerCount()-1);
        /*random neuron*/
        int rnd2;
        if(g.getLastActNeur(rnd)<30)
            rnd2 = Func.rnd(0, g.getLastActNeur(rnd)+1);     
        else
            rnd2 = Func.rnd(0, g.getLastActNeur(rnd));
        /*random connection*/
        int rnd3;
        if(rnd>0)
            //if(g.getLastActNeur(rnd-1)<30)
            //    rnd3 = Func.rnd(0, g.getLastActNeur(rnd-1)+1);
            //else
                rnd3 = Func.rnd(0, g.getLastActNeur(rnd-1));
        else
            rnd3 = Func.rnd(0,g.getInputNum()-1);
        
        if(g.isConnDeletable(rnd))
            g.getConnections()[rnd][1+rnd2][1+rnd3]^=true;
        else
            g.getConnections()[rnd][1+g.getLastActNeur(rnd)+1][1+rnd3]=true;//g.getConnections()[rnd][1+rnd2][1+rnd3]=true;
    }
    
    /**
     * Inverse state of neuron in random hidden layer.
     * If there are less than two neurons with value 1,
     * it set random neuron to 1.
     * @param g Gene to mutate
     */
    public static void NeurInv(Gene g) {
        System.out.println("MUT:NeurInv");
        int rnd = Func.rnd(0, g.getLayerCount()-2);
        if(g.getLayerCount()>1){
            /*random hidden layer*/
            if(Func.rnd(1, 10)<6 && g.isNeurDeletable(rnd)) {
                int rnd2;
                /*random neuron */
                if(g.getLastActNeur(rnd)<30)
                    rnd2 = Func.rnd(0, g.getLastActNeur(rnd)+1);
                else
                    rnd2 = Func.rnd(0, g.getLastActNeur(rnd));
                g.getConnections()[rnd][1+rnd2][0]^=true;
            }
            else {
                if(g.getLastActNeur(rnd)<30) {
                    int neur=g.getLastActNeur(rnd)+1;
                    g.getConnections()[rnd][1+neur][0]=true;
                    int rnd2;
                    if(rnd>0)
                        rnd2 = Func.rnd(0, g.getLastActNeur(rnd-1));
                    else
                        rnd2 = Func.rnd(0, g.getInputNum()-1);
                    int rnd3 = Func.rnd(0, g.getLastActNeur(rnd+1));
                    g.getConnections()[rnd][1+neur][1+rnd2]=true;
                    g.getConnections()[rnd+1][1+rnd3][1+neur]=true;
                }
            }
        }
    }
    
    /**
     * Copy-past one of the hidden layers if there are free place for it.
     * @param g Gene to mutate
     */    
    public static void LayerCopy(Gene g) {
        System.out.println("MUT:LayerCopy");
        if(g.getLayerCount()<10){
            /*random hidden layer*/
            int rnd = Func.rnd(0, g.getLayerCount()-2);
            /*copying layers*/
            for(int i=g.getLayerCount()-1; i>=rnd; i--)
                g.getConnections()[i+1]=g.getConnections()[i];
        }
    }

  /**
     * Split one of the hidden layers for two layers 
     * if there are free place for it.
     * @param g Gene to mutate
     */      
    public static void LayerSplit(Gene g) {
        System.out.println("MUT:LayerSplit");
        if(g.getLayerCount()<10){
            /*random hidden layer*/
            int rnd = Func.rnd(0, g.getLayerCount()-2);
            if(g.isLaySplitable(rnd)){
                /*copying layers*/
                for(int i=g.getLayerCount()-1; i>rnd; i--)  
                    for(int j=0; j<1+Gene.MAX_NEUR_IN_LAY; j++)
                        g.getConnections()[i+1][j]=g.getConnections()[i][j];
                /*free the layer*/
                for(int i=0; i<Gene.MAX_NEUR_IN_LAY; i++)    
                    for(int j=0; j<Gene.MAX_NEUR_IN_LAY+1; j++)
                        g.getConnections()[rnd+1][1+i][j] = false;
                /*copying half of the layer*/
                for(int i=g.getLastActNeur(rnd)/2, k=0; 
                        i<g.getLastActNeur(rnd); i++, k++)
                    for(int j=0; j<1+Gene.MAX_NEUR_IN_LAY; j++) {
                        g.getConnections()[rnd+1][1+k][j]=
                            g.getConnections()[rnd]
                                [1+i][j];
                            g.getConnections()[rnd]
                                [1+i][j] = false;
                    }
            }
        }
    }
    
    /**
     * Convolve two layers into one. "Throws" connections from previos layer to
     * next if they are passing through current layer.
     * @param g Gene to mutate
     */
    public static void LayerConvlt(Gene g) {
        System.out.println("MUT:LayerConvlt");
        if(g.getLayerCount()>2){
            /*random hidden & not previos to output or next to input layer*/
            int rnd = Func.rnd(1, g.getLayerCount()-2);
            
            /*save new layer configuration*/
            boolean tempLayer [][] = 
                    new boolean[1+Gene.MAX_NEUR_IN_LAY][1+Gene.MAX_NEUR_IN_LAY];
            tempLayer[0][0] = true;
            for(int i=0; i<g.getLastActNeur(rnd+1); i++)
                    tempLayer[1+i][0] = g.getConnections()[rnd+1][1+i][0];
            for(int i=0; i<g.getLastActNeur(rnd+1); i++)
                for(int j=0; j<g.getLastActNeur(rnd); j++)
                    for(int k=0; k<g.getLastActNeur(rnd-1); k++)
                    if(g.getConnections()[rnd+1][1+i][1+j] & 
                            g.getConnections()[rnd][1+j][1+k])
                        tempLayer[1+i][1+k] = true;
            
            /*write new layer and shift the rest*/
            g.getConnections()[rnd] = tempLayer;
            for(int i=rnd+1; i<g.getLayerCount()-1; i++)
                g.getConnections()[i] = g.getConnections()[i+1];
            
            /*clean last layer*/
            for(int i=Gene.MAX_NEUR_IN_LAY; i>=0; i--)
                for(int j=Gene.MAX_NEUR_IN_LAY; j>=0; j--)
                g.getConnections()[g.getLayerCount()-1][i][j] = false;
        }
    }

    /**
     * Merge two layers into one. One half of neurons in one layer becomes other
     * half in other layer.
     * @param g Gene to mutate
     */
    public static void LayerMerge(Gene g) {
        System.out.println("MUT:LayerMerge");
        if(g.getLayerCount()>2){
            /*random hidden & not previos to output or next to input layer*/
            int rnd = Func.rnd(1, g.getLayerCount()-2);
            
            /*save new layer configuration*/
            boolean tempLayer [][] = 
                    new boolean[1+Gene.MAX_NEUR_IN_LAY][1+Gene.MAX_NEUR_IN_LAY];
            tempLayer[0][0] = true;
            int k=0;
            for(int i=0; i<g.getLastActNeur(rnd)/2 && k<Gene.MAX_NEUR_IN_LAY; i++, k++)
                for(int j=0; j<1+Gene.MAX_NEUR_IN_LAY; j++)
                    tempLayer[1+i][j] = g.getConnections()[rnd][1+i][j];
            for(int i=g.getLastActNeur(rnd+1)/2; i<g.getLastActNeur(rnd+1) && k<Gene.MAX_NEUR_IN_LAY; i++, k++)
                for(int j=0; j<1+Gene.MAX_NEUR_IN_LAY; j++)
                    tempLayer[1+i][j] = g.getConnections()[rnd+1][1+i][j];
            
            /*write new layer and shift the rest*/
            g.getConnections()[rnd] = tempLayer;
            for(int i=rnd+1; i<g.getLayerCount()-1; i++)
                g.getConnections()[i] = g.getConnections()[i+1];
            
            /*clean last layer*/
            for(int i=Gene.MAX_NEUR_IN_LAY; i>=0; i--)
                for(int j=Gene.MAX_NEUR_IN_LAY; j>=0; j--)
                g.getConnections()[g.getLayerCount()-1][i][j] = false;            
        }
    }
    
    public static void MomentChange(Gene g) {
        int rnd = Func.rnd(0, 1);
        if(rnd==0) {
            if(g.getMomentum()<0.9)
                g.setMomentum(g.getMomentum()+0.1);
            else
                g.setMomentum(g.getMomentum()-0.1);
        }
        else {
            if(g.getMomentum()>=0.1)
                g.setMomentum(g.getMomentum()-0.1);
            else
                g.setMomentum(g.getMomentum()+0.1);
        }
    }
    
    public static void LRChange(Gene g) {
        int rnd = Func.rnd(0, 1);
        if(rnd==0) {
            if(g.getLearRate()<0.95)
                g.setLearRate(g.getLearRate()+0.05);
            else
                g.setLearRate(g.getLearRate()-0.05);
        }
        else {
            if(g.getLearRate()>=0.05)
                g.setLearRate(g.getLearRate()-0.05);
            else
                g.setLearRate(g.getLearRate()+0.05);
        }        
    }
    
    public static void LSChange(Gene g) {
        int rnd = Func.rnd(0, 1);
        if(rnd==0) {
            if(g.getLinearSlope()<0.9)
                g.setLinearSlope(g.getLinearSlope()+0.1);
            else
                g.setLinearSlope(g.getLinearSlope()-0.1);
        }
        else {
            if(g.getLinearSlope()>=0.1)
                g.setLinearSlope(g.getLinearSlope()-0.1);
            else
                g.setLinearSlope(g.getLinearSlope()+0.1);
        }        
    }
    
    public static void SSChange(Gene g) {
        int rnd = Func.rnd(0, 1);
        if(rnd==0) {
            if(g.getSigmoidSlope()<0.9)
                g.setSigmoidSlope(g.getSigmoidSlope()+0.1);
            else
                g.setSigmoidSlope(g.getSigmoidSlope()-0.1);
        }
        else {
            if(g.getSigmoidSlope()>=0.1)
                g.setSigmoidSlope(g.getSigmoidSlope()-0.1);
            else
                g.setSigmoidSlope(g.getSigmoidSlope()+0.1);
        }           
    }
}