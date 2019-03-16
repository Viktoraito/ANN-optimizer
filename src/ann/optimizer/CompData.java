package ann.optimizer;

import org.neuroph.core.data.DataSet;

public class CompData {
    private DataSet trainingSet;
    private DataSet controlSet;
    private double fitError;    
    private int MAX_ITERATIONS;

    public CompData(DataSet trainingSet, DataSet controlSet, double fitError, int MAX_ITERATIONS) {
        this.trainingSet = trainingSet;
        this.controlSet = controlSet;
        this.fitError = fitError;
        this.MAX_ITERATIONS=MAX_ITERATIONS;
    }

    /**
     * @return the trainingSet
     */
    public DataSet getTrainingSet() {
        return trainingSet;
    }

    /**
     * @param trainingSet the trainingSet to set
     */
    public void setTrainingSet(DataSet trainingSet) {
        this.trainingSet = trainingSet;
    }

    /**
     * @return the controlSet
     */
    public DataSet getControlSet() {
        return controlSet;
    }

    /**
     * @param controlSet the controlSet to set
     */
    public void setControlSet(DataSet controlSet) {
        this.controlSet = controlSet;
    }

    /**
     * @return the fitError
     */
    public double getFitError() {
        return fitError;
    }

    /**
     * @param fitError the fitError to set
     */
    public void setFitError(double fitError) {
        this.fitError = fitError;
    }

    /**
     * @return the MAX_ITERATIONS
     */
    public int getMAX_ITERATIONS() {
        return MAX_ITERATIONS;
    }

    /**
     * @param MAX_ITERATIONS the MAX_ITERATIONS to set
     */
    public void setMAX_ITERATIONS(int MAX_ITERATIONS) {
        this.MAX_ITERATIONS = MAX_ITERATIONS;
    }
}
