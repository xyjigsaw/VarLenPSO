package VarLenPSO;

/**
 *
 * This class is used to record the solution and its respected accurary (for classification problems)
 */
public class SolutionPack {
    public boolean[] sol;
    public double acc;
    public SolutionPack(boolean[] s, double a) {
        sol = s; acc =a;
    }
}
