import processing.core.PApplet;

import java.util.ArrayList;

public class ObjectTracking implements PixelFilter {

    public int numOfClusters = 1;

    ArrayList<ClusterForBW> clusterList = new ArrayList<>();


    public ObjectTracking() {
        for (int i = 0; i < numOfClusters; i++) {
            ClusterForBW c = new ClusterForBW();
            clusterList.add(c);
        }
    }


    @Override
    public DImage processImage(DImage img) {
        short[][] bwPixelGrid = img.getBWPixelGrid();

        ArrayList<Dot> dotArrayList = getWhitePixels(bwPixelGrid);


            assignPixels(dotArrayList, clusterList);
           do{
               updateLocationOfCluster(clusterList);
           }while (allClustersStable());
            cLearClusters(clusterList);





        return img;
    }

    private boolean allClustersStable() {
       boolean areClustersStable = true;
        for (int i = 0; i < clusterList.size(); i++) {
            if(!clusterList.get(i).isClusterStable()){
                areClustersStable = false;
            }
        }
        return areClustersStable;
    }

    private void cLearClusters(ArrayList<ClusterForBW> clusterList) {
        for (int i = 0; i < clusterList.size(); i++) {
            clusterList.get(i).clear();
        }
    }

    private void updateLocationOfCluster(ArrayList<ClusterForBW> clusterList) {


        for (ClusterForBW clusterForBW : clusterList) {
            clusterForBW.updatePosition();
        }


    }


    private void assignPixels(ArrayList<Dot> dotArrayList, ArrayList<ClusterForBW> clusterList) {
        for (Dot dot : dotArrayList) {
            ClusterForBW cluster = findClosestCluster(dot);
            cluster.addDot(dot);
        }

    }

    private ArrayList<Dot> getWhitePixels(short[][] bwPixelGrid) {
        ArrayList<Dot> list = new ArrayList<>();
        for (int r = 0; r < bwPixelGrid.length; r++) {
            for (int c = 0; c < bwPixelGrid[0].length; c++) {
                if (bwPixelGrid[r][c] == 255) {
                    Dot d = new Dot(r, c);
                    list.add(d);
                }

            }
        }
        return list;
    }

    @Override
    public void drawOverlay(PApplet window, DImage original, DImage filtered) {
        for (ClusterForBW c: clusterList) {
            window.fill(0,0,255);
            window.stroke(0,0,255);
            window.ellipse(c.getLocationOfCluster().getY(), c.getLocationOfCluster().getX(), 20,20);
        }

    }


    public ClusterForBW findClosestCluster(Dot d) {
        ClusterForBW closestCluster = clusterList.get(0);
        double closestDistance = clusterList.get(0).getDistance(d);
        for (int i = 1; i < clusterList.size(); i++) {
            ClusterForBW c = clusterList.get(i);
            if (c.getDistance(d) < closestDistance) {
                closestDistance = c.getDistance(d);
                closestCluster = c;
            }

        }
        return closestCluster;
    }
}
