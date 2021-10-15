import java.util.ArrayList;

public class PlayToBeatInfrequentPlays implements Player {

    private ArrayList<Integer> list = new ArrayList<>();
    private int checkBack;
    private double numOfRocks = 0;
    private double numOfPaper = 0;
    private double numOfScissor = 0;

    public PlayToBeatInfrequentPlays(int n){
        this.checkBack = n;
    }



    @Override
    public int getMove() {

        double total = list.size();

        double num = Math.random();

        for (int i = list.size()-1; i <= Math.min(checkBack,list.size()); i--) {

            if(list.get(i) == RPS.ROCK){
                numOfRocks++;
            }
            if(list.get(i) == RPS.PAPER){
                numOfPaper++;
            }
            if(list.get(i) == RPS.SCISSORS){
                numOfScissor++;
            }
        }

        double percentRock = numOfRocks / total;
        double percentPaper = numOfPaper / total;
        double percentScissor = numOfScissor / total;

        if (num < percentPaper) {
            return RPS.SCISSORS;
        } else if (num < percentPaper + percentScissor) {
            return RPS.ROCK;
        } else {
            return RPS.PAPER;
        }




    }

    @Override
    public void updateLastRoundInfo(int yourMove, int opponentMove, int outcome) {

        if (opponentMove == RPS.ROCK) {
            list.add(RPS.ROCK);
        } else if (opponentMove == RPS.PAPER) {
            list.add(RPS.PAPER);

        } else {
            list.add(RPS.SCISSORS);
        }

        if (list.size() > checkBack) {
            list.remove(0);
        }

    }
}
