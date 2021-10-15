public class BeatByFrequency implements Player{

    private double numOfRocks = 0;
    private double numOfPaper = 0;
    private double numOfScissor = 0;


    @Override
    public int getMove() {
       double total = (numOfScissor+numOfPaper+numOfRocks);
       double num = Math.random();
        if (total != 0) {

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
        return RPS.ROCK;

    }

    @Override
    public void updateLastRoundInfo(int yourMove, int opponentMove, int outcome) {

        if(opponentMove == RPS.ROCK){
            numOfRocks++;
        }
       else if (opponentMove == RPS.PAPER) {
            numOfPaper++;
        }
       else {
           numOfScissor++;
        }

    }
}
