public class BeatLastWinningMovePlayer implements Player{


    int outcome = 0;
    int move;

    @Override
    public int getMove() {
        return move;
    }

    @Override
    public void updateLastRoundInfo(int yourMove, int opponentMove, int outcome) {
        if(outcome == 2){
          move = opponentMove;
        }
        else{
            move = yourMove;
        }
    }
}
