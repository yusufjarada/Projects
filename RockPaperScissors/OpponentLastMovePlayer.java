public class OpponentLastMovePlayer implements Player{


    private int opponentLastMove = RPS.ROCK;

    @Override
    public int getMove() {
        return opponentLastMove;
    }

    @Override
    public void updateLastRoundInfo(int yourMove, int opponentMove, int outcome) {
        opponentLastMove = opponentMove;
    }
}
