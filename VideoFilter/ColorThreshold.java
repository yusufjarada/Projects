import processing.core.PApplet;

import javax.swing.*;
import java.util.GregorianCalendar;

public class ColorThreshold implements PixelFilter {
    public int threshold;
    private static final short RED = 152;
    private static final short GREEN = 52;
    private static final short BLUE = 235;


    public ColorThreshold(){

        String message = JOptionPane.showInputDialog("Enter Threshold");
        threshold = Integer.parseInt(message);
    }

    public DImage processImage(DImage img) {
        short[][] red = img.getRedChannel();
        short[][] green = img.getGreenChannel();
        short[][] blue = img.getBlueChannel();

        for (int r = 0; r < red.length; r++) {
            for (int c = 0; c < red[0].length; c++) {
                short redTemp = red[r][c];
                short greenTemp = green[r][c];
                short blueTemp = blue[r][c];


                if (distanceToColor(redTemp, greenTemp, blueTemp) <=threshold ) {
                    red[r][c] = RED;
                    green[r][c] = GREEN;
                    blue[r][c] = BLUE;
                } else if (distanceToBlack(redTemp, greenTemp, blueTemp) >=
                        distanceToWhite(redTemp, greenTemp, blueTemp)) {
                    red[r][c] = 0;
                    green[r][c] = 0;
                    blue[r][c] = 0;
                } else {
                    red[r][c] = 255;
                    green[r][c] = 255;
                    blue[r][c] = 255;
                }


            }
        }


        img.setColorChannels(red, green, blue);
        return img;
    }

    private double distanceToColor(short redTemp, short greenTemp, short blueTemp) {
        double deltaRed = Math.abs(redTemp - RED);
        double deltaGreen = Math.abs(greenTemp - GREEN);
        double deltaBlue = Math.abs(blueTemp - BLUE);
        return Math.sqrt((deltaRed * deltaRed) + (deltaGreen * deltaGreen) + (deltaBlue * deltaBlue));

    }

    private double distanceToWhite(short redTemp, short greenTemp, short blueTemp) {
        double deltaRed = Math.abs(redTemp - 255);
        double deltaGreen = Math.abs(greenTemp - 255);
        double deltaBlue = Math.abs(blueTemp - 255);

        return Math.sqrt((deltaRed * deltaRed) + (deltaGreen * deltaGreen) + (deltaBlue * deltaBlue));


    }

    private double distanceToBlack(short redTemp, short greenTemp, short blueTemp) {

        double deltaRed = Math.abs(redTemp);
        double deltaGreen = Math.abs(greenTemp);
        double deltaBlue = Math.abs(blueTemp);

        return Math.sqrt((deltaRed * deltaRed) + (deltaGreen * deltaGreen) + (deltaBlue * deltaBlue));
    }


    @Override
    public void drawOverlay(PApplet window, DImage original, DImage filtered) {
        window.fill(255, 0, 0);
        window.ellipse(original.getWidth(), original.getHeight(), 10, 10);

        window.fill(0, 255, 0);
        window.ellipse(0, 0, 10, 10);

    }
}
