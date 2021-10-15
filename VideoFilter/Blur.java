import processing.core.PApplet;

public class Blur implements PixelFilter {
    private static final short[][] BLUR_KERNEL =
            {{1,1,1}, {1, 1, 1}, {1, 1, 1}};

    private static final short weight = 9;

    public  DImage processImage(DImage img) {


        short[][] bwpixels = img.getBWPixelGrid();
        short[][] outputImage = new short[bwpixels.length][bwpixels[0].length];
        for (int r = 0; r < bwpixels.length - (BLUR_KERNEL.length - 1); r++) {
            for (int c = 0; c < bwpixels[0].length - (BLUR_KERNEL.length - 1); c++) {

                double output = 0;

                for (int kr = 0; kr < BLUR_KERNEL.length; kr++) {
                    for (int kc = 0; kc < BLUR_KERNEL[0].length; kc++) {


                        int kernelValue = BLUR_KERNEL[kr][kc];
                        int pixelValue = bwpixels[r + kr][c + kc];
                        output += kernelValue*pixelValue;
                    }
                }

                output = output / weight;
                if (output < 0) {
                    output = 0;
                }
                if (output > 255) {
                    output = 255;
                }
                outputImage[r + 1][c + 1] = (short) output;

            }
        }

        img.setPixels(outputImage);
        return img;
    }

    @Override
    public void drawOverlay(PApplet window, DImage original, DImage filtered) {
        window.fill(255, 0, 0);
        window.ellipse(original.getWidth(), original.getHeight(), 10, 10);

        window.fill(0, 255, 0);
        window.ellipse(0, 0, 10, 10);
    }
}
