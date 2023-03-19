import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;

public class Main {


    public static void main(String[] args) throws IOException {
//        learnModel();
        useTrainedModel();
    }

    public static Object[] dataPreparation() throws IOException {
        int qty = 60000;
        BufferedImage[] images = new BufferedImage[qty];
        int[] rightNumbers = new int[qty];
        File[] imagesFiles = new File("/home/oleg/train").listFiles();
        double[][] inputs = new double[qty][784];
        for (int i = 0; i < qty; i++) {
            images[i] = ImageIO.read(imagesFiles[i]);
            rightNumbers[i] = Integer.parseInt(imagesFiles[i].getName().charAt(10) + "");
            for (int x = 0; x < 28; x++) {
                for (int y = 0; y < 28; y++) {
                    inputs[i][x + y * 28] = (images[i].getRGB(x, y) & 0xff) / 255.0;
                }
            }
        }
       return new Object [] {inputs, rightNumbers};
    }

    public static void useTrainedModel() throws IOException {
        NeuralNetwork nn = new NeuralNetwork( null);
        Object[] data = dataPreparation();
        double[][] inputs = (double[][]) data[0];
        int[] rightDigits =(int[]) data[1];
        int right  = 0;
        for (int c = 0; c < 10000; c++) {
            double[] outputs = nn.feedForward(inputs[c]);
            int maxDigit = 0;
            double maxDigitWeight = -1;
            for (int k = 0; k < 10; k++) {
                if(outputs[k] > maxDigitWeight) {
                    maxDigitWeight = outputs[k];
                    maxDigit = k;
                }

            }
            if (maxDigit == rightDigits[c]){
                right++;
            }
            System.out.println(maxDigit +" "+rightDigits[c]);
        }
        System.out.println("всего: 10000 правильно: "+ right );
    }

    public static void learnModel() throws IOException {
        NeuralNetwork nn = new NeuralNetwork( 784, 32, 16, 10);
        Object[] data = dataPreparation();
        double[][] inputs = (double[][]) data[0];
        int[] rightDigits =(int[]) data[1];

        int epochs = 6000;
        for (int i = 1; i < epochs; i++) {
            int right = 0;
            double errorSum = 0;
            int batchSize = 100;
            for (int j = 0; j < batchSize; j++) {
                int imgIndex = (int)(Math.random() * 60000);
                double[] targets = new double[10];
                int digit = rightDigits[imgIndex];
                targets[digit] = 1;

                double[] outputs = nn.feedForward(inputs[imgIndex]);
                int maxDigit = 0;
                double maxDigitWeight = -1;
                for (int k = 0; k < 10; k++) {
                    if(outputs[k] > maxDigitWeight) {
                        maxDigitWeight = outputs[k];
                        maxDigit = k;
                    }
                }
                if(digit == maxDigit) right++;
                for (int k = 0; k < 10; k++) {
                    errorSum += (targets[k] - outputs[k]) * (targets[k] - outputs[k]);
                }
                nn.backpropagation(targets);
            }
            System.out.println("epoch: " + i + ". correct: " + right + ". error: " + errorSum);
        }
        FileOutputStream fos = new FileOutputStream("model");
        ObjectOutputStream oos = new ObjectOutputStream(fos);

        oos.writeObject(nn.NNModel());
        oos.close();
    }

}
