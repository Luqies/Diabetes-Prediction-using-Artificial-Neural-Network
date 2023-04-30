/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Project/Maven2/JavaApp/src/main/java/${packagePath}/${mainClassName}.java to edit this template
 */

package com.mycompany.backpropagationneuralnetwork;

/**
 *
 * @author ahmsy
 */
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class BackpropagationNeuralNetwork {
    private final int numInputNodes;
    private final int numHiddenNodes;
    private final int numOutputNodes;
    private final double[][] hiddenWeights;
    private final double[] hiddenBiases;
    private final double[][] outputWeights;
    private final double[] outputBiases;
    private final double learningRate;
    private final Random random = new Random();

    public BackpropagationNeuralNetwork(int numInputNodes, int numHiddenNodes, int numOutputNodes, double learningRate) {
        this.numInputNodes = numInputNodes;
        this.numHiddenNodes = numHiddenNodes;
        this.numOutputNodes = numOutputNodes;
        this.hiddenWeights = new double[numInputNodes][numHiddenNodes];
        this.hiddenBiases = new double[numHiddenNodes];
        this.outputWeights = new double[numHiddenNodes][numOutputNodes];
        this.outputBiases = new double[numOutputNodes];
        this.learningRate = learningRate;
        
        // Initialize weights and biases randomly
        for (int i = 0; i < numInputNodes; i++) {
            for (int j = 0; j < numHiddenNodes; j++) {
                hiddenWeights[i][j] =random.nextDouble() - 0.5;
            }
        }
        for (int i = 0; i < numHiddenNodes; i++) {
            hiddenBiases[i] = random.nextDouble() - 0.5;
        }
        for (int i = 0; i < numHiddenNodes; i++) {
            for (int j = 0; j < numOutputNodes; j++) {
                outputWeights[i][j] = random.nextDouble() - 0.5;
            }
        }
        for (int i = 0; i < numOutputNodes; i++) {
            outputBiases[i] =  random.nextDouble() - 0.5;
        }
    }

    public void train(double[][] inputs, double[][] targets, int numIterations) {
       
        double error = Double.MAX_VALUE;
        int iteration =0;
        int checkA = 0;
        int n =1;
        while (iteration < numIterations && error >0.01 && checkA!=1) {
            error=0.0;
            for (int example = 0; example < inputs.length; example++) {
                double[] target = targets[example];
                // Forward pass
                double[] hiddenOutputs = new double[numHiddenNodes];
                for (int j = 0; j < numHiddenNodes; j++) {
                    double weightedSum = 0;
                    for (int i = 0; i < numInputNodes; i++) {
                        weightedSum += inputs[example][i] * hiddenWeights[i][j];
                    }
                    hiddenOutputs[j] = sigmoid(weightedSum + hiddenBiases[j]);
                }
                double[] outputs = new double[numOutputNodes];
                for (int j = 0; j < numOutputNodes; j++) {
                    double weightedSum = 0;
                    for (int i = 0; i < numHiddenNodes; i++) {
                        weightedSum += hiddenOutputs[i] * outputWeights[i][j];
                    }
                    outputs[j] = sigmoid(weightedSum + outputBiases[j]);
                }

                // Backward pass
                double[] outputErrors = new double[numOutputNodes];
                for (int j = 0; j < numOutputNodes; j++) {
                    error+=0.5*Math.pow((target[0] - outputs[0])* sigmoidDerivative(outputs[j]),2);
                    outputErrors[j] = (targets[example][j] - outputs[j]) * sigmoidDerivative(outputs[j]);
                    
                }
                double[] hiddenErrors = new double[numHiddenNodes];
                for (int j = 0; j < numHiddenNodes; j++) {
                    double weightedSum = 0;
                    for (int i = 0; i < numOutputNodes; i++) {
                        weightedSum += outputErrors[i] * outputWeights[j][i];
                    }
                    hiddenErrors[j] = weightedSum * sigmoidDerivative(hiddenOutputs[j]);
                }
                
                
                // Update weights and biases
                for (int i = 0; i < numInputNodes; i++) {
                    for (int j = 0; j < numHiddenNodes; j++) {
                        hiddenWeights[i][j] += learningRate * inputs[example][i] * hiddenErrors[j];
                        
                    }
                }
                for (int i = 0; i < numHiddenNodes; i++) {
                    hiddenBiases[i] += learningRate * hiddenErrors[i];
                }
                for (int i = 0; i < numHiddenNodes; i++) {
                    for (int j = 0; j < numOutputNodes; j++) {
                        outputWeights[i][j] += learningRate * hiddenOutputs[i] * outputErrors[j];
                    }
                }
                for (int i = 0; i < numOutputNodes; i++) {
                    outputBiases[i] += learningRate * outputErrors[i];
                    
                }
               iteration++;
               n++;
               
               System.out.println("Epoch #" + iteration + ", Error: " + error); 
               if(error<0.01){
                   checkA=1;
                   break;
               }
            }
        }
        System.out.println("Final weights:");

            System.out.println("Input-hidden weights:");
            for (int i = 0; i < numInputNodes; i++) {
            for (int j = 0; j < numHiddenNodes; j++) {
                System.out.print(hiddenWeights[i][j] + " ");
            }
            System.out.println();
             }
    }

    public double[] predict(double[] input) {
        double[] hiddenOutputs = new double[numHiddenNodes];
        for (int j = 0; j < numHiddenNodes; j++) {
            double weightedSum = 0;
            for (int i = 0; i < numInputNodes; i++) {
                weightedSum += input[i] * hiddenWeights[i][j];
            }
            hiddenOutputs[j] = sigmoid(weightedSum + hiddenBiases[j]);
        }
        double[] outputs = new double[numOutputNodes];
        for (int j = 0; j < numOutputNodes; j++) {
            double weightedSum = 0;
            for (int i = 0; i < numHiddenNodes; i++) {
                weightedSum += hiddenOutputs[i] * outputWeights[i][j];
            }
            outputs[j] = sigmoid(weightedSum + outputBiases[j]);
        }
        return outputs;
    }

    private static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    private static double sigmoidDerivative(double x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }

    public static void main(String[] args) {
        // Load diabetes dataset
        double[][] inputs = new double[100][8];
        double[][] targets = new double[100][1];
        try (BufferedReader br = new BufferedReader(new FileReader("diabetes.csv"))) {
            String line;
            int row = 0;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                for (int i = 0; i < 8; i++) {
                    inputs[row][i] = Double.parseDouble(values[i]);
                }
                targets[row][0] = Double.parseDouble(values[8]);
                row++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Split dataset into training and testing sets
        int numExamples = inputs.length;
        int numTrainingExamples = (int) (0.7 * numExamples);
        int numTestingExamples = numExamples - numTrainingExamples;
        double[][] trainingInputs = Arrays.copyOfRange(inputs, 0, numTrainingExamples);
        double[][] trainingTargets = Arrays.copyOfRange(targets, 0, numTrainingExamples);
        double[][] testingInputs = Arrays.copyOfRange(inputs, numTrainingExamples, numExamples);
        double[][] testingTargets = Arrays.copyOfRange(targets, numTrainingExamples, numExamples);

        // Train neural network
        BackpropagationNeuralNetwork nn = new BackpropagationNeuralNetwork(8, 4, 1, 0.1);
        nn.train(trainingInputs, trainingTargets, 10000);

        
        // Test neural network
        int numCorrect = 0;
        for (int i = 0; i < numTestingExamples; i++) {
            double[] output = nn.predict(testingInputs[i]);
            if ((output[0] >= 0.5 && testingTargets[i][0] == 1) || (output[0] < 0.5 && testingTargets[i][0] == 0)) {
                numCorrect++;
            }
        }
        double accuracy = (double) numCorrect / numTestingExamples;
        System.out.println("Accuracy: " + accuracy);
    }
}