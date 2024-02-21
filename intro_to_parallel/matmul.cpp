#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <cstdlib> // For std::rand(), std::srand()
#include <ctime> // For std::time()
#include <chrono>


using Eigen::MatrixXd;
using std::vector;

std::vector<std::vector<int>> processesInCuda(const std::vector<std::vector<int>>& matrixA, 
            const std::vector<std::vector<int>>& matrixB);

vector<vector<int>> generateRandomMatrix(int rows, int cols) {
    vector<vector<int>> matrix(rows, vector<int>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = std::rand() % 10; // Random numbers between 0 and 9
        }
    }
    return matrix;
}

vector<vector<int>> manualMatrixMultiplication(const vector<vector<int>>& a, const vector<vector<int>>& b) {
    int rows = a.size();
    int cols = b[0].size();
    int temp = b.size(); // Common dimension
    vector<vector<int>> c(rows, vector<int>(cols, 0));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            for (int k = 0; k < temp; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return c;
}


void printMatrix(const Eigen::MatrixXd& matrix) {
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            std::cout << matrix(i, j) << " ";
        }
        std::cout << std::endl; // Move to the next line after each row
    }
}


bool areMatricesApproximatelyEqual(const MatrixXd& m1, const MatrixXd& m2, double tolerance = 1e-6) {
    return (m1 - m2).array().abs().maxCoeff() <= tolerance;
}


int main() {
    // std::srand(static_cast<unsigned int>(std::time(0))); // Seed random number generator
    std::srand(42);
    int rows = 4, cols = 4; // For simplicity, using square matrices

    // Generating two matrices
    auto matrixA = generateRandomMatrix(rows, cols);
    auto matrixB = generateRandomMatrix(rows, cols);

    // 1. Calculating matrix multiplication manually
    auto startManual = std::chrono::steady_clock::now();
    auto matrixC = manualMatrixMultiplication(matrixA, matrixB);
    auto endManual = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsedManual = endManual - startManual;

    MatrixXd eigenC(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            eigenC(i, j) = matrixC[i][j];
        }
    }

    // Convert vectors to Eigen matrices for simplicity in multiplication and approximation
    MatrixXd eigenA(rows, cols);
    MatrixXd eigenB(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            eigenA(i, j) = matrixA[i][j];
            eigenB(i, j) = matrixB[i][j];
        }
    }

    // 2. Calculating matrix multiplication with Eigen
    auto startEigen = std::chrono::steady_clock::now();
    MatrixXd resultEigen = eigenA * eigenB;
    auto endEigen = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsedEigen = endEigen - startEigen;

    // 3. Calculating matrix multiplication with CUDA
    auto startCuda = std::chrono::steady_clock::now();
    auto matrixCCuda = processesInCuda(matrixA, matrixB);
    auto endCuda = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsedCuda = endCuda - startCuda;

    // Print elapsed times neatly
    std::cout << "Manual Multiplication Time: " << elapsedManual.count() << " seconds" << std::endl;
    std::cout << "Eigen Multiplication Time: " << elapsedEigen.count() << " seconds" << std::endl;
    std::cout << "CUDA Multiplication Time: " << elapsedCuda.count() << " seconds" << std::endl;

    // Convert CUDA result to Eigen matrix for comparison
    MatrixXd eigenCCuda(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            eigenCCuda(i, j) = matrixCCuda[i][j];
        }
    }

    // Compare eigenC with resultEigen
    bool approximatelyEqual = resultEigen.isApprox(eigenCCuda, 1e-6); // Tolerance can be adjusted

    if (approximatelyEqual) {
        std::cout << "CUDA computed Matrix is approximately equal." << std::endl;
    } else {
        std::cout << "CUDA computed Matrix is not approximately equal." << std::endl;
    }

    if (rows < 9 and cols < 9)
    {
        std::cout << "Manual" << std::endl;
        printMatrix(eigenC);
        
        std::cout << "Eigen" << std::endl;
        printMatrix(resultEigen);

        // Print CUDA result matrix
        std::cout << "CUDA" << std::endl;
        printMatrix(eigenCCuda);

    }

    return 0;
}

