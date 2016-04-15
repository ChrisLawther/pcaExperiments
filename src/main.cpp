#include <stdio.h>
#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues> 

int main(int argc, char *argv[])
{
    Eigen::MatrixXd data(10,2);

    data <<
        2.5, 2.4,
        0.5, 0.7,
        2.2, 2.9,
        1.9, 2.2,
        3.1, 3.0,
        2.3, 2.7,
        2.0, 1.6,
        1.0, 1.1,
        1.5, 1.6,
        1.1, 0.9;

    auto originalMean = data.colwise().mean();
    Eigen::MatrixXd centered = data.rowwise() - originalMean;
    Eigen::MatrixXd covariant = (centered.adjoint() * centered) / double(data.rows() - 1);

    std::cout << "Mean:" << std::endl;
    std::cout << originalMean << std::endl;

    std::cout << "Centered:" << std::endl;
    std::cout << centered << std::endl;

    std::cout << "Covariance:" << std::endl;
    std::cout << covariant << std::endl;

    Eigen::EigenSolver<Eigen::MatrixXd> es(covariant);

    std::cout << "Eigen Vectors:" << std::endl;
    std::cout << es.eigenvectors() << std::endl << std::endl;

    std::cout << "Eigen Values:" << std::endl;
    std::cout << es.eigenvalues() << std::endl;

    // Make a 'Feature Vector' containing EigenVectors sorted by EigenValue
    // (optionally excluding columns with low corresponding EigenValues)
    auto eigenVectors = es.eigenvectors();
    auto eigenValues = es.eigenvalues();
    
    std::vector<unsigned> indices = {0, 1};

    std::sort(indices.begin(), indices.end(), [&eigenValues](int a, int b)
    {
        return eigenValues[b].real() < eigenValues[a].real();
    });

    Eigen::Matrix<std::complex<double>, -1, -1, 0, -1, -1> featureVector(eigenVectors.rows(), eigenVectors.cols());

    unsigned col = 0;
    for (auto idx : indices)
    {
        featureVector.col(col) = eigenVectors.col(idx);
        col += 1;
    }

    featureVector.transposeInPlace();

    std::cout << "Feature Vector:" << std::endl;
    std::cout << featureVector << std::endl << std::endl;



    auto finalData = featureVector * centered.transpose();
    
    std::cout << "Transformed data:" << std::endl;
    std::cout << finalData.transpose() << std::endl << std::endl;

    auto reconstructedAdjusted = featureVector.transpose() * finalData;

    std::cout << "Reconstructed, adjusted data:" << std::endl;
    std::cout << reconstructedAdjusted.transpose() << std::endl << std::endl;
  
    // Our original mean values don't have an imaginary component, so we need to introduce one.
    Eigen::Matrix<std::complex<double>, -1, -1, 0, -1, -1> complexMean(originalMean.rows(), originalMean.cols());
    
    complexMean.data()[0, 0] = std::complex<double>(originalMean.row(0).col(0).value(), 0);
    complexMean.data()[0, 1] = std::complex<double>(originalMean.row(0).col(1).value(), 0);

    std::cout << "ComplexMean: " << complexMean << std::endl;

    auto reconstructedOriginal = reconstructedAdjusted.transpose().rowwise() + complexMean.row(0);

    std::cout << "Reconstructed, original data:" << std::endl;
    std::cout << reconstructedOriginal << std::endl << std::endl;

    return 0;
}
