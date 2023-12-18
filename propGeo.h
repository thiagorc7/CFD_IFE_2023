// propGeo.h
#ifndef PROP_GEO_H
#define PROP_GEO_H

//#include <C:/Users/thiag/Desktop/SOFTWARES/VSCode/Test/Eigen/Dense>
//#include <C:/Users/thiag/Desktop/SOFTWARES/VSCode/Test/Eigen/Sparse>

#include <Eigen/Dense>
#include <Eigen/Sparse>

Eigen::MatrixXd somaMatrizes(const std::vector<Eigen::MatrixXd>& matrizes);
Eigen::MatrixXd produtoMatrizes(const std::vector<Eigen::MatrixXd>& matrizes);
Eigen::MatrixXd resolverSistema(const Eigen::MatrixXd& matriz, const Eigen::VectorXd& vetor);

#endif