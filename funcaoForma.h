//arquivo de cabecalho para a funcao: funcaoForma.h
#ifndef FUNCAO_FORMA_H
#define FUNCAO_FORMA_H

//#include <C:/Users/thiag/Desktop/SOFTWARES/VSCode/Test/Eigen/Dense>
//#include <C:/Users/thiag/Desktop/SOFTWARES/VSCode/Test/Eigen/Sparse>

#include <Eigen/Dense>
#include <Eigen/Sparse>

// Formato de chamada das funcoes do programa funcaoForma.cpp
 void FuncaoFormaTriangulo(Eigen::MatrixXd& coef, Eigen::MatrixXd& dcoef_dksi1,Eigen::MatrixXd& dcoef_dksi2, int grau, Eigen::VectorXd& ksi,
                          Eigen::VectorXd& phi,
                           Eigen::MatrixXd& dphi_dksi);
#endif