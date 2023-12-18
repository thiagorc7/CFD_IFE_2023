//arquivo de cabecalho para a funcao: funcaoCoeficientesTriangular.h
#ifndef FUNCAO_COEFICIENTES_TRIANGULAR_H
#define FUNCAO_COEFICIENTES_TRIANGULAR_H

//#include <C:/Users/thiag/Desktop/SOFTWARES/VSCode/Test/Eigen/Dense>
//#include <C:/Users/thiag/Desktop/SOFTWARES/VSCode/Test/Eigen/Sparse>

#include <Eigen/Dense>
#include <Eigen/Sparse>

// Formato de chamada das funcoes do programa 
void funcaoCoeficientesTriangular(Eigen::MatrixXd& coef, Eigen::MatrixXd& dcoef_dksi1,Eigen::MatrixXd& dcoef_dksi2, int grau);

#endif