// Eng Thiago Rodrigues Carvalho
// funcaoFormaLagrange.cpp
// Programa para geracao das funcoes de forma e suas matrizes gradientes para Linhas-curvas
// Numeracao original: numeracao em linha 
// Renumeracao para Gmsh: antihoraria, com hierarquia
//tipoNumeracao 0: numeracao linha alta ordem; 1: numeracao antihoraria hierarquica ate 3 ordem; 

#include "funcaoFormaLagrange.h"
#include <iostream>

//#include <C:/Users/thiag/Desktop/SOFTWARES/VSCode/Test/Eigen/Dense>
//#include <C:/Users/thiag/Desktop/SOFTWARES/VSCode/Test/Eigen/Sparse>

#include <Eigen/Dense>
#include <Eigen/Sparse>

//  Implementacao do programa funcaoForma incluindo a funcao FuncaoFormaTriangulo
// Para tetraedrico e necessario incluir dphi_dksi3 ou usar apenas dphi_dksi

    void FuncaoFormaLinha(int tipoFuncaoForma,int ndir, int grau, Eigen::VectorXd& ksi,
                          Eigen::VectorXd& phi,
                           Eigen::MatrixXd& dphi_dksi) {

    // Inicializacao
    int dim=0;
    //if (tipoFuncaoForma==0) {
      dim=grau+1; //vale apenas para triangulos de grau qualquer; dim = nnoselemSuperficie
    //} //tipoFuncaoForma=1 para Triangular; 2 para Tetraedrica; 3 para quadrangular; 4 para hexaedrica; 5 para prismatrica;

    Eigen::VectorXd adm1(dim);
    Eigen::VectorXd adm2(dim);
    
    //Eigen::VectorXd dphi_dksi1 = Eigen::VectorXd::Zero(dim);Eigen::VectorXd dphi_dksi2 = Eigen::VectorXd::Zero(dim);Eigen::VectorXd dphi_dksi3 = Eigen::VectorXd::Zero(dim);
    dphi_dksi = Eigen::MatrixXd::Zero(dim,ndir);
    phi = Eigen::VectorXd::Zero(dim);

    double ksi1=ksi(0); //double ksi2=ksi(1);
    //if (tipoFuncaoForma==2) {
    //   double ksi3=ksi(2);
    //} 
    
    // Coordenadas adimensionais entre 0 e 1
    //for (int i=0; i<dim; ++i){
         //ksi(i)=(i-1)/grau;
    //}
         
    //Eigen::MatrixXd matriz(dim, dim);Eigen::MatrixXd coef(dim, dim);Eigen::MatrixXd dcoef_dksi1(dim, dim);Eigen::MatrixXd dcoef_dksi2(dim, dim);
    
    // Matriz das potencias adimensionais Mksi

           //     matriz(linha, coluna - 1) = pow(adm1(linha), (i - j)) * pow(adm2(linha), (j - 1));
    //std::cout << "Matriz M:\n" << matriz << std::endl;

    // Obtencao da matriz dos coeficientes a partir da inversao de M
    //coef = matriz.inverse();
    //std::cout << "Matriz dos Coeficientes do Elemento:\n" << coef << std::endl;

    // 1ª derivada dos coeficientes em relação a ksi1

             //   dcoef_dksi1(coluna - 1, linha) = coef(coluna - 1, linha) * (1.0 * i - 1.0 * j);
  

    // 1ª derivada dos coeficientes em relação a ksi2
 
           //     dcoef_dksi2(coluna - 1, linha) = coef(coluna - 1, linha) * (1.0 * j - 1.0);
 

    // Funções forma e derivadas - numeracao em linha
    if (grau==3){
        phi(0)=-4.5*pow(ksi1, 3)+9.0*pow(ksi1, 2)-5.5*ksi1+1;   //no1
        phi(1)=13.5*pow(ksi1, 3)-22.5*pow(ksi1, 2)+9.0*ksi1;   //no2
        phi(2)= -13.5*pow(ksi1, 3)+18.0*pow(ksi1, 2)-4.5*ksi1;  //no3
        phi(3)= 4.5*pow(ksi1, 3)-4.5*pow(ksi1, 2)+ksi1;  //no4

        dphi_dksi(0,0)=-13.5*pow(ksi1, 2)+18.0*ksi1-5.5;  //no1
        dphi_dksi(1,0)= 40.5*pow(ksi1, 2)-45.0*ksi1+9.0; //no2
        dphi_dksi(2,0)= -40.5*pow(ksi1, 2)+36.0*ksi1-4.5; //no3
        dphi_dksi(3,0)= 13.5*pow(ksi1, 2)-9.0*ksi1+1.0; //no4

    } else if(grau==2){
        phi(0)= 2.0*pow(ksi1, 2)-3.0*ksi1+1;   //no1
        phi(1)= -4.0*pow(ksi1, 2)+4.0*ksi1;   //no2
        phi(2)= 2.0*pow(ksi1, 2)-1.0*ksi1;  //no3

        dphi_dksi(0,0)= 4.0*ksi1-3.0;  //no1
        dphi_dksi(1,0)= -8.0*ksi1+4.0; //no2
        dphi_dksi(2,0)= 4.0*ksi1-1.0; //no3

    } else if(grau==1){
        phi(0)=-1.0*ksi1+1;   //no1
        phi(1)=1.0*ksi1;   //no2

        dphi_dksi(0,0)=-1.0;  //no1
        dphi_dksi(1,0)= 1.0; //no2

    }
    

          //      phi(linha) += coef(coluna - 1, linha) * pow(ksi1, (i - j)) * pow(ksi2, (j - 1));
  

    // 1ª derivadas das funções forma em relação a ksi1
 
              //      dphi_dksi1(linha) += dcoef_dksi1(coluna - 1, linha) * pow(ksi1, (i - j - 1)) * pow(ksi2, (j - 1));

    //std::cout << "dphi_dksi2:\n" << dphi_dksi2 << std::endl;
    // 1ª derivadas das funções forma em relação a ksi2

                   // dphi_dksi2(linha) += dcoef_dksi2(coluna - 1, linha) * pow(ksi1, (i - j)) * pow(ksi2, (j - 2));
                
    // adc 1ª derivadas das funções forma em relação a ksi3


    //std::cout << "dphi_dksi2:\n" << dphi_dksi2 << std::endl;
    //std::cout << "phi: \n" << phi << std::endl;
    
    //GENERALIZAR PARA QUALQUER 1D OU 2D OU 3D
    //dphi_dksi(no,dir) - derivada da funcao Phi
    //for (int linha=0; linha <dim; ++linha){
            
    //primeira direcao
            //derivadas 

            //dphi_dksi(linha,1)+=dphi_dksi2(linha);

    //}

    int tipoNumeracao=1; //tipoNumeracao 0: numeracao linha alta ordem; 1: numeracao antihoraria hierarquica ate 3 ordem; 
    Eigen::VectorXd phiLinha = Eigen::VectorXd::Zero(dim);
    Eigen::MatrixXd dphiLinha = Eigen::MatrixXd::Zero(dim,ndir);
    phiLinha = phi;
    dphiLinha= dphi_dksi;
    if (tipoNumeracao==1){

    // Renumeracao elemento 1D
    if(grau==3){
        phi(0)=phiLinha(0);
        phi(1)=phiLinha(3);
        phi(2)=phiLinha(1);
        phi(3)=phiLinha(2);

        dphi_dksi(0,0)=dphiLinha(0,0);
        dphi_dksi(1,0)=dphiLinha(3,0);
        dphi_dksi(2,0)=dphiLinha(1,0);
        dphi_dksi(3,0)=dphiLinha(2,0);

    } else if (grau==2){
        phi(0)=phiLinha(0);
        phi(1)=phiLinha(2);
        phi(2)=phiLinha(1);

        dphi_dksi(0,0)=dphiLinha(0,0);
        dphi_dksi(1,0)=dphiLinha(2,0);
        dphi_dksi(2,0)=dphiLinha(1,0);

    } else if (grau==1){
        //basta manter, pois phi=phiLinha; dphi_dksi=dphiLinha;

    }
    } //fim tiponumeracao
    

} //final void funcaoFormaLagrange

