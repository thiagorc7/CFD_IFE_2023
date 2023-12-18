//ENG THIAGO R CARVALHO
// FUNCAO para geracao das funcoes aproximadoras, gradintes da funcao mudanca de configuracao inicial e atual
// Deformacao de Green Lagrange; DE1, DA1
//FUNCAO PARA GERACAO E AVALIACAO DE FUNCOES DE FORMAS DE ELEMENTOS TRIANGULARES DE ORDEM QUALQUER
//Segue triangulo de Pascal
//Base triangular 2D
// propGeo.cpp

// Inclusao dos programas auxiliares ou cabecalhos ou headers
#include "propGeo.h"
#include "funcaoForma.h"
#include "funcaoCoeficientesTriangular.h"

// Inclusao das bibliotecas auxiliares
//#include <C:/Users/thiag/Desktop/SOFTWARES/VSCode/Test/Eigen/Dense>
//#include <C:/Users/thiag/Desktop/SOFTWARES/VSCode/Test/Eigen/Sparse>
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Sparse>

  void main_propGeo(){
    // Exemplo de uso
    int tipoFuncaoForma=1; //1 para Triangular; 2 para Tetraedrica; 3 para quadrangular; 4 para hexaedrica; 5 para prismatrica;
    int ndir = 2;
    int grau = 3;
    double ksi1 = 0.333333333333333;
    double ksi2 = 0.333333333333333;
    double ksi3 = 0.0;
    int dim = ((grau + 1) * (grau + 2)) / 2;
    //Eigen::VectorXd phi(dim);
    //Eigen::VectorXd dphi_dksi1(dim);
    //Eigen::VectorXd dphi_dksi2(dim);
    std::cout << "status1: ok\n";

    Eigen::VectorXd ksi = Eigen::VectorXd::Zero(ndir);
    Eigen::VectorXd phi = Eigen::VectorXd::Zero(dim);
    Eigen::MatrixXd dphi_dksi = Eigen::MatrixXd::Zero(dim,ndir);
    Eigen::MatrixXd Xil = Eigen::MatrixXd::Zero(dim,ndir);
    Eigen::MatrixXd A0 = Eigen::MatrixXd::Zero(ndir,ndir);
    Eigen::MatrixXd inversaA0 = Eigen::MatrixXd::Zero(ndir,ndir);
    Eigen::VectorXd xi = Eigen::VectorXd::Zero(ndir);
    std::cout << "status1.5: ok\n";
    std::cout << "ksi:\n" << ksi << std::endl;
    ksi(0)=ksi1;
    std::cout << "status1.51: ok\n";
    std::cout << "ksi:\n" << ksi << std::endl;
    ksi(1)=ksi1;
    std::cout << "status1.55: ok\n";
    std::cout << "ksi:\n" << ksi << std::endl;
    if (tipoFuncaoForma==2 ||  tipoFuncaoForma==4 || tipoFuncaoForma==5){
      ksi(2)=ksi3; //se tiver 3 direcoes; ou seja solido 3D
    }
    std::cout << "status1.6: ok\n";
    //Xil << 0.0,0.0, 0.333333333333333,0.0, 0.666666666666667,0.0, 1.0,0.0, 0.0,0.5, 0.5,0.5, 1.0,0.5, 0.0,1.0, 1.0,1.0, 1.0,2.0;
    //Xil << 0.0,0.0, 0.003333333333333,0.0, 0.006666666666667,0.0, 1.0,0.0, 0.0,0.003333333333333, 
    //0.003333333333333,0.003333333333333, 0.006666666666667,0.003333333333333, 0.0,0.006666666666667, 0.003333333333333,0.006666666666667, 0.0,1.0;
    //Xil << 0.0,0.0, 0.333333333333333,0.0, 0.666666666666667,0.0, 1.0,0.0, 0.0,0.333333333333333, 0.333333333333333,0.333333333333333,
    //0.666666666666667,0.333333333333333, 0.0,0.666666666666667, 0.333333333333333,0.666666666666667, 0.0,1.0;
    Xil << -0.333333333333333,-0.333333333333333, 0.0,-0.333333333333333, 0.333333333333333,-0.333333333333333, 0.666666666666667,-0.333333333333333,
    -0.333333333333333,0.0, 0.0,0.0, 0.333333333333333,0.0, -0.333333333333333,0.333333333333333, 0.0,0.333333333333333,
    -0.333333333333333,0.666666666666667;

   std::cout << "status2: ok\n";
       //////////////////////FUNCAO DE MONTAGEM DOS COEFICIENTES DAS FUNCOES DE FORMA DE ORDEM QUALQUER //////////////////////////////////////////////
        int nnoselemSuperficie= dim;
        Eigen::MatrixXd coef = Eigen::MatrixXd::Zero(nnoselemSuperficie,nnoselemSuperficie); // retorna
        Eigen::MatrixXd dcoef_dksi1 = Eigen::MatrixXd::Zero(nnoselemSuperficie,nnoselemSuperficie); // retorna
        Eigen::MatrixXd dcoef_dksi2 = Eigen::MatrixXd::Zero(nnoselemSuperficie,nnoselemSuperficie); // retorna

        funcaoCoeficientesTriangular(coef, dcoef_dksi1, dcoef_dksi2, grau); // a funcao retorna coef,dcoef_ksi calculados ordem qualquer

    //////////////////////FUNCAO DE MONTAGEM DOS COEFICIENTES DAS FUNCOES DE FORMA DE ORDEM QUALQUER ///////////////////////////////////////////////////
    //FuncaoFormaTriangulo(tipoFuncaoForma, ndir,  grau,  ksi, phi, dphi_dksi);
    FuncaoFormaTriangulo(coef, dcoef_dksi1, dcoef_dksi2, grau, ksi,  phi,  dphi_dksi); // a funcao retorna phi e dphi_dksi calculados
    std::cout << "status3: ok\n";

   Eigen::MatrixXd Yil = Eigen::MatrixXd::Zero(dim,ndir);
   Eigen::MatrixXd Uil = Eigen::MatrixXd::Zero(dim,ndir);
   Eigen::VectorXd yi = Eigen::VectorXd::Zero(ndir);
   //Uil<< 0.0,0.0, 1.111111111111E-06,3.703703703704E-07, 
   //4.444444444444E-06,2.962962962963E-06, 1.000000000000E-05,1.000000000000E-05, 
   //1.250000000000E-06,2.500000000000E-06, 5.625000000000E-06,4.375000000000E-06, 
   //1.875000000000E-05,1.500000000000E-05, 1.000000000000E-05,1.000000000000E-05, 
   //5.000000000000E-05,3.000000000000E-05, 2.100000000000E-04,9.000000000000E-05;
   
   //Uil << 0.0,0.0, 1.111111111111E-10,3.703703703704E-13, 4.444444444444E-10,2.962962962963E-12, 1.0E-05,1.0E-05, 3.703703703704E-13, 1.111111111111E-10,
   //1.114851851852E-10,1.114827160494E-10, 4.448296296296E-10,1.140790123457E-10, 2.962962962963E-12,4.444444444444E-10, 
   //1.140888888889E-10,4.448197530864E-10, 1.0E-05,1.0E-05;
   //Uil << 0.0,0.0, 10.0,20.0, 10.0,20.0, 10.0,20.0, 10.0,20.0, 10.0,20.0, 10.0,20.0, 10.0,20.0, 10.0,20.0, 10.0,20.0;

   //Yil << 1,1, 3.5,0.5, 4.5,0.5, 5.5,0.5, 0.6,1.6, 3.6,1.6, 4.7,1.7, 0.7,2.7, 3.8,2.8, 0.9,3.9;
   //Yil<< 13,20, 13,23, 13,24, 13,25, 12,20, 12,23, 12,24, 11,20, 11,23, 10,20;
   Yil << 0.333333333333333,-0.333333333333333, 0.333333333333333,0.0, 0.333333333333333,0.333333333333333, 0.333333333333333,0.666666666666667,
   0.0,-0.333333333333333, 0.0,0.0, 0.0,0.333333333333333, -0.333333333333333,-0.333333333333333, 
    -0.333333333333333,0.0, -0.666666666666667,-0.333333333333333;
   //Yil=Xil+Uil;
   ////////////////////////////////////////
   // Propriedades Geometricas Iniciais


    
    //std::cout << "Xil: \n" << Xil << std::endl;

    // Vetor de posicoes reais iniciais
    xi=phi.transpose()*Xil;
    std::cout << "xi:\n" << xi << std::endl;

    
    // Matriz Gradiente da Funcao Mudanca de Configuracao Inicial A0
    A0=dphi_dksi.transpose()*Xil;
    std::cout << "A0:\n" << A0 << std::endl;
    
    //Jacobiano de A0 - Determinante da Matriz Mudanca de Configuracao Inicial
     double J0 = A0.determinant();


    ////////////////////////////////////////
    // Propriedades Geometricas ATUAIS Y, A1
    Eigen::MatrixXd A1 = Eigen::MatrixXd::Zero(ndir,ndir);

    //dphi_dksi - Gradiente da funcao Phi ja calculado

    // Vetor de posicoes reais Atuais
    yi=phi.transpose()*Yil;
    std::cout << "yi:\n" << yi << std::endl;

    // Matriz Gradiente da Funcao Mudanca de Configuracao Atual A1
    A1=dphi_dksi.transpose()*Yil;
    std::cout << "A1:\n" << A1 << std::endl;
    

    //Jacobiano de A1 - Determinante da Matriz Mudanca de Configuracao Atual
     double J1 = A1.determinant();


    ////////////////////////////////////////
    // Propriedades de Deformacao Geometrica E, DE
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(ndir,ndir);
    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(ndir,ndir);
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(ndir, ndir);
    Eigen::MatrixXd Enl = Eigen::MatrixXd::Zero(ndir,ndir);
    Eigen::MatrixXd Snl = Eigen::MatrixXd::Zero(ndir,ndir);
    
    A=A1*A0.inverse(); //gradiente da funcao mudanca de configuracao
    std::cout << "A total:\n" << A << std::endl;
    
    C=A.transpose()*A; //tensor de alongamento a direita de cauch green
    std::cout << "C:\n" << C << std::endl;

    Enl = 0.5*(C-I); // tensor de deformacoes de green lagrange
    std::cout << "Enl:\n" << Enl << std::endl;
    
    double Ev=0; //Traco do tensor de deformacoes ou deformacao volumetrica
    for (int i=0; i <ndir; ++i){
        for (int j=0; j <ndir; ++j){
       if (i==j){
         Ev+=Enl(i,j);
       }
     } 
    } 


    // Calculo tensor de tensoes SVK Snl ndirxndir
    //Snl=2G+lambda
    double Emat=10.0; double vmat=0.3;
    double Gmat=Emat/(2*(1+vmat));
    double lambdaMat=Emat*vmat/((1-2*vmat)*(1+vmat));
    double kmat=Emat/((1+vmat)*(1-2*vmat));
    int deltaij=0;
    int tipoelem=1; //tipoelem: 1 solido 3D; 2 chapa EPT; 3 chapa EPD; 4 placa; 5 casca; 6 portico 3D;
    if (tipoelem==1){
     //Solido 3D
    for (int i=0; i <ndir; ++i){
      for (int j=0; j <ndir; ++j){
       if (i==j){
         deltaij=1;
       }
       Snl(i,j)=2*Gmat*Enl(i,j)+lambdaMat*Ev*deltaij;
     } 
    } 

    }
    else if(tipoelem==2){
     //Chapa EPT
    Snl(1,1)=(2*Gmat/(1-vmat))*(Enl(1,1)+vmat*Enl(2,2));
    Snl(1,2)= 2*Gmat*Enl(1,2);
    Snl(2,1)=2*Gmat*Enl(2,1);
    Snl(2,2)=(2*Gmat/(1-vmat))*(Enl(2,2)+vmat*Enl(1,1));

    }
    else if(tipoelem==3){
     //Chapa EPD
    Snl(1,1)=kmat*((1-vmat)*Enl(1,1)+vmat*Enl(2,2));
    Snl(1,2)= 2*Gmat*Enl(1,2);
    Snl(2,1)=2*Gmat*Enl(2,1);
    Snl(2,2)=kmat*((1-vmat)*Enl(2,2)+vmat*Enl(1,1));

    }
    else if(tipoelem==4){

    }
    else if(tipoelem==5){

    }
    else if(tipoelem==6){

    }
    else {
        std::cout << "Tipo de elemento nao implementado." << std::endl;
        //return 50; //tipo de elemento nao implementado
    }
    std::cout << "Snl:\n" << Snl << std::endl;

    ////////////////////////////////////////// Propriedades geometricas para a Hessiana e forca interna
    //Tensor derivada do gradiente mudanca de configuracao: DA1
    Eigen::MatrixXd DA1 = Eigen::MatrixXd::Zero(ndir,ndir);
    Eigen::MatrixXd DEai = Eigen::MatrixXd::Zero(ndir,ndir);
    Eigen::MatrixXd D2Eaibj = Eigen::MatrixXd::Zero(ndir,ndir);
    //Eigen::MatrixXd DSaj = Eigen::MatrixXd::Zero(ndir,ndir);
    Eigen::MatrixXd DSai = Eigen::MatrixXd::Zero(ndir,ndir);
    Eigen::MatrixXd DA1ai = Eigen::MatrixXd::Zero(ndir,ndir);
    Eigen::MatrixXd DA1bj = Eigen::MatrixXd::Zero(ndir,ndir);
    Eigen::MatrixXd DEbj = Eigen::MatrixXd::Zero(ndir,ndir);
    int nnoselem=dim; int direcao=1; 
    for (int noelem=0; noelem <nnoselem; ++noelem){
        for (int dir=0; dir <ndir; ++dir){

        //Calculo DA1
        for (int i=0; i <ndir; ++i){
           for (int j=0; j <ndir; ++j){
              int deltakronecker=0;
              if (dir==i) {
                  deltakronecker=1;
              }
              DA1(i,j)=dphi_dksi(noelem,j)*deltakronecker;
            } 
        }
       std::cout << "noelem:\n" << noelem << std::endl;
       std::cout << "dir:\n" << dir << std::endl;
       std::cout << "DA1:\n" << DA1 << std::endl;
        //Calculo DEai
        inversaA0=A0.inverse();
        DEai=0.5*(inversaA0.transpose()*DA1.transpose()*A1*inversaA0+inversaA0.transpose()*A1.transpose()*DA1*inversaA0);
        std::cout << "DEai:\n" << DEai << std::endl;
        //Calculo flai

        //Montagem vetor de forcas global Fj
        


        } 
    }    


    //Lacos Globais da hessiana
    for (int noalpha=0; noalpha <nnoselem; ++noalpha){
    for (int diri=0; diri <ndir; ++diri){

     for (int nobeta=0; nobeta <nnoselem; ++nobeta){
     for (int dirj=0; dirj <ndir; ++dirj){
       
       //Calculo DAai
        for (int i=0; i <ndir; ++i){
           for (int j=0; j <ndir; ++j){
              int deltakronecker=0;
              if (diri==i) {
                  deltakronecker=1;
              }
              DA1ai(i,j)=dphi_dksi(noalpha,j)*deltakronecker;
            } 
        }

       //Calculo DAbj
        for (int i=0; i <ndir; ++i){
           for (int j=0; j <ndir; ++j){
              int deltakronecker=0;
              if (dirj==i) {
                  deltakronecker=1;
              }
              DA1bj(i,j)=dphi_dksi(nobeta,j)*deltakronecker;
            } 
        }
        
        // Calculo DEai
        DEai=0.5*(inversaA0.transpose()*DA1.transpose()*A1*inversaA0+inversaA0.transpose()*A1.transpose()*DA1*inversaA0);

        //Calculo DEbj
        DEbj=0.5*(inversaA0.transpose()*DA1bj.transpose()*A1*inversaA0+inversaA0.transpose()*A1.transpose()*DA1bj*inversaA0);
       
        //Traco DEai
         double DEv=0; //Traco do tensor de deformacoes ou deformacao volumetrica
         for (int i=0; i <ndir; ++i){
          for (int j=0; j <ndir; ++j){
           if (i==j){
            DEv+=DEai(i,j);
           }
         } 
        }

       //Calculo DSai
        if (tipoelem==1){
         //Solido 3D
        for (int i=0; i <ndir; ++i){
          for (int j=0; j <ndir; ++j){
            if (i==j){
              deltaij=1;
             }
            DSai(i,j)=2*Gmat*DEai(i,j)+lambdaMat*DEv*deltaij;
           } 
        } 

        }
        else if(tipoelem==2){
         //Chapa EPT
         DSai(1,1)=(2*Gmat/(1-vmat))*(DEai(1,1)+vmat*DEai(2,2));
         DSai(1,2)= 2*Gmat*DEai(1,2);
         DSai(2,1)=2*Gmat*DEai(2,1);
         DSai(2,2)=(2*Gmat/(1-vmat))*(DEai(2,2)+vmat*DEai(1,1));

        }
        else if(tipoelem==3){
         //Chapa EPD
         DSai(1,1)=kmat*((1-vmat)*DEai(1,1)+vmat*DEai(2,2));
         DSai(1,2)= 2*Gmat*DEai(1,2);
         DSai(2,1)=2*Gmat*DEai(2,1);
         DSai(2,2)=kmat*((1-vmat)*DEai(2,2)+vmat*DEai(1,1));

         }
        else if(tipoelem==4){

         }
       else if(tipoelem==5){

         }
       else if(tipoelem==6){

         }
       else {
         std::cout << "Tipo de elemento nao implementado." << std::endl;
        // return 50; //tipo de elemento nao implementado
        }


       //Calculo D2Eaibj
       D2Eaibj=0.5*(inversaA0.transpose()*DA1ai.transpose()*DA1bj*inversaA0+inversaA0.transpose()*DA1bj.transpose()*DA1ai*inversaA0);


       //Calculo haibj
       double haibj=0.0; 
        for (int i=0; i <ndir; ++i){
           for (int j=0; j <ndir; ++j){ 
             haibj+=DEbj(i,j)*DSai(i,j)+Snl(i,j)*D2Eaibj(i,j);
           }
        }      

       //Calculo e montagem Hessiana haibj :=> Hestatica


            }
        }
     } 
    }//final lacos globais da hessiana  







    //////////////////////////////////////// impressoes
    // Imprime os resultados
    std::cout << "phi:\n" << phi << std::endl;
    std::cout << "dphi_dksi: \n" << dphi_dksi << std::endl;
    
} // final void

