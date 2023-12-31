//------------------------------------------------------------------------------
//--------------------------Universidade de Sao Paulo---------------------------
//----------------------Escola de Engenharia de Sao Carlos----------------------
//----------------Departamento de Engenharia de Estruturas - SET----------------
//------------------------------Sao Carlos - 2018-------------------------------
//------------------------------------------------------------------------------
 
///-----------------------------------------------------------------------------
///-----Software developed for analysis of 2D incompressible flow problems------
///----in the Arbitrary Lagrangian-Eulerian (ALE) description. The governing----
///-----equations are approximated by the Finite Element Method with a mixed----
///-formulation, stabilized finite elements (PSPG, SUPG and LSIC) and quadratic-
///----------------approximation for both velocity and pressure.----------------
///-----------------------------------------------------------------------------
  
//------------------------------------------------------------------------------
//---------------------------------Developed by---------------------------------
//-------Jeferson Wilian Dossa Fernandes and Rodolfo Andre Kuche Sanches--------
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------STARTS MAIN PROGRAM-----------------------------
//------------------------------------------------------------------------------
static char help[] = "Solves the Incompressible flow problem";

// C++ standard libraries
#include <fstream> 
  
// Developed Header Files
#include "src/Fluid.hpp"  
#include "src/fluidDomain.h"


int main(int argc, char **args) {
 
    // Starts main program invoking PETSc
    PetscInitialize(&argc, &args, (char*)0, help);

    int rank, size; 
 
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
  
    if (rank == 0){
        std::cout << "2D Incompressible Flows Numerical Analysis" << std::endl;
        std::cout << "Starting.." << std::endl;
        std::cout << "Type the input file name:" << std::endl;
    };

    // Defines the problem dimension
    const int dimension = 2;

    //Type definition
    typedef Fluid<dimension>         FluidModel;
 
//  Create problem variables 
    FluidModel control;  


//================================================================================
//==================================PROBLEM MESH==================================
//================================================================================
    Geometry* fluid2 = new Geometry(0);
    double elSize = 2.e-2;

    Point*  p1001   =   fluid2  ->  addPoint({  0.984807753 ,   -0.173648178    },elSize,   false);
    Point*  p1002   =   fluid2  ->  addPoint({  0.981809834 ,   -0.173546784    },elSize,   false);
    Point*  p1003   =   fluid2  ->  addPoint({  0.978499256 ,   -0.173432989    },elSize,   false);
    Point*  p1004   =   fluid2  ->  addPoint({  0.97484592  ,   -0.173305518    },elSize,   false);
    Point*  p1005   =   fluid2  ->  addPoint({  0.97080956  ,   -0.173161727    },elSize,   false);
    Point*  p1006   =   fluid2  ->  addPoint({  0.96635002  ,   -0.172999369    },elSize,   false);
    Point*  p1007   =   fluid2  ->  addPoint({  0.961427004 ,   -0.172816315    },elSize,   false);
    Point*  p1008   =   fluid2  ->  addPoint({  0.95599035  ,   -0.172609407    },elSize,   false);
    Point*  p1009   =   fluid2  ->  addPoint({  0.949989726 ,   -0.17237576 },elSize,   false);
    Point*  p1010   =   fluid2  ->  addPoint({  0.943365062 ,   -0.172111078    },elSize,   false);
    Point*  p1011   =   fluid2  ->  addPoint({  0.936046113 ,   -0.171810083    },elSize,   false);
    Point*  p1012   =   fluid2  ->  addPoint({  0.92796281  ,   -0.171467528    },elSize,   false);
    Point*  p1013   =   fluid2  ->  addPoint({  0.919034965 ,   -0.171077194    },elSize,   false);
    Point*  p1014   =   fluid2  ->  addPoint({  0.909182397 ,   -0.170633167    },elSize,   false);
    Point*  p1015   =   fluid2  ->  addPoint({  0.898305272 ,   -0.170125665    },elSize,   false);
    Point*  p1016   =   fluid2  ->  addPoint({  0.886293449 ,   -0.169546027    },elSize,   false);
    Point*  p1017   =   fluid2  ->  addPoint({  0.873027667 ,   -0.16887932 },elSize,   false);
    Point*  p1018   =   fluid2  ->  addPoint({  0.858387681 ,   -0.168116524    },elSize,   false);
    Point*  p1019   =   fluid2  ->  addPoint({  0.842214611 ,   -0.16723775 },elSize,   false);
    Point*  p1020   =   fluid2  ->  addPoint({  0.825293842 ,   -0.166279937    },elSize,   false);
    Point*  p1021   =   fluid2  ->  addPoint({  0.808490113 ,   -0.165291992    },elSize,   false);
    Point*  p1022   =   fluid2  ->  addPoint({  0.791793912 ,   -0.164270204    },elSize,   false);
    Point*  p1023   =   fluid2  ->  addPoint({  0.775204484 ,   -0.163218501    },elSize,   false);
    Point*  p1024   =   fluid2  ->  addPoint({  0.758721715 ,   -0.162137881    },elSize,   false);
    Point*  p1025   =   fluid2  ->  addPoint({  0.742345604 ,   -0.161028343    },elSize,   false);
    Point*  p1026   =   fluid2  ->  addPoint({  0.726075804 ,   -0.159891857    },elSize,   false);
    Point*  p1027   =   fluid2  ->  addPoint({  0.709892593 ,   -0.158724942    },elSize,   false);
    Point*  p1028   =   fluid2  ->  addPoint({  0.6938056   ,   -0.157530316    },elSize,   false);
    Point*  p1029   =   fluid2  ->  addPoint({  0.67780508  ,   -0.156306259    },elSize,   false);
    Point*  p1030   =   fluid2  ->  addPoint({  0.6618908   ,   -0.155053747    },elSize,   false);
    Point*  p1031   =   fluid2  ->  addPoint({  0.646062762 ,   -0.153772775    },elSize,   false);
    Point*  p1032   =   fluid2  ->  addPoint({  0.630311336 ,   -0.152460636    },elSize,   false);
    Point*  p1033   =   fluid2  ->  addPoint({  0.614636463 ,   -0.151117314    },elSize,   false);
    Point*  p1034   =   fluid2  ->  addPoint({  0.599028168 ,   -0.149742067    },elSize,   false);
    Point*  p1035   =   fluid2  ->  addPoint({  0.583496425 ,   -0.148335644    },elSize,   false);
    Point*  p1036   =   fluid2  ->  addPoint({  0.568031721 ,   -0.146894329    },elSize,   false);
    Point*  p1037   =   fluid2  ->  addPoint({  0.552624139 ,   -0.145417393    },elSize,   false);
    Point*  p1038   =   fluid2  ->  addPoint({  0.53728377  ,   -0.143904582    },elSize,   false);
    Point*  p1039   =   fluid2  ->  addPoint({  0.521990949 ,   -0.142352432    },elSize,   false);
    Point*  p1040   =   fluid2  ->  addPoint({  0.506755945 ,   -0.140760718    },elSize,   false);
    Point*  p1041   =   fluid2  ->  addPoint({  0.491578812 ,   -0.13912844 },elSize,   false);
    Point*  p1042   =   fluid2  ->  addPoint({  0.476440235 ,   -0.137450157    },elSize,   false);
    Point*  p1043   =   fluid2  ->  addPoint({  0.461340416 ,   -0.13572489 },elSize,   false);
    Point*  p1044   =   fluid2  ->  addPoint({  0.446289709 ,   -0.133951417    },elSize,   false);
    Point*  p1045   =   fluid2  ->  addPoint({  0.431278484 ,   -0.13212703 },elSize,   false);
    Point*  p1046   =   fluid2  ->  addPoint({  0.416297516 ,   -0.130246035    },elSize,   false);
    Point*  p1047   =   fluid2  ->  addPoint({  0.401347007 ,   -0.128307455    },elSize,   false);
    Point*  p1048   =   fluid2  ->  addPoint({  0.386427653 ,   -0.126307352    },elSize,   false);
    Point*  p1049   =   fluid2  ->  addPoint({  0.371540004 ,   -0.124242772    },elSize,   false);
    Point*  p1050   =   fluid2  ->  addPoint({  0.356674689 ,   -0.12210902 },elSize,   false);
    Point*  p1051   =   fluid2  ->  addPoint({  0.34183223  ,   -0.119903141    },elSize,   false);
    Point*  p1052   =   fluid2  ->  addPoint({  0.327003866 ,   -0.117617498    },elSize,   false);
    Point*  p1053   =   fluid2  ->  addPoint({  0.312200094 ,   -0.11524988 },elSize,   false);
    Point*  p1054   =   fluid2  ->  addPoint({  0.297402262 ,   -0.112790909    },elSize,   false);
    Point*  p1055   =   fluid2  ->  addPoint({  0.28262113  ,   -0.1102374  },elSize,   false);
    Point*  p1056   =   fluid2  ->  addPoint({  0.267857681 ,   -0.107583433    },elSize,   false);
    Point*  p1057   =   fluid2  ->  addPoint({  0.253093842 ,   -0.104816687    },elSize,   false);
    Point*  p1058   =   fluid2  ->  addPoint({  0.238350291 ,   -0.101934713    },elSize,   false);
    Point*  p1059   =   fluid2  ->  addPoint({  0.223608941 ,   -0.098925182    },elSize,   false);
    Point*  p1060   =   fluid2  ->  addPoint({  0.208881882 ,   -0.095777026    },elSize,   false);
    Point*  p1061   =   fluid2  ->  addPoint({  0.19415114  ,   -0.092476924    },elSize,   false);
    Point*  p1062   =   fluid2  ->  addPoint({  0.17943851  ,   -0.08901653 },elSize,   false);
    Point*  p1063   =   fluid2  ->  addPoint({  0.164737616 ,   -0.085374415    },elSize,   false);
    Point*  p1064   =   fluid2  ->  addPoint({  0.150051411 ,   -0.081533837    },elSize,   false);
    Point*  p1065   =   fluid2  ->  addPoint({  0.135393909 ,   -0.077472895    },elSize,   false);
    Point*  p1066   =   fluid2  ->  addPoint({  0.121396035 ,   -0.073356653    },elSize,   false);
    Point*  p1067   =   fluid2  ->  addPoint({  0.108643915 ,   -0.069373761    },elSize,   false);
    Point*  p1068   =   fluid2  ->  addPoint({  0.09702595  ,   -0.065523835    },elSize,   false);
    Point*  p1069   =   fluid2  ->  addPoint({  0.086448738 ,   -0.061806647    },elSize,   false);
    Point*  p1070   =   fluid2  ->  addPoint({  0.076827139 ,   -0.058221406    },elSize,   false);
    Point*  p1071   =   fluid2  ->  addPoint({  0.068080441 ,   -0.054765047    },elSize,   false);
    Point*  p1072   =   fluid2  ->  addPoint({  0.060134863 ,   -0.051435735    },elSize,   false);
    Point*  p1073   =   fluid2  ->  addPoint({  0.052925161 ,   -0.048229067    },elSize,   false);
    Point*  p1074   =   fluid2  ->  addPoint({  0.046390055 ,   -0.045141349    },elSize,   false);
    Point*  p1075   =   fluid2  ->  addPoint({  0.040474339 ,   -0.042168935    },elSize,   false);
    Point*  p1076   =   fluid2  ->  addPoint({  0.035128072 ,   -0.039307088    },elSize,   false);
    Point*  p1077   =   fluid2  ->  addPoint({  0.030305433 ,   -0.036550772    },elSize,   false);
    Point*  p1078   =   fluid2  ->  addPoint({  0.025965527 ,   -0.03389582 },elSize,   false);
    Point*  p1079   =   fluid2  ->  addPoint({  0.022070584 ,   -0.031337606    },elSize,   false);
    Point*  p1080   =   fluid2  ->  addPoint({  0.01858596  ,   -0.028871033    },elSize,   false);
    Point*  p1081   =   fluid2  ->  addPoint({  0.015482576 ,   -0.026494025    },elSize,   false);
    Point*  p1082   =   fluid2  ->  addPoint({  0.012731061 ,   -0.024200384    },elSize,   false);
    Point*  p1083   =   fluid2  ->  addPoint({  0.010308243 ,   -0.021989072    },elSize,   false);
    Point*  p1084   =   fluid2  ->  addPoint({  0.008191299 ,   -0.019857077    },elSize,   false);
    Point*  p1085   =   fluid2  ->  addPoint({  0.00636048  ,   -0.017802952    },elSize,   false);
    Point*  p1086   =   fluid2  ->  addPoint({  0.004796061 ,   -0.015826264    },elSize,   false);
    Point*  p1087   =   fluid2  ->  addPoint({  0.003481296 ,   -0.01392812 },elSize,   false);
    Point*  p1088   =   fluid2  ->  addPoint({  0.002399386 ,   -0.012111652    },elSize,   false);
    Point*  p1089   =   fluid2  ->  addPoint({  0.001533659 ,   -0.010378693    },elSize,   false);
    Point*  p1090   =   fluid2  ->  addPoint({  0.000865532 ,   -0.008735003    },elSize,   false);
    Point*  p1091   =   fluid2  ->  addPoint({  0.000375873 ,   -0.007185433    },elSize,   false);
    Point*  p1092   =   fluid2  ->  addPoint({  4.41059E-05 ,   -0.005734377    },elSize,   false);
    Point*  p1093   =   fluid2  ->  addPoint({  -0.00015073 ,   -0.004384842    },elSize,   false);
    Point*  p1094   =   fluid2  ->  addPoint({  -0.000231683    ,   -0.00313855 },elSize,   false);
    Point*  p1095   =   fluid2  ->  addPoint({  -0.000220521    ,   -0.00199461 },elSize,   false);
    Point*  p1096   =   fluid2  ->  addPoint({  -0.000137402    ,   -0.000949993    },elSize,   false);
    Point*  p1097   =   fluid2  ->  addPoint({  0   ,   0   },elSize,   false);
    Point*  p1098   =   fluid2  ->  addPoint({  0.000195801 ,   0.000939696 },elSize,   false);
    Point*  p1099   =   fluid2  ->  addPoint({  0.000474975 ,   0.001949743 },elSize,   false);
    Point*  p1100   =   fluid2  ->  addPoint({  0.000855736 ,   0.003028513 },elSize,   false);
    Point*  p1101   =   fluid2  ->  addPoint({  0.001358065 ,   0.004171956 },elSize,   false);
    Point*  p1102   =   fluid2  ->  addPoint({  0.002002718 ,   0.005373467 },elSize,   false);
    Point*  p1103   =   fluid2  ->  addPoint({  0.002810768 ,   0.006623542 },elSize,   false);
    Point*  p1104   =   fluid2  ->  addPoint({  0.003800881 ,   0.007912188 },elSize,   false);
    Point*  p1105   =   fluid2  ->  addPoint({  0.00499089  ,   0.009228239 },elSize,   false);
    Point*  p1106   =   fluid2  ->  addPoint({  0.006397114 ,   0.010560592 },elSize,   false);
    Point*  p1107   =   fluid2  ->  addPoint({  0.008035046 ,   0.011897479 },elSize,   false);
    Point*  p1108   =   fluid2  ->  addPoint({  0.009919724 ,   0.013231474 },elSize,   false);
    Point*  p1109   =   fluid2  ->  addPoint({  0.012065864 ,   0.01455389  },elSize,   false);
    Point*  p1110   =   fluid2  ->  addPoint({  0.014488824 ,   0.01585796  },elSize,   false);
    Point*  p1111   =   fluid2  ->  addPoint({  0.017207285 ,   0.017137342 },elSize,   false);
    Point*  p1112   =   fluid2  ->  addPoint({  0.020240303 ,   0.018386643 },elSize,   false);
    Point*  p1113   =   fluid2  ->  addPoint({  0.023610353 ,   0.019600887 },elSize,   false);
    Point*  p1114   =   fluid2  ->  addPoint({  0.027339564 ,   0.020773124 },elSize,   false);
    Point*  p1115   =   fluid2  ->  addPoint({  0.031457657 ,   0.021899132 },elSize,   false);
    Point*  p1116   =   fluid2  ->  addPoint({  0.035992667 ,   0.022970919 },elSize,   false);
    Point*  p1117   =   fluid2  ->  addPoint({  0.040978892 ,   0.023981422 },elSize,   false);
    Point*  p1118   =   fluid2  ->  addPoint({  0.046453406 ,   0.024922072 },elSize,   false);
    Point*  p1119   =   fluid2  ->  addPoint({  0.052456063 ,   0.025782798 },elSize,   false);
    Point*  p1120   =   fluid2  ->  addPoint({  0.059031643 ,   0.026552659 },elSize,   false);
    Point*  p1121   =   fluid2  ->  addPoint({  0.066228696 ,   0.027219027 },elSize,   false);
    Point*  p1122   =   fluid2  ->  addPoint({  0.074100345 ,   0.027766447 },elSize,   false);
    Point*  p1123   =   fluid2  ->  addPoint({  0.082705438 ,   0.028177428 },elSize,   false);
    Point*  p1124   =   fluid2  ->  addPoint({  0.092106789 ,   0.028433796 },elSize,   false);
    Point*  p1125   =   fluid2  ->  addPoint({  0.102374359 ,   0.02851204  },elSize,   false);
    Point*  p1126   =   fluid2  ->  addPoint({  0.113585041 ,   0.028387435 },elSize,   false);
    Point*  p1127   =   fluid2  ->  addPoint({  0.125819109 ,   0.028031604 },elSize,   false);
    Point*  p1128   =   fluid2  ->  addPoint({  0.139164411 ,   0.027412817 },elSize,   false);
    Point*  p1129   =   fluid2  ->  addPoint({  0.153725947 ,   0.026493263 },elSize,   false);
    Point*  p1130   =   fluid2  ->  addPoint({  0.168888418 ,   0.02529614  },elSize,   false);
    Point*  p1131   =   fluid2  ->  addPoint({  0.184002492 ,   0.023882125 },elSize,   false);
    Point*  p1132   =   fluid2  ->  addPoint({  0.19906249  ,   0.022276591 },elSize,   false);
    Point*  p1133   =   fluid2  ->  addPoint({  0.214071365 ,   0.020496283 },elSize,   false);
    Point*  p1134   =   fluid2  ->  addPoint({  0.229042435 ,   0.018559153 },elSize,   false);
    Point*  p1135   =   fluid2  ->  addPoint({  0.243958077 ,   0.016480502 },elSize,   false);
    Point*  p1136   =   fluid2  ->  addPoint({  0.258839735 ,   0.014266696 },elSize,   false);
    Point*  p1137   =   fluid2  ->  addPoint({  0.273679834 ,   0.011932275 },elSize,   false);
    Point*  p1138   =   fluid2  ->  addPoint({  0.288499587 ,   0.009482636 },elSize,   false);
    Point*  p1139   =   fluid2  ->  addPoint({  0.303280401 ,   0.006927152 },elSize,   false);
    Point*  p1140   =   fluid2  ->  addPoint({  0.318043474 ,   0.00427122  },elSize,   false);
    Point*  p1141   =   fluid2  ->  addPoint({  0.332789905 ,   0.001520741 },elSize,   false);
    Point*  p1142   =   fluid2  ->  addPoint({  0.347510673 ,   -0.001317614    },elSize,   false);
    Point*  p1143   =   fluid2  ->  addPoint({  0.362226513 ,   -0.004241412    },elSize,   false);
    Point*  p1144   =   fluid2  ->  addPoint({  0.376928318 ,   -0.007244983    },elSize,   false);
    Point*  p1145   =   fluid2  ->  addPoint({  0.39162693  ,   -0.010324149    },elSize,   false);
    Point*  p1146   =   fluid2  ->  addPoint({  0.406322872 ,   -0.013475955    },elSize,   false);
    Point*  p1147   =   fluid2  ->  addPoint({  0.421026555 ,   -0.016699193    },elSize,   false);
    Point*  p1148   =   fluid2  ->  addPoint({  0.435738471 ,   -0.019990898    },elSize,   false);
    Point*  p1149   =   fluid2  ->  addPoint({  0.450459315 ,   -0.023347134    },elSize,   false);
    Point*  p1150   =   fluid2  ->  addPoint({  0.465189229 ,   -0.026766912    },elSize,   false);
    Point*  p1151   =   fluid2  ->  addPoint({  0.479938831 ,   -0.030248037    },elSize,   false);
    Point*  p1152   =   fluid2  ->  addPoint({  0.494718095 ,   -0.033791259    },elSize,   false);
    Point*  p1153   =   fluid2  ->  addPoint({  0.509517711 ,   -0.037391887    },elSize,   false);
    Point*  p1154   =   fluid2  ->  addPoint({  0.524337823 ,   -0.041048933    },elSize,   false);
    Point*  p1155   =   fluid2  ->  addPoint({  0.539198442 ,   -0.044763889    },elSize,   false);
    Point*  p1156   =   fluid2  ->  addPoint({  0.55409986  ,   -0.048535798    },elSize,   false);
    Point*  p1157   =   fluid2  ->  addPoint({  0.569032503 ,   -0.052360936    },elSize,   false);
    Point*  p1158   =   fluid2  ->  addPoint({  0.584016036 ,   -0.056242774    },elSize,   false);
    Point*  p1159   =   fluid2  ->  addPoint({  0.599041063 ,   -0.060177621    },elSize,   false);
    Point*  p1160   =   fluid2  ->  addPoint({  0.614117152 ,   -0.064168184    },elSize,   false);
    Point*  p1161   =   fluid2  ->  addPoint({  0.629254514 ,   -0.068214226    },elSize,   false);
    Point*  p1162   =   fluid2  ->  addPoint({  0.64444352  ,   -0.072313039    },elSize,   false);
    Point*  p1163   =   fluid2  ->  addPoint({  0.659693797 ,   -0.076467337    },elSize,   false);
    Point*  p1164   =   fluid2  ->  addPoint({  0.675005405 ,   -0.080677124    },elSize,   false);
    Point*  p1165   =   fluid2  ->  addPoint({  0.690388321 ,   -0.084943153    },elSize,   false);
    Point*  p1166   =   fluid2  ->  addPoint({  0.705842544 ,   -0.089265415    },elSize,   false);
    Point*  p1167   =   fluid2  ->  addPoint({  0.721367959 ,   -0.093644909    },elSize,   false);
    Point*  p1168   =   fluid2  ->  addPoint({  0.736974311 ,   -0.098083353    },elSize,   false);
    Point*  p1169   =   fluid2  ->  addPoint({  0.752651623 ,   -0.102580004    },elSize,   false);
    Point*  p1170   =   fluid2  ->  addPoint({  0.768419618 ,   -0.107138339    },elSize,   false);
    Point*  p1171   =   fluid2  ->  addPoint({  0.784277948 ,   -0.111760328    },elSize,   false);
    Point*  p1172   =   fluid2  ->  addPoint({  0.800226615 ,   -0.116445969    },elSize,   false);
    Point*  p1173   =   fluid2  ->  addPoint({  0.816265384 ,   -0.12119624 },elSize,   false);
    Point*  p1174   =   fluid2  ->  addPoint({  0.832393621 ,   -0.126015088    },elSize,   false);
    Point*  p1175   =   fluid2  ->  addPoint({  0.848621534 ,   -0.130902282    },elSize,   false);
    Point*  p1176   =   fluid2  ->  addPoint({  0.864119807 ,   -0.135608021    },elSize,   false);
    Point*  p1177   =   fluid2  ->  addPoint({  0.878137786 ,   -0.139898397    },elSize,   false);
    Point*  p1178   =   fluid2  ->  addPoint({  0.890831571 ,   -0.143809062    },elSize,   false);
    Point*  p1179   =   fluid2  ->  addPoint({  0.902317239 ,   -0.147372666    },elSize,   false);
    Point*  p1180   =   fluid2  ->  addPoint({  0.91271197  ,   -0.150615965    },elSize,   false);
    Point*  p1181   =   fluid2  ->  addPoint({  0.922122221 ,   -0.153568493    },elSize,   false);
    Point*  p1182   =   fluid2  ->  addPoint({  0.930645154 ,   -0.156255203    },elSize,   false);
    Point*  p1183   =   fluid2  ->  addPoint({  0.938358135 ,   -0.158697959    },elSize,   false);
    Point*  p1184   =   fluid2  ->  addPoint({  0.945338643 ,   -0.160918344    },elSize,   false);
    Point*  p1185   =   fluid2  ->  addPoint({  0.951654317 ,   -0.162935393    },elSize,   false);
    Point*  p1186   =   fluid2  ->  addPoint({  0.957372971 ,   -0.164768171    },elSize,   false);
    Point*  p1187   =   fluid2  ->  addPoint({  0.962552522 ,   -0.166433185    },elSize,   false);
    Point*  p1188   =   fluid2  ->  addPoint({  0.967241252 ,   -0.167944942    },elSize,   false);
    Point*  p1189   =   fluid2  ->  addPoint({  0.971487378 ,   -0.169317628    },elSize,   false);
    Point*  p1190   =   fluid2  ->  addPoint({  0.975329495 ,   -0.170563025    },elSize,   false);
    Point*  p1191   =   fluid2  ->  addPoint({  0.978806106 ,   -0.171692755    },elSize,   false);
    Point*  p1192   =   fluid2  ->  addPoint({  0.981955952 ,   -0.172718108    },elSize,   false);


    
    Line* l1001 = fluid2 -> addLine({p1001,p1002,p1003,p1004,p1005,p1006,p1007,p1008,p1009,p1010,
                                     p1011,p1012,p1013,p1014,p1015,p1016,p1017,p1018,p1019,p1020,
                                     p1021,p1022,p1023,p1024,p1025,p1026,p1027,p1028,p1029,p1030,
                                     p1031,p1032,p1033,p1034,p1035,p1036,p1037,p1038,p1039,p1040,
                                     p1041,p1042,p1043,p1044,p1045,p1046,p1047,p1048,p1049,p1050,
                                     p1051,p1052,p1053,p1054,p1055,p1056,p1057});

    Line* l1002 = fluid2 -> addLine({p1057,p1058,p1059,p1060,p1061,p1062,p1063,p1064,p1065,p1066,
                                    p1067,p1068,p1069,p1070,p1071,p1072,p1073,p1074,p1075,p1076,
                                    p1077,p1078,p1079,p1080,p1081,p1082,p1083,p1084,p1085,p1086,
                                    p1087,p1088,p1089,p1090,p1091,p1092,p1093,p1094,p1095,p1096,
                                    p1097});

    Line* l1003 = fluid2 -> addLine({p1097,p1098,p1099,p1100,p1101,p1102,p1103,p1104,p1105,p1106,
                                    p1107,p1108,p1109,p1110,p1111,p1112,p1113,p1114,p1115,p1116,
                                    p1117,p1118,p1119,p1120,p1121,p1122,p1123,p1124,p1125,p1126,
                                    p1127,p1128,p1129,p1130,p1131,p1132,p1133,p1134,p1135,p1136,
                                    p1137});

    Line* l1004 = fluid2 -> addLine({p1137,p1138,p1139,p1140,p1141,p1142,p1143,p1144,p1145,p1146,
                                    p1147,p1148,p1149,p1150,p1151,p1152,p1153,p1154,p1155,p1156,
                                    p1157,p1158,p1159,p1160,p1161,p1162,p1163,p1164,p1165,p1166,
                                    p1167,p1168,p1169,p1170,p1171,p1172,p1173,p1174,p1175,p1176,
                                    p1177,p1178,p1179,p1180,p1181,p1182,p1183,p1184,p1185,p1186,
                                    p1187,p1188,p1189,p1190,p1191,p1192,p1001});

    Point*  p1193   =   fluid2  ->  addPoint({  0.984300724 ,   -0.188639606    },elSize,   false);
    Point*  p1194   =   fluid2  ->  addPoint({  0.981298469 ,   -0.188538065    },elSize,   false);
    Point*  p1195   =   fluid2  ->  addPoint({  0.977979889 ,   -0.188423995    },elSize,   false);
    Point*  p1196   =   fluid2  ->  addPoint({  0.974317106 ,   -0.188296193    },elSize,   false);
    Point*  p1197   =   fluid2  ->  addPoint({  0.970269387 ,   -0.188151997    },elSize,   false);
    Point*  p1198   =   fluid2  ->  addPoint({  0.965798181 ,   -0.187989215    },elSize,   false);
    Point*  p1199   =   fluid2  ->  addPoint({  0.960862768 ,   -0.187805699    },elSize,   false);
    Point*  p1200   =   fluid2  ->  addPoint({  0.955412989 ,   -0.187598291    },elSize,   false);
    Point*  p1201   =   fluid2  ->  addPoint({  0.949398126 ,   -0.187364089    },elSize,   false);
    Point*  p1202   =   fluid2  ->  addPoint({  0.942757029 ,   -0.187098749    },elSize,   false);
    Point*  p1203   =   fluid2  ->  addPoint({  0.935419917 ,   -0.186797007    },elSize,   false);
    Point*  p1204   =   fluid2  ->  addPoint({  0.927317166 ,   -0.186453626    },elSize,   false);
    Point*  p1205   =   fluid2  ->  addPoint({  0.918369214 ,   -0.186062413    },elSize,   false);
    Point*  p1206   =   fluid2  ->  addPoint({  0.908494595 ,   -0.18561739 },elSize,   false);
    Point*  p1207   =   fluid2  ->  addPoint({  0.897593629 ,   -0.185108774    },elSize,   false);
    Point*  p1208   =   fluid2  ->  addPoint({  0.885554752 ,   -0.184527827    },elSize,   false);
    Point*  p1209   =   fluid2  ->  addPoint({  0.872260282 ,   -0.183859677    },elSize,   false);
    Point*  p1210   =   fluid2  ->  addPoint({  0.857589686 ,   -0.183095282    },elSize,   false);
    Point*  p1211   =   fluid2  ->  addPoint({  0.841383445 ,   -0.182214705    },elSize,   false);
    Point*  p1212   =   fluid2  ->  addPoint({  0.824429843 ,   -0.181255034    },elSize,   false);
    Point*  p1213   =   fluid2  ->  addPoint({  0.807591846 ,   -0.180265071    },elSize,   false);
    Point*  p1214   =   fluid2  ->  addPoint({  0.790861312 ,   -0.179241185    },elSize,   false);
    Point*  p1215   =   fluid2  ->  addPoint({  0.774239366 ,   -0.17818742 },elSize,   false);
    Point*  p1216   =   fluid2  ->  addPoint({  0.757724126 ,   -0.177104672    },elSize,   false);
    Point*  p1217   =   fluid2  ->  addPoint({  0.741316044 ,   -0.175992968    },elSize,   false);
    Point*  p1218   =   fluid2  ->  addPoint({  0.725013829 ,   -0.174854216    },elSize,   false);
    Point*  p1219   =   fluid2  ->  addPoint({  0.708797818 ,   -0.173684938    },elSize,   false);
    Point*  p1220   =   fluid2  ->  addPoint({  0.692678132 ,   -0.172487883    },elSize,   false);
    Point*  p1221   =   fluid2  ->  addPoint({  0.676644578 ,   -0.171261299    },elSize,   false);
    Point*  p1222   =   fluid2  ->  addPoint({  0.660697384 ,   -0.170006197    },elSize,   false);
    Point*  p1223   =   fluid2  ->  addPoint({  0.644835186 ,   -0.168722459    },elSize,   false);
    Point*  p1224   =   fluid2  ->  addPoint({  0.629048366 ,   -0.167407372    },elSize,   false);
    Point*  p1225   =   fluid2  ->  addPoint({  0.613337828 ,   -0.166060993    },elSize,   false);
    Point*  p1226   =   fluid2  ->  addPoint({  0.597693565 ,   -0.164682577    },elSize,   false);
    Point*  p1227   =   fluid2  ->  addPoint({  0.582124108 ,   -0.163272737    },elSize,   false);
    Point*  p1228   =   fluid2  ->  addPoint({  0.566620115 ,   -0.16182776 },elSize,   false);
    Point*  p1229   =   fluid2  ->  addPoint({  0.551172475 ,   -0.160346984    },elSize,   false);
    Point*  p1230   =   fluid2  ->  addPoint({  0.535790422 ,   -0.158830061    },elSize,   false);
    Point*  p1231   =   fluid2  ->  addPoint({  0.520454322 ,   -0.157273517    },elSize,   false);
    Point*  p1232   =   fluid2  ->  addPoint({  0.505174654 ,   -0.155677136    },elSize,   false);
    Point*  p1233   =   fluid2  ->  addPoint({  0.48995045  ,   -0.154039793    },elSize,   false);
    Point*  p1234   =   fluid2  ->  addPoint({  0.474762471 ,   -0.152356032    },elSize,   false);
    Point*  p1235   =   fluid2  ->  addPoint({  0.459611383 ,   -0.150624905    },elSize,   false);
    Point*  p1236   =   fluid2  ->  addPoint({  0.444507206 ,   -0.14884513 },elSize,   false);
    Point*  p1237   =   fluid2  ->  addPoint({  0.429439294 ,   -0.147013849    },elSize,   false);
    Point*  p1238   =   fluid2  ->  addPoint({  0.414398754 ,   -0.145125373    },elSize,   false);
    Point*  p1239   =   fluid2  ->  addPoint({  0.39938606  ,   -0.143178725    },elSize,   false);
    Point*  p1240   =   fluid2  ->  addPoint({  0.38440091  ,   -0.141169798    },elSize,   false);
    Point*  p1241   =   fluid2  ->  addPoint({  0.369444171 ,   -0.139095633    },elSize,   false);
    Point*  p1242   =   fluid2  ->  addPoint({  0.35450654  ,   -0.136951498    },elSize,   false);
    Point*  p1243   =   fluid2  ->  addPoint({  0.339587132 ,   -0.134734174    },elSize,   false);
    Point*  p1244   =   fluid2  ->  addPoint({  0.324676869 ,   -0.132435902    },elSize,   false);
    Point*  p1245   =   fluid2  ->  addPoint({  0.309786197 ,   -0.130054376    },elSize,   false);
    Point*  p1246   =   fluid2  ->  addPoint({  0.294896077 ,   -0.127580061    },elSize,   false);
    Point*  p1247   =   fluid2  ->  addPoint({  0.280017392 ,   -0.12500969 },elSize,   false);
    Point*  p1248   =   fluid2  ->  addPoint({  0.265149197 ,   -0.122336877    },elSize,   false);
    Point*  p1249   =   fluid2  ->  addPoint({  0.250273544 ,   -0.119549165    },elSize,   false);
    Point*  p1250   =   fluid2  ->  addPoint({  0.235411176 ,   -0.116643948    },elSize,   false);
    Point*  p1251   =   fluid2  ->  addPoint({  0.220540825 ,   -0.113608052    },elSize,   false);
    Point*  p1252   =   fluid2  ->  addPoint({  0.205674357 ,   -0.110430073    },elSize,   false);
    Point*  p1253   =   fluid2  ->  addPoint({  0.190794339 ,   -0.107096496    },elSize,   false);
    Point*  p1254   =   fluid2  ->  addPoint({  0.175917641 ,   -0.103597458    },elSize,   false);
    Point*  p1255   =   fluid2  ->  addPoint({  0.16103634  ,   -0.099910596    },elSize,   false);
    Point*  p1256   =   fluid2  ->  addPoint({  0.146151214 ,   -0.096017914    },elSize,   false);
    Point*  p1257   =   fluid2  ->  addPoint({  0.131277776 ,   -0.091897091    },elSize,   false);
    Point*  p1258   =   fluid2  ->  addPoint({  0.117049328 ,   -0.087713051    },elSize,   false);
    Point*  p1259   =   fluid2  ->  addPoint({  0.104054003 ,   -0.083654262    },elSize,   false);
    Point*  p1260   =   fluid2  ->  addPoint({  0.092185511 ,   -0.079721376    },elSize,   false);
    Point*  p1261   =   fluid2  ->  addPoint({  0.081348872 ,   -0.075913077    },elSize,   false);
    Point*  p1262   =   fluid2  ->  addPoint({  0.07145782  ,   -0.07222749 },elSize,   false);
    Point*  p1263   =   fluid2  ->  addPoint({  0.062431618 ,   -0.068660758    },elSize,   false);
    Point*  p1264   =   fluid2  ->  addPoint({  0.054194717 ,   -0.06520943 },elSize,   false);
    Point*  p1265   =   fluid2  ->  addPoint({  0.046679701 ,   -0.061867037    },elSize,   false);
    Point*  p1266   =   fluid2  ->  addPoint({  0.039825463 ,   -0.058628607    },elSize,   false);
    Point*  p1267   =   fluid2  ->  addPoint({  0.033574509 ,   -0.055487807    },elSize,   false);
    Point*  p1268   =   fluid2  ->  addPoint({  0.027874259 ,   -0.052436528    },elSize,   false);
    Point*  p1269   =   fluid2  ->  addPoint({  0.022677589 ,   -0.049466499    },elSize,   false);
    Point*  p1270   =   fluid2  ->  addPoint({  0.017942204 ,   -0.046569663    },elSize,   false);
    Point*  p1271   =   fluid2  ->  addPoint({  0.013628259 ,   -0.043736281    },elSize,   false);
    Point*  p1272   =   fluid2  ->  addPoint({  0.00970075  ,   -0.040956274    },elSize,   false);
    Point*  p1273   =   fluid2  ->  addPoint({  0.006128365 ,   -0.03822001 },elSize,   false);
    Point*  p1274   =   fluid2  ->  addPoint({  0.002881493 ,   -0.035513475    },elSize,   false);
    Point*  p1275   =   fluid2  ->  addPoint({  -6.1251E-05 ,   -0.032827595    },elSize,   false);
    Point*  p1276   =   fluid2  ->  addPoint({  -0.002721436    ,   -0.030148443    },elSize,   false);
    Point*  p1277   =   fluid2  ->  addPoint({  -0.00511197 ,   -0.027466434    },elSize,   false);
    Point*  p1278   =   fluid2  ->  addPoint({  -0.007244   ,   -0.0247726  },elSize,   false);
    Point*  p1279   =   fluid2  ->  addPoint({  -0.00912292 ,   -0.022060386    },elSize,   false);
    Point*  p1280   =   fluid2  ->  addPoint({  -0.010750961    ,   -0.019327494    },elSize,   false);
    Point*  p1281   =   fluid2  ->  addPoint({  -0.012123342    ,   -0.016582428    },elSize,   false);
    Point*  p1282   =   fluid2  ->  addPoint({  -0.013236185    ,   -0.013847885    },elSize,   false);
    Point*  p1283   =   fluid2  ->  addPoint({  -0.014091815    ,   -0.011145987    },elSize,   false);
    Point*  p1284   =   fluid2  ->  addPoint({  -0.014697552    ,   -0.008506291    },elSize,   false);
    Point*  p1285   =   fluid2  ->  addPoint({  -0.015066782    ,   -0.005969573    },elSize,   false);
    Point*  p1286   =   fluid2  ->  addPoint({  -0.015225293    ,   -0.003576339    },elSize,   false);
    Point*  p1287   =   fluid2  ->  addPoint({  -0.015206621    ,   -0.00134902 },elSize,   false);
    Point*  p1288   =   fluid2  ->  addPoint({  -0.01504656 ,   0.000698337 },elSize,   false);
    Point*  p1289   =   fluid2  ->  addPoint({  -0.014772116    ,   0.002604723 },elSize,   false);
    Point*  p1290   =   fluid2  ->  addPoint({  -0.014377987    ,   0.004490004 },elSize,   false);
    Point*  p1291   =   fluid2  ->  addPoint({  -0.013828158    ,   0.006468635 },elSize,   false);
    Point*  p1292   =   fluid2  ->  addPoint({  -0.013083916    ,   0.008568016 },elSize,   false);
    Point*  p1293   =   fluid2  ->  addPoint({  -0.01211643 ,   0.010762706 },elSize,   false);
    Point*  p1294   =   fluid2  ->  addPoint({  -0.010901859    ,   0.013020158 },elSize,   false);
    Point*  p1295   =   fluid2  ->  addPoint({  -0.009429823    ,   0.015293486 },elSize,   false);
    Point*  p1296   =   fluid2  ->  addPoint({  -0.00770169 ,   0.017539797 },elSize,   false);
    Point*  p1297   =   fluid2  ->  addPoint({  -0.00572069 ,   0.019728813 },elSize,   false);
    Point*  p1298   =   fluid2  ->  addPoint({  -0.003492207    ,   0.021838949 },elSize,   false);
    Point*  p1299   =   fluid2  ->  addPoint({  -0.001027644    ,   0.023850204 },elSize,   false);
    Point*  p1300   =   fluid2  ->  addPoint({  0.001665595 ,   0.025756224 },elSize,   false);
    Point*  p1301   =   fluid2  ->  addPoint({  0.004590393 ,   0.027558402 },elSize,   false);
    Point*  p1302   =   fluid2  ->  addPoint({  0.007754062 ,   0.029261056 },elSize,   false);
    Point*  p1303   =   fluid2  ->  addPoint({  0.011170142 ,   0.030868798 },elSize,   false);
    Point*  p1304   =   fluid2  ->  addPoint({  0.014854041 ,   0.032386221 },elSize,   false);
    Point*  p1305   =   fluid2  ->  addPoint({  0.018830793 ,   0.033819037 },elSize,   false);
    Point*  p1306   =   fluid2  ->  addPoint({  0.023123594 ,   0.035168457 },elSize,   false);
    Point*  p1307   =   fluid2  ->  addPoint({  0.027765064 ,   0.036437521 },elSize,   false);
    Point*  p1308   =   fluid2  ->  addPoint({  0.032787919 ,   0.037624573 },elSize,   false);
    Point*  p1309   =   fluid2  ->  addPoint({  0.038228502 ,   0.038727112 },elSize,   false);
    Point*  p1310   =   fluid2  ->  addPoint({  0.044127585 ,   0.03974066  },elSize,   false);
    Point*  p1311   =   fluid2  ->  addPoint({  0.050527666 ,   0.040658324 },elSize,   false);
    Point*  p1312   =   fluid2  ->  addPoint({  0.057475858 ,   0.041471759 },elSize,   false);
    Point*  p1313   =   fluid2  ->  addPoint({  0.065024343 ,   0.0421706   },elSize,   false);
    Point*  p1314   =   fluid2  ->  addPoint({  0.073229314 ,   0.042741135 },elSize,   false);
    Point*  p1315   =   fluid2  ->  addPoint({  0.082149894 ,   0.043167137 },elSize,   false);
    Point*  p1316   =   fluid2  ->  addPoint({  0.091851643 ,   0.043431626 },elSize,   false);
    Point*  p1317   =   fluid2  ->  addPoint({  0.102406737 ,   0.043512005 },elSize,   false);
    Point*  p1318   =   fluid2  ->  addPoint({  0.113892361 ,   0.043384287 },elSize,   false);
    Point*  p1319   =   fluid2  ->  addPoint({  0.126390221 ,   0.043020728 },elSize,   false);
    Point*  p1320   =   fluid2  ->  addPoint({  0.13999002  ,   0.042390078 },elSize,   false);
    Point*  p1321   =   fluid2  ->  addPoint({  0.154791414 ,   0.041455375 },elSize,   false);
    Point*  p1322   =   fluid2  ->  addPoint({  0.170177278 ,   0.040240666 },elSize,   false);
    Point*  p1323   =   fluid2  ->  addPoint({  0.185496097 ,   0.038807578 },elSize,   false);
    Point*  p1324   =   fluid2  ->  addPoint({  0.200740926 ,   0.03718239  },elSize,   false);
    Point*  p1325   =   fluid2  ->  addPoint({  0.215917191 ,   0.03538228  },elSize,   false);
    Point*  p1326   =   fluid2  ->  addPoint({  0.231039985 ,   0.033425551 },elSize,   false);
    Point*  p1327   =   fluid2  ->  addPoint({  0.246096828 ,   0.031327243 },elSize,   false);
    Point*  p1328   =   fluid2  ->  addPoint({  0.261108725 ,   0.029094093 },elSize,   false);
    Point*  p1329   =   fluid2  ->  addPoint({  0.276068425 ,   0.026740875 },elSize,   false);
    Point*  p1330   =   fluid2  ->  addPoint({  0.29100042  ,   0.024272695 },elSize,   false);
    Point*  p1331   =   fluid2  ->  addPoint({  0.305886109 ,   0.021699094 },elSize,   false);
    Point*  p1332   =   fluid2  ->  addPoint({  0.320746619 ,   0.019025644 },elSize,   false);
    Point*  p1333   =   fluid2  ->  addPoint({  0.335585019 ,   0.016258018 },elSize,   false);
    Point*  p1334   =   fluid2  ->  addPoint({  0.350392204 ,   0.01340301  },elSize,   false);
    Point*  p1335   =   fluid2  ->  addPoint({  0.365189323 ,   0.010463069 },elSize,   false);
    Point*  p1336   =   fluid2  ->  addPoint({  0.379967351 ,   0.007443934 },elSize,   false);
    Point*  p1337   =   fluid2  ->  addPoint({  0.394737469 ,   0.004349792 },elSize,   false);
    Point*  p1338   =   fluid2  ->  addPoint({  0.409501613 ,   0.001183363 },elSize,   false);
    Point*  p1339   =   fluid2  ->  addPoint({  0.424270142 ,   -0.002054086    },elSize,   false);
    Point*  p1340   =   fluid2  ->  addPoint({  0.439043252 ,   -0.005359479    },elSize,   false);
    Point*  p1341   =   fluid2  ->  addPoint({  0.453822633 ,   -0.00872906 },elSize,   false);
    Point*  p1342   =   fluid2  ->  addPoint({  0.468608174 ,   -0.012161748    },elSize,   false);
    Point*  p1343   =   fluid2  ->  addPoint({  0.483410176 ,   -0.015655239    },elSize,   false);
    Point*  p1344   =   fluid2  ->  addPoint({  0.498239622 ,   -0.01921049 },elSize,   false);
    Point*  p1345   =   fluid2  ->  addPoint({  0.513087534 ,   -0.022822867    },elSize,   false);
    Point*  p1346   =   fluid2  ->  addPoint({  0.527953611 ,   -0.026491252    },elSize,   false);
    Point*  p1347   =   fluid2  ->  addPoint({  0.542857797 ,   -0.030217098    },elSize,   false);
    Point*  p1348   =   fluid2  ->  addPoint({  0.557801386 ,   -0.033999681    },elSize,   false);
    Point*  p1349   =   fluid2  ->  addPoint({  0.572774606 ,   -0.037835212    },elSize,   false);
    Point*  p1350   =   fluid2  ->  addPoint({  0.587797095 ,   -0.041727141    },elSize,   false);
    Point*  p1351   =   fluid2  ->  addPoint({  0.602860294 ,   -0.045671985    },elSize,   false);
    Point*  p1352   =   fluid2  ->  addPoint({  0.617972991 ,   -0.049672236    },elSize,   false);
    Point*  p1353   =   fluid2  ->  addPoint({  0.633145236 ,   -0.053727602    },elSize,   false);
    Point*  p1354   =   fluid2  ->  addPoint({  0.648368801 ,   -0.05783574 },elSize,   false);
    Point*  p1355   =   fluid2  ->  addPoint({  0.663653346 ,   -0.061999373    },elSize,   false);
    Point*  p1356   =   fluid2  ->  addPoint({  0.678998    ,   -0.066218245    },elSize,   false);
    Point*  p1357   =   fluid2  ->  addPoint({  0.694412731 ,   -0.070493097    },elSize,   false);
    Point*  p1358   =   fluid2  ->  addPoint({  0.70989886  ,   -0.074824283    },elSize,   false);
    Point*  p1359   =   fluid2  ->  addPoint({  0.725455826 ,   -0.079212677    },elSize,   false);
    Point*  p1360   =   fluid2  ->  addPoint({  0.741093809 ,   -0.083660117    },elSize,   false);
    Point*  p1361   =   fluid2  ->  addPoint({  0.756802357 ,   -0.088165726    },elSize,   false);
    Point*  p1362   =   fluid2  ->  addPoint({  0.772601135 ,   -0.092732961    },elSize,   false);
    Point*  p1363   =   fluid2  ->  addPoint({  0.788490706 ,   -0.097364055    },elSize,   false);
    Point*  p1364   =   fluid2  ->  addPoint({  0.804470634 ,   -0.102058881    },elSize,   false);
    Point*  p1365   =   fluid2  ->  addPoint({  0.820542384 ,   -0.106818922    },elSize,   false);
    Point*  p1366   =   fluid2  ->  addPoint({  0.836703512 ,   -0.111647596    },elSize,   false);
    Point*  p1367   =   fluid2  ->  addPoint({  0.852962914 ,   -0.116544273    },elSize,   false);
    Point*  p1368   =   fluid2  ->  addPoint({  0.868492974 ,   -0.121259662    },elSize,   false);
    Point*  p1369   =   fluid2  ->  addPoint({  0.882540264 ,   -0.125559005    },elSize,   false);
    Point*  p1370   =   fluid2  ->  addPoint({  0.8952615   ,   -0.129478126    },elSize,   false);
    Point*  p1371   =   fluid2  ->  addPoint({  0.90677304  ,   -0.133049753    },elSize,   false);
    Point*  p1372   =   fluid2  ->  addPoint({  0.917190553 ,   -0.13630016 },elSize,   false);
    Point*  p1373   =   fluid2  ->  addPoint({  0.926621867 ,   -0.139259294    },elSize,   false);
    Point*  p1374   =   fluid2  ->  addPoint({  0.935163994 ,   -0.141952053    },elSize,   false);
    Point*  p1375   =   fluid2  ->  addPoint({  0.942895533 ,   -0.144400686    },elSize,   false);
    Point*  p1376   =   fluid2  ->  addPoint({  0.949893364 ,   -0.14662658 },elSize,   false);
    Point*  p1377   =   fluid2  ->  addPoint({  0.956224705 ,   -0.148648631    },elSize,   false);
    Point*  p1378   =   fluid2  ->  addPoint({  0.96195693  ,   -0.150485758    },elSize,   false);
    Point*  p1379   =   fluid2  ->  addPoint({  0.967148985 ,   -0.152154792    },elSize,   false);
    Point*  p1380   =   fluid2  ->  addPoint({  0.971849522 ,   -0.153670354    },elSize,   false);
    Point*  p1381   =   fluid2  ->  addPoint({  0.976106756 ,   -0.155046631    },elSize,   false);
    Point*  p1382   =   fluid2  ->  addPoint({  0.979959686 ,   -0.156295533    },elSize,   false);
    Point*  p1383   =   fluid2  ->  addPoint({  0.983445287 ,   -0.157428184    },elSize,   false);
    Point*  p1384   =   fluid2  ->  addPoint({  0.986599011 ,   -0.158454798    },elSize,   false);
    Point*  p1385   =   fluid2  ->  addPoint({  0.989458672 ,   -0.159387429    },elSize,   false);


   
    Line* l1005 = fluid2 -> addLine({p1193,p1194,p1195,p1196,p1197,p1198,p1199,p1200,p1201,p1202,
                                    p1203,p1204,p1205,p1206,p1207,p1208,p1209,p1210,p1211,p1212,
                                    p1213,p1214,p1215,p1216,p1217,p1218,p1219,p1220,p1221,p1222,
                                    p1223,p1224,p1225,p1226,p1227,p1228,p1229,p1230,p1231,p1232,
                                    p1233,p1234,p1235,p1236,p1237,p1238,p1239,p1240,p1241,p1242,
                                    p1243,p1244,p1245,p1246,p1247,p1248,p1249});

    Line* l1006 = fluid2 -> addLine({p1249,p1250,p1251,p1252,p1253,p1254,p1255,p1256,p1257,p1258,
                                    p1259,p1260,p1261,p1262,p1263,p1264,p1265,p1266,p1267,p1268,
                                    p1269,p1270,p1271,p1272,p1273,p1274,p1275,p1276,p1277,p1278,
                                    p1279,p1280,p1281,p1282,p1283,p1284,p1285,p1286,p1287,p1288,
                                    p1289});
    
    Line* l1007 = fluid2 -> addLine({p1289,p1290,p1291,p1292,p1293,p1294,p1295,p1296,p1297,p1298,
                                    p1299,p1300,p1301,p1302,p1303,p1304,p1305,p1306,p1307,p1308,
                                    p1309,p1310,p1311,p1312,p1313,p1314,p1315,p1316,p1317,p1318,
                                    p1319,p1320,p1321,p1322,p1323,p1324,p1325,p1326,p1327,p1328,
                                    p1329});

    Line* l1008 = fluid2 -> addLine({p1329,p1330,p1331,p1332,p1333,p1334,p1335,p1336,p1337,p1338,
                                    p1339,p1340,p1341,p1342,p1343,p1344,p1345,p1346,p1347,p1348,
                                    p1349,p1350,p1351,p1352,p1353,p1354,p1355,p1356,p1357,p1358,
                                    p1359,p1360,p1361,p1362,p1363,p1364,p1365,p1366,p1367,p1368,
                                    p1369,p1370,p1371,p1372,p1373,p1374,p1375,p1376,p1377,p1378,
                                    p1379,p1380,p1381,p1382,p1383,p1384,p1385});


    Line* l1009 = fluid2 -> addLine({p1001,p1193});
    Line* l1010 = fluid2 -> addLine({p1057,p1249});
    Line* l1011 = fluid2 -> addLine({p1097,p1289});
    Line* l1012 = fluid2 -> addLine({p1137,p1329});
    Line* l1013 = fluid2 -> addLine({p1001,p1385});

    LineLoop* ll1001 = fluid2 -> addLineLoop({ l1009 -> operator-(), l1005 -> operator-(), l1010, l1001 });
    LineLoop* ll1002 = fluid2 -> addLineLoop({ l1010 -> operator-(), l1006 -> operator-(), l1011, l1002 });
    LineLoop* ll1003 = fluid2 -> addLineLoop({ l1011 -> operator-(), l1007 -> operator-(), l1012, l1003 });
    LineLoop* ll1004 = fluid2 -> addLineLoop({ l1012 -> operator-(), l1008 -> operator-(), l1013, l1004 });
    
    elSize = 0.6;
    double elSize2 = 0.15;

    Point* p1390 = fluid2 -> addPoint({-1.0,-1.0},elSize2,false);
    Point* p1391 = fluid2 -> addPoint({ 4.0,-1.0},elSize2,false);
    Point* p1392 = fluid2 -> addPoint({ 4.0, 1.0},elSize2,false);
    Point* p1393 = fluid2 -> addPoint({-1.0, 1.0},elSize2,false);
    Point* p1394 = fluid2 -> addPoint({-6.0,-6.0},elSize,false);
    Point* p1395 = fluid2 -> addPoint({21.0,-6.0},elSize,false);
    Point* p1396 = fluid2 -> addPoint({21.0, 6.0},elSize,false);
    Point* p1397 = fluid2 -> addPoint({-6.0, 6.0},elSize,false);
    
    Line* l1014 = fluid2 -> addLine({p1391,p1392});
    Line* l1015 = fluid2 -> addLine({p1390,p1391});
    Line* l1016 = fluid2 -> addLine({p1393,p1390});
    Line* l1017 = fluid2 -> addLine({p1392,p1393});
    Line* l1018 = fluid2 -> addLine({p1394,p1395});
    Line* l1019 = fluid2 -> addLine({p1395,p1396});
    Line* l1020 = fluid2 -> addLine({p1396,p1397});
    Line* l1021 = fluid2 -> addLine({p1397,p1394});

    LineLoop* ll1005 = fluid2 -> addLineLoop({ l1015, l1014, l1017, l1016, l1005, l1006, l1007, l1008, l1013 -> operator-(), l1009 });
    LineLoop* ll1006 = fluid2 -> addLineLoop({ l1018, l1019, l1020, l1021, l1015 -> operator-(), l1016 -> operator-(), l1017 -> operator-(), l1014 -> operator-() });

    //Transfinite lines 
    int t1 = 40; int t2 = 65; int t3 = 35; int t4 = 4;
    //corners
    fluid2 -> transfiniteLine({ l1001 }, t2, 1.02);
    fluid2 -> transfiniteLine({ l1002 -> operator-()}, t3, 1.05);
    fluid2 -> transfiniteLine({ l1003 }, t3, 1.05);
    fluid2 -> transfiniteLine({ l1004 -> operator-()}, t2, 1.02);
    fluid2 -> transfiniteLine({ l1005 }, t2, 1.02);
    fluid2 -> transfiniteLine({ l1006 -> operator-()}, t3, 1.04);
    fluid2 -> transfiniteLine({ l1007 }, t3, 1.04);
    fluid2 -> transfiniteLine({ l1008 -> operator-()}, t2, 1.02);
    fluid2 -> transfiniteLine({ l1009 }, t4, 1.25);
    fluid2 -> transfiniteLine({ l1010 }, t4, 1.25);
    fluid2 -> transfiniteLine({ l1011 }, t4, 1.25);
    fluid2 -> transfiniteLine({ l1012 }, t4, 1.05);

    fluid2 -> transfiniteLine({ l1013 }, t4, 1.25);
    
    PlaneSurface* s1001 = fluid2 -> addPlaneSurface({ll1001});
    PlaneSurface* s1002 = fluid2 -> addPlaneSurface({ll1002});
    PlaneSurface* s1003 = fluid2 -> addPlaneSurface({ll1003});
    PlaneSurface* s1004 = fluid2 -> addPlaneSurface({ll1004});
    PlaneSurface* s1005 = fluid2 -> addPlaneSurface({ll1005});
    PlaneSurface* s1006 = fluid2 -> addPlaneSurface({ll1006});


/*    fluid2 -> addBoundaryCondition("NEUMANN", l1014, {}, {}, "GLOBAL");
    fluid2 -> addBoundaryCondition("NEUMANN", l1015, {}, {}, "GLOBAL");
    fluid2 -> addBoundaryCondition("NEUMANN", l1016, {}, {}, "GLOBAL");
    fluid2 -> addBoundaryCondition("NEUMANN", l1017, {}, {}, "GLOBAL");
    fluid2 -> addBoundaryCondition("NEUMANN", l1005, {}, {}, "GLOBAL");
    fluid2 -> addBoundaryCondition("NEUMANN", l1006, {}, {}, "GLOBAL");
    fluid2 -> addBoundaryCondition("NEUMANN", l1007, {}, {}, "GLOBAL");
    fluid2 -> addBoundaryCondition("NEUMANN", l1008, {}, {}, "GLOBAL");
    fluid2 -> addBoundaryCondition("NEUMANN", l1009, {}, {}, "GLOBAL");
    fluid2 -> addBoundaryCondition("NEUMANN", l1010, {}, {}, "GLOBAL");
    fluid2 -> addBoundaryCondition("NEUMANN", l1011, {}, {}, "GLOBAL");
    fluid2 -> addBoundaryCondition("NEUMANN", l1012, {}, {}, "GLOBAL");
    fluid2 -> addBoundaryCondition("NEUMANN", l1013, {}, {}, "GLOBAL");
  */  
    
    // retangulo intermediario
    fluid2 -> addBoundaryCondition("GEOMETRY", l1014, {}, {}, "GLOBAL");
    fluid2 -> addBoundaryCondition("GEOMETRY", l1015, {}, {}, "GLOBAL");
    fluid2 -> addBoundaryCondition("GEOMETRY", l1016, {}, {}, "GLOBAL");
    fluid2 -> addBoundaryCondition("GEOMETRY", l1017, {}, {}, "GLOBAL");


     // parte externa camada limite
    fluid2 -> addBoundaryCondition("GEOMETRY", l1005, {}, {}, "GLOBAL");
    fluid2 -> addBoundaryCondition("GEOMETRY", l1006, {}, {}, "GLOBAL");
    fluid2 -> addBoundaryCondition("GEOMETRY", l1007, {}, {}, "GLOBAL");
    fluid2 -> addBoundaryCondition("GEOMETRY", l1008, {}, {}, "GLOBAL");

     // conexão da camada limite
    fluid2 -> addBoundaryCondition("GEOMETRY", l1009, {}, {}, "GLOBAL");
    fluid2 -> addBoundaryCondition("GEOMETRY", l1010, {}, {}, "GLOBAL");
    fluid2 -> addBoundaryCondition("GEOMETRY", l1011, {}, {}, "GLOBAL");
    fluid2 -> addBoundaryCondition("GEOMETRY", l1012, {}, {}, "GLOBAL");
    fluid2 -> addBoundaryCondition("GEOMETRY", l1013, {}, {}, "GLOBAL");


    // paredes globais 
    fluid2 -> addBoundaryCondition("DIRICHLET", l1018, {}, {0}, "GLOBAL");
    fluid2 -> addBoundaryCondition("NEUMANN", l1019, {}, {}, "GLOBAL");
    fluid2 -> addBoundaryCondition("DIRICHLET", l1020, {}, {0}, "GLOBAL");
    fluid2 -> addBoundaryCondition("DIRICHLET", l1021, {1}, {0}, "GLOBAL");
        
 //    fluid2 -> addBoundaryCondition("DIRICHLET", l1001, {0.0}, {0.0}, "GLOBAL");
 //   fluid2 -> addBoundaryCondition("DIRICHLET", l1002, {0}, {0.0}, "GLOBAL");
 //   fluid2 -> addBoundaryCondition("DIRICHLET", l1003, {0}, {0}, "GLOBAL");
 //   fluid2 -> addBoundaryCondition("DIRICHLET", l1004, {0}, {0}, "GLOBAL");

     // aerofolio
    fluid2 -> addBoundaryCondition("MOVING", l1001, {0.0}, {0.0}, "GLOBAL");
    fluid2 -> addBoundaryCondition("MOVING", l1002, {0}, {0.0}, "GLOBAL");
    fluid2 -> addBoundaryCondition("MOVING", l1003, {0}, {0}, "GLOBAL");
    fluid2 -> addBoundaryCondition("MOVING", l1004, {0}, {0}, "GLOBAL");


    fluid2 -> transfiniteSurface({s1001}, "Right", {});
    fluid2 -> transfiniteSurface({s1002}, "Right", {});
    fluid2 -> transfiniteSurface({s1003}, "Right", {});
    fluid2 -> transfiniteSurface({s1004}, "Right", {});

    if (rank == 0){
  
        FluidDomain* problem = new FluidDomain(fluid2);
        problem -> generateMesh("T6", "DELAUNAY", "cylinder", "", false, true);
    };

//================================================================================
//=================================PROBLEM MESH===================================
//================================================================================

    MPI_Barrier(PETSC_COMM_WORLD);

    control.dataReading(fluid2,"cylinder_data.txt","cylinder.msh","mirror.txt",0);

    //Data reading     
    //control.dataReading("control.txt","mirror_control.txt");
            
    //solveTransientProblem function needs three parameters:  
    //1- The maximum number of iterations in the Newton-Raphson process
    //2- The maximum relative error in the Newton-Raphson process (DU)  
    
    // control.solveTransientProblem(3, 1.e-6); 
    control.solveTransientProblem(3,1.e-6);
    //control.solveTransientProblemMoving(3,1.e-6);
     
    //Finalize main program   
    PetscFinalize();
 
    return 0; 
}
 
 
  




 
