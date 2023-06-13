#include <iostream>
#include<fstream>
#include <cmath>
#include <vector>
#include <array>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues> 
#include <eigen3/unsupported/Eigen/SpecialFunctions>


using namespace std;
using namespace Eigen;


class momentmatching 
{
public:
    static constexpr int num_classes = 5; 
    // param struct
    struct Params {
        Array<double, num_classes,1> a;
        Array<double, num_classes,1> alphas;
        Array<double, num_classes,1> kappas;
        Array<double, num_classes,1> betas;
        Array<double, num_classes,1> gammas;
    }post_params, prior_params, temp_params;

    Array<double , num_classes, 1> c_array;

    Params & next_params = prior_params;
    
    Matrix<double , num_classes, 2> & input_dataSetRef;
    Matrix<double , num_classes, 1> & input_aRef;

    vector<double> measurements = {4.99164777 ,4.77152141 ,3.91641394, 4.55800433 ,3.8606777 , 3.74498796,
    5.72879082 ,5.7962263 , 5.07160451 ,4.90461636};

    Matrix<double, num_classes, num_classes > exp_weigh_full;  
    Matrix<double, num_classes, num_classes >  exp_weigh_sq_full; 
    Matrix<double, num_classes, num_classes >  exp_mu_full ;
    Matrix<double, num_classes, num_classes >  exp_mu_lamda_sq_full; 
    Matrix<double, num_classes, num_classes > exp_lambda_full ;
    Matrix<double, num_classes, num_classes > exp_lambda_sq_full;

    Array<double, num_classes,1> exp_weigh;  
    Array<double, num_classes,1>  exp_weigh_sq; 
    Array<double, num_classes,1>  exp_mu ;
    Array<double, num_classes,1>  exp_mu_lamda_sq; 
    Array<double, num_classes,1> exp_lambda ;
    Array<double, num_classes,1> exp_lambda_sq;

    // decide later
    void initialize_prior();

    void analytical_posterior(double &meaurement);
    
    void swap_elementInParams(int i);

    void evaluate_moments();

    void update_aAndDataset();

    void print_params(Params param, string name)
    {
        cout<<param.a<< " "<< name <<" final a"<<endl;
        cout<<param.alphas<< " "<< name <<" final alphas"<<endl;
        cout<<param.kappas<< " "<< name <<" final kappas"<<endl;
        cout<<param.betas<< " "<< name <<" final betas"<<endl;
        cout<<param.gammas<< " "<< name <<" final gammas"<<endl;
    };

    momentmatching(Matrix<double , num_classes, 2> & input_dataSet, Matrix<double , num_classes, 1> & input_a): input_dataSetRef{input_dataSet}, input_aRef{input_a}
    {
        initialize_prior();

        for(double meas :measurements)
        {   
            analytical_posterior(meas);
            evaluate_moments();

        };
        update_aAndDataset();
        print_params(next_params, "next");
    };

};