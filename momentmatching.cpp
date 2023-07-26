#include "momentmatching.h"

void momentmatching:: initialize_prior()
{   prior_params.a = input_aRef;
    prior_params.alphas = input_dataSetRef.col(0);
    prior_params.kappas.setOnes();
    
    // Tuning parameter which determines the sigma for tau sampled from gamma distribution. If input sigma is very high >0.14 increase sigma_tau to avoid truedivide error
    double sigma_tau = 40.;
    prior_params.betas = (1/(sigma_tau* input_dataSetRef.col(1).array().square()* prior_params.kappas)).square();
    prior_params.gammas = prior_params.betas.sqrt()/sigma_tau;
    // cout<<prior_params.betas<<endl;
   
}

void momentmatching:: analytical_posterior(double &meas)
{  
    post_params.a = prior_params.a +1;
    post_params.alphas = (prior_params.kappas*prior_params.alphas + meas)/ (prior_params.kappas +1);
    post_params.kappas = prior_params.kappas +1;
    post_params.betas = prior_params.betas +0.5;
    post_params.gammas = prior_params.gammas + prior_params.kappas  *  (prior_params.alphas - meas).square() / (2* (prior_params.kappas +1));
    c_array = (prior_params.kappas / post_params.kappas).sqrt()  * 
                (post_params.betas.lgamma().exp()/prior_params.betas.lgamma().exp()) * 
                 (prior_params.gammas .pow(prior_params.betas) / post_params.gammas .pow(post_params.betas)) ; 
}
void  momentmatching::swap_elementInParams(int i)
{   temp_params = prior_params;
    temp_params.a[i] = post_params.a[i];
    temp_params.alphas[i] = post_params.alphas[i];
    temp_params.kappas[i] = post_params.kappas[i];
    temp_params.betas[i] = post_params.betas[i];
    temp_params.gammas[i] = post_params.gammas[i];
}

void  momentmatching::evaluate_moments()
{
    c_array = (c_array/ c_array.sum());

    for (int i =0; i< num_classes; i++)
    {   
        swap_elementInParams(i);
        // print_params( temp_params, "temp" + to_string(i));

        exp_weigh_full.row(i) = temp_params.a/temp_params.a.sum();
        exp_weigh_sq_full.row(i) = temp_params.a * (temp_params.a +1) / ((temp_params.a.sum()) * (temp_params.a.sum() +1));
        exp_lambda_full.row(i) = temp_params.betas / temp_params.gammas;
        exp_lambda_sq_full.row(i) = temp_params.betas * (temp_params.betas +1) / temp_params.gammas.square();

        exp_mu_full.row(i) = temp_params.alphas;
        exp_mu_lamda_sq_full.row(i) =  1/temp_params.kappas + (temp_params.alphas.square())   *  temp_params.betas/temp_params.gammas;

    };

    exp_weigh = (c_array.matrix().transpose()*exp_weigh_full).transpose();
    exp_weigh_sq = (c_array.matrix().transpose()*exp_weigh_sq_full).transpose();
    exp_lambda = (c_array.matrix().transpose()*exp_lambda_full).transpose();
    exp_lambda_sq = (c_array.matrix().transpose()*exp_lambda_sq_full).transpose();
    exp_mu = (c_array.matrix().transpose()*exp_mu_full).transpose();
    exp_mu_lamda_sq = (c_array.matrix().transpose()*exp_mu_lamda_sq_full).transpose();
    
    next_params.a = (exp_weigh)*((exp_weigh- exp_weigh_sq)/(exp_weigh_sq - exp_weigh.square()));
    next_params.alphas = exp_mu;
    next_params.betas = (exp_lambda.square())/(exp_lambda_sq- (exp_lambda.square()));
    next_params.gammas = (exp_lambda)/(exp_lambda_sq- (exp_lambda.square()));
    next_params.kappas =  1/(exp_mu_lamda_sq- (next_params.alphas.square())*exp_lambda);
    // print_params(next_params, "next");

}   

void  momentmatching::update_aAndDataset()
{
    input_aRef = next_params.a;
    //mu
    input_dataSetRef.col(0) = next_params.alphas;
    // sigma 
    input_dataSetRef.col(1) = 1/(next_params.kappas*next_params.betas / next_params.gammas).sqrt();

}

int main(int argc, char** argv)
{   Matrix<double , 7, 1> input_a ;
    input_a << 5.,5.,5.,5.,5.,5.,5.;
    Matrix<double , 7, 2>input_dataSet ;
    input_dataSet<< 0.543, 0.065, //Concrete
                    0.577, 0.077, //Grass
                    0.428, 0.059, //Pebbles
                    0.478, 0.113, //Rocks
                    0.372, 0.055, //Wood
                    0.616, 0.048, //Rubber
                    0.583, 0.068; //Rug
    vector<double> measurements = {0.62,0.62,0.62,0.62,0.62,0.62,0.62,0.62,0.62,0.62};

    momentmatching(input_dataSet, input_a,measurements);
    cout<<input_dataSet<<endl;
}