#include <iostream>
#include <vector>
#include <complex> //this contains complex numbers and trig functions
#include <fstream>
using namespace std;
typedef complex<double> dcomp;
//void function(vector<vector<vector<double>>> &green_function )

class SelfEnergy
{
private:
    double onsite_lead, hopping_lead;
    int principal_layers, steps1;

public:
    vector<dcomp> self_energy;
    vector<dcomp> surface_gf; //can I define the size here???
    vector<dcomp> transfer_matrix;
    vector<dcomp> old_t;
    vector<dcomp> next_t;
    vector<dcomp> energy;

    SelfEnergy(double onsite1, double hopping1, int principle_layers1, int steps) : onsite_lead(onsite1), hopping_lead(hopping1), principal_layers(principle_layers1), steps1(steps)
    {
        for (int i = 0; i < steps; i++)
        {
            self_energy.push_back(0);
            surface_gf.push_back(0);
            transfer_matrix.push_back(0);
            old_t.push_back(0);
            next_t.push_back(0);
        }
    }

    double onsite() const { return onsite_lead; }
    double hopping() const { return hopping_lead; }
    int steps() const { return steps1; }
};

void get_transfer(SelfEnergy &u, vector<dcomp> &energy)
{
    
    int steps = u.steps();
    vector<dcomp> t_n(steps);
    vector<dcomp> t_prod(steps);
    vector<dcomp> next_t(steps);
    for (int i = 0; i < steps; i++)
    {
        
        t_n[i] = operator/( u.hopping() , ( energy[i] - u.onsite() )  );
        t_prod[i] = t_n[i];
        u.transfer_matrix[i] = t_prod[i];
    }
                  
                   ofstream myfile;
    myfile.open("C:\\Users\\user\\.spyder-py3\\printing code\\self_energy_lead_real.txt");
    for (int i = 0; i < steps; i++)
    {
        myfile << energy[i].real() << " " << u.next_t[i].real() << "\n";
    }

    myfile.close();

    myfile.open("C:\\Users\\user\\.spyder-py3\\printing code\\self_energy_lead_imag.txt");
    for (int i = 0; i < steps; i++)
    {
        myfile << energy[i].real() << " " << u.next_t[i].imag() << "\n";
    }

    myfile.close();

    vector<double> differencelist(2 * steps);
    vector<dcomp> old_transfer(steps);
    double difference = 1.0;
    int count=0;
    do
    {
        difference=0;
        for (int i = 0; i < steps; i++)
        {
            t_n[i] = operator/( t_n[i] * t_n[i] , ( 1.0 - 2.0 * t_n[i] * t_n[i] ) );
            t_prod[i] = t_n[i] * t_prod[i];
            u.transfer_matrix[i] = u.transfer_matrix[i]+t_prod[i];
        }
        
        for (int i = 0; i < steps; i++)
        {
            differencelist[i] = abs(u.transfer_matrix[i].real() - old_transfer[i].real());
            differencelist[steps + i] = abs(u.transfer_matrix[i].imag() - old_transfer[i].imag());
            old_transfer[i] = u.transfer_matrix[i];
        }

        for (int i = 0; i < 2 * steps; i++)
        {            
            //cout<<differencelist[i]<< " ";
            if (difference < differencelist[i])
            {
                difference = differencelist[i];
            }
            //cout<<differencelist[i] << " ";
        }
        
        count++;
        
        cout<< "The difference is " << difference <<endl;
    }
    while(difference>0.01);
}

void get_self_energy(SelfEnergy &u, vector<dcomp> &energy)
{
    int steps = u.steps();
    for (int i = 0; i < steps; i++)
    {
        u.surface_gf[i] = operator/(1.0, (  energy[i] - u.onsite() - u.hopping() * u.transfer_matrix[i] )  );
        u.self_energy[i] = u.hopping() * u.hopping() * u.surface_gf[i]; //this assumes the coupling in the leads is the same as the coupling to the scattering region
    }
}

int main()
{

    dcomp j1;
    j1 = -1;
    j1 = sqrt(j1);
    /*
    dcomp a, b;
    double pi;
    pi = 2 * asin(1);

    a = exp(2 * pi * j1);
    b = 5;
    
    cout << "i is " << j1 << "and Euler was right: e(i pi) = " << a << endl;
    dcomp z = operator+(b , j1);
    cout<< "b = " << b << endl;

    cout<< operator/( 1.0 , z ) << endl;
    */
    int steps = 1005, principle_layers = 1;
    double onsite = 0.0, hopping = -1.0;
    double e_upper_bound = 5.0, e_lower_bound = -5.0;
    vector<dcomp> energy(steps);
    vector<dcomp> energy2(steps);

    for (int i = 0; i < steps; i++)
    {
        energy[i] = e_lower_bound + (e_upper_bound - e_lower_bound) / steps * i +0.0001 + 0.0000001* j1;
        //cout<< energy[i]<<endl;
    }

    SelfEnergy left_lead(onsite, hopping, principle_layers, steps);
    get_transfer(left_lead, energy);
    get_self_energy(left_lead, energy);

                   ofstream myfile;
    myfile.open("C:\\Users\\user\\.spyder-py3\\printing code\\self_energy_lead_real.txt");
    for (int i = 0; i < steps; i++)
    {
        myfile << energy[i].real() << " " << left_lead.self_energy[i].real() << "\n";
    }

    myfile.close();

    myfile.open("C:\\Users\\user\\.spyder-py3\\printing code\\self_energy_lead_imag.txt");
    for (int i = 0; i < steps; i++)
    {
        myfile << energy[i].real() << " " << left_lead.self_energy[i].imag() << "\n";
    }

    myfile.close();
    
}
