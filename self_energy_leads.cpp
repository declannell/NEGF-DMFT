#include <iostream>
#include <vector>
using namespace std;

class hamiltonian
{
private:
    double onsite, hopping;
    int chain_length;

public:
    vector<vector<double>> matrix;
    
    hamiltonian(double onsite1, double hopping1, int chain_length1) : onsite(onsite1), hopping(hopping1), chain_length(chain_length1)
    {

        for (int i = 0; i < chain_length; i++)
        {
            vector<double> row(chain_length);
            matrix.push_back(row);
        }

        for (int i = 0; i < chain_length; i++)

        {
            matrix.at(i).at(i) = onsite;
        }
        for (int i = 0; i < chain_length - 1; i++)
        {
            matrix[i + 1][i] = hopping;
            matrix[i][i + 1] = hopping;
        }
    }
    double onsite_energy() const { return onsite; }
    double hopping_energy() const { return hopping; }
    double length_of_chain() const { return chain_length; }
};

//void function(vector<vector<vector<double>>> &green_function )
                                                                
int main()
{
    //mat	 = 	Mat<double>
    int number_of_atoms = 7;
    double onsite_energy = 1.0, hopping = -1.0;
    //vector<vector<vector<double>>> green_function();//parathensis initialise the array
    hamiltonian chain(onsite_energy, hopping, number_of_atoms);
    for (int i = 0; i < number_of_atoms; i++)
    {
        for (int j = 0; j < number_of_atoms; j++)
        {
            cout << chain.matrix[i][j] << " ";
        }
        cout << "\n";
    }
}