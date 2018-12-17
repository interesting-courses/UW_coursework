#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <stdlib.h>     /* srand, rand */
#include <algorithm>    // std::max


using namespace std;

int main() {

ifstream inFile("b3.in");

int size_E = 500000;
int size_V = 10000;

int16_t** E = new int16_t*[size_E];
for(int i = 0; i < size_E; ++i)
    E[i] = new int16_t[2];

int k = 0;
while (inFile >> E[k][0] >> E[k][1] ){
//    cout << k << ":" << E[k][0] << " " << E[k][1] << endl;
    k += 1;
}
inFile.close();


int16_t** G = new int16_t*[size_V];
for(int i = 0; i < size_V; ++i)
    G[i] = new int16_t[size_V];

for(int i=0;i<size_V;++i){
    for(int j=i;j<size_V;++j){
        G[i][j] = 0;
    }
}

bool mask[size_V];
for(int i=0;i<size_V;++i){
    mask[i] = true;
}

for(k=0;k<size_E;++k){
    int i = E[k][0];
    int j = E[k][1];
    G[min(i,j)][max(i,j)] += 1;
}


int rand_index;
for(int k=0;k<size_V-2;++k){
    if(k%10==0){
        cout << "iteration: "<< k << "\n";
        cout << size_E << "\n";
    }

    int a = 0;
    
    rand_index = rand() % size_E;

    int ii,jj;
    int i=0;
    while(i<size_V){
        if(mask[i]){
            int j=i;
            while(j<size_V){
                if(mask[j]){
                    rand_index -= G[i][j];
                    if(rand_index <= 0){
                        ii = min(i,j);
                        jj = max(i,j);
                        goto end_rand_search;
                    }
                }
                ++j;
            }
        }
        ++i;
    }
    cout << "error";
    for(int i=0;i<size_V;++i){
        for(int j=i;j<size_V;++j){
            if(mask[i] && mask[j]){
                a += G[i][j];
            }
        }
    }
    cout << a;
    return 0;

    end_rand_search:
//    cout << "exit random search " << ii << "," << jj << " G: " << G[ii][jj] << " \n";

    mask[jj] = false;

    size_E -= G[ii][jj];

    G[ii][jj] = 0;
    
    for(int j=jj;j<size_V;++j){
        G[ii][j] += G[jj][j];
    }
    for(int j=ii;j<jj;++j){
        G[ii][j] += G[j][jj];
    }
    for(int j=0;j<ii;++j){
        G[j][ii] += G[j][jj];
    }
}

cout << "size_cut: " << size_E;

return 0;
}
