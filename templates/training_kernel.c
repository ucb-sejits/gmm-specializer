#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>



#define PI  3.1415926535897931
#define COVARIANCE_DYNAMIC_RANGE 1E6
#define MINVALUEFORMINUSLOG -1000.0

typedef struct components_struct {
                float* N;        // expected # of pixels in component: [M]
                float* pi;       // probability of component in GMM: [M]
                float* CP; //cluster probability [M]
                float* constant; // Normalizing constant [M]
                float* avgvar;    // average variance [M]
                float* means;   // Spectral mean for the component: [M*D]
                float* R;      // Covariance matrix: [M*D*D]
                float* Rinv;   // Inverse of covariance matrix: [M*D*D]
            } components_t;



/*
 * Inverts a square matrix (stored as a 1D float array)
 * 
 * actualsize - the dimension of the matrix
 *
 * written by Mike Dinolfo 12/98
 * version 1.0
 */
void invert_matrix(float* data, int actualsize, float* log_determinant)  {
    int maxsize = actualsize;
    int n = actualsize;
    *log_determinant = 0.0;
    
    if (actualsize == 1) { // special case, dimensionality == 1
        *log_determinant = logf(data[0]);
        data[0] = 1.0 / data[0];
    } else if(actualsize >= 2) { // dimensionality >= 2
        for (int i=1; i < actualsize; i++) { data[i] /= data[0]; } // normalize row 0
        for (int i=1; i < actualsize; i++) { 
            for (int j=i; j < actualsize; j++) { // do a column of L
                float sum = 0.0;
                for (int k = 0; k < i; k++)  
                  sum += data[j*maxsize+k] * data[k*maxsize+i];
                data[j*maxsize+i] -= sum;
            }
            if (i == actualsize-1) continue;
            for (int j=i+1; j < actualsize; j++)  {  // do a row of U
                float sum = 0.0;
                for (int k = 0; k < i; k++)
                    sum += data[i*maxsize+k]*data[k*maxsize+j];
                data[i*maxsize+j] = 
                  (data[i*maxsize+j]-sum) / data[i*maxsize+i];
            }
        }
      
        for(int i=0; i<actualsize; i++) {
            *log_determinant += logf(fabs(data[i*n+i]));
        }
        for ( int i = 0; i < actualsize; i++ )  // invert L
            for ( int j = i; j < actualsize; j++ )  {
                float x = 1.0;
                if ( i != j ) {
                    x = 0.0;
                    for ( int k = i; k < j; k++ ) 
                        x -= data[j*maxsize+k]*data[k*maxsize+i];
                }
                data[j*maxsize+i] = x / data[j*maxsize+j];
            }
        for ( int i = 0; i < actualsize; i++ )   // invert U
            for ( int j = i; j < actualsize; j++ )  {
                if ( i == j ) continue;
                float sum = 0.0;
                for ( int k = i; k < j; k++ )
                    sum += data[k*maxsize+j]*( (i==k) ? 1.0 : data[i*maxsize+k] );
                data[i*maxsize+j] = -sum;
            }
        for ( int i = 0; i < actualsize; i++ )   // final inversion
            for ( int j = 0; j < actualsize; j++ )  {
                float sum = 0.0;
                for ( int k = ((i>j)?i:j); k < actualsize; k++ )  
                    sum += ((j==k)?1.0:data[j*maxsize+k])*data[k*maxsize+i];
                data[j*maxsize+i] = sum;
            }
    } else {
        printf("Error: Invalid dimensionality for invert(...)\n");
    }
}

void normalize_pi(components_t* components, int num_components) {
    float total = 0;
    for(int i=0; i < num_components; i++) {
         total += components->pi[i];
    }
    
    for(int m=0; m < num_components; m++){
         components->pi[m] /= total; 
    }
}


float log_add(float log_a, float log_b) {
    if(log_a < log_b) {
        float tmp = log_a;
        log_a = log_b;
        log_b = tmp;
    }
    return (((log_b - log_a) <= MINVALUEFORMINUSLOG) ? log_a : log_a + (float)(logf(1.0 + (double)(expf((double)(log_b - log_a))))));
}


void constants (components_t* components,int M, int D) {
    float log_determinant;
    float* matrix = (float*)malloc(sizeof(float) * D * D);

    for(int m = 0; m < M; m++) {
        // Invert covariance matrix
        memcpy(matrix,&(components->R[m*D*D]),sizeof(float) * D * D);
        invert_matrix(matrix,D,&log_determinant);
        memcpy(&(components->Rinv[m*D*D]),matrix,sizeof(float) * D * D);
    
        // Compute constant
        components->constant[m] = -D * 0.5f * logf(2 * PI) - 0.5f * log_determinant;
        components->CP[m] = components->constant[m] * 2.0;
    }
    normalize_pi(components, M);
    free(matrix);
}

void mvtmeans(float* data_by_event, int num_dimensions,int num_events, float* means) {
    for(int d=0; d < num_dimensions; d++) {
         means[d] = 0.0;
         for(int n=0; n < num_events; n++) {
             means[d] += data_by_event[n * num_dimensions + d];
         }
         means[d] /= (float) num_events;
    }
}


void seed_covars(components_t* components,float* data_by_event,float* means,int num_dimensions,int num_events,float* avgvar,int num_components) {

    for(int i = 0; i < num_dimensions*num_dimensions; i++) {
      int row = (i) / num_dimensions;
      int col = (i) % num_dimensions;


      components->R[row*num_dimensions+col] = 0.0f;

      for(int j=0; j < num_events; j++) {
        if(row==col) {
          components->R[row*num_dimensions+col] +=(data_by_event[j*num_dimensions + row])*(data_by_event[j*num_dimensions + row]);
        }
      }

      
      if(row==col) {
        components->R[row*num_dimensions+col] /=(float) (num_events -1);
        components->R[row*num_dimensions+col] -=((float)(num_events)*means[row]*means[row]) / (float)(num_events-1);
        components->R[row*num_dimensions+col] /=(float)num_components;
      }
    }
}

void average_variance(float* data_by_event,float* means,int num_dimensions, int num_events,float* avgvar) {

    float total = 0.0f;
    // Compute average variance for each dimension
    for(int i = 0; i < num_dimensions; i++) {
        float variance = 0.0f;
        for(int j=0; j < num_events; j++) {
            variance += data_by_event[j*num_dimensions + i] * data_by_event[j*num_dimensions + i];
        }
        variance /= (float) num_events;
        variance -= means[i] * means[i];
        total += variance;
    }
    *avgvar = total/ (float) num_dimensions;
}


void seed_components(float *data_by_event,components_t* components,int num_dimensions,int num_components,int num_events) {



    float* means = (float*)malloc(sizeof(float) * num_dimensions);
    float avgvar;


    
    // Compute means
    mvtmeans(data_by_event, num_dimensions, num_events, means);
    // Compute the average variance
    seed_covars(components,data_by_event,means,num_dimensions,num_events,&avgvar,num_components);
    average_variance(data_by_event,means,num_dimensions,num_events,&avgvar);
    float seed;
    if(num_components > 1) {
       seed = (num_events)/(num_components);
    } else {
       seed = 0.0f;
    }

    memcpy(components->means, means, sizeof(float) * num_dimensions);



    for(int c = 1; c < num_components; c++) {
        memcpy(&components->means[c*num_dimensions],&data_by_event[((int)(c * seed)) * num_dimensions],sizeof(float) * num_dimensions);
        for(int i = 0; i < num_dimensions * num_dimensions; i++) {
          components->R[c * num_dimensions * num_dimensions+i] = components->R[i];
          components->Rinv[c * num_dimensions * num_dimensions+i] = 0.0f;

        }
        
    }
    //compute pi, N
    for(int c =0; c<num_components; c++) {
        components->pi[c] = 1.0f / ((float)num_components);
        components->N[c] = ((float) num_events) / ((float)num_components);
        components->avgvar[c] = avgvar / COVARIANCE_DYNAMIC_RANGE;
    }

    free(means);

    printf("safely quit seed_components");
}

void compute_average_variance (float* data_by_event,components_t* components,int num_dimensions,int num_components,int num_events) {

    float* means = (float*)malloc(sizeof(float) * num_dimensions);
    float avgvar;
    
    // Compute the means
    mvtmeans(data_by_event, num_dimensions, num_events, means);
   
    average_variance(data_by_event,means,num_dimensions,num_events,&avgvar);
    
    for(int c = 0; c < num_components; c++) {
        components->avgvar[c] = avgvar / COVARIANCE_DYNAMIC_RANGE;
    }
}

void estep1 (float* data,components_t* components,float* component_memberships,int D, int M, int N,float* loglikelihoods, char* cvtype) {
    // Compute likelihood for every data point in each component
    float* temploglikelihoods = (float*)malloc(M * N * sizeof(float));
    char diag[5];
    strcpy(diag, "diag");
    for(int m = 0; m < M; m++) {
        float component_pi = components->pi[m];
        float component_constant = components->constant[m];
        float* means = &(components->means[m*D]);
        float* Rinv = &(components->Rinv[m*D*D]);
        for(int n=0; n < N; n++) {
            float like = 0.0;
            if (strcmp(cvtype,diag) == 0){
              for(int i = 0; i < D; i++) {
                like += (data[i*N+n] - means[i]) *
                    (data[i*N+n] - means[i]) * Rinv[i*D+i];
            }
            }
            else{
              for(int i = 0; i < D; i++) {
                for(int j = 0; j < D; j++) {
                    like += (data[i*N+n] - means[i]) *
                        (data[j*N+n] - means[j]) * Rinv[i*D+j];
                }
            }
            }
            component_memberships[m*N+n] = (component_pi > 0.0f) ? -0.5 * like + component_constant + logf(component_pi) :MINVALUEFORMINUSLOG;
        }
    }

    //estep1 log_add()
    for(int n = 0; n < N; n++) {
        float finalloglike = MINVALUEFORMINUSLOG;
        for(int m = 0; m < M; m++) {
            finalloglike = log_add(finalloglike, component_memberships[m*N+n]);
        }
        loglikelihoods[n] = finalloglike;
    }
}

float estep2_events (components_t* components,float* component_memberships,int M, int n, int N) {
  // Finding maximum likelihood for this data point
  float temp = 0.0f;
  float thread_likelihood = 0.0f;
  float max_likelihood = -10000;
  float denominator_sum = 0.0f;

  for (int i = n; i < M*N; i += N){
    if (component_memberships[i]>max_likelihood){
      max_likelihood = component_memberships[i];
    }

  }

  // Computes sum of all likelihoods for this event
  for(int m = 0; m < M; m++) {
        temp = expf(component_memberships[m*N+n] - max_likelihood);
        denominator_sum += temp;
  }
  temp = max_likelihood + logf(denominator_sum);
    thread_likelihood += temp;

  // Divide by denominator to get each membership
  for(int m = 0; m < M; m++) {
      component_memberships[m*N+n] = expf(component_memberships[m*N+n] - temp);
  }
  return thread_likelihood;
}

void estep2 (float* data,components_t* components,float* component_memberships,int D, int M, int N,float* likelihood) {

    float total = 0.0f;
    for(int n=0; n < N; n++) {
        total += estep2_events(components,component_memberships,M, n, N);
    }
    *likelihood = total;
}

void mstep_mean (float* data,components_t* components,float* component_memberships,int D, int M, int N) {
    for(int m = 0; m < M; m++) {
        for(int d = 0; d < D; d++) {
          components->means[m*D+d] = 0.0;
          for(int n = 0; n < N; n++) {
            components->means[m*D+d] += data[d*N+n]*component_memberships[m*N+n];
          }
          components->means[m*D+d] /= components->N[m];
        }
    }
}


void mstep_n  (float* data,components_t* components,float* component_memberships,int D, int M, int N) {
    for(int m = 0; m < M; m++) {
      components->N[m] = 0.0;
      for(int n = 0; n < N; n++) {
        components->N[m] += component_memberships[m*N+n];
      }
      components->pi[m] = components->N[m];
    }
}


void mstep_covar(float* data,components_t* components,float* component_memberships,int D, int M, int N, char* cvtype) {
    char diag[5];
    strcpy(diag, "diag");
    for(int m = 0; m < M; m++) {
        float* means = &(components->means[m*D]);
        for(int i = 0; i < D; i++) {
            for(int j = 0; j <= i; j++) {
              float sum = 0.0f;

              if (strcmp(cvtype,diag) == 0){
                if(i != j) {
                    components->R[m*D*D+i*D+j] = 0.0f;
                    components->R[m*D*D+j*D+i] = 0.0f;
                    continue;
                }
              }
              
              
              for(int n = 0; n < N; n++) {
                    sum += (data[i*N+n] - means[i]) *(data[j*N+n]-means[j]) * component_memberships[m*N+n];
              }

              if(components->N[m] >= 1.0f) {
                    components->R[m*D*D+i*D+j] = sum / components->N[m];
                    components->R[m*D*D+j*D+i] = sum / components->N[m];
              } else {
                    components->R[m*D*D+i*D+j] = 0.0f;
                    components->R[m*D*D+j*D+i] = 0.0f;
              }
              if(i == j) {
                    components->R[m*D*D+j*D+i] += components->avgvar[m];
              }
            }
        }
    }
}

void em_train(float *input_data, float *component_memberships, float *loglikelihoods,int num_components, int num_dimensions,int num_events,int min_iters,int max_iters, char* cvtype, float *ret_likelihood) {
    float* N =  (float*)malloc(sizeof(float) * num_components);  // expected # of pixels in component: [M]
    float* pi = (float*)malloc(sizeof(float) * num_components);       // probability of component in GMM: [M]
    float* CP = (float*)malloc(sizeof(float) * num_components); //cluster probability [M]
    float* constant = (float*)malloc(sizeof(float) * num_components); // Normalizing constant [M]
    float* avgvar = (float*)malloc(sizeof(float) * num_components);    // average variance [M]
    float* means = (float*)malloc(sizeof(float) * num_components * num_dimensions);   // Spectral mean for the component: [M*D]
    float* R = (float*)malloc(sizeof(float) * num_components * num_dimensions * num_dimensions);      // Covariance matrix: [M*D*D]
    float* Rinv = (float*)malloc(sizeof(float) * num_components * num_dimensions * num_dimensions);   //


    float* data_by_dimension;
    components_t components;

    components.N = N;
    components.pi = pi;
    components.CP = CP;
    components.constant = constant;
    components.avgvar = avgvar;
    components.means = means;
    components.R = R;
    components.Rinv = Rinv;
    
    data_by_dimension  = (float*)malloc(sizeof(float) * num_events * num_dimensions);
    
    for(int e = 0; e < num_events; e++) {
        for(int d = 0; d < num_dimensions; d++) {
            data_by_dimension[d * num_events + e] = input_data[e * num_dimensions + d];
        }
    }

    seed_components(input_data,&components,num_dimensions,num_components,num_events);

    // Computes the R matrix inverses, and the gaussian constant
    constants (&components,num_components,num_dimensions);
    // Compute average variance based on the data
    compute_average_variance(input_data,&components,num_dimensions,num_components,num_events);
    // Calculate an epsilon value
    //int ndata_points = num_events*num_dimensions;
    float epsilon = (1 + num_dimensions + 0.5 * (num_dimensions + 1) * num_dimensions) *log((float)num_events * num_dimensions) * 0.0001;

    printf ("%f\n",epsilon);
    int iters;
    float likelihood = -100000;
    float old_likelihood = likelihood * 10;
    
    float change = epsilon*2;
  
    iters = 0;

    // This is the iterative loop for the EM algorithm.
    // It re-estimates parameters, re-computes constants, and then regroups the events
    // These steps keep repeating until the change in likelihood is less than some epsilon        
    // while(iters < min_iters || (fabs(change) > epsilon && iters < max_iters)) {
    while(iters < min_iters || (iters < max_iters && change > epsilon)) {
        printf("loop");
        printf("%d\n",iters);
        //printf("Training iteration: %u\n", iters);
        old_likelihood = likelihood;

        estep1(data_by_dimension,&components, component_memberships,num_dimensions,num_components,num_events,loglikelihoods,cvtype);
        printf("estep1\n");
        estep2(data_by_dimension,&components,component_memberships,num_dimensions,num_components,num_events, &likelihood);
        printf("estep2\n");
        //printf("Likelihood: %g\n", likelihood);
        
        // This kernel computes a new N, pi isn't updated until compute_constants though
        mstep_n(data_by_dimension,&components,component_memberships,num_dimensions,num_components,num_events);
        printf("mstep_n\n");
        mstep_mean(data_by_dimension,&components,component_memberships, num_dimensions, num_components,num_events);
        printf("mstep_mean\n");
        mstep_covar(data_by_dimension,&components,component_memberships,num_dimensions,num_components,num_events,cvtype);
        printf("mstep_covar\n");
        
        // Inverts the R matrices, computes the constant, normalizes cluster probabilities
        constants(&components,num_components,num_dimensions);
        printf("constants");
        change = likelihood - old_likelihood;
        printf("%f\n",change);
        iters++;
    }

    printf("%f\n", likelihood);
    estep1(data_by_dimension,&components,component_memberships,num_dimensions,num_components,num_events,loglikelihoods,cvtype);
    estep2(data_by_dimension,&components,component_memberships,num_dimensions,num_components,num_events,&likelihood);
      
    *ret_likelihood = likelihood;
    free(data_by_dimension);
}
