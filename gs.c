#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

/*** Skeleton for Lab 1 ***/

/***** Globals ******/
float **a; /* The coefficients */
float *x;  /* The unknowns */
float *b;  /* The constants */
float err; /* The absolute relative error */
int num = 0;  /* number of unknowns */


/****** Function declarations */
void check_matrix(); /* Check whether the matrix will converge */
void get_input();  /* Read input from file */
float calc_unknown(int index);
int within_error(float newVal, float oldVal);

/********************************/



/* Function definitions: functions are ordered alphabetically ****/
/*****************************************************************/

/*
    
 */
float calc_unknown(int unknown) {
  
  int index = unknown - 1;
  float solution = b[index];

  for(int j = 0; j < num; j++) {
    if(j == index) {
      continue;
    }
    solution -= (a[index][j] * x[j]);
  }

  solution = solution / a[index][index];
  return solution;
}

/* 
   Conditions for convergence (diagonal dominance):
   1. diagonal element >= sum of all other elements of the row
   2. At least one diagonal element > sum of all other elements of the row
 */
void check_matrix() {
  int bigger = 0; /* Set to 1 if at least one diag element > sum  */
  int i, j;
  float sum = 0;
  float aii = 0;
  
  for(i = 0; i < num; i++) {
    sum = 0;
    aii = fabs(a[i][i]);
    
    for(j = 0; j < num; j++) {
      if( j != i) {
        sum += fabs(a[i][j]);
      }
    }

    if( aii < sum) {
      printf("The matrix will not converge.\n");
      exit(1);
    }
    
    if(aii > sum) {
      bigger++;
    }  
  }
  
  if( !bigger ) {
    printf("The matrix will not converge\n");
    exit(1);
  }
}


/******************************************************/
/* Read input from file */
/* After this function returns:
 * a[][] will be filled with coefficients and you can access them using a[i][j] for element (i,j)
 * x[] will contain the initial values of x
 * b[] will contain the constants (i.e. the right-hand-side of the equations
 * num will have number of variables
 * err will have the absolute error that you need to reach
 */
void get_input(char filename[]) {
  FILE * fp;
  int i,j;  
 
  fp = fopen(filename, "r");
  if(!fp) {
    printf("Cannot open file %s\n", filename);
    exit(1);
  }

  fscanf(fp,"%d ",&num);
  fscanf(fp,"%f ",&err);

  /* Now, time to allocate the matrices and vectors */
  a = (float**)malloc(num * sizeof(float*));
  if( !a) {
    printf("Cannot allocate a!\n");
    exit(1);
  }

  for(i = 0; i < num; i++) {
    a[i] = (float *)malloc(num * sizeof(float)); 
    if( !a[i]) {
		  printf("Cannot allocate a[%d]!\n",i);
		  exit(1);
  	}
  }
 
  x = (float *) malloc(num * sizeof(float));
  if( !x) {
    printf("Cannot allocate x!\n");
    exit(1);
  }


  b = (float *) malloc(num * sizeof(float));
  if( !b) {
    printf("Cannot allocate b!\n");
    exit(1);
  }

  /* Now .. Filling the blanks */ 

  /* The initial values of Xs */
  for(i = 0; i < num; i++) {
    fscanf(fp,"%f ", &x[i]);
  }
 
  for(i = 0; i < num; i++) {
    for(j = 0; j < num; j++) {
      fscanf(fp,"%f ",&a[i][j]);
    }
   
    /* reading the b element */
    fscanf(fp,"%f ",&b[i]);
  }
 
  fclose(fp); 
}

/*
    Returns 1 if the percent error between the new and old value is
      less than or equal to the given relative error
    Returns 0 otherwise
 */
int within_error(float newVal, float oldVal) {
  float error = fabs((newVal - oldVal) / newVal);
  if(error <= err) {
    return 1;
  } else {
    return 0;
  }
}


/************************************************************/


int main(int argc, char *argv[]) {

  int i;
  int nit = 0; /* number of iterations */
  int comm_sz, my_rank, my_first_i, my_last_i, local_num;
  int all_done = 0;

  
  if( argc != 2) {
    printf("Usage: gsref filename\n");
    exit(1);
  }
  
  /* Read the input file and fill the global data structure above */ 
  get_input(argv[1]);
 
  /* Check for convergence condition */
  /* This function will exit the program if the coffeicient will never converge to 
   * the needed absolute error. 
   * This is not expected to happen for this programming assignment.
   */
  check_matrix();

  //Initialize MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  local_num = num / comm_sz;

  do {

    /****OTHER PROCESSES****/

    if(my_rank != 0) {
      //printf("Enter Process %d\n", my_rank);
      int done = 0;

      float *local_new = (float *) malloc(local_num * sizeof(float));
      if( !local_new) {
        printf("Cannot allocate local new!\n");
        exit(1);
      }

      //printf("Still good\n");

      my_first_i = my_rank * local_num;
      my_last_i = (my_rank + 1) * local_num;

      //printf("my_first_i: %d\nmy_last_i: %d\n", my_first_i, my_last_i);
      int counter = 0;
      for(int i = my_first_i; i < my_last_i; i++) {
        //printf("Still good %f\n", calc_unknown(i + 1));
        local_new[counter] = calc_unknown(i + 1);
        //printf("Still good LOCAL %f\n", local_new[counter]);
        counter++;
      }

      //printf("Still good\n");
      counter = 0;
      for(int i = my_first_i; i < my_last_i; i++) {
        //printf("X[%d] = %f\n", i, local_new[counter]);
        if(within_error(local_new[counter], x[i]) == 0) {
          break;
        }
        if(i == my_last_i - 1) {
          done = 1;
        }
        counter++;
      }

      //printf("Still good\n");
      // printf("Sending: %f\n", *(&local_new));
      // printf("Sending: %f\n", local_new[0]);
      printf("Sending: %d\n", (&local_new[0]));
      MPI_Ssend(&local_new, local_num, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
      MPI_Ssend(&done, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
      //printf("Sent from process %d\n", my_rank);

    } else {

      /****MAIN PROCESS****/

      float *new = (float *) malloc(num * sizeof(float));
      if( !new) {
        printf("Cannot allocate new!\n");
        exit(1);
      }

      int *procs_done = (int *) malloc(comm_sz * sizeof(int));
      if( !procs_done) {
        printf("Cannot allocate processes done!\n");
        exit(1);
      }

      procs_done[0] = 0;

      my_first_i = my_rank * (num / comm_sz);
      my_last_i = (my_rank + 1) * (num / comm_sz);

      //printf("Still good RANK\n");

      //printf("my_first_i: %d\nmy_last_i: %d\n", my_first_i, my_last_i);

      for(int i = my_first_i; i < my_last_i; i++) {
        new[i] = calc_unknown(i + 1);
        //printf("%f\n", calc_unknown(i + 1));
      }

      //printf("Still good CALCS\n");

      for(int i = my_first_i; i < my_last_i; i++) {
        if(within_error(new[i], x[i]) == 0) {
          break;
        }
        if(i == my_last_i - 1) {
          procs_done[0] = 1;
        }
      }

      //printf("Still good ERROR CHECK\n");

      float *temp;// = (float *) malloc(local_num * sizeof(float));
      int status;

      for(int p = 1; p < comm_sz; p++) {
        MPI_Recv(&temp/*&new + (sizeof(float) * local_num * p)*/, local_num, MPI_FLOAT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Recieved\n");
        int counter = 0;
        for(int i = local_num * p; i < local_num * (p + 1); i++) {
          //new[i] = temp[i]/**(temp + sizeof(float) * counter)*/;
          printf("temp: %f\n", temp[i]);
          counter++;
        }
        MPI_Recv(&status, 1, MPI_INT, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        procs_done[p] = status;
        printf("Received\n");
      }

      //printf("Still good RECVS\n");

      for(int i = 0; i < num; i++) {
        printf("Received new[%d] = %f done\n", i, new[i]);
        x[i] = new[i];
      }

      //printf("Still good UPDATE X\n");

      for(int i = 0; i < comm_sz; i++) {
        printf("Process %d done: %d\n", i, procs_done[i]);
        if(procs_done[i] == 0) {
          break;
        }
        if(i == comm_sz - 1) {
          all_done = 1;
          printf("All done\n");
        }
      }
      printf("Still good CHECK ALL DONE\n");

    }
    nit++;
  } while(all_done != 1);

  MPI_Finalize();

  /* Writing to the stdout */
  /* Keep that same format */
  for(int i = 0; i < num; i++) {
    printf("%f\n",x[i]);
  }
 
  printf("total number of iterations: %d\n", nit);

  exit(0);
}
