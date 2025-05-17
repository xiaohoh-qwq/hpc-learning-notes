#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 1000000;
    int chunk = n / size;

    float *a = NULL;
    float *b = NULL;
    float *c = NULL;
    float *sub_a = (float*)malloc(chunk * sizeof(float));
    float *sub_b = (float*)malloc(chunk * sizeof(float));
    float *sub_c = (float*)malloc(chunk * sizeof(float));

    if (rank == 0) {
        a = (float*)malloc(n * sizeof(float));
        b = (float*)malloc(n * sizeof(float));
        c = (float*)malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) {
            a[i] = i * 1.0f;
            b[i] = i * 2.0f;
        }
    }

    MPI_Scatter(a, chunk, MPI_FLOAT, sub_a, chunk, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatter(b, chunk, MPI_FLOAT, sub_b, chunk, MPI_FLOAT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < chunk; i++) {
        sub_c[i] = sub_a[i] + sub_b[i];
    }

    MPI_Gather(sub_c, chunk, MPI_FLOAT, c, chunk, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("c[0]=%f, c[n-1]=%f\n", c[0], c[n-1]);
        free(a);
        free(b);
        free(c);
    }

    free(sub_a);
    free(sub_b);
    free(sub_c);

    MPI_Finalize();
    return 0;
}
